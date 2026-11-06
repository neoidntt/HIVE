"""HIVE FastAPI backend.

Security note: the legacy ``/download/{filename}`` endpoint joined the
user-supplied name straight onto the output directory, so
``GET /download/../../../../etc/passwd`` served arbitrary files. Paths are now
resolved and checked to be inside the output directory before anything is read.

Concurrency note: the legacy ``/analyze-2d`` wiped shared upload/output
directories at the start of every request, so two simultaneous requests deleted
each other's data and returned each other's results. Each request now gets its
own isolated job directory.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from hive.config import ArborConfig, HiveConfig, PreprocessConfig, SegmentationConfig
from hive.hardware import get_backend
from hive.pipeline import analyze_folder

app = FastAPI(
    title="HIVE API",
    description="Fluorescence image analysis: segmentation, intensity, arborization.",
    version="2.0.0",
)

BASE = Path(__file__).resolve().parents[1]
JOBS_DIR = BASE / "api_jobs"


@app.get("/")
def root() -> dict[str, str]:
    """Report service status and detected hardware.

    Returns:
        Service metadata.
    """
    backend = get_backend()
    return {"name": "HIVE API", "version": "2.0.0", "status": "running",
            "compute": backend.name, "device": backend.device}


@app.post("/analyze-2d")
async def analyze_2d(
    file: UploadFile = File(...),
    channel: str = Form("auto"),
    method: str = Form("threshold"),
    threshold_scale: float = Form(1.0),
    min_size: int = Form(80),
    voxel_size_xy: float = Form(1.0),
    unit: str = Form("px"),
    arborization: bool = Form(True),
) -> JSONResponse:
    """Analyse a single uploaded 2D image.

    Args:
        file: The uploaded image.
        channel: Channel to analyse.
        method: ``threshold`` or ``cellpose``.
        threshold_scale: Otsu multiplier; 1.0 is unmodified Otsu.
        min_size: Debris threshold in pixels.
        voxel_size_xy: Physical pixel size; 1.0 reports pixels.
        unit: Unit label.
        arborization: Run branching/Sholl analysis.

    Returns:
        Job id, object count, measurement records and artifact names.

    Raises:
        HTTPException: 400 if the upload is unnamed or cannot be analysed.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Upload has no filename")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    input_dir, output_dir = job_dir / "input", job_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Use only the basename: an upload named "../../x.tif" must not escape.
    target = input_dir / Path(file.filename).name
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        cfg = HiveConfig(
            input_path=str(input_dir),
            output_folder=str(output_dir),
            channel=channel,
            voxel_size_xy=voxel_size_xy,
            voxel_unit=unit,
            preprocess=PreprocessConfig(),
            segmentation=SegmentationConfig(
                method=method, threshold_scale=threshold_scale, min_size=min_size
            ),
            arbor=ArborConfig(enabled=arborization),
        )
        result = analyze_folder(cfg)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    measurements = result["measurements"]
    return JSONResponse({
        "status": "complete",
        "job_id": job_id,
        "objects_detected": int(len(measurements)),
        "errors": result["errors"],
        "measurements": measurements.to_dict(orient="records"),
        "files": sorted(p.name for p in output_dir.iterdir() if p.is_file()),
    })


@app.get("/download/{job_id}/{filename}")
def download(job_id: str, filename: str) -> FileResponse:
    """Serve an artifact from a completed job.

    The resolved path is verified to lie inside the job's output directory, which
    defeats ``../`` traversal, absolute paths and symlink escapes alike -- the
    legacy endpoint joined the raw name onto the output folder and served
    whatever came out.

    Args:
        job_id: Job identifier returned by ``/analyze-2d``.
        filename: Artifact name.

    Returns:
        The file.

    Raises:
        HTTPException: 400 on a malformed id, 404 if the file does not exist or
            resolves outside the job directory.
    """
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Malformed job id")

    output_dir = (JOBS_DIR / job_id / "output").resolve()
    if not output_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unknown job id")

    candidate = (output_dir / filename).resolve()
    if not candidate.is_relative_to(output_dir) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)


@app.delete("/job/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    """Delete a job's uploads and results.

    Args:
        job_id: Job identifier.

    Returns:
        Deletion status.

    Raises:
        HTTPException: 400 on a malformed id, 404 if the job is unknown.
    """
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Malformed job id")
    job_dir = (JOBS_DIR / job_id).resolve()
    if not job_dir.is_relative_to(JOBS_DIR.resolve()) or not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unknown job id")
    shutil.rmtree(job_dir)
    return {"status": "deleted", "job_id": job_id}
