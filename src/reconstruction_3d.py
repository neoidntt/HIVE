"""LEGACY (HIVE 1.x) 3D z-stack reconstruction.

.. deprecated:: 2.0
    Superseded by :func:`hive.pipeline.analyze_zstack`.

Known broken: the RGB z-stack path
----------------------------------
:func:`analyze_zstack` calls :func:`src.baseline_subtraction.apply_baseline_subtraction`
on the *raw* array before any channel selection happens. That function accepts
only 2D images or (Z, Y, X) stacks, so a channel-last (Z, Y, X, C) stack raises
``ValueError: Expected a 2D image or 3D z-stack`` (verified against this
package). The 4D branches inside :func:`zstack_to_volume` are consequently
unreachable through ``analyze_zstack`` -- they can only be exercised by
calling ``zstack_to_volume`` directly. Split RGB stacks into single channels
before use, or use ``hive.pipeline`` instead.

Intensity caveat
----------------
As in the 2D legacy path, volumes are measured on the enhanced, per-slice
normalised array produced by :func:`src.preprocessing.preprocess`, so 3D
intensity columns are not fluorescence. Volumes and counts are valid.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from skimage.measure import marching_cubes
import plotly.graph_objects as go
from .preprocessing import select_channel, preprocess, ensure_2d
from src.baseline_subtraction import apply_baseline_subtraction
from .io_utils import load_zstack, save_uint8
from .excel_report import export_excel_report


def normalize_volume(volume):
    """Rescale a volume to [0, 1] by its own min and max.

    Args:
        volume: Any array-like.

    Returns:
        ``float32`` array in [0, 1]; a flat volume is returned unchanged.

    Warning:
        Scaling by the true maximum makes the result hostage to a single hot
        voxel. It also rescales each volume independently, so intensities are
        not comparable between stacks.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if vol.max() > vol.min():
        vol = (vol - vol.min()) / (vol.max() - vol.min())
    return vol


def zstack_to_volume(zstack, channel="auto", gaussian_kernel=3):
    """Reduce a raw z-stack to a 3D (Z, Y, X) enhanced volume, slice by slice.

    Args:
        zstack: 2D (promoted to a single-slice stack), 3D (Z, Y, X), or 4D
            channel-last/channel-first array.
        channel: Channel to select for multi-channel input.
        gaussian_kernel: Passed through to :func:`src.preprocessing.preprocess`.

    Returns:
        3D ``float32`` volume, enhanced per slice.

    Raises:
        ValueError: On an unsupported shape.

    Note:
        Each slice is normalised and CLAHE'd *independently*, so a genuine
        axial intensity gradient through the stack is flattened away. This is
        why the 1.x path reports a 33% axial gradient as 1.000x.

        Both 4D branches are dead code when reached via :func:`analyze_zstack`
        -- see the module docstring.
    """
    arr = np.asarray(zstack)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    # If shape is Z,Y,X,C
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        slices = [preprocess(select_channel(arr[i], channel), gaussian_kernel) for i in range(arr.shape[0])]
        return np.stack(slices, axis=0)
    # If shape is C,Z,Y,X, use first channel-ish fallback
    if arr.ndim == 4 and arr.shape[0] <= 4:
        arr = np.moveaxis(arr, 0, -1)
        slices = [preprocess(select_channel(arr[i], channel), gaussian_kernel) for i in range(arr.shape[0])]
        return np.stack(slices, axis=0)
    if arr.ndim == 3:
        slices = [preprocess(ensure_2d(arr[i]), gaussian_kernel) for i in range(arr.shape[0])]
        return np.stack(slices, axis=0)
    raise ValueError(f"Unsupported z-stack shape: {arr.shape}")


def segment_volume(volume, threshold_scale=1.2, min_size=500, hole_size=100):
    """Threshold a volume with a single global scaled-Otsu cut.

    Args:
        volume: 3D array.
        threshold_scale: Multiplier on the Otsu threshold.
        min_size: Minimum object size in voxels.
        hole_size: Accepted for signature compatibility but **unused** --
            :func:`scipy.ndimage.binary_fill_holes` fills every enclosed
            cavity regardless of size, so this argument has no effect.

    Returns:
        Boolean 3D mask.

    Note:
        One threshold is applied to the whole stack. Confocal stacks usually
        dim with depth, so a global cut over-segments the top planes and
        under-segments the bottom ones.
    """
    vol = normalize_volume(volume)
    thresh = filters.threshold_otsu(vol) * float(threshold_scale)
    mask = vol > thresh
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=int(min_size))
    mask = ndi.binary_fill_holes(mask)
    return mask.astype(bool)


def measure_3d(labeled, intensity_volume, voxel_size_xy=1.0, voxel_size_z=1.0):
    """Measure per-object volume, centroid and intensity in 3D.

    Args:
        labeled: Integer 3D label image.
        intensity_volume: Volume sampled for intensity (see module caveat).
        voxel_size_xy: Physical XY voxel size.
        voxel_size_z: Physical Z step.

    Returns:
        :class:`pandas.DataFrame`, one row per object. ``volume_physical`` is
        ``voxel count * voxel_size_xy^2 * voxel_size_z``, so leaving the voxel
        sizes at 1.0 reports voxels, and setting them wrong silently reports
        wrong physical volumes.
    """
    rows = []
    voxel_volume = float(voxel_size_xy) * float(voxel_size_xy) * float(voxel_size_z)
    for r in measure.regionprops(labeled, intensity_image=intensity_volume):
        rows.append({
            "label": int(r.label),
            "volume_voxels": int(r.area),
            "volume_physical": float(r.area) * voxel_volume,
            "centroid_z": float(r.centroid[0]),
            "centroid_y": float(r.centroid[1]),
            "centroid_x": float(r.centroid[2]),
            "mean_intensity": float(r.mean_intensity),
            "max_intensity": float(r.max_intensity),
        })
    return pd.DataFrame(rows)


def export_obj(path, verts, faces):
    """Write a triangle mesh as a Wavefront OBJ file.

    Args:
        path: Destination ``.obj`` path.
        verts: ``(N, 3)`` vertices in marching-cubes (Z, Y, X) order; written
            reversed to (X, Y, Z) for conventional 3D viewers.
        faces: ``(M, 3)`` vertex indices, 0-based; incremented by 1 because
            the OBJ format indexes from 1.
    """
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[2]:.6f} {v[1]:.6f} {v[0]:.6f}\n")
        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a} {b} {c}\n")


def create_3d_html(path, volume, mask, voxel_size_xy=1.0, voxel_size_z=1.0, step_size=2):
    """Render the mask surface to a standalone interactive Plotly HTML page.

    Extracts an isosurface with marching cubes (Lorensen & Cline, 1987) at
    level 0.5 -- the midpoint of a binary mask -- using anisotropic ``spacing``
    so the mesh is in physical units rather than voxels.

    Args:
        path: Destination ``.html`` path.
        volume: Accepted for signature compatibility; **unused**. The surface
            is built from ``mask`` alone and coloured by Z height.
        mask: Boolean 3D mask to surface.
        voxel_size_xy: Physical XY voxel size.
        voxel_size_z: Physical Z step.
        step_size: Marching-cubes stride; higher gives a coarser, smaller mesh.

    Returns:
        ``(verts, faces)``, or ``(None, None)`` when the mask is empty (an
        explanatory placeholder page is still written).

    Note:
        Written with ``include_plotlyjs="cdn"``, so the page needs internet
        access to render. The 30 MB self-contained viewer in this repository's
        demo output came from the 2.0 renderer, not from here.
    """
    if not mask.any():
        fig = go.Figure()
        fig.update_layout(title="HIVE 3D Viewer: no segmented volume detected")
        fig.write_html(str(path))
        return None, None

    spacing = (float(voxel_size_z), float(voxel_size_xy), float(voxel_size_xy))
    verts, faces, normals, values = marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing, step_size=max(1, int(step_size)))
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.55,
            intensity=verts[:, 0],
            colorscale="Plasma",
            name="Segmented PDF-positive volume",
        )
    ])
    fig.update_layout(
        title="HIVE Amira/Imaris-style 3D Reconstruction Viewer",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(str(path), include_plotlyjs="cdn")
    return verts, faces


def analyze_zstack(zstack_path, output_folder="output_results", channel="auto", threshold_scale=1.2, min_size=500, hole_size=100, voxel_size_xy=1.0, voxel_size_z=1.0, gaussian_kernel=3, mesh_step_size=2):
    """Run the full legacy 3D pipeline and write every artifact to disk.

    Steps: load -> top-hat baseline subtraction -> per-slice enhancement ->
    global Otsu -> label -> measure -> export TIFF/PNG/CSV/XLSX/HTML/OBJ.

    Args:
        zstack_path: Multi-page TIFF or folder of slice images.
        output_folder: Destination directory, created if absent.
        channel: Channel to analyse for multi-channel input.
        threshold_scale: Multiplier on the Otsu threshold.
        min_size: Minimum object size in voxels.
        hole_size: Forwarded to :func:`segment_volume`, where it is unused.
        voxel_size_xy: Physical XY voxel size.
        voxel_size_z: Physical Z step.
        gaussian_kernel: Smoothing kernel for per-slice enhancement.
        mesh_step_size: Marching-cubes stride.

    Returns:
        ``(measurements, summary)`` DataFrames.

    Raises:
        ValueError: For channel-last RGB stacks -- see the module docstring.

    Note:
        Callers must unpack **two** values. ``frontend/streamlit_app.py``
        assigns the result to a single name and then renders the tuple, which
        is a latent bug in that legacy front end.

        The top-hat radius is hard-coded to ``(2, 25, 25)`` voxels and cannot
        be configured, unlike ``hive.baseline``.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    raw = load_zstack(zstack_path)

    # Baseline runs on the RAW array, before channel selection -- this is the
    # step that rejects (Z, Y, X, C) input outright.
    raw, baseline_info = apply_baseline_subtraction(
        raw,
        method="tophat",
        tophat_radius=(2, 25, 25),
        normalize=False,
    )
    volume = zstack_to_volume(raw, channel=channel, gaussian_kernel=gaussian_kernel)
    mask = segment_volume(volume, threshold_scale=threshold_scale, min_size=min_size, hole_size=hole_size)
    labeled = measure.label(mask)
    measurements = measure_3d(labeled, volume, voxel_size_xy=voxel_size_xy, voxel_size_z=voxel_size_z)
    summary = pd.DataFrame([{
        "object_count": int(len(measurements)),
        "total_volume_voxels": float(measurements["volume_voxels"].sum()) if not measurements.empty else 0,
        "total_volume_physical": float(measurements["volume_physical"].sum()) if not measurements.empty else 0,
    }])

    tifffile.imwrite(str(output_folder / "HIVE_3D_volume.tif"), normalize_volume(volume).astype(np.float32))
    tifffile.imwrite(str(output_folder / "HIVE_3D_mask.tif"), mask.astype(np.uint8) * 255)
    save_uint8(output_folder / "HIVE_3D_max_projection.png", np.max(volume, axis=0))
    save_uint8(output_folder / "HIVE_3D_mask_projection.png", np.max(mask, axis=0).astype(float))
    measurements.to_csv(output_folder / "HIVE_3D_measurements.csv", index=False)
    summary.to_csv(output_folder / "HIVE_3D_summary.csv", index=False)
    export_excel_report(output_folder / "HIVE_3D_report.xlsx", summary, measurements, settings={
        "mode": "3d", "channel": channel, "threshold_scale": threshold_scale,
        "min_size": min_size, "voxel_size_xy": voxel_size_xy, "voxel_size_z": voxel_size_z,
    })
    verts, faces = create_3d_html(output_folder / "HIVE_Amira_Imaris_style_3D_viewer.html", volume, mask, voxel_size_xy, voxel_size_z, mesh_step_size)
    if verts is not None:
        export_obj(output_folder / "HIVE_3D_mesh.obj", verts, faces)
    print("3D reconstruction complete.")
    print(f"Files saved in: {output_folder}")
    return measurements, summary
