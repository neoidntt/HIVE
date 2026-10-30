# HIVE - High-Processing Imaging Visualization Engine

A general-purpose fluorescence microscopy analysis pipeline: segmentation, intensity
quantification, arborization (Sholl) analysis, 3D reconstruction and rendering.

Works on any fluorescence imaging — cells, neurons, glia, organoids, tissue sections.
Nothing in the pipeline assumes a particular biology.

---

## Changes in the `Improvements` branch

This branch compares against `main` as follows. The analysis pipeline in `hive/` is unchanged
apart from one new module; the work is in the Streamlit app, a new test file and the notebooks.

### Streamlit app (`app.py`)

- **Built-in synthetic demo.** The sidebar has a new *Data source* choice, *Synthetic demo* or
  *Upload files*. The demo needs no uploads, so the app can be tried straight away in 2D or 3D mode.
  A *Synthetic data seed* input makes the sample images repeatable.
- **Runs itself.** `python app.py` now relaunches itself through `streamlit run` when it is not
  already running inside Streamlit.
- **Safer uploads.** Uploaded file names are reduced to their base name (no path tricks), and
  duplicate names get a numeric suffix instead of overwriting each other.
- **Stale results are cleared.** The app records a signature of the data source, mode, seed,
  settings and uploaded files. Changing any of them discards the old result and asks you to press
  *Run analysis* again, so a result never sits next to settings that did not produce it.
- **Cleaner runs.** The input and output folders are recreated on every run, and the app now shows
  any error from the pipeline, not only file, value and import errors.
- **Better 2D → 3D projection.** When a stack is collapsed to 2D, labels are taken from the plane
  where each pixel is brightest. Before, labels and intensity were each reduced with `max` separately.
- **New results display.**
  - *2D image inspection*: pick an image and see its measurement image next to its boundary overlay.
  - *Interactive measurement plots*: Plotly scatter plot with selectable axes, plus a histogram of
    mean intensity.
  - *3D point cloud*: an interactive view of segmented voxels, with a slider that limits how many
    points are drawn.
  - An **Excel report** download button (`HIVE_2D_report.xlsx` or `HIVE_3D_report.xlsx`) next to
    the existing downloads.

### New files

- `hive/synthetic.py` generates the reproducible demo data: three 160×160 fields with six
  cell-like objects each (dim, reference and bright), and a 20×90×90 anisotropic z-stack with a
  sphere and a bar.
- `tests/test_frontend_demo.py` tests the app with Streamlit's `AppTest`: the phantoms are
  reproducible, the 2D and 3D demos run end to end, and an uploaded TIFF runs through the 2D app.

### Launcher and notebooks

- `Run HIVE App.command` tries the project's own `.venv/bin/python` before any system Python.
- `HIVE_walkthrough.ipynb` is saved with its outputs (29 executed code cells). Same 47 cells as
  `main`, but the file grows from about 63 KB to about 2 MB.
- `Bee1004_DEMO.ipynb` was re-run on a machine where the example TIFF is missing, so its first
  cell now stores a `FileNotFoundError` instead of the file report. It needs re-running with the
  data present.
- `Bee1004_DEMO_annotated.ipynb` and `Bee1004_analysis.ipynb` only have notebook metadata changes
  (kernel name and language info).

## Future improvements

Planned next steps, roughly grouped. Nothing here is implemented yet.

### Planned

- **Better machine-learning segmentation (Cellpose).** Cellpose is already available as an optional
  `--method cellpose` in the command line. The next step is to bring it into the Streamlit app
  with a model picker, GPU use when available, and the option to fine-tune a model on a lab's
  own corrected masks. Add other models (such as StarDist) for comparison, and benchmark
  them against Otsu + watershed on the synthetic data.
- **Quality control.** Check every image before analysis and flag problems instead of silently
  measuring them: out-of-focus or blurry planes, saturated pixels, empty or nearly empty
  fields, heavy noise, uneven illumination, missing or implausible voxel sizes, and
  objects the pipeline could not identify with confidence. Show a pass/warn/fail summary
  in the app and write it to the report.
- **Upload a whole folder in the Streamlit app.** Let users point the app at a folder (or drop
  one in) instead of selecting files one by one, with batch progress, a file list and the
  option to skip files that do not match.
- **Interactive preprocessing and segmentation correction, optional.** After the automatic
  run, offer the user the choice to review and correct the result, or to accept it as is.
  Corrections would include adjusting preprocessing live with a preview, merging or splitting
  objects, deleting false detections, drawing missed ones, and re-measuring from the corrected
  labels. Corrections should be saved so they can be reused or used as training data.
- **Timelapse microscopy analysis.** Support time series (T×Z×Y×X) so users can watch how cells
  change over time:
  - track cell movement and report speed, distance and direction;
  - follow growth, in area or volume, per cell;
  - follow arborization over time (Sholl curves, branch counts and lengths, extension and
    retraction of processes), such as honeybee brain cells in action;
  - detect divisions, merges and cells entering or leaving the field;
  - show tracks as an animation and as per-cell plots.

### Other ideas

- **Saved settings and reproducibility.** Save every run's settings, software versions and seed
  in the output folder and reload them, so any result can be reproduced.
- **Multi-channel analysis.** Choose and measure more than one channel, with colocalization
  statistics between channels.
- **Group comparison and statistics.** Label images by condition and get comparison plots and
  basic statistics (effect sizes, with tests that suit the sample size) in the report.
- **Background subtraction preview.** Show the effect of each background method on the image
  before running the whole analysis.
- **Large images.** Tiled or chunked processing for stacks that do not fit in memory, rather than
  refusing them.
- **More input formats.** Direct support for more microscope formats (for example LIF, ND2 and
  OME-TIFF) with voxel size and channel names read from metadata.
- **Export options.** Export labels as ImageJ ROIs and OME-TIFF so results open in Fiji and napari.
- **Automated checks.** Continuous integration that runs the test suite on every push, and tests
  for the new features as they are added.
- **Easier install.** One-step packaging (a conda environment file or installer) so the
  app can be started without using the command line.

---

## Read this first: how HIVE measures

Fluorescence analysis asks two questions that want **opposite** things from the data:

| Question | Wants | Operations |
|---|---|---|
| *Where are the objects?* | Maximum contrast — distort freely | blur, CLAHE, normalize |
| *How bright are they?* | Zero distortion — keep the physical scale | background subtraction only |

HIVE builds **two images** and never confuses them:

```
raw → select_channel → baseline subtract ──→ MEASUREMENT image  (quantitative)
                                │
                                └→ blur → CLAHE → normalize ──→ SEGMENTATION image  (detection only)
```

Objects are **found** on the segmentation image and **measured** on the measurement image.
Enhancement never touches a reported number.

This matters more than it sounds. HIVE 1.x measured intensity on the enhanced,
per-image min-max normalized image. Measured on synthetic data with known answers:

| Scenario | True | HIVE 1.x reported | HIVE 2.0 |
|---|---|---|---|
| Two samples, 3× brightness difference | 3.00× | **1.00×** | 3.00× |
| Z-stack with a 33% axial gradient | 1.333× | **1.000×** | 1.333× |

A real biological difference reported as *no difference at all*. If you have results from
HIVE 1.x, **every intensity number in them is invalid** — object counts and areas are fine.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 minimum, 3.10+ recommended | 3.9 is the hard syntax floor (`list[Path]` built-in generics are evaluated at runtime in `frontend/streamlit_app.py`). The `hive/` package itself uses `from __future__ import annotations`, so its `float \| tuple[...]` unions are never evaluated. 3.10+ is recommended because current releases of the dependencies below require it. Verified on 3.14.5. |
| pip + venv | any recent | `venv` ships with CPython. |
| RAM | ≥ 8 GB | The pipeline holds several copies of a volume at once. A 1 GB stack can peak at many GB — `hive.io_utils.memory_report` estimates this before a run and refuses stacks that would not fit. |
| OS | macOS / Linux / Windows | No compiled extensions of HIVE's own; everything comes from wheels. |

No system packages, JVM, or GPU are required. Core dependencies (installed automatically from
`requirements.txt`) are numpy, scipy, scikit-image, pandas, tifffile, imagecodecs, matplotlib,
pillow, plotly and openpyxl, plus streamlit / fastapi / uvicorn for the app and API, and pytest
for the test suite. Each lower bound in `requirements.txt` marks a specific API HIVE relies on —
see the comments in that file.

**Optional extras**, none required and all degrade cleanly:

| Extra | Install | Enables |
|---|---|---|
| Cellpose | `pip install cellpose torch torchvision` | `--method cellpose` deep-learning segmentation. ~2 GB. Without it, that flag raises a clear `ImportError`; everything else works. |
| GPU | (comes with `torch`) | CUDA / Apple MPS detection in `hive.hardware`. Rarely worth it — the CPU path runs a 31 MB stack in ~1 s. |
| CZI files | `pip install czifile` | Reading Zeiss `.czi`. |
| Fiji bridge | `pip install pyimagej jpype1 scyjava jgo` | `--pyimagej-check`. Needs a JVM and Maven; isolated on purpose because it breaks often. |

---

## Install

Clone the repository and install into a virtual environment:

```bash
git clone https://github.com/neoidntt/HIVE.git
cd HIVE

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

For a reproducible analysis environment, install the exact tested versions instead. An
unpinned scientific stack means a result you cannot reproduce in six months:

```bash
pip install -r requirements-lock.txt
```

HIVE runs from the repository root — there is no `pip install .` step, because `hive/` is
imported directly by `app.py`, `hive_cli.py` and `api/main.py`. To import it from elsewhere,
add the repository root to `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/HIVE:$PYTHONPATH"
python -c "import hive; print(hive.__version__)"    # 2.0.0
```

Verify the installation:

```bash
pytest tests/ -q                 # 88 tests, all should pass
python hive_cli.py --hardware    # prints the detected compute device
```

---

## Quick start

### Interactive notebook (start here)

```bash
jupyter lab HIVE_walkthrough.ipynb
```

Self-contained: it generates its own phantoms with known ground truth, so it runs with no
data of your own. Explains the theory behind each step and includes an interactive
threshold explorer.

### Web app

```bash
streamlit run app.py
```

Upload TIF/PNG/CZI, tune parameters in the sidebar, inspect segmentation side-by-side with
the raw image, render a rotating 3D GIF, export CSV.

### Command line

```bash
# 2D batch, physical units
python hive_cli.py --mode 2d --input input_images --channel magenta \
    --voxel-size-xy 0.325 --unit um

# 3D reconstruction with a rotating GIF
python hive_cli.py --mode 3d --input input_zstacks/my_stack \
    --voxel-size-xy 0.2 --voxel-size-z 1.0 --unit um --gif

# What hardware will it use?
python hive_cli.py --hardware
```

### API

```bash
uvicorn api.main:app --reload    # docs at http://127.0.0.1:8000/docs
```

---

## What you get

**Per object:** area/volume (px and physical), centroid, bbox, eccentricity, solidity,
axis lengths, orientation, extent, `intensity_mean` / `max` / `min`, and
**`integrated_intensity`**.

> Prefer `integrated_intensity` over `intensity_mean`. Because background is already
> removed, it is the background-corrected integrated density — the *total* signal in the
> object. Mean intensity alone confounds a small bright object with a large dim one.

**Arborization** (when `--no-arbor` is not set): `neurite_length`, `n_branch_points`,
`n_endpoints`, `n_branches`, `max`/`mean_branch_length`, plus Sholl:
`sholl_max_intersections`, `sholl_critical_radius`, `sholl_enclosing_radius`,
`sholl_n_primary`, `sholl_ramification_index`, `sholl_regression_coeff_k`. Full
radius-vs-intersections curves are exported separately.

**3D:** volume + mask TIFFs with embedded voxel spacing (open correctly in Fiji/napari/Imaris),
max projections, interactive Plotly viewer, OBJ mesh, rotating GIF.

**Reports:** CSV, and a multi-sheet Excel workbook whose **Settings** sheet records every
one of the ~42 parameters that produced the run.

---

## Methods

| Step | Method |
|---|---|
| Background | Morphological white top-hat (grey opening) — ImageJ rolling-ball family. Also percentile and Gaussian. |
| Contrast | CLAHE (Zuiderveld, 1994) — segmentation image only |
| Threshold | Otsu (1979) |
| Splitting | Distance-transform watershed (Beucher & Lantuéjoul, 1979) |
| Skeleton | Zhang–Suen-family thinning (2D); Lee, Kashyap & Chu (1994) (3D) |
| Branch topology | Rutovitz crossing number (2D); 26-neighbour degree (3D), as Fiji AnalyzeSkeleton (Arganda-Carreras et al., 2010) |
| Sholl | Sholl (1953); semi-log regression coefficient per Ferreira et al. (2014) |
| Surface | Marching cubes (Lorensen & Cline, 1987) |
| Compositing | Porter & Duff (1984) *over* operator |
| Deep learning | Cellpose (Stringer et al., 2021) — optional |

---

## Things that will bite you

**`threshold_scale` is not part of Otsu.** It's an empirical multiplier. `1.0` is unmodified
Otsu; HIVE 1.x silently defaulted to `1.2`. Report whatever you used. If your object count
sits on a steep part of the count-vs-threshold curve, it's an artifact of the threshold, not
a property of your sample — the notebook plots this.

**Set your voxel sizes.** Defaults report pixels. Wrong voxel sizes silently give wrong
physical units *and* distorted 3D renders. Confocal z-steps are routinely 3–10× the xy
pixel size; HIVE resamples to isotropic before rendering, but only if you tell it the truth.

**Top-hat radius must exceed your largest real feature**, or real signal gets eaten as
background.

**Leave watershed off for branched cells.** It assumes convex objects and will shatter a
neuron into fragments.

**`channel="auto"` is a heuristic** (highest variance), not a measurement. The resolved
choice is recorded in the report — check it.

**`magenta` is a display-space proxy** (mean of R and B), not spectral unmixing. If you have
the raw single-channel acquisition, analyse that instead.

**Per-object Sholl assumes one arbor per object with a soma-like centre.** In dense neuropil
where processes from many cells intermingle, segmentation cannot assign a neurite to its
parent soma, and no skeleton analysis fixes that.

**Saved PNGs are contrast-stretched per image** — QC previews, not data. Read the CSV.

---

## Testing

```bash
pytest tests/ -q     # 88 tests
```

Every arborization phantom has an analytically known answer (a 4-arm star must report
exactly 4 tips and 1 junction), so a failure means the measurement is *wrong*, not merely
different. `tests/test_pipeline.py::TestIntensityFidelity` pins the intensity contract — if
those fail, the exported numbers are not fluorescence.

---

## Layout

```
hive/               the package
  config.py         one config for CLI, app and API
  preprocessing.py  channel selection + the two-image contract
  baseline.py       background subtraction
  segmentation.py   Otsu, watershed, Cellpose
  cleanup.py        debris/edge/hole filters, display normalization
  measurements.py   morphology + intensity
  arborization.py   skeleton, branch topology, Sholl
  neurites.py       bouton + axon detection
  render3d.py       volume rendering + GIF
  visualization.py  overlays, Sholl plots, 3D viewer
  reporting.py      CSV/Excel export
  pipeline.py       2D and 3D orchestration
  io_utils.py       readers/writers, memory guard
  hardware.py       CUDA/MPS/CPU detection
app.py              Streamlit application
hive_cli.py         command line
prep_stack.py       stack preparation utility
api/main.py         FastAPI backend
tests/              ground-truth test suite (88 tests)
presentation/       figure + slide-deck generation
HIVE_walkthrough.ipynb
src/                LEGACY 1.x — superseded, see below
frontend/           LEGACY 1.x Streamlit app — superseded by app.py
hive_app.py         LEGACY 1.x CLI — superseded by hive_cli.py
```

### Migrating from `src/`

`hive/` **replaces** `src/`. The old package is left in place only so you can compare
outputs; it is not maintained. Once you're satisfied, delete it.

The legacy modules are annotated with their specific failure modes. In summary:

- **Intensity values from `src/` are not fluorescence.** `src.preprocessing.preprocess`
  min-max normalizes and CLAHEs each image, and `src.pipeline` then reuses that same
  enhanced array as the intensity source for `regionprops`. Counts and areas are fine.
- **The 3D RGB path raises.** `src.reconstruction_3d.analyze_zstack` runs baseline
  subtraction on the raw array before channel selection, and
  `src.baseline_subtraction._validate` accepts only 2D or `(Z, Y, X)` input — so a
  channel-last `(Z, Y, X, C)` stack fails with `ValueError: Expected a 2D image or 3D
  z-stack`. The 4D branches in `zstack_to_volume` are unreachable through that path.
- **`frontend/streamlit_app.py` mishandles 3D results.** It assigns the
  `(measurements, summary)` tuple to a single name and renders it, and its
  `*_3d_view.html` glob never matches the `HIVE_..._3D_viewer.html` the pipeline writes,
  so the interactive-mesh link never appears.
- **`src` defaults `threshold_scale` to `1.2`**, not `1.0`. Object counts from 1.x and
  2.0 are not comparable unless you match this.

Since this repository is under version control, snapshot before deleting anything:

```bash
git checkout -b remove-legacy-src
git rm -r src frontend hive_app.py
git commit -m "Remove legacy 1.x package"
```

| HIVE 1.x | HIVE 2.0 |
|---|---|
| `hive_app.py` | `hive_cli.py` |
| `frontend/streamlit_app.py` | `app.py` |
| `src.pipeline.analyze_folder(config)` | `hive.pipeline.analyze_folder(cfg)` → dict |
| `src.reconstruction_3d.analyze_zstack(...)` → tuple | `hive.pipeline.analyze_zstack(cfg)` → `AnalysisResult` |
| `HiveConfig(flat)` | `HiveConfig` with nested `preprocess`/`segmentation`/`arbor`/`render` |
