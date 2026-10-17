# HIVE - High-Processing Imaging Visualization Engine

A general-purpose fluorescence microscopy analysis pipeline: segmentation, intensity
quantification, arborization (Sholl) analysis, 3D reconstruction and rendering.

Works on any fluorescence imaging — cells, neurons, glia, organoids, tissue sections.
Nothing in the pipeline assumes a particular biology.

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
