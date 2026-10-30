# HIVE - High-Processing Imaging Visualization Engine

A general-purpose fluorescence microscopy analysis pipeline: segmentation, intensity
quantification, arborization (Sholl) analysis, 3D reconstruction and rendering.

Works on any fluorescence imaging — cells, neurons, glia, organoids, tissue sections.
Nothing in the pipeline assumes a particular biology.

---

## What's new in 0.0.3

0.0.3 adds the nine items below to 0.0.2. The measurement engine and its intensity contract
are unchanged: the 92 tests of 0.0.2 still pass untouched, next to 107 new ones.

| Feature | What it does | Where |
|---|---|---|
| **Saved settings** | Every run writes `hive_run.json` into its output folder: all settings, the random seed, the HIVE release, the versions of the libraries, and a SHA-256 of each input file. Reload it in the app sidebar, or with `--settings`, to repeat the run; the output is identical. Settings this version does not know are refused rather than ignored. | `hive/provenance.py` |
| **Multi-channel + colocalization** | Find objects on one channel, measure more (`--extra-channels`). Each extra channel gets its intensity per object and Pearson's r and Manders' M1/M2 against the main channel, per object and per image. Channels are chosen by the name stored in the file. | `hive/channels.py` |
| **Group comparison** | Label images by condition (`--conditions conditions.csv`, or the table in the app). The report gets tests, effect sizes and plots. Images are the unit of replication by default (see below). | `hive/stats.py` |
| **Background preview** | Shows none / percentile / smooth / top-hat side by side on your own image before you run anything (app expander, or `--preview-baseline IMAGE out.png`). | `hive/baseline_preview.py` |
| **Large stacks** | A 3D stack that would not fit in memory is analysed in tiles instead of being refused. | `hive/tiled.py` |
| **More formats** | OME-TIFF, ImageJ TIFF, Zeiss CZI, Nikon ND2 and Leica LIF, with channel names and pixel size read from the file. `--inspect FILE` shows what a file contains. | `hive/formats.py` |
| **Export** | Labels as OME-TIFF (napari, Fiji) and every object as an ImageJ ROI in one zip (Fiji ROI Manager). | `hive/exports.py` |
| **Automated checks** | GitHub Actions runs the whole suite on every push and pull request, on Linux, Windows and macOS. | `.github/workflows/` |
| **Easier install** | Double-click launcher that installs everything on first start (see Install). | `launch_hive.py`, `HIVE.bat`, `HIVE.sh`, `environment.yml` |

### What was and was not verified

Tested with files and data created in the test suite: OME-TIFF, ImageJ TIFF, the tiled-versus-whole
agreement, ROIs rasterised back to the original masks, every statistic against SciPy or a hand
calculation, the settings round trip, the launcher's first-time install on Linux, and the whole
suite on Python 3.11, 3.12, 3.13 and 3.14 on Linux.

**Not verified**, so treat with care:

* **CZI, ND2 and LIF** are written against the libraries' documented APIs (`czifile`, `nd2`,
  `readlif`) and tested with stand-ins for those libraries, not with real microscope files. Run
  `python hive_cli.py --inspect yourfile.nd2` first and check the channels and pixel size it reports.
* **Fiji and napari** were not opened here. The ROI zip and OME-TIFF are checked by reading them back
  with `roifile` and `tifffile`.
* **Windows.** `HIVE.exe` is built and checked on a GitHub Windows runner on every release: the first
  start installs the libraries, the second start skips the install, and the app answers on its port.
  The test suite also passes on the Windows and macOS runners. Nobody has yet double-clicked it on a
  real Windows desktop, so the browser opening and the SmartScreen prompt are unchecked there.
* **macOS and Linux launchers** (`Run HIVE App.command`, `HIVE.sh`) are tested on Linux only; the
  macOS one is untested.

### Colocalization

For a channel `X` next to the main channel, per object and per image:
`coloc_X_pearson`, `coloc_X_manders_m1`, `coloc_X_manders_m2`, plus `X_intensity_mean` and
`X_integrated_intensity`. Extra channels get the same background subtraction as the main one and no
contrast enhancement. Pearson's r is NaN (not 0) when it is undefined (fewer than 3 pixels, or no
variation in a channel). Manders' thresholds are each channel's Otsu threshold; Costes' iterative
threshold is not implemented. Objects come from the main channel, so these numbers describe
colocalization *inside the main channel's objects*.

### Comparing conditions

* **Unit of replication.** Cells in one image are not independent, so the default compares one value
  per image (`--conditions` / the app's "Image" choice). Comparing objects is available but is
  labelled in the output as treating cells as independent, which makes p-values too small.
* **Choice of test.** A group needs at least 3 replicates for a p-value. Two groups: Welch's t-test
  only if both have at least 8 replicates and Shapiro–Wilk does not reject normality, otherwise
  Mann–Whitney U. With few replicates the smallest possible p-value can be large, so
  `min_possible_p` is shown beside the result. Three or more groups: Kruskal–Wallis overall, then every
  pair with Holm's correction.
* **Effect sizes**, always: Hedges' g with a 95 % confidence interval, and Cliff's delta. Positive
  means the first-listed group is higher.
* Output: `group_comparison/` (tables and a plot per metric) and the `Group_Stats` / `Group_Data`
  sheets of the Excel report.

### Large stacks

`large_image.mode` is `auto` by default: a 3D stack is tiled when ten times its float32 size (the
measured peak of the whole-volume pipeline) exceeds 85 % of RAM. `--tiling force` and
`--tile-size N` are there for testing. Each tile is read with a halo of extra pixels, and **one
threshold is computed over the whole volume** so every tile is cut at the same intensity. On the test
stack the tiled run found the same nine objects, with the same volumes and an identical mask, as the
whole-volume run.

Differences from a whole-volume run, all recorded in `hive_run.json`:

* CLAHE is switched off (it rescales per image, which would make one threshold mean different things
  in different tiles). Compare object counts with `--no-clahe` on your own data.
* Arborization / Sholl, extra channels, the 3D viewer, the GIF and the ROI zip are not produced.
  The label image is written as an OME-TIFF, which Fiji and napari open without loading it all.
* An object that reaches further from its centre than the halo can be cut at a tile edge. Those rows
  have `tile_clipped = True` and a warning is logged; raise `--halo` or exclude them.
* A compressed TIFF is decoded again for every tile, which is slow. Uncompressed TIFFs are read
  directly from disk.
* Only a single z-stack is supported (no time series or scenes), and only threshold segmentation.

### Formats

| Format | Channel names | Pixel size | Notes |
|---|---|---|---|
| OME-TIFF | from the OME-XML | from the OME-XML | tested |
| ImageJ TIFF | none (numbered) | from the TIFF tags | tested |
| Zeiss CZI | from the file's XML | from the file's XML | needs `czifile` |
| Nikon ND2 | from the file | from the file | needs `nd2` |
| Leica LIF | the lookup-table colour Leica stores | from the file | needs `readlif`; reads the first image in the file |

A file with several time points, scenes or tiles is refused with a message, not flattened. A pixel size
typed by the user always wins over the one in the file.

---

## Changes in 0.0.2 (the first `Improvements` commit)

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

Planned next steps. The first group is not implemented yet; the items of the old "Other ideas" list that 0.0.3 built are described under "What's new in 0.0.3".

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
| Python | 3.11 or newer | The suite passes on 3.11, 3.12, 3.13 and 3.14. 3.10 is not supported: the older `tifffile` it is limited to cannot write the OME label file the tiled mode uses. The one-click launcher installs Python 3.12 for you if needed. |
| pip + venv | any recent | `venv` ships with CPython. |
| RAM | ≥ 8 GB | The pipeline holds several copies of a volume at once. A 1 GB stack can peak at many GB — `hive.io_utils.memory_report` estimates this before a run and refuses stacks that would not fit. |
| OS | macOS / Linux / Windows | No compiled extensions of HIVE's own; everything comes from wheels. |

No system packages, JVM, or GPU are required. Core dependencies (installed automatically from
`requirements.txt`) are numpy, scipy, scikit-image, pandas, tifffile, imagecodecs, matplotlib,
pillow, plotly and openpyxl, the file-format readers czifile, nd2, readlif and roifile, plus streamlit / fastapi / uvicorn for the app and API, and pytest
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

### One step (no command line)

Put the HIVE folder anywhere and double-click the launcher for your system:

| System | Double-click | Needs |
|---|---|---|
| Windows | `HIVE.exe` (included) or `HIVE.bat` | `HIVE.exe` needs nothing installed; `HIVE.bat` needs any Python 3 installed. Both need an internet connection on the first start |
| macOS | `Run HIVE App.command` | Python 3 (preinstalled with the Xcode command line tools) |
| Linux | `HIVE.sh` (after `chmod +x HIVE.sh`) | Python 3 |

The first start takes a few minutes and about 1 GB of disk: it downloads `uv` (a small installer; its SHA-256 is checked),
creates `.venv` with Python 3.12 (downloading that Python if the computer has none) and installs
`requirements.txt` into it. Everything, including the download cache and any Python it fetches, stays inside the HIVE folder (`.venv` and `.hive_tools`). Later starts open HIVE straight
away, and the environment is rebuilt only if `requirements.txt` changes. `python launch_hive.py
--setup-only` installs without starting; `--reinstall` rebuilds from scratch.

`HIVE.exe` is `launch_hive.py` frozen with PyInstaller, so it runs on a computer without Python. A
copy sits in the HIVE folder next to `app.py`, so it is already inside the zip and tar.gz you download
from GitHub (and is also attached to the release). Double-click it. It is not signed, so Windows
SmartScreen may warn the first time ("More info", then "Run anyway"). To rebuild it, push to the
`exe-build` branch (the **Build launcher** workflow builds and tests it on Windows) or run
`pip install pyinstaller pillow` then `python packaging/build_exe.py` on Windows.

When the app is ready the launcher opens your browser. If none opens, it prints the address
(`http://localhost:8501`, or the next free port). If the app cannot start, the window stays open and
shows the last lines it printed.

With conda: `conda env create -f environment.yml`, then `conda activate hive`.

### With pip

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
python -c "import hive; print(hive.__release__)"    # 0.0.3
```

Verify the installation:

```bash
python -m pytest -q              # 199 tests, all should pass
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

Upload TIF/PNG/OME-TIFF/CZI/ND2/LIF, tune parameters in the sidebar, preview the background
methods on your image, inspect segmentation side-by-side with the raw image, compare conditions,
render a rotating 3D GIF, and download CSV, Excel, labels (OME-TIFF), ImageJ ROIs and the
`hive_run.json` that reproduces the run.

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

# What is in this file?  (channels, size, pixel size)
python hive_cli.py --inspect scan.nd2

# Compare the four background methods on one image before running anything
python hive_cli.py --preview-baseline scan.ome.tif preview.png --channel DAPI

# Two channels, colocalization, and a comparison of conditions
python hive_cli.py --input scans --channel DAPI --extra-channels GFP --conditions conditions.csv

# Repeat an earlier run exactly (settings, seed) from the record it saved
python hive_cli.py --settings old_results/hive_run.json --output rerun
```

`conditions.csv` has two columns, `image` (file name) and `condition`.

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
python -m pytest -q     # 199 tests
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
  provenance.py     saved settings and run records (hive_run.json)        [0.0.3]
  formats.py        OME-TIFF, ImageJ TIFF, CZI, ND2, LIF readers           [0.0.3]
  channels.py       extra channels and colocalization                      [0.0.3]
  stats.py          group comparison: tests, effect sizes, plots           [0.0.3]
  baseline_preview.py  side-by-side background methods                     [0.0.3]
  exports.py        OME-TIFF labels, ImageJ ROIs                           [0.0.3]
  tiled.py          tiled analysis of stacks larger than memory            [0.0.3]
app.py              Streamlit application
hive_cli.py         command line
launch_hive.py      one-step installer + launcher (HIVE.bat, HIVE.sh, Run HIVE App.command)
packaging/          builds HIVE.exe from launch_hive.py
environment.yml     conda environment
.github/workflows/  tests on every push; HIVE.exe build
prep_stack.py       stack preparation utility
api/main.py         FastAPI backend
tests/              ground-truth test suite (199 tests)
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
