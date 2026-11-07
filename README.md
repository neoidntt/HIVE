# HIVE - High-Processing Imaging Visualization Engine

A general-purpose fluorescence microscopy analysis pipeline: segmentation, intensity
quantification, arborization (Sholl) analysis, 3D reconstruction and rendering.

Works on any fluorescence imaging — cells, neurons, glia, organoids, tissue sections.
Nothing in the pipeline assumes a particular biology.

---

## What's new in 0.0.5

0.0.5 is 0.0.4 plus four changes to the app, and one fix to the tiled analysis that turned up while checking them.

* **Don't save results.** With the data source *File or folder on this computer* there is a checkbox, **Don't save
  results to this computer**. The analysis runs and the results appear on screen, but no results folder, tables,
  pictures or exports are written, and downloads (CSV, Excel, settings) are made in memory. Only a stack too large for
  memory needs a disk file while it is analysed in tiles; that goes in a scratch folder that is removed before the
  analysis returns, so its region and 3D views (which need the label image) are not available in that case. From the
  command line and Python, the same switch is `save=False`.
* **2D and 3D options are separate.** Options that only mean something for one kind of analysis are greyed out (not
  removed) for the other, and flip the moment you switch the Mode: the Z step, the 3D render settings, the rotating GIF
  and "each file is its own stack" are 3D only; the parallel-workers slider is 2D only; everything else applies to both.
* **A professional volume viewer.** The 3D volume view (`hive/volume_view.py`) keeps rotate / zoom / pan and now has:
  alpha-composited volume rendering, a surface mode and a brightest-voxel mode; Blinn-Phong lighting with an aimable light
  (or a light that follows the camera), specular, soft shadows and ambient occlusion; brightness, black and white points,
  gamma, opacity, colour maps and a colour for each channel; a crop box and two free cutting planes; full quality when
  still and a cheaper render while the view moves; background colours, orientation axes, a scale bar, a bounding box,
  auto-rotate and PNG screenshots. The controls are grouped and collapsed so the picture stays clear. It is a
  self-contained WebGL2 page rather than VTK.js, three.js or Plotly: it needs no download and no internet (so it also
  works from the saved `HIVE_3D_volume_viewer.html`), it was already verified on the real stack, and Plotly's volume
  type sends coordinates for every voxel, which is far too heavy at these sizes.
* **The 3D segmentation is real 3D.** The old view drew a random 12,000 of the segmented voxels as dots. Dots have no
  surface, no light and no depth cue, and a thin stack drawn true to scale puts them all in a slab a few percent as deep as
  it is wide, so it read as a flat picture on a plane. Now every region is a closed surface built with marching cubes from
  all the z-planes it spans (`hive/regions3d.py`, `hive/regions_view.py`, saved as `HIVE_3D_regions_viewer.html`). Each
  region has its own colour, a show / hide switch and its own transparency (order-independent, so see-through regions look
  right from every angle); click a region in the picture to find it in the list. For a stack analysed in tiles the
  surfaces are built from the full-resolution label image, reading only around each region.
* **Fix: objects at tile seams were counted twice.** An object lying where two tiles overlap is segmented by both,
  and both could keep it. On the real 4.8 GB stack, at the same tile size, 788 objects were listed, 16 of them with no voxels
  in the label image; now 724 are listed, every one in the image, and the 64 repeats are reported in the notes. Tiled
  counts now agree with whole-volume counts on test stacks (before, they were too high). Results from 0.0.4 tiled runs
  of large objects are therefore a few percent too high.

## What's new in 0.0.4: faster analysis

0.0.4 does the same analysis as 0.0.3, much faster. Nothing about the results changed: on a test set of
8 RGB photographs (2000 x 1100-1270 px, 889 objects) all 85 output files (every table, the Excel report's
data, all 56 PNGs pixel for pixel, the OME-TIFF labels and the ROI zips) are identical to the 0.0.3 run, apart
from the output folder path written into the report's Settings sheet.

| 8 images, default settings, 4-core computer | time |
|---|---|
| 0.0.3, one worker | 232 s |
| 0.0.3, uploaded in the app (its default of 2 workers) | 141 s |
| 0.0.4, one worker | about 50-65 s |
| 0.0.4, uploaded in the app (4 workers, chosen automatically) | about 26 s |

The timings were taken on a computer that was also busy with other work, so treat them as rough; a quiet
machine should do a little better. Speed also depends on your images: it grows with pixel count and with how
much branching the objects have.

### The 3D view is now a real, rotatable volume

The reconstruction used to look flat for three reasons. The rotating GIF is a maximum-intensity projection, which keeps
only the brightest voxel along each line of sight, so it has no depth cue (on the bee stack it correlates 0.93 with one
single plane); a GIF cannot be rotated or zoomed; and the "interactive" plot showed only 12,000 random dots of the
segmented objects, not the tissue. The sample itself is also thin (109 planes against 4953 pixels wide, 2.2%).

Now every 3D result, including tiled ones, has a **3D volume (rotate, zoom, pan)** view built from every z-plane and
saved as `HIVE_3D_volume_viewer.html` (it also opens by itself in a browser with no internet and no app). Drag to
rotate, scroll to zoom, right-drag or Shift+drag to pan. Nearer structures hide farther ones and surfaces are shaded, so
depth can be seen; sliders set depth exaggeration (thin samples are stretched, 1 = true to scale), brightness, how
faint a signal is hidden, opacity, which planes to show, and colours; segmented objects can be highlighted; "Brightest
voxel" gives the GIF's look. Big volumes are averaged in x and y to about 12 million voxels, never in z. It needs a
browser with WebGL2 turned on (current Firefox, Chrome, Edge, Safari); otherwise the page says so. For looking only:
measurements always come from the full analysis. Verified in a browser on the real 4.8 GB stack (rotate, zoom, side view).

### Which channel did "auto" choose?

When the channel is left on **auto**, HIVE picks the channel with the most contrast, and now tells you which one after
the run: a blue note in the app (2D batch, 3D result and folder-of-stacks views), a `note:` line on the command line,
and the Notes sheet of the Excel report and `hive_run.json`. Nothing is added when you name a channel yourself or the
image is grayscale. In a tiled folder of colour slices the choice is made once, on the first slice, and used for all
slices (before, each slice could pick a different channel).

### The app no longer dies when the computer runs out of memory

When memory runs out, the operating system ends the biggest process. Until now that was the web app itself, and the
browser showed only "Connection error. Is Streamlit still running?". A 2D batch (and a folder of stacks) now runs in
a process of its own (`hive/isolate.py`), so the analysis is the one that gets ended: the app stays up and says
"The analysis ended unexpectedly... almost certainly because the computer ran out of memory", and you can run it
again at once. The app itself also stays small while it works: a 32-megapixel JPEG took the app 1.7 GB before and about
0.4 GB now. This does not make more memory appear. If your computer is short, close other programs (a browser with
many tabs is often the biggest user) or analyse fewer images at a time. A single 3D stack still runs inside the app,
because its views need the whole result.

### Viewing a stack that was analysed in tiles

A stack too big for memory used to give only tables and a label file. The app now builds its views from the files on
disk, so they work on a laptop:

* **Region (full resolution):** pick a plane and a square region (256 to 2048 px) and see the raw pixels next to
  the segmentation. It reads only that rectangle (0.2 s on a 4.8 GB stack).
* **Whole stack (reduced):** press *Load the whole-stack overview* once (36 s on a 4.8 GB stack). It keeps every
  n-th pixel of every plane (about 30 million voxels; n = 7 for that stack) and gives a plane-by-plane view and a
  max projection.
* **3D view and GIF:** the interactive point view and the rotation GIF are built from the reduced copy (the GIF
  took about 60 s; the whole thing stayed under 0.6 GB).

These are for looking only; every measurement comes from the full-resolution analysis. The reduced views show the raw
image, not the background-corrected one the measurements use.

### A folder of whole stacks, one after another

In 3D mode a folder normally means "the slices of one stack, one plane per file". If every file is a whole stack,
tick **Each file in this folder is its own stack** (the app ticks it for you when it sees multi-plane files), or on
the command line add `--each-file`:

```bash
python hive_cli.py --mode 3d --each-file --input my_stacks --output my_results
```

The stacks run one at a time, on purpose: one stack already uses every core, and two at once would double the
memory. Each stack gets its own folder (`my_results/<file name>/`, exactly as if run alone), the tables of all stacks
are also combined in `combined_3d_measurements.csv` and `combined_3d_summary.csv`, and a file that fails is listed
while the others carry on. In the app you can pick any finished stack and look at it (region, reduced overview,
3D view, GIF). Without the option, a folder of multi-plane files stops straight away with a message, instead of
trying to read every file whole.

### 3D analysis is much faster (second speed-up in 0.0.4)

The 3D analysis spent most of its time on one number: **solidity** (object volume divided by the volume of its
convex hull). scikit-image tests every voxel of an object's bounding box against every face of the hull, so one
object that spans the volume cost minutes. HIVE now finds, for each row of voxels, the stretch that lies inside the
hull and counts it. It uses the same hull and the same inequality, so the numbers are identical: checked on 169
real objects (maximum difference 0.0) and 300 random shapes. Other changes, all checked to give the same result as
the code they replace: hole filling looks up the nearest object only near each hole, the opening and closing of the
mask use shifted array operations (8 times faster), and the branch neighbour count is a separable sum.

| 3D, 4-core computer | before | now |
|---|---|---|
| 109 x 512 x 512 crop, whole volume in memory | 281 s (2.5 GB) | 81 s (1.0 GB) |
| the same crop, in tiles | 93 s | 19 s |
| a real 109 x 2217 x 4953 stack (4.8 GB as float32), in tiles | 1727 s | about 250 s |

In-memory 3D and 2D outputs are byte-identical to before (all tables and the label image).

One behaviour changed on purpose in tiled runs. The global threshold is estimated from a sample of the image. The
sample used to be random per tile, so the threshold (and a few objects) depended on the tile size, which depends on
how much memory is free when you start. It is now every n-th pixel counted from the image corner, so it does not
depend on tiling, and a volume smaller than 20 million voxels is used in full.

### 2D analysis: second round of speed-ups (0.0.4)

On a 20-megapixel image with 1,469 objects the whole run went from 20.5 s to 16.2 s, with every output file
(tables, all PNGs pixel for pixel, OME-TIFF, ROI zip) identical to the run before. What changed, each checked to give
exactly what the code it replaces gave:

* **CLAHE** (`hive/clahe.py`) is scikit-image's algorithm with its last step spread over threads: 4.3 s to about
  2.0 s on that image, and lower peak memory. On first use it checks itself against scikit-image on a small image and
  falls back to scikit-image's own function if they ever differ (for example after an upgrade).
* **PNG files** are written straight through zlib: about twice as fast, the same pixels, and the files are a little
  smaller. Each picture is now built and encoded inside one worker thread, as many at once as free memory allows.
* **Overlays:** the label overlay uses a table lookup instead of a binary search (3.0 s to under 1 s), and the thick
  boundary outline uses shifted comparisons instead of grey dilation and erosion.

What is left is spread thinly (measurement, branch analysis, cleanup, each about 2 s on that image), so further gains
would come from running more images at once rather than from one image.

Where the 2D time went, and what changed (first round):

* **Branch measurement (the biggest cost).** To measure each branch of a skeleton the code built a mask the size
  of the whole image and re-scanned it, once per branch, which is thousands of full-image passes for a dense
  arbor. All branch lengths are now computed in one pass, with the same arithmetic.
* **Spur pruning and Sholl analysis** work on the skeleton's own pixels instead of whole images and whole shells.
* **One skeleton, not two.** The combined skeleton drawn on the overlay is assembled from the per-object skeletons
  that were already computed, instead of skeletonising the whole mask again. (With `--watershed` the old route is
  kept, so that overlay is unchanged.)
* **Pictures.** QC PNGs are written at PNG compression level 1 (same pixels, about 3 times faster to encode, files a
  little larger) and several are written at once while the OME-TIFF / ROI export runs.
* **Workers are automatic.** A 2D batch now uses one worker per CPU core by default, limited by the number of images
  and by memory (the app slider and `--workers auto`). Before, the app used 2 on a 4-core machine and the command
  line used 1. `--workers 1` still runs everything in one process.
* **The app no longer re-zips your results on every click.** It used to rebuild the 69 MB results zip (about 6 s)
  every time any widget on the results page was touched. The zip is now built only when you press the download
  button, and already-compressed files are stored rather than compressed again. The Sholl chart draws many curves
  as one line trace and is cached; a results-page click went from about 5 s to about 1 s.
* **Excel report and Sholl table** are written faster (`xlsxwriter` is now in `requirements.txt` and used when
  installed; the report has the same sheets and values, and `openpyxl` is the fallback).

### Very large files (25 GB and more)

**Do not upload them through the browser.** Streamlit keeps every uploaded file in the computer's memory
(there is no option to stream it to disk), so a 25 GB upload would use up the RAM and the operating system would
kill the app, the same "Connection error" as above. The upload box is limited to 1 GB for that reason.

Instead choose **Data source: File or folder on this computer** in the sidebar, then **Choose a file...** /
**Choose a folder...** (or type or paste a path). HIVE opens the data where it is: nothing is uploaded or
copied, so there is no size limit. Results are written to **Save results in** (default `~/HIVE_results`), in a
new sub-folder for every run; nothing that is already there is touched or deleted. From the command line it is
the same: `python hive_cli.py --input /data/huge.tif --output /data/results`.

How a file larger than memory is analysed:

* **In tiles.** The image is cut into tiles that each fit comfortably in memory, together with a border (halo) so
  background subtraction sees the same neighbourhood as in a whole-image run. Several tiles run at the same time
  (one process per core, at most 4 by default; `--workers N` or the `large_image.workers` setting) and tile size is
  chosen from the memory that is free, so the total stays bounded. A progress bar (or progress lines on the command
  line) shows how far a long run has got.
* **One threshold for the whole image**, found in a first pass over all tiles, so every tile is cut at the same
  intensity. Each object is counted once, by the tile that contains its centre.
* **What is read:** 3D stacks and single huge 2D images in **TIFF / OME-TIFF** (BigTIFF too), a folder of slice
  images, and multi-channel OME-TIFF by channel name. An uncompressed TIFF is read straight from disk. A
  **compressed or tiled** TIFF is read in pieces through the `zarr` package (now in `requirements.txt`); without
  it HIVE refuses a compressed plane that would not fit. **CZI, ND2 and LIF** can only be read whole: if one is
  larger than the free memory, export it as an OME-TIFF (Fiji or the microscope software) and open that.
* **What you get:** the measurement table (`<name>_measurements.csv`, with a `tile_clipped` column), the Excel
  report, `hive_run.json`, and the labels as an OME-TIFF that opens in Fiji or napari. The label file is created
  at its full size but only the parts holding objects are written, so it occupies far less disk than it appears to.
  Because the results can be many GB, the app leaves them in the results folder instead of offering a download.
* **What is not done** (the same as the 3D tiled mode): CLAHE is off, and there is no branching/Sholl analysis,
  overlays, ImageJ ROIs or extra-channel colocalization. An object larger than the halo can be cut at a tile edge;
  those rows are flagged `tile_clipped`.
* **Millions of objects.** A very large image can hold more objects than an Excel sheet takes (1,048,575 rows). The
  Excel sheet then holds a note instead, and the complete table is in the CSV. The app shows the first 20,000 rows and
  plots a random 20,000 objects, and says so.

Measured (4-core laptop, 16 GB RAM, other programs running, uncompressed uint16 BigTIFF mosaics of bright round
cells on a noisy background):

| File | Pixels | Time | Peak memory (all processes together) | Objects found / planted |
|---|---|---|---|---|
| 0.29 GB | 144 million | 27 s | 1.8 GB | 475 / 480 (5 touching pairs merge) |
| 3.2 GB | 1.6 billion | 223 s | 1.8 GB | 1600 / 1600 |
| **26.0 GB** | **13.0 billion** | **1669 s (28 min)** | **2.1 GB** | **4558 / 4560** (2 overlapping pairs merge) |

Speed is roughly 8 million pixels (16 MB of 16-bit data) per second on four cores, so expect about half an hour
per 25 GB; the first pass reads the whole file once (11 of the 28 minutes) and the second reads it again. The
26 GB file's labels took 418 MB of real disk (49 GB apparent), and its CSV and Excel report were a few hundred KB. A compressed file is slower, and a
file on a network drive is limited by the network.

**Not verified:** compressed or pyramidal slide-scanner formats (SVS, NDPI) at this size; the interactive file
dialogs on Windows and macOS (the commands are tested, not clicked) and the Linux one only as a command line;
whole-slide RGB TIFFs (an RGB plane is not supported by the tile reader: convert to one channel or an OME-TIFF
with channels); and any 3D stack of 25 GB, which uses the same tiles as the 2D test but was tested only at
a few hundred MB. Run a small crop first to check the settings before committing to a long run. Keep the browser tab open during a long run: closing it stops the analysis.

### Large images no longer crash the app (fix in 0.0.4)

A 66-megapixel photograph (10555 x 6231 px RGB) made the analysis use more than 8 GB, and when the computer
had less than that free, the operating system killed the whole Streamlit process. In the browser that shows
as **"Connection error. Is Streamlit still running?"**, with no explanation. It is fixed three ways:

* **Much less memory, same results.** That image now needs about 4.0 GB at its peak (2.6 GB without CLAHE),
  and takes about a minute. The savings: the overlay pictures are built and written one at a time instead of
  all at once and in bands of rows; images are loaded in the type they are stored in (a JPEG is 0.2 GB as
  stored, 0.8 GB as float32) and converted one channel at a time; hole filling no longer allocates an
  unused full-size array. Every one of these was checked against the calculation it replaced (tables,
  overlay pixels, channel choice, hole filling), and the 8-image comparison is still identical.
* **A check before each image.** HIVE now estimates the memory an image needs from its header and compares
  it with the memory that is free *right now* (not the size of the computer, which is how the earlier tiling
  and worker choices went wrong when a browser was using half of it). If it fits, nothing changes. If it
  fits only without CLAHE, CLAHE is switched off for that image and a note says so in the app, in the
  Excel report (`Notes` sheet) and in `hive_run.json`; measurements are never taken from the CLAHE image,
  but the objects found can differ a little. If it cannot fit either way, that image is skipped with a message
  saying how much it needs and how much is free; the rest of the batch still runs.
* **Free memory also decides the number of workers** and when a 3D stack is analysed in tiles.

The estimate is from the image header, so it is approximate (a Linux computer reports free memory exactly;
Windows is read from the system; macOS falls back to the machine's total). If you still see "Connection error"
on a very large image, close other programs (a web browser can hold several GB) or crop / downsample the image.

What 0.0.4 does not change: images are still analysed in full detail. If you want it faster still, the options
from before apply: untick **Analyse branching / Sholl** (the largest remaining cost), untick CLAHE, and untick the
exports you do not need.

**Not verified:** the timings on Windows or macOS, and on machines with fewer than 4 cores (automatic workers then
use fewer). The automatic worker count is limited by a memory estimate taken from each image's header; for CZI, ND2
and LIF files that estimate is not available, so check `--workers` yourself for very large files of those types.

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
  macOS one is untested. `HIVE.sh` was tested on Linux with no Python on the PATH (it fetched `uv`, Python 3.12 and
  all libraries and started), with a terminal opened for it, and with the shortcut files; `HIVE.bat` was run under
  Wine's command interpreter (Python found, a `.bat` Python shim, Python missing, a crash exit code), but its
  PowerShell download step (`packaging/bootstrap_uv.ps1`) has not been run on Windows.

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
| Windows | `HIVE.exe` (included) or `HIVE.bat` | Nothing installed. Both need an internet connection on the first start |
| macOS | `Run HIVE App.command` | Nothing installed (a Python you already have is used if there is one) |
| Linux | `HIVE.sh` | Nothing installed; `curl` or `wget` for the first start if there is no Python 3.10+ |

`HIVE.sh` and `HIVE.bat` behave like `HIVE.exe`:

* **They check, then start.** If the libraries are already in `.venv` they open HIVE straight away; if not, they
  download what is missing and then open it.
* **They need no Python.** If the computer has Python 3.10 or newer it is used to run the launcher; if not,
  the script downloads `uv` itself (checksum checked; `HIVE.bat` does it with `packaging/bootstrap_uv.ps1`), and
  `uv` brings its own Python 3.12. All of it stays inside the HIVE folder.
* **Started without a window, they open one.** When a Linux file manager runs `HIVE.sh` with no terminal, the
  script opens one (so you can see the progress and any message) and runs itself there.
* **A desktop shortcut, offered once.** When HIVE starts (`HIVE.exe`, `HIVE.bat` or `HIVE.sh`) and there is no HIVE
  shortcut on the desktop, it asks **Add HIVE to your desktop? (Yes / No)**. **Yes** makes the shortcut (`HIVE.lnk` on
  Windows, `HIVE.desktop` on Linux, `HIVE.command` on macOS) and never asks again. **No** leads to **Remind you next
  time you open HIVE? (Yes / No)**: Yes asks again at the next start; No stops the question for good. The answer is
  kept in `.hive_settings.json` next to the launcher (delete the file to be asked again). If a shortcut is already on
  the desktop, nothing is asked. On a Mac, `Run HIVE App.command` is different: whenever there is no shortcut it asks
  **Add HIVE to your desktop? (Yes / No)** every time, offers no "don't remind me" and ignores the saved answer, so it
  is always the way to add the shortcut after you have said "never" in the others.
* **Linux:** some file managers show a `.sh` file in a text editor instead of running it. Open `HIVE.sh` once with
  right-click, **Run as a Program** (or `./HIVE.sh` in a terminal; if there is no such menu entry, make the file
  executable first with `chmod +x HIVE.sh`). Answer Yes to the shortcut question and double-click the icon from then on.
  (If the icon does nothing, right-click it once and choose **Allow Launching**.)
* **Windows:** `HIVE.bat` needs no one-time step, and an unsigned script may show a SmartScreen prompt the first time
  ("More info", then "Run anyway"). A Python installed through `pyenv-win` is recognised too.

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

**If a `.venv` folder already exists** and cannot be used on this computer (made on another system,
for example Linux or macOS when you are on Windows, or damaged), the launcher never deletes it
unasked. It tells you why and offers two choices: **[R]** replace it with a new one, or **[K]** keep it
and use a separate `.venv-windows` / `.venv-linux` / `.venv-mac` next to it (the default, and the answer
when there is no keyboard to ask). Once that separate one exists it is used on later starts without
asking again. `--reinstall` rebuilds the environment without asking.

**On Linux, use `HIVE.sh`.** `HIVE.exe` also runs under Wine, but Wine older than 10.1 cannot run
numpy 2.1 or newer (it crashes with `unimplemented function ucrtbase.dll.crealf`). So when the launcher
finds it is running under such a Wine it installs numpy 2.0.x instead and says so; real Windows and
Wine 10.1+ are not limited. Tested: with numpy 2.0.2 the whole stack imports and the app starts and
answers under Wine 10.0, and HIVE's tests pass on numpy 2.0.2.

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
python -c "import hive; print(hive.__release__)"    # 0.0.5
```

Verify the installation:

```bash
python -m pytest -q              # 402 tests, all should pass
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
python -m pytest -q     # 402 tests
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
tests/              ground-truth test suite (402 tests)
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
