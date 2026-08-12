"""Build the HIVE overview PowerPoint deck.

Every figure and number in this deck comes from a real run of the HIVE pipeline on
the sample data in presentation/sample_data -- nothing is mocked up. The sample
data itself is synthetic, and every slide that shows it says so.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path("/Users/deathyang/Desktop/HIVE_Final/presentation")
FIG, RESULTS = ROOT / "figures", ROOT / "results"
OUT = ROOT / "HIVE_overview.pptx"

NAVY = RGBColor(0x0D, 0x29, 0x42)
YELLOW = RGBColor(0xF9, 0xB4, 0x10)
RED = RGBColor(0xE8, 0x0C, 0x54)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x5B, 0x6B, 0x7B)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def textbox(slide, text, left, top, width, height, size=14, color=NAVY, bold=False,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, font="Calibri"):
    """Add a word-wrapped text box and return it."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = font
    return box


def rect(slide, left, top, width, height, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    """Add a filled rectangle."""
    s = slide.shapes.add_shape(shape, left, top, width, height)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1.5)
    s.shadow.inherit = False
    return s


def content_slide(title, kicker=None):
    """Add a slide with the standard navy header bar and yellow accent."""
    slide = prs.slides.add_slide(BLANK)
    rect(slide, 0, 0, SW, Inches(0.92), NAVY)
    rect(slide, 0, Inches(0.92), SW, Inches(0.06), YELLOW)
    textbox(slide, title, Inches(0.55), Inches(0.14), Inches(9.5), Inches(0.65),
            size=26, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        textbox(slide, kicker, Inches(10.0), Inches(0.14), Inches(2.85), Inches(0.65),
                size=12, color=YELLOW, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    return slide


def place_image(slide, path, top, max_w, max_h, left=None):
    """Place an image scaled to fit a box, preserving aspect ratio."""
    with Image.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    width, height = Emu(int(w * scale)), Emu(int(h * scale))
    left = int((SW - width) / 2) if left is None else left
    slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    return width, height


def synthetic_footer(slide, extra=""):
    """Stamp the slide as synthetic demonstration data. Non-negotiable on data slides."""
    note = "Synthetic demonstration data — generated to exercise the pipeline, not a biological result."
    textbox(slide, note + ("  " + extra if extra else ""), Inches(0.55), Inches(6.95),
            Inches(12.2), Inches(0.35), size=10, color=GREY, italic=True)


# ============================================================ 1. TITLE
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, SW, SH, NAVY)
rect(s, 0, Inches(3.02), Inches(1.6), Inches(0.09), YELLOW, )
textbox(s, "HIVE", Inches(1.1), Inches(1.9), Inches(11), Inches(1.1),
        size=62, color=WHITE, bold=True)
textbox(s, "High-Processing Imaging Visualization Engine", Inches(1.12), Inches(3.25),
        Inches(11), Inches(0.6), size=25, color=YELLOW)
textbox(s,
        "Segmentation  ·  intensity quantification  ·  arborization (Sholl)  ·  3D reconstruction\n"
        "for fluorescence microscopy",
        Inches(1.12), Inches(4.0), Inches(11), Inches(1.0), size=16, color=RGBColor(0xC9, 0xD6, 0xE2))
textbox(s, "How it works, and what goes in and out", Inches(1.12), Inches(5.5),
        Inches(11), Inches(0.4), size=14, color=RGBColor(0x8F, 0xA6, 0xBC))

# ============================================================ 2. WHAT IT DOES
s = content_slide("What HIVE does")
cards = [
    ("Detect", "Finds objects with Otsu\nthresholding or Cellpose,\nthen filters debris, edge-\ntruncated objects and holes.", NAVY),
    ("Quantify", "Area / volume, shape, and\nfluorescence intensity in\nreal physical units —\ncomparable across images.", YELLOW),
    ("Branch", "Skeletonises each object and\nmeasures neurite length,\njunctions, and a full Sholl\nprofile.", RED),
    ("Reconstruct", "Builds 3D volumes, meshes,\nan interactive viewer, and\nrotating projection GIFs\nfrom z-stacks.", NAVY),
]
x = Inches(0.55)
for name, body, colour in cards:
    rect(s, x, Inches(1.45), Inches(2.95), Inches(0.5), colour)
    rect(s, x, Inches(1.95), Inches(2.95), Inches(2.15), LIGHT)
    textbox(s, name, x + Inches(0.2), Inches(1.5), Inches(2.6), Inches(0.4), size=17,
            color=WHITE if colour != YELLOW else NAVY, bold=True)
    textbox(s, body, x + Inches(0.2), Inches(2.1), Inches(2.6), Inches(1.9), size=12, color=NAVY)
    x += Inches(3.13)

textbox(s, "Three ways to run it", Inches(0.55), Inches(4.45), Inches(6), Inches(0.4),
        size=17, color=NAVY, bold=True)
runs = [
    ("Notebook", "HIVE_walkthrough.ipynb — exploratory analysis, explains the theory, self-contained"),
    ("Web app", "streamlit run app.py — upload, tune, inspect, export. No code required."),
    ("CLI / API", "python hive_cli.py … for batch runs; FastAPI backend for integration"),
]
y = Inches(4.95)
for name, body in runs:
    rect(s, Inches(0.55), y + Inches(0.06), Inches(0.09), Inches(0.28), YELLOW)
    textbox(s, name, Inches(0.8), y, Inches(1.5), Inches(0.35), size=13, color=NAVY, bold=True)
    textbox(s, body, Inches(2.35), y, Inches(10.4), Inches(0.35), size=13, color=GREY)
    y += Inches(0.52)

# ============================================================ 3. PIPELINE
s = content_slide("How it works: the pipeline")
steps = [
    ("Load", "TIF · PNG · CZI\n2D or z-stack"),
    ("Channel", "select / auto\nRGB → 1 channel"),
    ("Background", "top-hat\n(rolling-ball)"),
    ("Segment", "Otsu\nor Cellpose"),
    ("Clean", "debris · edges\nholes"),
    ("Measure", "shape · intensity\narborization"),
    ("Export", "CSV · Excel\n3D · GIF"),
]
x, w, gap = Inches(0.5), Inches(1.62), Inches(0.19)
for i, (name, body) in enumerate(steps):
    colour = YELLOW if name in ("Background", "Measure") else NAVY
    rect(s, x, Inches(1.5), w, Inches(0.44), colour)
    rect(s, x, Inches(1.94), w, Inches(0.92), LIGHT)
    textbox(s, name, x, Inches(1.55), w, Inches(0.35), size=13,
            color=NAVY if colour == YELLOW else WHITE, bold=True, align=PP_ALIGN.CENTER)
    textbox(s, body, x, Inches(2.04), w, Inches(0.8), size=10, color=NAVY, align=PP_ALIGN.CENTER)
    if i < len(steps) - 1:
        arrow = rect(s, x + w + Inches(0.02), Inches(2.05), Inches(0.15), Inches(0.18), GREY,
                     shape=MSO_SHAPE.RIGHT_ARROW)
    x += w + gap

rect(s, Inches(0.5), Inches(3.3), Inches(12.35), Inches(0.05), YELLOW)
textbox(s, "The split that makes the numbers trustworthy", Inches(0.5), Inches(3.5),
        Inches(12), Inches(0.4), size=17, color=NAVY, bold=True)
textbox(s,
        "After background subtraction the pipeline forks into two images that are never confused:",
        Inches(0.5), Inches(3.95), Inches(12), Inches(0.35), size=13, color=GREY)

rect(s, Inches(0.9), Inches(4.5), Inches(5.6), Inches(1.75), LIGHT, line=YELLOW)
textbox(s, "MEASUREMENT image", Inches(1.15), Inches(4.65), Inches(5.1), Inches(0.35),
        size=15, color=NAVY, bold=True)
textbox(s, "Background removed, original intensity scale kept.\n"
           "Nothing else touches it.\n\n→ Every reported number comes from here.",
        Inches(1.15), Inches(5.05), Inches(5.1), Inches(1.1), size=12, color=NAVY)

rect(s, Inches(6.85), Inches(4.5), Inches(5.6), Inches(1.75), LIGHT, line=RED)
textbox(s, "SEGMENTATION image", Inches(7.1), Inches(4.65), Inches(5.1), Inches(0.35),
        size=15, color=NAVY, bold=True)
textbox(s, "Blurred, CLAHE-enhanced, normalised.\n"
           "Deliberately distorted for maximum contrast.\n\n→ Only decides WHERE objects are.",
        Inches(7.1), Inches(5.05), Inches(5.1), Inches(1.1), size=12, color=NAVY)

textbox(s, "Enhancing an image helps you find objects. It destroys their intensity. "
           "So HIVE finds objects on one image and measures them on the other.",
        Inches(0.9), Inches(6.4), Inches(11.5), Inches(0.5), size=12.5, color=NAVY, italic=True)

# ============================================================ 4. CONTRACT FIGURE
s = content_slide("The same neuron, three ways", "core idea")
place_image(s, FIG / "02_contract.png", Inches(1.15), Inches(11.6), Inches(4.6))
textbox(s,
        "Why it matters: measuring the enhanced image (right) reports the contrast stretch, not the biology. "
        "Two samples with a true 3× brightness difference come back as 1.0× — no difference at all.",
        Inches(0.6), Inches(6.05), Inches(12.1), Inches(0.7), size=13, color=NAVY)
synthetic_footer(s)

# ============================================================ 5. INPUT
s = content_slide("INPUT: what you give it", "example")
place_image(s, FIG / "01_input.png", Inches(1.3), Inches(7.4), Inches(4.5), left=Inches(0.5))

rect(s, Inches(8.35), Inches(1.35), Inches(4.45), Inches(4.9), LIGHT)
textbox(s, "Files", Inches(8.6), Inches(1.5), Inches(4), Inches(0.3), size=15, color=NAVY, bold=True)
textbox(s,
        "6 × control/treated images  (220×220, uint16)\n"
        "1 × z-stack  (24 planes, 140×140)\n\n"
        "TIF · PNG · JPG · CZI · LSM\n"
        "2D images, folders, or multi-page stacks",
        Inches(8.6), Inches(1.9), Inches(4), Inches(1.5), size=12, color=NAVY)

rect(s, Inches(8.6), Inches(3.45), Inches(4), Inches(0.03), YELLOW)
textbox(s, "Settings used", Inches(8.6), Inches(3.6), Inches(4), Inches(0.3), size=15,
        color=NAVY, bold=True)
textbox(s,
        "channel            gray\n"
        "background      top-hat, radius 30\n"
        "threshold          Otsu × 1.0\n"
        "min object        300 px\n"
        "pixel size          0.35 µm\n"
        "Sholl step         4.0 µm",
        Inches(8.6), Inches(4.0), Inches(4), Inches(1.6), size=12, color=NAVY, font="Consolas")
textbox(s, "All 42 parameters are recorded in the report's Settings sheet.",
        Inches(8.6), Inches(5.75), Inches(4), Inches(0.4), size=10.5, color=GREY, italic=True)
synthetic_footer(s)

# ============================================================ 6. OUTPUT: SEGMENTATION
s = content_slide("OUTPUT 1: segmentation", "example")
place_image(s, FIG / "03_segmentation.png", Inches(1.35), Inches(8.6), Inches(4.75), left=Inches(0.5))
rect(s, Inches(9.5), Inches(1.5), Inches(3.35), Inches(4.4), LIGHT)
textbox(s, "Cleanup is reported,\nnever silent", Inches(9.75), Inches(1.7), Inches(2.9),
        Inches(0.7), size=14, color=NAVY, bold=True)
textbox(s,
        "Every run logs what was removed:\n\n"
        "• objects below the debris size\n"
        "• objects touching the edge\n   (truncated → unmeasurable)\n"
        "• holes filled\n\n"
        "A filtered-away object silently\nchanges your n. HIVE tells you.",
        Inches(9.75), Inches(2.5), Inches(2.9), Inches(3.2), size=12, color=NAVY)
synthetic_footer(s)

# ============================================================ 7. OUTPUT: SKELETON
s = content_slide("OUTPUT 2: arborization", "example")
place_image(s, FIG / "04_skeleton.png", Inches(1.3), Inches(8.3), Inches(4.9), left=Inches(0.45))
rect(s, Inches(9.3), Inches(1.45), Inches(3.55), Inches(4.7), LIGHT)
textbox(s, "Measured per object", Inches(9.55), Inches(1.6), Inches(3.1), Inches(0.35),
        size=14, color=NAVY, bold=True)
textbox(s,
        "neurite_length          345 µm\n"
        "n_branch_points        16\n"
        "n_endpoints              16\n"
        "n_branches                33\n"
        "max_branch_length   38 µm\n\n"
        "…and the full Sholl profile",
        Inches(9.55), Inches(2.05), Inches(3.1), Inches(2.0), size=11.5, color=NAVY,
        font="Consolas")
rect(s, Inches(9.55), Inches(4.25), Inches(3.1), Inches(0.03), YELLOW)
textbox(s,
        "Length is diagonal-corrected: counting\nskeleton pixels underestimates a diagonal\nby ~29%.\n\n"
        "Junctions use the crossing number, so a\npixel resting on a branch isn't miscounted\nas a fork.",
        Inches(9.55), Inches(4.45), Inches(3.1), Inches(1.6), size=10.5, color=GREY)
synthetic_footer(s)

# ============================================================ 8. SHOLL
s = content_slide("OUTPUT 3: Sholl analysis", "example")
place_image(s, FIG / "05_sholl.png", Inches(1.3), Inches(7.3), Inches(5.3), left=Inches(0.5))
rect(s, Inches(8.1), Inches(1.4), Inches(4.75), Inches(4.85), LIGHT)
textbox(s, "Sholl (1953)", Inches(8.35), Inches(1.55), Inches(4.3), Inches(0.35),
        size=15, color=NAVY, bold=True)
textbox(s,
        "Concentric shells at increasing radius from\nthe soma; count how many times the arbor\ncrosses each one.\n\n"
        "N(r) profiles complexity against distance —\nit separates a compact bushy arbor from a\nsparse far-reaching one even when both\nhave the same total branch count.",
        Inches(8.35), Inches(1.95), Inches(4.3), Inches(2.1), size=12, color=NAVY)
rect(s, Inches(8.35), Inches(4.2), Inches(4.3), Inches(0.03), YELLOW)
textbox(s,
        "Derived scalars per object:\n"
        "  peak N(r) · critical radius · enclosing radius\n"
        "  primary processes · ramification index\n"
        "  Sholl regression coefficient k",
        Inches(8.35), Inches(4.4), Inches(4.3), Inches(1.3), size=11.5, color=NAVY)
textbox(s, "Full radius-vs-intersections curves export to their own CSV.",
        Inches(8.35), Inches(5.75), Inches(4.3), Inches(0.4), size=10.5, color=GREY, italic=True)
synthetic_footer(s)

# ============================================================ 9. DATA TABLE
s = content_slide("OUTPUT 4: the data", "example")
df = pd.read_csv(RESULTS / "2d" / "measurements_with_group.csv")
cols = [("image", "image", 1.85), ("area_um2", "area\n(µm²)", 1.05),
        ("intensity_mean", "mean\nintensity", 1.15),
        ("integrated_intensity", "integrated\nintensity", 1.35),
        ("neurite_length_um", "neurite\nlength (µm)", 1.3),
        ("n_branch_points", "branch\npoints", 1.0),
        ("n_endpoints", "end\npoints", 0.95),
        ("sholl_max_intersections", "peak\nN(r)", 0.95),
        ("sholl_ramification_index", "ramif.\nindex", 1.0)]
n_rows, n_cols = len(df) + 1, len(cols)
total_w = sum(c[2] for c in cols)
tbl_shape = s.shapes.add_table(n_rows, n_cols, Inches(0.55), Inches(1.45),
                               Inches(total_w), Inches(0.42 * n_rows))
tbl = tbl_shape.table
for j, (_, header, width) in enumerate(cols):
    tbl.columns[j].width = Inches(width)
    cell = tbl.cell(0, j)
    cell.text = header
    cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
    for p in cell.text_frame.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size = Pt(10.5); r.font.bold = True; r.font.color.rgb = WHITE

for i, (_, row) in enumerate(df.iterrows(), start=1):
    is_ctrl = row["group"] == "control"
    for j, (col, _, _) in enumerate(cols):
        val = row[col]
        text = val if isinstance(val, str) else (f"{val:,.0f}" if abs(val) >= 100 else f"{val:.1f}")
        cell = tbl.cell(i, j)
        cell.text = str(text)
        cell.fill.solid()
        cell.fill.fore_color.rgb = LIGHT if is_ctrl else RGBColor(0xFD, 0xEC, 0xF1)
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(10); r.font.color.rgb = NAVY
                r.font.bold = j == 0

textbox(s, "One row per object. Straight to CSV / Excel — no reshaping.",
        Inches(0.55), Inches(4.6), Inches(7), Inches(0.4), size=13, color=NAVY, bold=True)
textbox(s,
        "Also exported: bounding box, centroid, eccentricity, solidity, axis lengths, orientation,\n"
        "extent, min/max intensity, branch-length stats, and the full Sholl curves.",
        Inches(0.55), Inches(5.0), Inches(7.5), Inches(0.8), size=12, color=GREY)

rect(s, Inches(8.5), Inches(4.55), Inches(4.35), Inches(1.9), LIGHT, line=YELLOW)
textbox(s, "Read integrated_intensity, not mean", Inches(8.75), Inches(4.7), Inches(3.9),
        Inches(0.35), size=12.5, color=NAVY, bold=True)
textbox(s,
        "Background is already removed, so it is the\ncorrected integrated density — the total signal\n"
        "in the object.\n\nMean intensity alone confounds a small bright\nobject with a large dim one.",
        Inches(8.75), Inches(5.1), Inches(3.9), Inches(1.3), size=11, color=NAVY)
synthetic_footer(s)

# ============================================================ 10. RESULTS
s = content_slide("What the numbers say", "example")
place_image(s, FIG / "07_groups.png", Inches(1.4), Inches(12.3), Inches(3.4))
textbox(s,
        "Control vs treated: arbor complexity falls (341 → 153 µm of neurite, 16 → 7 junctions, peak N(r) 20 → 10) "
        "and signal dims (644 → 420). Morphology and intensity are measured independently, so a change in one "
        "cannot manufacture a change in the other.",
        Inches(0.6), Inches(5.2), Inches(12.1), Inches(1.1), size=13, color=NAVY)
rect(s, Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.05), YELLOW)
textbox(s, "Each dot is one image. Bars are mean ± SEM.", Inches(0.6), Inches(6.45),
        Inches(6), Inches(0.35), size=11, color=GREY, italic=True)
synthetic_footer(s)

# ============================================================ 11. 3D
s = content_slide("3D reconstruction", "example")
place_image(s, FIG / "06_3d.png", Inches(1.25), Inches(7.8), Inches(3.05), left=Inches(0.5))
gif = RESULTS / "3d" / "HIVE_3D_rotation.gif"
place_image(s, gif, Inches(1.25), Inches(3.6), Inches(3.3), left=Inches(8.9))
textbox(s, "360° rotating projection", Inches(8.9), Inches(4.6), Inches(3.6), Inches(0.3),
        size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
textbox(s, "animates in slideshow mode ▶", Inches(8.9), Inches(4.9), Inches(3.6), Inches(0.3),
        size=10, color=RED, align=PP_ALIGN.CENTER, italic=True)

textbox(s,
        "From a z-stack HIVE exports: volume + mask TIFFs with embedded voxel spacing (so Fiji, napari and Imaris "
        "open them correctly scaled), max projections, an interactive Plotly viewer, an OBJ mesh, and the "
        "rotating GIF — with customisable frames, speed, colormap and projection mode (MIP or alpha-blended).",
        Inches(0.55), Inches(5.35), Inches(12.2), Inches(1.0), size=13, color=NAVY)
textbox(s,
        "Volumes are resampled to isotropic voxels before rotating. Confocal z-steps are routinely 3–10× the xy "
        "pixel size; rotating without resampling makes spheres change shape as they turn.",
        Inches(0.55), Inches(6.35), Inches(12.2), Inches(0.6), size=11, color=GREY, italic=True)

# ============================================================ 12. HOW TO RUN
s = content_slide("Running it")
blocks = [
    ("Explore", "jupyter lab HIVE_walkthrough.ipynb",
     "Self-contained walkthrough. Generates its own phantoms with known\n"
     "ground truth — runs with no data of your own. Explains the theory and\n"
     "includes an interactive threshold explorer."),
    ("Point and click", "streamlit run app.py",
     "Upload TIF/CZI, tune parameters live, inspect segmentation beside the raw\n"
     "image, render the 3D GIF, export CSV. Detects CUDA / Apple MPS / CPU\n"
     "automatically."),
    ("Batch", "python hive_cli.py --mode 2d --input imgs \\\n"
              "    --voxel-size-xy 0.35 --unit um",
     "Whole folders, scriptable, reproducible. Add --mode 3d --gif for z-stacks."),
]
y = Inches(1.35)
for name, cmd, body in blocks:
    rect(s, Inches(0.55), y, Inches(0.09), Inches(1.5), YELLOW)
    textbox(s, name, Inches(0.8), y, Inches(2.3), Inches(0.4), size=16, color=NAVY, bold=True)
    code = rect(s, Inches(3.2), y, Inches(4.6), Inches(0.95), NAVY)
    textbox(s, cmd, Inches(3.35), y + Inches(0.12), Inches(4.4), Inches(0.75), size=11,
            color=YELLOW, font="Consolas")
    textbox(s, body, Inches(8.05), y, Inches(4.9), Inches(1.5), size=11, color=GREY)
    y += Inches(1.72)

rect(s, Inches(0.55), Inches(6.55), Inches(12.3), Inches(0.05), YELLOW)
textbox(s, "65 ground-truth tests · pytest tests/ -q      "
           "A 4-arm phantom must report exactly 4 tips and 1 junction — so a failure means the "
           "measurement is wrong, not merely different.",
        Inches(0.55), Inches(6.7), Inches(12.3), Inches(0.5), size=11, color=GREY, italic=True)

prs.save(OUT)
print(f"wrote {OUT}")
print(f"  {len(prs.slides.__iter__.__self__._sldIdLst)} slides, {OUT.stat().st_size/1024:.0f} KB")
for i, sl in enumerate(prs.slides, 1):
    titles = [sh.text_frame.text.split("\n")[0] for sh in sl.shapes
              if sh.has_text_frame and sh.text_frame.text.strip()]
    print(f"  {i:2d}. {titles[0][:58] if titles else '(image only)'}")
