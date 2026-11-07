"""Generate synthetic sample data and run the real HIVE pipeline on it.

Everything the slide deck shows is produced here by the actual pipeline -- no
mock-ups. The data is synthetic and labelled as such on every slide.

Scenario: a plausible-looking two-group comparison (Control vs Treated) where the
treated group has a genuinely less complex arbor AND dimmer signal. This exercises
both halves of HIVE's contract: morphology and intensity.
"""
from __future__ import annotations

import math
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, "/Users/deathyang/Desktop/HIVE_Final")
warnings.filterwarnings("ignore")

from skimage.draw import line, disk
from skimage.morphology import dilation, disk as sedisk

ROOT = Path("/Users/deathyang/Desktop/HIVE_Final/presentation")
DATA = ROOT / "sample_data"
RESULTS = ROOT / "results"
for d in (DATA, RESULTS):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)

S, C = 220, 110


def grow(mask, y, x, angle, length, depth, max_depth, rng, spread=0.55):
    """Recursively draw a bifurcating arbor onto a boolean mask."""
    y2 = y + length * math.sin(angle)
    x2 = x + length * math.cos(angle)
    yi, xi = int(round(y)), int(round(x))
    y2i, x2i = int(round(np.clip(y2, 1, S - 2))), int(round(np.clip(x2, 1, S - 2)))
    rr, cc = line(yi, xi, y2i, x2i)
    mask[rr, cc] = True
    if depth >= max_depth:
        return
    for sign in (-1, 1):
        jitter = rng.normal(0, 0.08)
        grow(mask, y2, x2, angle + sign * spread + jitter,
             length * 0.72, depth + 1, max_depth, rng, spread)


def make_neuron(max_depth: int, n_primary: int, seed: int, brightness: int,
                background: int = 120, noise: float = 6.0) -> np.ndarray:
    """Render one synthetic neuron as a uint16 fluorescence image."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((S, S), bool)
    rr, cc = disk((C, C), 5, shape=mask.shape)
    mask[rr, cc] = True
    for k in range(n_primary):
        angle = 2 * math.pi * k / n_primary + rng.normal(0, 0.05)
        grow(mask, C, C, angle, 34, 0, max_depth, rng)
    mask = dilation(mask, sedisk(1))

    img = np.full((S, S), float(background))
    img[mask] = brightness
    # A gentle illumination gradient + shot noise, so the demo is not trivially clean.
    yy, xx = np.mgrid[0:S, 0:S]
    img += 25 * (xx / (S - 1))
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


# --- 2D groups -------------------------------------------------------------
# Control: arbors branch to 3 levels, bright. Treated: 1 level, dimmer.
specs = [("control", 3, 1000), ("treated", 1, 700)]
for group, depth, bright in specs:
    for rep in range(3):
        img = make_neuron(max_depth=depth, n_primary=5, seed=hash((group, rep)) % 10_000,
                          brightness=bright)
        tifffile.imwrite(DATA / f"{group}_{rep + 1:02d}.tif", img)
print(f"wrote {len(list(DATA.glob('*.tif')))} sample images to {DATA}")

# --- 3D z-stack ------------------------------------------------------------
Z = 24
rng = np.random.default_rng(7)
stack = np.full((Z, 140, 140), 120.0)
zz, yy, xx = np.ogrid[:Z, :140, :140]
# A soma plus two processes travelling through z -- something worth rotating.
stack[((zz - 12) ** 2 * 6 + (yy - 70) ** 2 + (xx - 70) ** 2) < 320] = 1100
for t in np.linspace(0, 1, 220):
    z = int(4 + t * 15); y = int(70 + 45 * t); x = int(70 + 40 * math.sin(3 * t))
    stack[z - 1:z + 2, y - 2:y + 3, x - 2:x + 3] = 900
for t in np.linspace(0, 1, 220):
    z = int(19 - t * 14); y = int(70 - 42 * t); x = int(70 - 38 * math.sin(2.5 * t))
    stack[z - 1:z + 2, y - 2:y + 3, x - 2:x + 3] = 850
stack += rng.normal(0, 5, stack.shape)
zstack_path = DATA / "zstack_demo.tif"
tifffile.imwrite(zstack_path, np.clip(stack, 0, 65535).astype(np.uint16))
print(f"wrote z-stack {stack.shape} to {zstack_path}")

# --- Run the REAL pipeline -------------------------------------------------
from hive.config import ArborConfig, HiveConfig, PreprocessConfig, RenderConfig, SegmentationConfig
from hive.pipeline import analyze_folder, analyze_zstack

VOXEL_XY, UNIT = 0.35, "um"

cfg2d = HiveConfig(
    input_path=str(DATA), output_folder=str(RESULTS / "2d"), mode="2d", channel="gray",
    voxel_size_xy=VOXEL_XY, voxel_unit=UNIT,
    preprocess=PreprocessConfig(baseline_method="tophat", tophat_radius=30,
                                gaussian_sigma=1.0, use_clahe=True),
    segmentation=SegmentationConfig(min_size=300, hole_size=30, threshold_scale=1.0),
    arbor=ArborConfig(enabled=True, sholl_step=4.0, prune_length=4),
)
# The z-stack lives in the same folder; exclude it from the 2D batch.
tmp2d = DATA.parent / "_2d_only"
if tmp2d.exists():
    shutil.rmtree(tmp2d)
tmp2d.mkdir()
for p in sorted(DATA.glob("*.tif")):
    if not p.name.startswith("zstack"):
        shutil.copy(p, tmp2d / p.name)
cfg2d.input_path = str(tmp2d)

out2d = analyze_folder(cfg2d)
m = out2d["measurements"]
m["group"] = m["image"].str.split("_").str[0]
m.to_csv(RESULTS / "2d" / "measurements_with_group.csv", index=False)
print(f"\n2D: {len(m)} objects across {len(out2d['summary'])} images; errors={out2d['errors']}")

cols = ["image", "group", "area_um2", "intensity_mean", "integrated_intensity",
        "neurite_length_um", "n_branch_points", "n_endpoints", "sholl_max_intersections",
        "sholl_ramification_index"]
print(m[cols].round(2).to_string(index=False))

print("\n--- group means ---")
gm = m.groupby("group")[["intensity_mean", "neurite_length_um", "n_branch_points",
                         "n_endpoints", "sholl_max_intersections"]].mean().round(2)
print(gm.to_string())

cfg3d = HiveConfig(
    input_path=str(zstack_path), output_folder=str(RESULTS / "3d"), mode="3d", channel="gray",
    voxel_size_xy=0.35, voxel_size_z=1.2, voxel_unit=UNIT,
    preprocess=PreprocessConfig(baseline_method="percentile", baseline_percentile=25,
                                gaussian_sigma=0.8, use_clahe=False),
    segmentation=SegmentationConfig(min_size=200, hole_size=20),
    arbor=ArborConfig(enabled=False),
    render=RenderConfig(mode="mip", n_frames=36, fps=15, colormap="magma", downsample=1),
)
r3d = analyze_zstack(cfg3d, render_gif=True)
print(f"\n3D: {r3d.object_count} objects; artifacts: {list(r3d.outputs)}")
print(r3d.measurements.round(2).to_string(index=False))
