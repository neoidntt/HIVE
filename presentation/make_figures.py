"""Build presentation figures from the real HIVE run in presentation/results."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, "/Users/deathyang/Desktop/HIVE_Final")
warnings.filterwarnings("ignore")

from hive.arborization import analyze_arbor, find_branch_points, skeleton_metrics
from hive.cleanup import clean_labels, normalize_for_display
from hive.config import ArborConfig, PreprocessConfig, SegmentationConfig
from hive.preprocessing import prepare_image
from hive.segmentation import segment
from hive.visualization import create_boundary_overlay

ROOT = Path("/Users/deathyang/Desktop/HIVE_Final/presentation")
DATA, RESULTS = ROOT / "sample_data", ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

NAVY, YELLOW, RED = "#0d2942", "#f9b410", "#e80c54"
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "axes.edgecolor": NAVY,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

PRE = PreprocessConfig(baseline_method="tophat", tophat_radius=30, gaussian_sigma=1.0,
                       use_clahe=True)
SEG = SegmentationConfig(min_size=300, hole_size=30, threshold_scale=1.0)
ARB = ArborConfig(enabled=True, sholl_step=4.0, prune_length=4)


def analyse(name):
    """Run the shared preprocess + segment + cleanup steps for one figure image.

    Args:
        name: Filename inside the sample-data directory.

    Returns:
        ``(raw, prep, labels)`` -- the untouched raw array, the
        :class:`~hive.preprocessing.PreparedImage` holding both the
        segmentation and measurement images, and the cleaned label image.

    Note:
        Returns ``raw`` alongside the processed results so figures can show
        the true input next to what the pipeline actually measured. The module
        constants ``PRE``/``SEG``/``ARB`` are shared by every figure, so all
        panels in the deck come from identical parameters.
    """
    raw = tifffile.imread(DATA / name).astype(np.float32)
    prep = prepare_image(raw, PRE, channel="gray")
    labels, _ = clean_labels(segment(prep.segmentation, SEG), SEG)
    return raw, prep, labels


ctrl_raw, ctrl_prep, ctrl_lab = analyse("control_01.tif")
trt_raw, trt_prep, trt_lab = analyse("treated_01.tif")

# ---------------------------------------------------------------- 1. INPUT
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
for ax, img, title in [(axes[0], ctrl_raw, "control_01.tif"), (axes[1], trt_raw, "treated_01.tif")]:
    im = ax.imshow(img, cmap="magma")
    ax.set_title(f"{title}\n{img.shape[0]}x{img.shape[1]} px  |  uint16", fontsize=11, color=NAVY)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, label="raw intensity")
fig.suptitle("INPUT: raw 16-bit fluorescence images (identical display scale)",
             fontsize=13, color=NAVY, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(FIG / "01_input.png"); plt.close(fig)

# ---------------------------------------------------------------- 2. CONTRACT
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8))
panels = [
    (ctrl_raw, "RAW", "uneven illumination + noise", None),
    (ctrl_prep.measurement, "MEASUREMENT image",
     "background removed, TRUE scale kept\nevery reported number comes from here", YELLOW),
    (ctrl_prep.segmentation, "SEGMENTATION image",
     "blurred + CLAHE + normalised\nonly decides WHERE objects are", RED),
]
for ax, (img, title, caption, colour) in zip(axes, panels):
    ax.imshow(img, cmap="magma")
    ax.set_title(title, fontsize=12, color=NAVY, weight="bold", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(caption, fontsize=9.5, color=NAVY, labelpad=8)
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(colour is not None)
        if colour:
            ax.spines[side].set(color=colour, linewidth=3.5)
fig.suptitle("The two-image contract: HIVE never measures the image it enhanced",
             fontsize=14, color=NAVY, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "02_contract.png"); plt.close(fig)

# ---------------------------------------------------------------- 3. SEGMENTATION
fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
axes[0].imshow(normalize_for_display(ctrl_prep.measurement), cmap="magma")
axes[0].set_title("measurement image")
axes[1].imshow(create_boundary_overlay(ctrl_prep.measurement, ctrl_lab))
axes[1].set_title(f"segmentation boundary ({ctrl_lab.max()} object)")
for ax in axes:
    ax.axis("off")
fig.suptitle("OUTPUT 1: segmentation", fontsize=13, color=NAVY, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(FIG / "03_segmentation.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. SKELETON
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
for ax, prep, lab, title in [(axes[0], ctrl_prep, ctrl_lab, "control_01"),
                             (axes[1], trt_prep, trt_lab, "treated_01")]:
    m = skeleton_metrics(lab > 0, ARB.prune_length, 0.35)
    rgb = np.dstack([normalize_for_display(prep.measurement)] * 3) * 0.45
    rgb[lab > 0] += np.array([0.0, 0.10, 0.28])
    rgb[m.skeleton] = [1.0, 0.85, 0.06]
    ax.imshow(np.clip(rgb, 0, 1))
    bp = np.argwhere(find_branch_points(m.skeleton))
    ax.scatter(bp[:, 1], bp[:, 0], s=42, facecolors="none", edgecolors=RED, lw=1.3)
    ax.set_title(f"{title}\n{m.total_length:.0f} um neurite  |  {m.n_branch_points} junctions",
                 fontsize=11, color=NAVY)
    ax.axis("off")
fig.suptitle("OUTPUT 2: skeleton (yellow) + detected junctions (red)",
             fontsize=13, color=NAVY, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(FIG / "04_skeleton.png"); plt.close(fig)

# ---------------------------------------------------------------- 5. SHOLL
sholl_all = pd.read_csv(RESULTS / "2d" / "combined_sholl_curves.csv")
sholl_all["group"] = sholl_all["image"].str.split("_").str[0]
fig, ax = plt.subplots(figsize=(7.0, 5.1))

# Bound the x-axis by the data: the arbors reach ~30 um, so a grid out to 100 um
# would leave 70% of the panel as a dead zero tail.
max_r = sholl_all.loc[sholl_all.intersections > 0, "radius"].max()
grid = np.arange(4, max_r + 8, 4)

styles = {"control": (NAVY, "-", "o", 3), "treated": (RED, "--", "s", 2)}
for group, (colour, ls, marker, z) in styles.items():
    sub = sholl_all[sholl_all.group == group]
    curves = [g.sort_values("radius") for _, g in sub.groupby("image")]
    stacked = np.vstack([np.interp(grid, c.radius, c.intersections, right=0) for c in curves])
    mean, sem = stacked.mean(0), stacked.std(0) / np.sqrt(len(stacked))
    # Distinct linestyle + marker: both groups have exactly 5 primary processes, so
    # the curves sit on top of each other near the soma and a solid line would hide one.
    ax.plot(grid, mean, color=colour, lw=2.2, ls=ls, marker=marker, ms=5,
            label=f"{group} (n={len(curves)})", zorder=z, markerfacecolor="white",
            markeredgewidth=1.6)
    ax.fill_between(grid, mean - sem, mean + sem, color=colour, alpha=0.18, zorder=1)

# The synthetic arbor doubles at each fork, and Sholl reads that structure directly.
for y in (5, 10, 20):
    ax.axhline(y, color="#9aa7b4", lw=0.8, ls=":", zorder=0)
ax.text(4.5, 5.7, "5 primary processes", fontsize=9, color="#5b6b7b")
ax.text(12.6, 10.7, "10 after 1st fork", fontsize=9, color="#5b6b7b")
ax.text(24.8, 20.3, "20 after 2nd fork", fontsize=9, color="#5b6b7b")

ax.set_xlabel("Distance from soma (um)")
ax.set_ylabel("Intersections N(r)")
ax.set_xlim(0, grid.max())
ax.set_ylim(0, 23)
ax.set_title("OUTPUT 3: Sholl analysis (mean +/- SEM)", fontsize=13, color=NAVY, weight="bold")
ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none",
          framealpha=0.95)
fig.tight_layout()
fig.savefig(FIG / "05_sholl.png"); plt.close(fig)

# ---------------------------------------------------------------- 6. 3D
vol = tifffile.imread(RESULTS / "3d" / "HIVE_3D_volume.tif")
mask = tifffile.imread(RESULTS / "3d" / "HIVE_3D_mask.tif")
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
axes[0].imshow(vol.max(axis=0), cmap="magma"); axes[0].set_title("max projection (XY)")
axes[1].imshow(vol.max(axis=1), cmap="magma", aspect="auto"); axes[1].set_title("max projection (XZ)")
axes[2].imshow(mask.max(axis=0), cmap="gray"); axes[2].set_title("segmented mask")
for ax in axes:
    ax.axis("off")
fig.suptitle("3D z-stack: 24 planes, voxel 0.35 x 0.35 x 1.20 um",
             fontsize=13, color=NAVY, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(FIG / "06_3d.png"); plt.close(fig)

# ---------------------------------------------------------------- 7. GROUP BARS
m = pd.read_csv(RESULTS / "2d" / "measurements_with_group.csv")
metrics = [("neurite_length_um", "Neurite length (um)"),
           ("n_branch_points", "Junctions"),
           ("sholl_max_intersections", "Peak Sholl N(r)"),
           ("intensity_mean", "Mean intensity")]
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, (col, label) in zip(axes, metrics):
    means = m.groupby("group")[col].mean()
    sems = m.groupby("group")[col].sem()
    order = ["control", "treated"]
    ax.bar(order, [means[o] for o in order], yerr=[sems[o] for o in order],
           color=[NAVY, RED], capsize=5, width=0.6)
    for i, o in enumerate(order):
        pts = m[m.group == o][col]
        ax.scatter(np.full(len(pts), i) + np.random.uniform(-.08, .08, len(pts)), pts,
                   color="white", edgecolor="black", zorder=3, s=22, lw=.7)
    ax.set_title(label, fontsize=11, color=NAVY)
    ax.tick_params(labelsize=9)
fig.suptitle("OUTPUT 4: quantification, n=3 per group (synthetic demonstration data)",
             fontsize=13, color=NAVY, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.86]); fig.savefig(FIG / "07_groups.png"); plt.close(fig)

print("figures written:")
for p in sorted(FIG.glob("*.png")):
    print(f"  {p.name:22s} {p.stat().st_size/1024:6.1f} KB")
