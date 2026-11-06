"""Small builders shared by the 0.0.3 tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from hive.config import ArborConfig, HiveConfig, PreprocessConfig, SegmentationConfig
from hive.synthetic import make_synthetic_2d


def quick_config(tmp_path: Path, **overrides) -> HiveConfig:
    """A fast, deterministic 2D configuration (no CLAHE, no arborization)."""
    values = dict(
        input_path=str(tmp_path / "in"), output_folder=str(tmp_path / "out"),
        preprocess=PreprocessConfig(use_clahe=False, tophat_radius=15.0, gaussian_sigma=0.0),
        segmentation=SegmentationConfig(min_size=20, hole_size=0),
        arbor=ArborConfig(enabled=False),
    )
    values.update(overrides)
    return HiveConfig(**values)


def write_synthetic_folder(folder: Path, names=None) -> list[Path]:
    """Write the three synthetic 2D fields as plain TIFFs."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, image in make_synthetic_2d(0):
        path = folder / (names.pop(0) if names else name)
        tifffile.imwrite(path, image.astype(np.float32))
        paths.append(path)
    return paths


def write_ome(path: Path, data: np.ndarray, axes: str, channels=None, xy=0.5, z=None) -> Path:
    """Write an OME-TIFF with channel names and physical sizes."""
    meta = {"axes": axes, "PhysicalSizeX": xy, "PhysicalSizeY": xy,
            "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"}
    if z:
        meta.update(PhysicalSizeZ=z, PhysicalSizeZUnit="µm")
    if channels:
        meta["Channel"] = {"Name": list(channels)}
    tifffile.imwrite(str(path), data, ome=True, metadata=meta)
    return path


def blob_volume(shape=(10, 200, 220), seed=1) -> np.ndarray:
    """A noisy z-stack with nine bright spheres, one of them on a tile seam."""
    rng = np.random.default_rng(seed)
    vol = 20 + rng.normal(0, 2, shape).astype(np.float32)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    for cz, cy, cx in [(5, 30, 30), (5, 30, 110), (4, 100, 60), (5, 100, 100), (5, 170, 40),
                       (4, 170, 170), (5, 60, 200), (5, 110, 5), (5, 100, 128)]:
        vol[((zz - cz) ** 2 * 4 + (yy - cy) ** 2 + (xx - cx) ** 2) < 81] += 150
    return vol
