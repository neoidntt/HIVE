"""Reproducible microscopy phantoms for the app's built-in demo."""

from __future__ import annotations

import numpy as np


def make_synthetic_2d(seed: int = 0) -> list[tuple[str, np.ndarray]]:
    """Return three noisy 2D fields with six cell-like objects each."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:160, :160]
    centers = (
        (40, 40, 9, 120), (40, 110, 8, 90), (100, 55, 10, 150),
        (120, 120, 7, 60), (75, 85, 9, 110), (135, 35, 6, 75),
    )
    fields = []

    for name, scale in (("dim", 0.75), ("reference", 1.0), ("bright", 1.5)):
        image = np.full((160, 160), 30, dtype=np.float32)
        image += (xx / 159 * 25).astype(np.float32)
        image += rng.normal(0, 3, image.shape).astype(np.float32)
        for cy, cx, radius, amplitude in centers:
            cell = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
            image[cell] += amplitude * scale
        fields.append((f"synthetic_{name}.tif", np.clip(image, 0, None)))

    return fields


def make_synthetic_3d(seed: int = 0) -> np.ndarray:
    """Return an anisotropic sphere-and-bar z-stack with reproducible noise."""
    rng = np.random.default_rng(seed)
    z_size, y_size, x_size = 20, 90, 90
    zz, yy, xx = np.ogrid[:z_size, :y_size, :x_size]

    volume = np.full((z_size, y_size, x_size), 30, dtype=np.float32)
    volume += np.linspace(0, 8, z_size, dtype=np.float32)[:, None, None]
    sphere = ((zz - 10) ** 2 * 9 + (yy - 35) ** 2 + (xx - 35) ** 2) < 200
    volume[sphere] += 180
    volume[8:13, 60:66, 20:70] += 110
    volume += rng.normal(0, 2, volume.shape).astype(np.float32)

    return np.clip(volume, 0, None)