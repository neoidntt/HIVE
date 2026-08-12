"""HIVE: High-Processing Imaging Visualization Engine.

A general-purpose fluorescence microscopy analysis package: segmentation,
intensity quantification, arborization (Sholl) analysis, 3D reconstruction
and rendering.

This package supersedes the legacy ``src/`` package. The central design
change is the **two-image contract** (see :mod:`hive.preprocessing`):
contrast-enhanced images are used *only* to find objects, while all
intensity measurements are taken from a background-corrected image that
retains its original physical scale. The legacy code measured intensity on
per-image min-max-normalised data, which made a real 3x biological
difference report as 1.0x.
"""

from .config import HiveConfig

__version__ = "2.0.0"
__all__ = ["HiveConfig", "__version__"]
