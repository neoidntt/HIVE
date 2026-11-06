"""Interchange exports: labels as OME-TIFF and objects as ImageJ ROIs.

Both open directly in the tools most labs already use: the OME-TIFF in napari and
Fiji (as a label/mask image with physical pixel size), the ROI zip in Fiji's ROI
Manager (``Analyze > Tools > ROI Manager > More > Open`` or just drag the zip in).

ROIs are the outer outline of each object in each plane, written as freehand
polygons with sub-pixel coordinates on the pixel edges, so rasterising them in
Fiji gives back the original mask. Holes inside an object are not represented
(an ImageJ polygon has no holes), and an object that falls apart into separate
pieces in one plane gets one ROI per piece with the same name.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure

_OME_UNITS = {"um": "µm", "µm": "µm", "μm": "µm", "nm": "nm", "mm": "mm"}


def export_labels_ome_tiff(
    path: str | Path, labels: np.ndarray, voxel_size_xy: float = 1.0,
    voxel_size_z: float = 1.0, unit: str = "px",
) -> Path:
    """Write a label image as an OME-TIFF.

    Args:
        path: Output path, normally ending ``.ome.tif``.
        labels: Integer labels, 2D ``(Y, X)`` or 3D ``(Z, Y, X)``; 0 is background.
        voxel_size_xy: Pixel size in ``unit``.
        voxel_size_z: Z step in ``unit`` (3D only).
        unit: ``"um"``, ``"nm"`` or ``"mm"`` writes physical sizes; anything else
            (``"px"``) leaves the scale out rather than label pixels as micrometres.

    Returns:
        The written path.

    Raises:
        ValueError: If the array is not 2D or 3D, or holds negative labels.
    """
    arr = np.asarray(labels)
    if arr.ndim not in (2, 3):
        raise ValueError(f"labels must be 2D or 3D, got shape {arr.shape}")
    if arr.size and arr.min() < 0:
        raise ValueError("labels must not be negative")
    dtype = np.uint16 if (arr.size == 0 or arr.max() <= np.iinfo(np.uint16).max) else np.uint32
    metadata: dict = {"axes": "ZYX" if arr.ndim == 3 else "YX"}
    ome_unit = _OME_UNITS.get(unit.strip().lower())
    if ome_unit:
        metadata.update(PhysicalSizeX=voxel_size_xy, PhysicalSizeY=voxel_size_xy,
                        PhysicalSizeXUnit=ome_unit, PhysicalSizeYUnit=ome_unit)
        if arr.ndim == 3:
            metadata.update(PhysicalSizeZ=voxel_size_z, PhysicalSizeZUnit=ome_unit)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr.astype(dtype), ome=True, photometric="minisblack",
                     metadata=metadata)
    return path


def _outline(mask: np.ndarray) -> np.ndarray | None:
    """The outer outline of one connected piece, as (row, col) on pixel edges.

    ``find_contours`` at level 0.5 passes through the midpoint between an inside and
    an outside pixel centre, which is the pixel edge. The mask is padded so objects
    touching the crop edge still close.
    """
    padded = np.pad(mask, 1).astype(np.float32)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return None
    outer = max(contours, key=len)      # the outer boundary is the longest; holes are shorter
    return outer - 1.0                  # undo the padding


def export_imagej_rois(path: str | Path, labels: np.ndarray) -> int:
    """Write every object's outline as an ImageJ ROI, zipped.

    Args:
        path: Output ``.zip`` path.
        labels: Integer label image, 2D or 3D.

    Returns:
        The number of ROIs written (0 means no file was created).

    Raises:
        ImportError: If the ``roifile`` package is missing.
        ValueError: If the array is not 2D or 3D.
    """
    try:
        import roifile
    except ImportError as exc:
        raise ImportError("ImageJ ROI export needs roifile: pip install roifile") from exc
    arr = np.asarray(labels)
    if arr.ndim not in (2, 3):
        raise ValueError(f"labels must be 2D or 3D, got shape {arr.shape}")
    volume = arr[None] if arr.ndim == 2 else arr
    rois, names = [], []
    for label, box in enumerate(ndimage.find_objects(volume), start=1):
        if box is None:
            continue
        for z in range(box[0].start, box[0].stop):
            crop = volume[z, box[1], box[2]] == label
            if not crop.any():
                continue
            pieces, count = ndimage.label(crop)
            for piece in range(1, count + 1):
                outline = _outline(pieces == piece)
                if outline is None or len(outline) < 3:
                    continue
                # ImageJ's pixel (r, c) covers x in [c, c+1], y in [r, r+1].
                xy = np.column_stack([outline[:, 1] + box[2].start + 0.5,
                                      outline[:, 0] + box[1].start + 0.5])
                name = f"{label:04d}" + (f"-z{z + 1:03d}" if arr.ndim == 3 else "")
                rois.append(roifile.ImagejRoi.frompoints(
                    xy.astype(np.float32), name=name, position=z if arr.ndim == 3 else None))
                names.append(f"{name}-{len(rois)}")
    if not rois:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    roifile.roiwrite(str(path), rois, name=names, mode="w")
    return len(rois)


def export_label_files(folder: str | Path, stem: str, labels: np.ndarray, cfg) -> dict[str, Path]:
    """Write the label OME-TIFF and ROI zip the configuration asks for.

    A missing optional package is reported as a warning rather than failing a long
    analysis whose measurements are already done.

    Args:
        folder: Output folder.
        stem: File name stem, for example the image name.
        labels: Final label image.
        cfg: A :class:`~hive.config.HiveConfig`.

    Returns:
        Kind -> path for every file written.
    """
    written: dict[str, Path] = {}
    folder = Path(folder)
    if cfg.export.ome_tiff_labels:
        written["labels_ome_tiff"] = export_labels_ome_tiff(
            folder / f"{stem}_labels.ome.tif", labels,
            cfg.voxel_size_xy, cfg.voxel_size_z, cfg.voxel_unit)
    if cfg.export.imagej_rois:
        try:
            if export_imagej_rois(folder / f"{stem}_rois.zip", labels):
                written["imagej_rois"] = folder / f"{stem}_rois.zip"
        except ImportError as exc:
            warnings.warn(f"{exc}; ImageJ ROIs were not written.", stacklevel=2)
    return written
