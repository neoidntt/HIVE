"""LEGACY (HIVE 1.x) optional Cellpose segmentation.

.. deprecated:: 2.0
    Superseded by :mod:`hive.segmentation`.

Cellpose (Stringer et al., 2021) is an optional dependency: it pulls in torch
and is large and slow to install, so it is excluded from ``requirements.txt``
and imported lazily inside the function below. HIVE degrades cleanly to Otsu
thresholding without it.
"""

import numpy as np
from skimage.morphology import remove_small_objects
from .preprocessing import ensure_2d


def segment_with_cellpose(
    image_2d,
    model_type="cyto3",
    diameter=0,
    cellprob_threshold=0.0,
    flow_threshold=0.4,
    gpu=False,
    min_size=0,
):
    """Segment a 2D fluorescence image with a pretrained Cellpose model.

    Args:
        image_2d: Image to segment; flattened to 2D and rescaled to [0, 1]
            because Cellpose expects normalised input.
        model_type: Pretrained model name (``cyto3``, ``cyto2``, ``cyto``,
            ``nuclei``).
        diameter: Expected object diameter in px. ``0`` or ``None`` lets
            Cellpose estimate it, which is slower and less reliable on sparse
            fields -- supply a measured value when you know it.
        cellprob_threshold: Lower values recover more, dimmer objects.
        flow_threshold: Maximum allowed flow error; lower is stricter.
        gpu: Use GPU if a supported device is available.
        min_size: Objects below this area (px) are dropped after inference.

    Returns:
        Boolean mask of detected objects.

    Raises:
        ImportError: If Cellpose is not installed, with the install command.

    Note:
        Cellpose returns instance labels, which are collapsed to a boolean
        mask here and relabelled downstream. Objects that Cellpose separated
        but which physically touch may therefore merge again.
    """
    # Imported lazily so the rest of HIVE runs without torch installed.
    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError(
            "Cellpose is not installed. Install with: python3 -m pip install cellpose torch torchvision"
        ) from exc

    img = ensure_2d(image_2d).astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    diameter_value = None if diameter is None or float(diameter) <= 0 else float(diameter)

    # Cellpose renamed its entry-point class across versions; try the modern
    # CellposeModel first and fall back to the older Cellpose wrapper.
    try:
        model = models.CellposeModel(gpu=bool(gpu), model_type=model_type)
    except Exception:
        model = models.Cellpose(gpu=bool(gpu), model_type=model_type)

    result = model.eval(
        img,
        channels=[0, 0],
        diameter=diameter_value,
        cellprob_threshold=float(cellprob_threshold),
        flow_threshold=float(flow_threshold),
    )
    masks = result[0]
    binary = masks > 0
    if min_size and int(min_size) > 0:
        binary = remove_small_objects(binary.astype(bool), min_size=int(min_size))
    return binary.astype(bool)
