"""LEGACY (HIVE 1.x) channel selection and image enhancement.

.. deprecated:: 2.0
    Superseded by :mod:`hive.preprocessing`.

This module is the origin of the HIVE 1.x intensity defect. :func:`preprocess`
performs per-image min-max normalisation *and* CLAHE, and the legacy pipeline
then reuses that same enhanced array as the intensity source for
``regionprops``. Enhancement is a non-linear, per-image remapping, so it
destroys exactly the quantity intensity measurement is trying to preserve.

HIVE 2.0 replaced this with the *two-image contract*: enhancement feeds only
object detection, while measurement reads a background-subtracted image that
keeps its original physical scale.
"""

from pathlib import Path
import numpy as np
from skimage import color, exposure, filters, img_as_float
from skimage.io import imread


def load_image(path: str):
    """Read an image from disk as float.

    Args:
        path: Image file path.

    Returns:
        Array scaled to [0, 1] by :func:`skimage.img_as_float`, which divides
        by the *dtype* maximum (e.g. 65535 for uint16) rather than the data
        maximum, so relative intensities between images survive this step.
    """
    img = imread(path)
    return img_as_float(img)


def ensure_2d(image):
    """Collapse any input to a single 2D plane.

    Args:
        image: 2D, 3D or higher-dimensional array.

    Returns:
        2D ``float32`` array. RGB(A) input is converted with the luminance-
        weighted :func:`skimage.color.rgb2gray`; a 1-2 channel trailing axis
        takes channel 0. Extra singleton axes are squeezed recursively.

    Raises:
        ValueError: If the shape cannot be reduced to 2D.

    Warning:
        A 3D array is treated as channel-last, **not** as a z-stack. Passing a
        (Z, Y, X) volume here silently keeps only the first plane if Z <= 2, or
        luminance-blends the first three planes if Z >= 3.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] >= 3:
            return color.rgb2gray(arr[:, :, :3]).astype(np.float32)
        return arr[:, :, 0].astype(np.float32)
    if arr.ndim > 3:
        return ensure_2d(np.squeeze(arr))
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def select_channel(image, channel="auto"):
    """Extract one analysis channel from an RGB(A) image.

    Args:
        image: 2D (returned unchanged) or channel-last RGB(A) array.
        channel: One of ``auto``, ``red``, ``green``, ``blue``, ``magenta``
            (alias ``pink``), ``gray``.

    Returns:
        2D ``float32`` channel image.

    Raises:
        ValueError: On an unsupported shape or an unknown channel name.

    Note:
        ``magenta`` is the mean of the red and blue display channels -- a
        display-space proxy, not spectral unmixing. Analyse the raw
        single-channel acquisition instead when you have it.

        ``auto`` picks the channel with the highest standard deviation. That
        is a contrast heuristic, not a measurement, and a noisy empty channel
        can win. Worse, the legacy version discards *which* channel it chose
        (it takes ``max`` over ``.values()``), so the choice is unrecoverable
        from the outputs. HIVE 2.0 records the resolved channel in the report.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.shape[-1] < 3:
        return arr[:, :, 0].astype(np.float32)

    rgb = arr[:, :, :3].astype(np.float32)
    channel = channel.lower()

    if channel == "red":
        out = rgb[:, :, 0]
    elif channel == "green":
        out = rgb[:, :, 1]
    elif channel == "blue":
        out = rgb[:, :, 2]
    elif channel in {"magenta", "pink"}:
        out = (rgb[:, :, 0] + rgb[:, :, 2]) / 2.0
    elif channel == "gray":
        out = color.rgb2gray(rgb)
    elif channel == "auto":
        # Choose the channel with the highest contrast/spread.
        candidates = {
            "red": rgb[:, :, 0],
            "green": rgb[:, :, 1],
            "blue": rgb[:, :, 2],
            "magenta": (rgb[:, :, 0] + rgb[:, :, 2]) / 2.0,
            "gray": color.rgb2gray(rgb),
        }
        out = max(candidates.values(), key=lambda x: float(np.std(x)))
    else:
        raise ValueError("channel must be auto, red, green, blue, magenta, or gray")
    return out.astype(np.float32)


def preprocess(image_2d, gaussian_kernel=3, use_clahe=True):
    """Build a contrast-enhanced image for thresholding.

    Applies, in order: per-image min-max normalisation, optional Gaussian
    smoothing, then optional CLAHE (Zuiderveld, 1994).

    Args:
        image_2d: Input image; reduced to 2D first.
        gaussian_kernel: Kernel size in px, converted to ``sigma = kernel / 3``
            (the usual 3-sigma rule for a truncated Gaussian). ``0`` disables.
        use_clahe: Apply adaptive histogram equalisation.

    Returns:
        Enhanced ``float32`` image in [0, 1].

    Warning:
        **Detection only.** Both the normalisation and CLAHE steps are
        non-linear, per-image remappings. Two images differing by a true 3x in
        brightness both emerge spanning [0, 1], so any intensity measured on
        this output reports that difference as 1.0x. The legacy pipeline does
        exactly that -- see the module docstring and
        :func:`src.pipeline.analyze_single_image`.
    """
    img = ensure_2d(image_2d)
    img = img.astype(np.float32)
    # Per-image rescale: this is the step that erases cross-image brightness.
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    if gaussian_kernel and gaussian_kernel > 0:
        # Suppress shot noise so Otsu sees a cleaner bimodal histogram.
        sigma = max(float(gaussian_kernel) / 3.0, 0.1)
        img = filters.gaussian(img, sigma=sigma, preserve_range=True)
    if use_clahe:
        # Local, non-linear contrast: lifts dim objects under uneven
        # illumination, and makes the result unusable for quantification.
        img = exposure.equalize_adapthist(np.clip(img, 0, 1), clip_limit=0.01)
    return img.astype(np.float32)


def save_uploaded_file(uploaded_file, folder: Path):
    """Persist a Streamlit upload to disk under its original filename.

    Args:
        uploaded_file: Streamlit ``UploadedFile``.
        folder: Destination directory, created if absent.

    Returns:
        :class:`~pathlib.Path` of the written file.

    Note:
        Reads the whole upload into memory via ``getbuffer()``. The legacy
        Streamlit front end streams in 1 MB chunks instead for large stacks.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path
