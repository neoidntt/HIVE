"""LEGACY (HIVE 1.x) image reading and writing helpers.

.. deprecated:: 2.0
    Superseded by :mod:`hive.io_utils`, which preserves voxel spacing metadata
    on export so the resulting TIFFs open at the correct scale in
    Fiji/napari/Imaris. Nothing written here carries physical calibration.
"""

from pathlib import Path
import numpy as np
import tifffile
from skimage.io import imread, imsave

# Extensions accepted anywhere the legacy pipeline scans a folder. Note the
# absence of .czi and other vendor formats -- 2.0 added those via hive.io_utils.
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def save_uint8(path, image):
    """Write ``image`` as an 8-bit PNG/TIFF preview.

    Args:
        path: Destination file path.
        image: Any array-like. Empty arrays are skipped silently.

    Warning:
        This is a **display** writer, not a data writer. Any array whose
        maximum exceeds 1 is divided by *its own maximum*, so each file is
        independently contrast-stretched and intensities are not comparable
        between saved images. Read the exported CSV for quantitative values.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return
    # Per-image rescale: convenient for eyeballing, fatal for quantification.
    if arr.max() > 1:
        arr = arr / arr.max()
    arr = np.clip(arr, 0, 1)
    imsave(str(path), (arr * 255).astype(np.uint8), check_contrast=False)


def collect_image_paths(folder):
    """List supported image files in ``folder``, sorted by name.

    Args:
        folder: Directory to scan. A missing directory yields ``[]`` rather
            than raising, so batch callers can report "no images found".

    Returns:
        Sorted list of :class:`~pathlib.Path`. Sorting matters for z-stacks
        built from one file per slice, where filename order *is* slice order.
    """
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])


def load_zstack(path):
    """Load a z-stack from a multi-page TIFF or a folder of slice images.

    Args:
        path: Either a single file (read whole) or a directory whose supported
            images are stacked along a new leading Z axis in filename order.

    Returns:
        A NumPy array. Axis 0 is Z when a folder was given; for a single file
        the on-disk axis order is returned unchanged.

    Raises:
        FileNotFoundError: If ``path`` does not exist, or a directory contains
            no supported images.
        ValueError: If the slice images differ in XY shape, which would make
            the stack ragged.
    """
    path = Path(path)
    if path.is_file():
        arr = tifffile.imread(str(path)) if path.suffix.lower() in {".tif", ".tiff"} else imread(str(path))
        return np.asarray(arr)
    if path.is_dir():
        files = collect_image_paths(path)
        if not files:
            raise FileNotFoundError(f"No z-stack image slices found in {path}")
        slices = [imread(str(f)) for f in files]
        shapes = [np.asarray(s).shape[:2] for s in slices]
        if len(set(shapes)) != 1:
            raise ValueError(f"All z-stack slices must have same XY shape. Found: {set(shapes)}")
        return np.stack(slices, axis=0)
    raise FileNotFoundError(f"Invalid z-stack path: {path}")
