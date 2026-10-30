"""Image reading and writing.

Memory note: the legacy loader built a Python list of decoded slices and then
called :func:`numpy.stack`, holding both the list and the stacked copy at once.
Combined with downstream copies this peaked at **7.8x the input size** (measured:
31.5 MB stack -> 243.9 MB peak). :func:`load_zstack` here preallocates the output
array and decodes each slice directly into its row, so the peak is the array plus
one slice.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import tifffile
from skimage.io import imread, imsave

VALID_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".czi", ".lsm", ".nd2"}
)
_TIFF_EXTENSIONS: frozenset[str] = frozenset({".tif", ".tiff", ".lsm"})

# Measured peak resident memory of the 3D front-end (baseline subtraction ->
# preprocessing -> segmentation), as a multiple of the float32 volume size.
# Measured by real RSS across 0.03-0.7 GB volumes: 11.5x, 10.4x, 10.1x, 9.0x.
# Used only to make the "too large" error message actionable.
PIPELINE_MEMORY_FACTOR: float = 10.0

# Fraction of physical RAM a single loaded volume may occupy by default. The
# pipeline needs several more copies on top, so this is a loading guard, not a
# promise that the run will fit.
_DEFAULT_MEMORY_FRACTION: float = 0.5


class ImageLoadError(RuntimeError):
    """Raised when an image or stack cannot be read or is malformed."""


class StackTooLargeError(MemoryError):
    """Raised when a stack would not fit in memory.

    Subclasses :class:`MemoryError` so it is caught by generic out-of-memory
    handling, but carries an actionable message instead of dying silently. The
    alternative is worse: NumPy asks for more than RAM, the OS starts swapping,
    and a Jupyter kernel eventually gets killed with no explanation.
    """


def _physical_memory_bytes() -> int | None:
    """Total physical RAM, or ``None`` if it cannot be determined.

    Uses :func:`os.sysconf`, which works on macOS and Linux without pulling in
    ``psutil`` as a dependency.

    Returns:
        Bytes of physical RAM, or ``None`` on platforms that do not report it.
    """
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        return None


def _default_max_bytes() -> int | None:
    """Default allocation ceiling for one loaded volume.

    Returns:
        Half of physical RAM in bytes, or ``None`` if RAM is unknown (in which
        case no guard is applied).
    """
    total = _physical_memory_bytes()
    return int(total * _DEFAULT_MEMORY_FRACTION) if total else None


def _fmt_bytes(n: float) -> str:
    """Format a byte count in the unit a human would use.

    Args:
        n: Byte count.

    Returns:
        A short string such as ``"70.0 GB"`` or ``"1.9 MB"``.
    """
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n:.0f} B"


def _check_allocation(n_bytes: int, shape: tuple[int, ...], max_bytes: int | None,
                      source: str) -> None:
    """Refuse an allocation that cannot plausibly fit in memory.

    Args:
        n_bytes: Size of the float32 array that would be allocated.
        shape: Shape of that array, for the message.
        max_bytes: Ceiling. ``None`` means "no guard".
        source: File or folder name, for the message.

    Raises:
        StackTooLargeError: If ``n_bytes`` exceeds ``max_bytes``.
    """
    if max_bytes is None or n_bytes <= max_bytes:
        return
    total = _physical_memory_bytes()
    ram = _fmt_bytes(total) if total else "unknown"
    raise StackTooLargeError(
        f"'{source}' is {shape} which needs {_fmt_bytes(n_bytes)} as float32 — over the "
        f"{_fmt_bytes(max_bytes)} load limit on this machine ({ram} RAM).\n"
        f"Running the full 3D pipeline on it would peak near "
        f"{_fmt_bytes(n_bytes * PIPELINE_MEMORY_FACTOR)} "
        f"(~{PIPELINE_MEMORY_FACTOR:.0f}x the volume, measured).\n"
        "HIVE loads volumes whole; it has no chunked/out-of-core path. Options:\n"
        "  • crop to a region of interest, or downsample (Fiji: Image > Scale)\n"
        "  • split a time series and analyse one timepoint at a time\n"
        "  • for genuinely huge volumes use chunked tooling (zarr / dask-image / BigDataViewer)\n"
        "  • override deliberately with load_zstack(path, max_bytes=...) if you know it fits"
    )


def _peek_stack_bytes(path: Path) -> tuple[int, tuple[int, ...]] | None:
    """Read a TIFF's shape from its header without loading the pixel data.

    Returns:
        Tuple of (float32 size in bytes, shape), or ``None`` for formats whose
        size cannot be determined without decoding.
    """
    if path.suffix.lower() not in _TIFF_EXTENSIONS:
        return None
    try:
        with tifffile.TiffFile(str(path)) as tif:
            shape = tuple(tif.series[0].shape)
    except Exception:
        return None
    return int(np.prod(shape)) * 4, shape


def memory_report(source: str | Path | tuple[int, ...], workers: int = 1,
                  use_clahe: bool = True) -> dict[str, Any]:
    """Estimate whether an image or stack can be processed on this machine.

    All multipliers here are measured, not guessed:

    * The pipeline peaks at roughly ``PIPELINE_MEMORY_FACTOR`` x the float32
      volume (measured across 0.03-0.7 GB volumes after the in-place rewrite).
    * CLAHE adds a further ~18x on a large 2D image and is the dominant cost
      there; it is not reducible via ``kernel_size``.
    * Each parallel worker holds its own full working set.

    Args:
        source: Path to a TIFF, or an explicit ``(Z, Y, X)`` / ``(Y, X)`` shape.
        workers: Planned parallel workers.
        use_clahe: Whether CLAHE will run. For a large 2D image this dominates.

    Returns:
        Dict with ``float32_bytes``, ``peak_bytes``, ``total_bytes``,
        ``ram_bytes``, ``fits``, ``max_workers`` and a printable ``summary``.

    Raises:
        ValueError: If a path's shape cannot be determined.

    Example:
        >>> report = memory_report((109, 2048, 2048), workers=4, use_clahe=False)
        >>> report["fits"] in (True, False)
        True
    """
    if isinstance(source, (str, Path)):
        peek = _peek_stack_bytes(Path(source))
        if peek is None:
            raise ValueError(
                f"Cannot determine the shape of {source} without decoding it; "
                "pass an explicit shape tuple instead."
            )
        f32_bytes, shape = peek
    else:
        shape = tuple(source)
        f32_bytes = int(np.prod(shape)) * 4

    # A large 2D image is dominated by CLAHE; a volume by the pipeline itself.
    is_big_2d = len(shape) == 2 and int(np.prod(shape)) > 50_000_000
    factor = 18.0 if (use_clahe and is_big_2d) else PIPELINE_MEMORY_FACTOR
    peak = f32_bytes * factor
    total = peak * max(1, workers)
    ram = _physical_memory_bytes()
    usable = ram * 0.85 if ram else None
    fits = bool(usable and total <= usable)
    max_workers = max(1, int(usable // peak)) if usable else 1

    lines = [
        f"shape            {shape}",
        f"as float32       {_fmt_bytes(f32_bytes)}",
        f"peak per worker  {_fmt_bytes(peak)}  ({factor:.0f}x"
        f"{', CLAHE-dominated' if factor == 18.0 else ''})",
        f"workers          {workers}  ->  {_fmt_bytes(total)} total",
        f"machine RAM      {_fmt_bytes(ram) if ram else 'unknown'}",
        f"verdict          {'FITS' if fits else 'DOES NOT FIT'}"
        f"   (max safe workers here: {max_workers})",
    ]
    if not fits and use_clahe and is_big_2d:
        lines.append("hint             use_clahe=False cuts this ~2.7x on large 2D images")

    return {
        "shape": shape, "float32_bytes": f32_bytes, "peak_bytes": peak,
        "total_bytes": total, "ram_bytes": ram, "fits": fits,
        "max_workers": max_workers, "summary": "\n".join(lines),
    }


def collect_image_paths(folder: str | Path) -> list[Path]:
    """List readable image files in a folder, sorted by name.

    Sorting is lexicographic, which orders zero-padded slice names correctly
    (``slice_02`` before ``slice_10``) but NOT unpadded ones (``slice_10``
    before ``slice_2``). A warning is emitted when unpadded numbering is
    detected, because a silently mis-ordered z-stack produces a plausible but
    wrong reconstruction.

    Args:
        folder: Directory to scan. Non-existent directories yield ``[]``.

    Returns:
        Sorted list of image paths.
    """
    folder = Path(folder)
    if not folder.exists():
        return []
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and not p.name.startswith(".")
    )
    _warn_on_ambiguous_ordering(paths)
    return paths


def _warn_on_ambiguous_ordering(paths: Sequence[Path]) -> None:
    """Warn when filenames carry unpadded numbers that will sort incorrectly.

    Args:
        paths: Candidate slice paths, already lexicographically sorted.
    """
    import re

    numbers = []
    for p in paths:
        found = re.findall(r"\d+", p.stem)
        if found:
            numbers.append(found[-1])
    if len(numbers) < 2:
        return
    if len({len(n) for n in numbers}) > 1:
        warnings.warn(
            "Slice filenames use inconsistent zero-padding (e.g. 'z2' and 'z10'), so "
            "lexicographic ordering may not match acquisition order. Rename to a fixed "
            "width (z02, z10) to guarantee correct z-ordering.",
            stacklevel=3,
        )


def load_image(path: str | Path) -> np.ndarray:
    """Read a single image, preserving its original intensity scale.

    Unlike the legacy loader this does NOT call ``img_as_float``, which rescales
    by the dtype maximum (dividing a uint16 image by 65535). That conversion is
    lossless in ratio terms but obscures the true scale; conversion is deferred
    to the point of use.

    Args:
        path: Image file path.

    Returns:
        The image as float32, in its original intensity scale.

    Raises:
        FileNotFoundError: If the path does not exist.
        ImageLoadError: If the file cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        if path.suffix.lower() == ".czi":
            arr = _read_czi(path)
        elif path.suffix.lower() in _TIFF_EXTENSIONS:
            arr = tifffile.imread(str(path))
        else:
            arr = imread(str(path))
    except Exception as exc:
        raise ImageLoadError(f"Could not decode {path.name}: {exc}") from exc
    return np.asarray(arr, dtype=np.float32)


def _read_czi(path: Path) -> np.ndarray:
    """Read a Zeiss CZI file and drop singleton acquisition axes.

    CZI stores a full acquisition hypercube (scene, time, channel, z, y, x, ...);
    squeezing singleton axes reduces the common single-scene single-timepoint
    case to something the rest of the pipeline understands.

    Args:
        path: CZI file path.

    Returns:
        Squeezed image array.

    Raises:
        ImageLoadError: If ``czifile`` / ``aicspylibczi`` are not installed.
    """
    try:
        import czifile
    except ImportError as exc:
        raise ImageLoadError(
            "Reading .czi requires the 'czifile' package: pip install czifile"
        ) from exc
    with czifile.CziFile(str(path)) as czi:
        return np.squeeze(czi.asarray())


def load_zstack(path: str | Path, max_bytes: int | None = None) -> np.ndarray:
    """Load a z-stack from a multi-page file or a folder of slices.

    Preallocates the destination array and decodes each slice into it, avoiding
    the list-plus-stack duplication in the legacy loader.

    Args:
        path: Multi-page TIFF/CZI file, or a directory of single-plane images.
        max_bytes: Refuse to allocate more than this many bytes for the float32
            volume. ``None`` (default) uses half of physical RAM. Pass a large
            value to override deliberately. Note this guards the *load*; the full
            pipeline needs roughly PIPELINE_MEMORY_FACTOR times more on top.

    Returns:
        Array of shape ``(Z, Y, X)`` for grayscale or ``(Z, Y, X, C)`` for
        multi-channel input, float32.

    Raises:
        FileNotFoundError: If the path does not exist or a folder holds no images.
        ValueError: If slices have mismatched XY shapes.
        StackTooLargeError: If the stack would exceed ``max_bytes``.
        ImageLoadError: If a slice cannot be decoded.
    """
    path = Path(path)
    limit = _default_max_bytes() if max_bytes is None else max_bytes

    if path.is_file():
        # Check the header BEFORE decoding. The previous version returned here
        # without any guard, so a single oversized TIFF went straight to
        # tifffile.imread and took the process (or the Jupyter kernel) down with it.
        peek = _peek_stack_bytes(path)
        if peek is not None:
            _check_allocation(peek[0], peek[1], limit, path.name)
        arr = load_image(path)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return arr

    if not path.is_dir():
        raise FileNotFoundError(f"Z-stack path does not exist: {path}")

    files = collect_image_paths(path)
    if not files:
        raise FileNotFoundError(
            f"No readable image slices in {path} "
            f"(looked for: {', '.join(sorted(VALID_EXTENSIONS))})"
        )

    probe = load_image(files[0])
    if probe.ndim not in (2, 3):
        raise ValueError(
            f"Slice {files[0].name} has shape {probe.shape}; expected 2D grayscale "
            "or 3D channel-last."
        )

    out_shape = (len(files), *probe.shape)
    _check_allocation(int(np.prod(out_shape)) * 4, out_shape, limit, path.name)

    stack = np.empty(out_shape, dtype=np.float32)
    stack[0] = probe
    del probe
    for index, file_path in enumerate(files[1:], start=1):
        plane = load_image(file_path)
        if plane.shape != stack.shape[1:]:
            raise ValueError(
                f"Slice '{file_path.name}' has shape {plane.shape} but '{files[0].name}' "
                f"has {stack.shape[1:]}. All slices must share XY dimensions."
            )
        stack[index] = plane
    return stack


def iter_image_paths(folder: str | Path) -> Iterator[Path]:
    """Yield image paths one at a time.

    Lets batch callers process a folder without holding every result in memory.

    Args:
        folder: Directory to scan.

    Yields:
        Image paths in sorted order.
    """
    yield from collect_image_paths(folder)


def save_uint8(path: str | Path, image: np.ndarray, rescale: bool = True) -> None:
    """Write an array as an 8-bit PNG for visual QC.

    IMPORTANT: with ``rescale=True`` each image is scaled to its own min/max, so
    brightness is NOT comparable between saved PNGs. These are QC previews, not
    quantitative data -- read the CSV for numbers. The legacy version rescaled
    unconditionally and silently.

    Args:
        path: Output path.
        image: 2D grayscale or 3D RGB array.
        rescale: Stretch to the array's own min/max. When ``False``, values are
            assumed already in ``[0, 1]`` and are merely clipped.

    Raises:
        ValueError: If the array is empty or not 2D/3D.
    """
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"Refusing to write an empty array to {path}")
    if arr.ndim not in (2, 3):
        raise ValueError(f"save_uint8 expects a 2D or 3D array, got shape {arr.shape}")

    if rescale:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    arr = np.clip(np.nan_to_num(arr), 0.0, 1.0)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # skimage's low-contrast warning is expected for masks
        imsave(str(path), (arr * 255).astype(np.uint8), check_contrast=False)


def save_volume_tiff(path: str | Path, volume: np.ndarray, voxel_size_xy: float = 1.0,
                     voxel_size_z: float = 1.0, unit: str = "px") -> None:
    """Write a 3D volume as an ImageJ-compatible TIFF with voxel spacing.

    Embedding spacing means Fiji, napari and Imaris open the stack with correct
    physical scaling instead of assuming isotropic unit voxels.

    Args:
        path: Output ``.tif`` path.
        volume: 3D array ``(Z, Y, X)``.
        voxel_size_xy: Physical pixel size in XY.
        voxel_size_z: Physical z-step.
        unit: Unit label (e.g. ``"um"``).

    Raises:
        ValueError: If ``volume`` is not 3D.
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"save_volume_tiff expects a 3D volume, got shape {vol.shape}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        vol.astype(np.float32),
        imagej=True,
        resolution=(1.0 / voxel_size_xy, 1.0 / voxel_size_xy),
        metadata={"spacing": voxel_size_z, "unit": unit, "axes": "ZYX"},
    )
