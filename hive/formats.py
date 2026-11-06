"""Microscope file formats: OME-TIFF, ImageJ TIFF, Zeiss CZI, Nikon ND2 and Leica LIF.

Each reader returns a :class:`MicroscopeImage`: pixels reordered to
``(C, Z, Y, X)``, the channel names stored in the file, and the voxel size in
micrometres. That removes the two guesses the plain-image path has to make: which
axis is the channel, and what one pixel is worth.

Time series and multi-scene files are refused with a clear message instead of
being flattened, because silently mixing timepoints into one z-stack gives a
plausible but wrong 3D reconstruction.

Support status (see the README): OME-TIFF and ImageJ TIFF are exercised by tests
with files written by tifffile. CZI, ND2 and LIF are written against the
documented APIs of ``czifile``, ``nd2`` and ``readlif``; the tests cover the
metadata and axis handling with stand-ins, not real vendor files.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from .config import HiveConfig
from .io_utils import ImageLoadError

# Formats read through this module regardless of content.
VENDOR_EXTENSIONS: frozenset[str] = frozenset({".nd2", ".lif", ".czi"})
_TIFF_EXTENSIONS = frozenset({".tif", ".tiff", ".lsm"})

# Length units to micrometres.
_TO_MICRONS = {
    "nm": 1e-3, "µm": 1.0, "μm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0,
    "mm": 1e3, "cm": 1e4, "m": 1e6, "angstrom": 1e-4, "å": 1e-4,
}


@dataclass
class MicroscopeImage:
    """One image or stack with its acquisition metadata.

    Attributes:
        data: Pixels as ``(C, Z, Y, X)`` in the file's own dtype (not converted,
            so nothing is rescaled or copied until a channel is asked for).
            ``None`` when only metadata was requested.
        shape: ``(C, Z, Y, X)``; set even when ``data`` is ``None``.
        channel_names: One name per channel, from the file where it stores any.
        voxel_size_xy: Pixel size in micrometres, or ``None`` if the file has none.
        voxel_size_z: Z step in micrometres, or ``None``.
        unit: Always ``"um"`` once a size is known.
        source_format: ``"ome-tiff"``, ``"imagej-tiff"``, ``"czi"``, ``"nd2"``,
            ``"lif"`` or ``"tiff"``.
        notes: Anything worth telling the user (names invented, series skipped).
    """

    data: np.ndarray | None
    shape: tuple[int, int, int, int]
    channel_names: list[str]
    voxel_size_xy: float | None = None
    voxel_size_z: float | None = None
    unit: str = "um"
    source_format: str = "tiff"
    notes: list[str] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        """int: Number of channels."""
        return self.shape[0]

    def channel_index(self, key: str | int) -> int:
        """Resolve a channel name or number to an index.

        Args:
            key: A channel name from the file (case-insensitive), or a 0-based
                number as an int, ``"3"`` or ``"channel_3"``.

        Returns:
            The 0-based channel index.

        Raises:
            ValueError: If nothing matches.
        """
        text = str(key).strip()
        lowered = [name.lower() for name in self.channel_names]
        if text.lower() in lowered:
            return lowered.index(text.lower())
        match = re.fullmatch(r"(?:channel_)?(\d+)", text.lower())
        if match and int(match.group(1)) < self.n_channels:
            return int(match.group(1))
        raise ValueError(
            f"Channel {key!r} not found. This file has: "
            + ", ".join(f"{i}={name}" for i, name in enumerate(self.channel_names))
        )

    def volume(self, key: str | int) -> np.ndarray:
        """One channel as a float32 ``(Z, Y, X)`` volume.

        Args:
            key: Channel name or number, see :meth:`channel_index`.

        Returns:
            A new float32 array in the original intensity scale.

        Raises:
            ImageLoadError: If the pixels were not loaded.
        """
        if self.data is None:
            raise ImageLoadError("Only metadata was read; load the file to get pixels")
        return np.asarray(self.data[self.channel_index(key)], dtype=np.float32)

    def plane(self, key: str | int) -> np.ndarray:
        """One channel of a single-plane image as float32 ``(Y, X)``.

        Raises:
            ImageLoadError: If the file has more than one z-plane.
        """
        if self.shape[1] != 1:
            raise ImageLoadError(
                f"This file has {self.shape[1]} z-planes; use 3D mode or pick one plane."
            )
        return self.volume(key)[0]


def is_vendor_file(path: str | Path) -> bool:
    """Whether a path is a CZI, ND2 or LIF file (by extension)."""
    return Path(path).suffix.lower() in VENDOR_EXTENSIONS


def is_microscope_file(path: str | Path) -> bool:
    """Whether the file should be read through this module.

    True for CZI/ND2/LIF, and for TIFFs that declare themselves as OME-TIFF or as
    a multi-channel ImageJ hyperstack. Plain TIFF, PNG and JPEG use the ordinary
    image path.

    Args:
        path: File to test.

    Returns:
        ``True`` if :func:`read_microscope` should be used.
    """
    path = Path(path)
    if is_vendor_file(path):
        return True
    if path.suffix.lower() not in _TIFF_EXTENSIONS or not path.is_file():
        return False
    try:
        with tifffile.TiffFile(str(path)) as tif:
            if tif.is_ome:
                return True
            return bool(tif.is_imagej and "C" in tif.series[0].axes.upper())
    except Exception:  # noqa: BLE001 - not a readable TIFF: let the normal loader report it
        return False


def _to_um(value: float | None, unit: str | None) -> float | None:
    """Convert a length to micrometres; ``None`` when the unit is not a length."""
    if value is None or value <= 0:
        return None
    factor = _TO_MICRONS.get((unit or "µm").strip().lower())
    return float(value) * factor if factor else None


def _to_czyx(arr: np.ndarray, axes: str, source: str) -> tuple[np.ndarray, list[str]]:
    """Reorder an array to ``(C, Z, Y, X)``.

    Args:
        arr: Pixels in the file's axis order.
        axes: One letter per axis (``T``, ``C``, ``Z``, ``Y``, ``X`` and others).
        source: File name, for messages.

    Returns:
        The reordered array (a view where possible) and a list of notes.

    Raises:
        ImageLoadError: If an axis other than C/Z/Y/X is longer than 1.
    """
    axes = axes.upper()
    if arr.ndim != len(axes):
        raise ImageLoadError(f"{source}: axes '{axes}' do not match array shape {arr.shape}")
    notes: list[str] = []

    # tifffile labels a plain multi-page TIFF 'I' or 'Q'; in a stack these are planes.
    if "Z" not in axes:
        for stand_in in ("I", "Q"):
            if stand_in in axes and arr.shape[axes.index(stand_in)] > 0:
                axes = axes.replace(stand_in, "Z", 1)
                break
    # 'S' = samples per pixel (an RGB camera): treat as channels.
    if "C" not in axes and "S" in axes:
        axes = axes.replace("S", "C", 1)

    for letter in [a for a in axes if a not in "CZYX"]:
        size = arr.shape[axes.index(letter)]
        if size != 1:
            meaning = {"T": "time points", "S": "samples", "P": "positions",
                       "M": "mosaic tiles", "V": "views", "B": "blocks"}.get(letter, "values")
            raise ImageLoadError(
                f"{source} has {size} {meaning} (axis {letter}). HIVE reads one at a time; "
                "export a single time point / position from the microscope software or "
                "with Fiji first. (Time-lapse analysis is a planned feature.)"
            )
        arr = np.squeeze(arr, axis=axes.index(letter))
        axes = axes.replace(letter, "", 1)

    for letter in "CZ":
        if letter not in axes:
            arr = arr[np.newaxis, ...]
            axes = letter + axes
    if "Y" not in axes or "X" not in axes:
        raise ImageLoadError(f"{source}: no Y/X axes in '{axes}'")
    order = [axes.index(letter) for letter in "CZYX"]
    return np.transpose(arr, order), notes


def _strip_ns(tag: str) -> str:
    """Drop the ``{namespace}`` prefix from an XML tag."""
    return tag.rsplit("}", 1)[-1]


def _read_ome(tif: tifffile.TiffFile, load: bool, name: str) -> MicroscopeImage:
    """Read a TIFF that carries OME-XML."""
    series = tif.series[0]
    notes = []
    if len(tif.series) > 1:
        notes.append(f"{len(tif.series)} image series in the file; read the first only.")
    root = ET.fromstring(tif.ome_metadata)
    pixels = next((e for e in root.iter() if _strip_ns(e.tag) == "Pixels"), None)
    names, vx, vz = [], None, None
    if pixels is not None:
        names = [c.get("Name") or "" for c in pixels if _strip_ns(c.tag) == "Channel"]
        vx = _to_um(_float(pixels.get("PhysicalSizeX")), pixels.get("PhysicalSizeXUnit"))
        vz = _to_um(_float(pixels.get("PhysicalSizeZ")), pixels.get("PhysicalSizeZUnit"))
    return _assemble(series, load, names, vx, vz, "ome-tiff", name, notes)


def _read_imagej(tif: tifffile.TiffFile, load: bool, name: str) -> MicroscopeImage:
    """Read an ImageJ TIFF/hyperstack, including the scale it stores."""
    series = tif.series[0]
    meta = tif.imagej_metadata or {}
    page = tif.pages[0]
    unit = meta.get("unit")
    vx = None
    resolution = page.tags.get("XResolution")
    if resolution is not None:
        num, den = resolution.value
        vx = _to_um(den / num if num else None, unit)
    vz = _to_um(_float(meta.get("spacing")), unit)
    return _assemble(series, load, [], vx, vz, "imagej-tiff", name, [])


def _float(text: object) -> float | None:
    """Parse a number from metadata, or ``None``."""
    try:
        return float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _assemble(series, load: bool, names: list[str], vx, vz, kind: str, name: str,
              notes: list[str]) -> MicroscopeImage:
    """Turn a tifffile series into a :class:`MicroscopeImage`."""
    axes = series.axes
    if load:
        data, extra = _to_czyx(series.asarray(), axes, name)
        notes = notes + extra
        shape = tuple(data.shape)
    else:
        data, shape = None, _czyx_shape(series.shape, axes, name)
    count = shape[0]
    if len(names) != count or not all(names):
        if names:
            notes.append("Some channel names are missing in the file; numbers used for those.")
        names = [(names[i] if i < len(names) and names[i] else f"channel_{i}")
                 for i in range(count)]
    return MicroscopeImage(data, shape, names, vx, vz, "um", kind, notes)  # type: ignore[arg-type]


def _czyx_shape(shape: tuple[int, ...], axes: str, name: str) -> tuple[int, int, int, int]:
    """The ``(C, Z, Y, X)`` shape for a given file shape, without reading pixels."""
    probe = np.lib.stride_tricks.as_strided(np.zeros(1, dtype=np.uint8), shape=shape,
                                            strides=(0,) * len(shape), writeable=False)
    arr, _ = _to_czyx(probe, axes, name)
    return tuple(arr.shape)  # type: ignore[return-value]


def _read_tiff(path: Path, load: bool) -> MicroscopeImage:
    """Read OME-TIFF, an ImageJ TIFF, or a plain TIFF with whatever it carries."""
    with tifffile.TiffFile(str(path)) as tif:
        if tif.is_ome:
            return _read_ome(tif, load, path.name)
        if tif.is_imagej:
            return _read_imagej(tif, load, path.name)
        return _assemble(tif.series[0], load, [], None, None, "tiff", path.name, [])


def _read_czi(path: Path, load: bool) -> MicroscopeImage:
    """Read a Zeiss CZI: pixels from ``czifile``, scale and channel names from its XML."""
    try:
        import czifile
    except ImportError as exc:
        raise ImageLoadError("Reading .czi requires the 'czifile' package: pip install czifile") from exc
    with czifile.CziFile(str(path)) as czi:
        axes = czi.axes
        try:
            xml = czi.metadata(raw=True)
        except TypeError:
            xml = czi.metadata()
        array = czi.asarray() if load else None
        shape_in = tuple(czi.shape)
    root = ET.fromstring(xml)

    def distance(axis: str) -> float | None:
        for node in root.iter():
            if _strip_ns(node.tag) == "Distance" and node.get("Id") == axis:
                value = next((c.text for c in node if _strip_ns(c.tag) == "Value"), None)
                return _to_um(_float(value), "m")  # CZI stores metres
        return None

    names = [n.get("Name") or "" for n in root.iter()
             if _strip_ns(n.tag) == "Channel" and n.get("Name")
             and _inside(root, n, "DisplaySetting")]
    notes: list[str] = []
    if load:
        data, extra = _to_czyx(array, axes, path.name)
        notes += extra
        shape = tuple(data.shape)
    else:
        data, shape = None, _czyx_shape(shape_in, axes, path.name)
    names = _fit_names(names, shape[0], notes)
    return MicroscopeImage(data, shape, names, distance("X"), distance("Z"), "um", "czi", notes)  # type: ignore[arg-type]


def _inside(root: ET.Element, node: ET.Element, ancestor_tag: str) -> bool:
    """Whether ``node`` sits under an element called ``ancestor_tag``."""
    for parent in root.iter():
        if _strip_ns(parent.tag) == ancestor_tag and any(n is node for n in parent.iter()):
            return True
    return False


def _fit_names(names: list[str], count: int, notes: list[str]) -> list[str]:
    """Make exactly ``count`` names, falling back to ``channel_N`` where unknown."""
    if len(names) != count:
        if names:
            notes.append(f"File lists {len(names)} channel names for {count} channels; "
                         "using numbers instead.")
        return [f"channel_{i}" for i in range(count)]
    return [n or f"channel_{i}" for i, n in enumerate(names)]


def _read_nd2(path: Path, load: bool) -> MicroscopeImage:
    """Read a Nikon ND2 with the ``nd2`` package."""
    try:
        import nd2
    except ImportError as exc:
        raise ImageLoadError("Reading .nd2 requires the 'nd2' package: pip install nd2") from exc
    notes: list[str] = []
    with nd2.ND2File(str(path)) as handle:
        axes = "".join(handle.sizes.keys())
        shape_in = tuple(handle.sizes.values())
        voxel = handle.voxel_size()
        names: list[str] = []
        try:
            names = [c.channel.name for c in handle.metadata.channels]
        except Exception:  # noqa: BLE001 - metadata layouts vary between NIS versions
            pass
        array = handle.asarray() if load else None
    if "S" in axes and "C" not in axes and shape_in[axes.index("S")] == 3:
        names = ["red", "green", "blue"]
    if load:
        data, extra = _to_czyx(array, axes, path.name)
        notes += extra
        shape = tuple(data.shape)
    else:
        data, shape = None, _czyx_shape(shape_in, axes, path.name)
    names = _fit_names(names, shape[0], notes)
    vz = float(voxel.z) if getattr(voxel, "z", None) and voxel.z > 0 else None
    vx = float(voxel.x) if getattr(voxel, "x", None) and voxel.x > 0 else None
    return MicroscopeImage(data, shape, names, vx, vz, "um", "nd2", notes)  # type: ignore[arg-type]


def _read_lif(path: Path, load: bool, image_index: int = 0) -> MicroscopeImage:
    """Read one image from a Leica LIF with ``readlif``.

    A LIF file is a container of many images; the first is read and the rest are
    named in the notes. Channel names are the lookup-table colours Leica records
    (the LIF header holds no free-text channel names).
    """
    try:
        from readlif.reader import LifFile
    except ImportError as exc:
        raise ImageLoadError("Reading .lif requires the 'readlif' package: pip install readlif") from exc
    lif = LifFile(str(path))
    images = list(lif.get_iter_image())
    if not images:
        raise ImageLoadError(f"{path.name} contains no images")
    image = images[image_index]
    notes = []
    if len(images) > 1:
        notes.append(f"{len(images)} images in the file; read '{image.name}'. Others: "
                     + ", ".join(i.name for i in images if i is not image))
    if image.nt > 1:
        raise ImageLoadError(f"{path.name}: '{image.name}' has {image.nt} time points; "
                             "time-lapse analysis is a planned feature.")
    if getattr(image, "n_mosaic", 1) > 1:
        raise ImageLoadError(f"{path.name}: '{image.name}' is a mosaic of {image.n_mosaic} tiles.")

    n_c, n_z = int(image.channels), int(image.nz)
    height, width = int(image.dims.y), int(image.dims.x)
    data = None
    if load:
        first = np.asarray(image.get_frame(z=0, t=0, c=0, m=0))
        data = np.empty((n_c, n_z, height, width), dtype=first.dtype)
        for c in range(n_c):
            for z in range(n_z):
                data[c, z] = np.asarray(image.get_frame(z=z, t=0, c=c, m=0))

    names = []
    for element in lif.xml_root.iter("Element"):
        if element.get("Name") == image.name:
            found = element.findall("./Data/Image/ImageDescription/Channels/ChannelDescription")
            if len(found) == n_c:
                names = [c.get("LUTName") or "" for c in found]
                break
    names = _fit_names(names, n_c, notes)

    def size(axis: int) -> float | None:
        scale = (image.scale_n or {}).get(axis)  # pixels per micrometre
        return 1.0 / scale if scale else None

    return MicroscopeImage(data, (n_c, n_z, height, width), names, size(1), size(3), "um",
                           "lif", notes)


def read_microscope(path: str | Path, load_pixels: bool = True) -> MicroscopeImage:
    """Read an image or stack with its channel names and voxel size.

    Args:
        path: OME-TIFF, ImageJ TIFF, plain TIFF, CZI, ND2 or LIF.
        load_pixels: ``False`` reads only the metadata, which is quick even for
            very large files (not for LIF, where the library reads the header).

    Returns:
        A :class:`MicroscopeImage` with pixels in ``(C, Z, Y, X)`` order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImageLoadError: If it cannot be read, needs a missing package, or holds
            several time points or scenes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix in _TIFF_EXTENSIONS:
            return _read_tiff(path, load_pixels)
        if suffix == ".czi":
            return _read_czi(path, load_pixels)
        if suffix == ".nd2":
            return _read_nd2(path, load_pixels)
        if suffix == ".lif":
            return _read_lif(path, load_pixels)
    except ImageLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - report any decoder failure by file name
        raise ImageLoadError(f"Could not read {path.name}: {exc}") from exc
    raise ImageLoadError(f"{path.name}: not a microscope format HIVE knows ({suffix})")


def apply_metadata(cfg: HiveConfig, meta: MicroscopeImage) -> list[str]:
    """Fill in voxel size and unit from a file, unless the user already set them.

    A value is only replaced while it is still the default (1.0 / ``px``), so a
    size typed by the user always wins over the file.

    Args:
        cfg: Configuration to update in place.
        meta: Metadata of the file about to be analysed.

    Returns:
        A readable line for each value taken from the file.
    """
    applied = []
    if meta.voxel_size_xy and cfg.voxel_size_xy == 1.0 and cfg.voxel_unit == "px":
        cfg.voxel_size_xy = meta.voxel_size_xy
        cfg.voxel_unit = meta.unit
        applied.append(f"pixel size {meta.voxel_size_xy:.4g} {meta.unit} (from file)")
        if meta.voxel_size_z and cfg.voxel_size_z == 1.0:
            cfg.voxel_size_z = meta.voxel_size_z
            applied.append(f"z step {meta.voxel_size_z:.4g} {meta.unit} (from file)")
    return applied
