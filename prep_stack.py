#!/usr/bin/env python3
"""Prepare a large multi-channel OME-TIFF for HIVE, without ever loading it whole.

HIVE loads volumes into memory, so a 38 GB stack cannot go in directly. This tool
reads the file **one page at a time**, so peak memory is a single plane (a few
hundred MB) regardless of how large the file is.

Three commands:

``--inspect``
    Print shape, axes, dtype, channels and physical voxel size from the OME
    metadata. Run this first: it tells you which channel index is which dye, and
    the real micron-per-pixel so you never have to guess it.

``--extract-2d``
    Write one channel out as one TIFF per z-plane, ready for ``hive_cli.py --mode 2d``.

``--extract-3d``
    Write one channel as a single downsampled volume, ready for
    ``hive_cli.py --mode 3d``. Downsampling is a **block mean**, which preserves
    the intensity scale (unlike striding, which aliases, or a block sum, which
    would inflate values by the block area).

Examples:
    python3 prep_stack.py --inspect "path/to/stack.ome.tif"

    python3 prep_stack.py --extract-2d "path/to/stack.ome.tif" \\
        --channel 1 --out planes_pdf

    python3 prep_stack.py --extract-3d "path/to/stack.ome.tif" \\
        --channel 1 --downsample 4 --out pdf_volume_ds4.tif
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile


def _series(tif: tifffile.TiffFile):
    """Return the first image series.

    Args:
        tif: An open TiffFile.

    Returns:
        The series describing the full hypercube.
    """
    return tif.series[0]


def _ome_field(metadata: str | None, name: str) -> str | None:
    """Pull one attribute out of OME-XML.

    Args:
        metadata: The OME-XML string, or None.
        name: Attribute name, e.g. ``PhysicalSizeX``.

    Returns:
        The attribute value, or None if absent.
    """
    if not metadata:
        return None
    hit = re.search(rf'{name}="([^"]+)"', metadata)
    return hit.group(1) if hit else None


def inspect(path: Path) -> dict:
    """Print everything you need to configure a HIVE run.

    Args:
        path: OME-TIFF path.

    Returns:
        Dict of the discovered properties.
    """
    with tifffile.TiffFile(str(path)) as tif:
        series = _series(tif)
        meta = tif.ome_metadata if tif.is_ome else None
        shape, axes, dtype = tuple(series.shape), series.axes, series.dtype

        print(f"file        : {path.name}")
        print(f"size on disk: {path.stat().st_size / 1e9:.1f} GB")
        print(f"shape       : {shape}")
        print(f"axes        : {axes}   (T=time C=channel Z=depth Y/X=image)")
        print(f"dtype       : {dtype}")
        print(f"pages       : {len(series.pages)}")
        print(f"z-planes    : {_resolve_dims(axes, shape)[0]}")
        print(f"compression : {series.pages[0].compression}")

        sizes = {a: s for a, s in zip(axes, shape)}
        n_z, n_c, _ = _resolve_dims(axes, shape)
        plane_px = sizes.get("Y", 1) * sizes.get("X", 1)

        print(f"\nchannels    : {n_c}")
        names = re.findall(r'<Channel[^>]*?Name="([^"]+)"', meta or "")
        exc = re.findall(r'ExcitationWavelength="([^"]+)"', meta or "")
        for c in range(n_c):
            label = names[c] if c < len(names) else "?"
            wave = f"  ex={exc[c]}" if c < len(exc) else ""
            print(f"  channel {c} : {label}{wave}")
        if not names:
            print("  (no channel names in metadata — infer from the filename order)")

        px_x = _ome_field(meta, "PhysicalSizeX")
        px_z = _ome_field(meta, "PhysicalSizeZ")
        unit_x = _ome_field(meta, "PhysicalSizeXUnit") or "um"
        print(f"\nvoxel size  : xy={px_x or '?'} {unit_x}   z={px_z or '?'}")
        if px_x:
            print(f"  -> use --voxel-size-xy {float(px_x):.4f}"
                  + (f" --voxel-size-z {float(px_z):.4f}" if px_z else ""))
        else:
            print("  -> NOT in metadata; get it from the acquisition software.")
            print("     Wrong voxel size silently gives wrong physical units.")

        print(f"\nmemory per ONE channel:")
        f32_plane = plane_px * 4
        f32_vol = plane_px * n_z * 4
        print(f"  one plane float32 : {f32_plane / 1e6:.0f} MB  -> ~{f32_plane * 10 / 1e9:.1f} GB peak (2D: fine)")
        print(f"  full volume float32: {f32_vol / 1e9:.1f} GB  -> ~{f32_vol * 10 / 1e9:.0f} GB peak (3D full-res)")
        for ds in (2, 4, 8):
            v = f32_vol / ds ** 2
            print(f"  downsampled {ds}x xy : {v / 1e9:.2f} GB  -> ~{v * 10 / 1e9:.1f} GB peak")

        return {"shape": shape, "axes": axes, "dtype": dtype,
                "n_c": n_c, "n_z": n_z, "px_x": px_x, "px_z": px_z}


def _resolve_dims(axes: str, shape: tuple[int, ...]) -> tuple[int, int, str | None]:
    """Work out which axis is depth and which is channel.

    Not every TIFF labels its axes. A plain 3D ``imwrite`` produces axes ``"QYX"``
    (Q = unspecified), so looking up ``"Z"`` returns nothing. Trusting a default of
    1 there is silently catastrophic: the caller writes ONE plane out of 20 and
    reports success. So an unlabelled leading axis is treated as depth, and a
    genuinely ambiguous file raises instead of guessing.

    Args:
        axes: Axis string, e.g. ``"ZCYX"`` or ``"QYX"``.
        shape: Matching shape.

    Returns:
        Tuple of (n_z, n_channels, depth axis name or None).

    Raises:
        ValueError: If several leading axes could be depth.
    """
    sizes = dict(zip(axes, shape))
    n_c = sizes.get("C", 1)
    leading = [a for a in axes[:-2] if a != "C"]

    if "Z" in sizes:
        z_axis = "Z"
    elif not leading:
        z_axis = None
    elif len(leading) == 1:
        z_axis = leading[0]  # unlabelled (Q/I) or T -- the only candidate
    else:
        raise ValueError(
            f"Cannot tell which axis is depth in axes={axes!r} shape={shape}. "
            "Re-save with explicit OME axis metadata."
        )
    return (sizes.get(z_axis, 1) if z_axis else 1), n_c, z_axis


def read_crop_volume(
    path: Path, crop: tuple[int, int, int, int], channel: int = 0,
    planes: tuple[int, int] | None = None,
) -> np.ndarray:
    """Read a small spatial crop across z **without decoding whole planes**.

    For an *uncompressed* TIFF (``compression: 1``, which the bee stack is) a
    memory map exposes the pixels directly, so slicing a window reads only that
    window's bytes off disk. A 1024x1024 crop of a 8871x19815 plane is 0.6% of it
    -- so this reads ~230 MB instead of the ~38 GB a full-plane pass would. On a
    Google Drive file, where disk I/O dominates, that is the difference between a
    demo you can run now and one you wait 20 minutes for.

    Falls back to per-page decode+crop for compressed files, where random access
    into a plane is not possible.

    Args:
        path: TIFF path.
        crop: ``(y0, y1, x0, x1)`` window in full-resolution pixels.
        channel: Channel index (single-channel files: 0).
        planes: ``(z_start, z_stop)`` half-open z-range, or None for all planes.

    Returns:
        Float32 volume ``(Z, y1-y0, x1-x0)``, held in memory (a crop is small).

    Raises:
        ValueError: If the crop is malformed or exceeds the image.
    """
    y0, y1, x0, x1 = crop
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"crop bounds inverted: {crop}")

    with tifffile.TiffFile(str(path)) as tif:
        series = _series(tif)
        axes, shape = series.axes, tuple(series.shape)
        n_z, n_c, z_axis = _resolve_dims(axes, shape)
        y, x = dict(zip(axes, shape))["Y"], dict(zip(axes, shape))["X"]
        if y1 > y or x1 > x:
            raise ValueError(f"crop {crop} exceeds the image ({y} x {x})")
        z_lo, z_hi = (0, n_z) if planes is None else planes
        uncompressed = series.pages[0].compression == 1

        if uncompressed and n_c == 1 and z_axis is not None:
            # The fast path: map the file and slice only the ROI.
            mm = tif.asarray(out="memmap")
            block = mm[z_lo:z_hi, y0:y1, x0:x1] if mm.ndim == 3 else mm[z_lo:z_hi, ..., y0:y1, x0:x1]
            return np.array(block, dtype=np.float32)  # materialise just the crop

        # Fallback: decode each page, then crop.
        out = np.empty((z_hi - z_lo, y1 - y0, x1 - x0), dtype=np.float32)
        for i, z in enumerate(range(z_lo, z_hi)):
            plane = series.pages[_page_index(axes, shape, z, channel, z_axis)].asarray()
            out[i] = plane[y0:y1, x0:x1]
        return out


def _page_index(axes: str, shape: tuple[int, ...], z: int, c: int,
                z_axis: str | None) -> int:
    """Map a (depth, channel) pair to a flat page index.

    Pages are flattened over the leading axes and the order differs between files
    (``ZCYX`` vs ``CZYX``), so this reads the actual axis order rather than
    assuming one.

    Args:
        axes: Axis string from the series.
        shape: Matching shape.
        z: Depth index.
        c: Channel index.
        z_axis: Name of the depth axis, from :func:`_resolve_dims`.

    Returns:
        Index into ``series.pages``.
    """
    lead_axes, lead_shape = axes[:-2], shape[:-2]
    coords = [z if a == z_axis else (c if a == "C" else 0) for a in lead_axes]
    return int(np.ravel_multi_index(coords, lead_shape)) if lead_shape else 0


def _parse_crop(spec: str | None) -> tuple[int, int, int, int] | None:
    """Parse a ``y0,y1,x0,x1`` crop spec.

    Args:
        spec: The string, or None for no crop.

    Returns:
        Tuple of bounds, or None.

    Raises:
        ValueError: If the spec is malformed or the bounds are inverted.
    """
    if not spec:
        return None
    try:
        y0, y1, x0, x1 = (int(v) for v in spec.split(","))
    except ValueError as exc:
        raise ValueError(f"--crop needs 'y0,y1,x0,x1', got {spec!r}") from exc
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"--crop bounds inverted: {spec}")
    return y0, y1, x0, x1


def _apply_crop(plane: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    """Slice a plane to the crop window.

    Args:
        plane: Full plane.
        crop: Bounds, or None.

    Returns:
        The cropped view.
    """
    if crop is None:
        return plane
    y0, y1, x0, x1 = crop
    return plane[y0:y1, x0:x1]


def extract_2d(path: Path, channel: int, out_dir: Path,
               crop: tuple[int, int, int, int] | None = None) -> None:
    """Write one channel as one TIFF per z-plane.

    Args:
        path: OME-TIFF path.
        channel: Channel index to extract.
        out_dir: Destination folder.

    Raises:
        ValueError: If the channel index is out of range.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffFile(str(path)) as tif:
        series = _series(tif)
        axes, shape = series.axes, tuple(series.shape)
        n_z, n_c, z_axis = _resolve_dims(axes, shape)
        if not 0 <= channel < n_c:
            raise ValueError(f"channel {channel} out of range; file has {n_c} (0..{n_c - 1})")

        for z in range(n_z):
            # asarray() on ONE page decodes only that plane.
            plane = _apply_crop(
                series.pages[_page_index(axes, shape, z, channel, z_axis)].asarray(), crop)
            # Zero-padded so lexicographic order == acquisition order.
            tifffile.imwrite(out_dir / f"z{z:03d}.tif", plane, photometric="minisblack")
            if z % 10 == 0 or z == n_z - 1:
                print(f"  plane {z + 1}/{n_z}", end="\r", flush=True)
    print(f"\nwrote {n_z} planes to {out_dir.resolve()}")


def extract_3d(path: Path, channel: int, downsample: int, out_file: Path,
               px_xy: float | None = None, px_z: float | None = None,
               crop: tuple[int, int, int, int] | None = None) -> None:
    """Write one channel as a single (optionally downsampled) volume.

    Downsampling uses a block MEAN, which preserves the intensity scale — a block
    sum would multiply values by the block area and a stride would alias. Only xy
    is reduced; z is kept, because 109 planes is already small and z resolution is
    usually what you care about.

    Args:
        path: OME-TIFF path.
        channel: Channel index.
        downsample: Integer xy reduction factor. ``1`` keeps full resolution.
        out_file: Destination ``.tif``.

    Raises:
        ValueError: If the channel index is out of range or downsample < 1.
    """
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, got {downsample}")
    from skimage.measure import block_reduce

    with tifffile.TiffFile(str(path)) as tif:
        series = _series(tif)
        axes, shape = series.axes, tuple(series.shape)
        sizes = dict(zip(axes, shape))
        n_z, n_c, z_axis = _resolve_dims(axes, shape)
        y, x = sizes["Y"], sizes["X"]
        if not 0 <= channel < n_c:
            raise ValueError(f"channel {channel} out of range; file has {n_c} (0..{n_c - 1})")
        if crop:
            y0, y1, x0, x1 = crop
            if y1 > y or x1 > x:
                raise ValueError(f"--crop {crop} exceeds the image ({y} x {x})")
            y, x = y1 - y0, x1 - x0
            print(f"cropping to y[{y0}:{y1}] x[{x0}:{x1}]  ->  {y} x {x}")

        out_y, out_x = -(-y // downsample), -(-x // downsample)
        est = n_z * out_y * out_x * 4 / 1e9
        print(f"output volume: ({n_z}, {out_y}, {out_x}) float32 = {est:.2f} GB")
        print(f"  HIVE will peak near {est * 10:.1f} GB on this — check that fits your RAM.\n")

        # Preallocate: one plane in flight at a time, never the whole source.
        volume = np.empty((n_z, out_y, out_x), dtype=np.float32)
        for z in range(n_z):
            plane = _apply_crop(
                series.pages[_page_index(axes, shape, z, channel, z_axis)].asarray(), crop)
            if downsample > 1:
                plane = block_reduce(plane, (downsample, downsample), np.mean)
            volume[z] = plane[:out_y, :out_x]
            if z % 10 == 0 or z == n_z - 1:
                print(f"  plane {z + 1}/{n_z}", end="\r", flush=True)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_file), volume.astype(np.float32), imagej=True,
                     metadata={"axes": "ZYX"})
    print(f"\nwrote {out_file.resolve()}  ({out_file.stat().st_size / 1e9:.2f} GB)")

    # Downsampling changes the effective pixel size. Passing the ORIGINAL value to
    # HIVE would silently under-report every area and length by downsample^2.
    if px_xy:
        new_xy = float(px_xy) * downsample
        z_flag = f" --voxel-size-z {float(px_z):.4f}" if px_z else ""
        print(f"\nIMPORTANT: xy was reduced {downsample}x, so the pixel size is now "
              f"{new_xy:.4f} (was {float(px_xy):.4f}).")
        print(f"Run HIVE with:  --voxel-size-xy {new_xy:.4f}{z_flag}")
    else:
        print(f"\nIMPORTANT: xy was reduced {downsample}x -> multiply your original "
              f"pixel size by {downsample} for --voxel-size-xy. z is unchanged.")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; ``None`` uses ``sys.argv``.

    Returns:
        Exit code.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="OME-TIFF file")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true", help="Print shape/channels/voxel size")
    mode.add_argument("--extract-2d", action="store_true", help="One TIFF per z-plane")
    mode.add_argument("--extract-3d", action="store_true", help="One downsampled volume")
    p.add_argument("--channel", type=int, default=0, help="Channel index (see --inspect)")
    p.add_argument("--downsample", type=int, default=4, help="xy reduction for --extract-3d")
    p.add_argument("--crop", default=None, metavar="y0,y1,x0,x1",
                   help="Analyse only this window. A FULL-RESOLUTION crop is usually better "
                        "science than downsampling the whole field: it keeps fine structure "
                        "where it matters instead of blurring everything.")
    p.add_argument("--out", default=None, help="Output folder (2D) or file (3D)")
    args = p.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    try:
        crop = _parse_crop(args.crop)
        if args.inspect:
            inspect(path)
        elif args.extract_2d:
            extract_2d(path, args.channel, Path(args.out or "planes"), crop=crop)
        else:
            with tifffile.TiffFile(str(path)) as _tf:
                _meta = _tf.ome_metadata if _tf.is_ome else None
            extract_3d(path, args.channel, args.downsample,
                       Path(args.out or f"volume_ch{args.channel}_ds{args.downsample}.tif"),
                       px_xy=_ome_field(_meta, "PhysicalSizeX"),
                       px_z=_ome_field(_meta, "PhysicalSizeZ"), crop=crop)
    except PermissionError:
        print(
            f"\nPERMISSION DENIED reading:\n  {path}\n\n"
            "macOS blocks Terminal from Google Drive's CloudStorage folder by default.\n"
            "Fix: System Settings > Privacy & Security > Full Disk Access > enable Terminal,\n"
            "then restart Terminal. (Or copy the file to ~/Desktop first.)",
            file=sys.stderr,
        )
        return 1
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
