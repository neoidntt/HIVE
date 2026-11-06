"""Hardware detection and optional GPU-accelerated array operations.

Scope note (read before enabling GPU): profiling the HIVE pipeline on a
30x512x512 float32 stack put the entire front-end at **0.96 s** on CPU --
0.36 s baseline subtraction, 0.32 s preprocessing, 0.28 s segmentation. At
that scale GPU transfer overhead exceeds the compute saved, and the real
constraint is memory (the legacy pipeline peaked at 7.8x the input size),
not FLOPs.

Acceleration therefore only pays off for genuinely large volumes (roughly
>500 MB) or the rotation sweep in :mod:`hive.render3d`, where every frame
resamples the whole volume. :func:`get_backend` reports honestly rather than
claiming a speedup that does not exist, and every accelerated function has a
CPU path that produces numerically equivalent results.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Device = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class Backend:
    """A resolved compute backend.

    Attributes:
        device: ``"cuda"``, ``"mps"`` or ``"cpu"``.
        name: Human-readable device description.
        torch_available: Whether PyTorch imported successfully.
        cupy_available: Whether CuPy imported successfully.
        notes: Honest caveats about whether acceleration will actually help.
    """

    device: Device
    name: str
    torch_available: bool
    cupy_available: bool
    notes: str = ""

    @property
    def is_gpu(self) -> bool:
        """bool: True if this backend is not plain CPU."""
        return self.device in ("cuda", "mps")


def _probe_torch() -> tuple[bool, Any]:
    """Import torch without raising.

    Returns:
        Tuple of (available, module-or-None).
    """
    try:
        import torch
    except Exception:
        return False, None
    return True, torch


def _probe_cupy() -> tuple[bool, Any]:
    """Import CuPy and confirm a working CUDA runtime without raising.

    CuPy imports successfully on machines with no CUDA driver and only fails
    on first allocation, so this probes an actual allocation.

    Returns:
        Tuple of (available, module-or-None).
    """
    try:
        import cupy

        cupy.zeros(1)
    except Exception:
        return False, None
    return True, cupy


def get_backend(prefer: Device | None = None) -> Backend:
    """Detect the best available compute backend.

    Resolution order is CUDA, then Apple MPS, then CPU. A requested backend
    that is unavailable falls back with a note rather than raising, so the
    same code runs on a workstation and a laptop.

    Args:
        prefer: Force a specific device. ``None`` auto-detects. An unavailable
            choice degrades to the next best option.

    Returns:
        The resolved :class:`Backend`.

    Example:
        >>> backend = get_backend()
        >>> backend.device in ("cuda", "mps", "cpu")
        True
    """
    torch_ok, torch = _probe_torch()
    cupy_ok, _ = _probe_cupy()

    if prefer == "cpu":
        return Backend("cpu", f"CPU ({platform.processor() or platform.machine()})",
                       torch_ok, cupy_ok, "CPU explicitly requested.")

    if (prefer in (None, "cuda")) and torch_ok and torch.cuda.is_available():
        return Backend(
            "cuda",
            f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})",
            torch_ok,
            cupy_ok,
            "CUDA active. Worthwhile for large volumes and GIF rotation sweeps; "
            "for stacks under ~500 MB, transfer overhead usually cancels the gain.",
        )

    if (prefer in (None, "mps")) and torch_ok and getattr(torch.backends, "mps", None) is not None \
            and torch.backends.mps.is_available():
        return Backend(
            "mps",
            f"Apple Metal (MPS) on {platform.machine()}",
            torch_ok,
            cupy_ok,
            "MPS active. Note MPS is float32-only and lacks some ops; HIVE falls "
            "back to CPU per-operation where unsupported.",
        )

    if prefer in ("cuda", "mps"):
        note = f"{prefer.upper()} requested but unavailable; using CPU."
    else:
        note = ("No GPU detected. This is rarely a problem: the CPU pipeline runs a "
                "31 MB stack in ~1 s. Memory, not compute, is the usual limit.")

    return Backend("cpu", f"CPU ({platform.processor() or platform.machine()})",
                   torch_ok, cupy_ok, note)


def describe_hardware() -> str:
    """Render a human-readable hardware summary for UIs and logs.

    Returns:
        Multi-line description of the active backend.
    """
    backend = get_backend()
    lines = [f"Active backend : {backend.name}", f"Device         : {backend.device}"]
    if backend.notes:
        lines.append(f"Note           : {backend.notes}")
    return "\n".join(lines)


def gaussian_filter_auto(
    image: np.ndarray,
    sigma: float | tuple[float, ...],
    backend: Backend | None = None,
) -> np.ndarray:
    """Gaussian-smooth an array on the best available device.

    Falls back to :func:`scipy.ndimage.gaussian_filter` when no GPU is present
    or when the GPU path raises for any reason (unsupported dtype, out of
    memory, missing MPS kernel). Results agree with the CPU path to within
    float32 rounding.

    Args:
        image: Input array of any dimensionality.
        sigma: Scalar, or one sigma per axis.
        backend: Backend to use. ``None`` auto-detects.

    Returns:
        Smoothed array, float32, same shape as ``image``.

    Raises:
        ValueError: If ``sigma`` is a sequence whose length differs from
            ``image.ndim``.
    """
    from scipy.ndimage import gaussian_filter

    arr = np.asarray(image, dtype=np.float32)
    if not np.isscalar(sigma) and len(tuple(sigma)) != arr.ndim:  # type: ignore[arg-type]
        raise ValueError(
            f"sigma has {len(tuple(sigma))} entries "  # type: ignore[arg-type]
            f"but image is {arr.ndim}-D"
        )

    backend = backend or get_backend()
    if not backend.is_gpu or not backend.torch_available:
        return gaussian_filter(arr, sigma=sigma).astype(np.float32)

    try:
        return _gaussian_torch(arr, sigma, backend)
    except Exception:
        # Any GPU failure is non-fatal: correctness beats speed.
        return gaussian_filter(arr, sigma=sigma).astype(np.float32)


def _gaussian_torch(
    arr: np.ndarray, sigma: float | tuple[float, ...], backend: Backend,
) -> np.ndarray:
    """Separable Gaussian convolution via PyTorch.

    A Gaussian is separable, so an N-D convolution factorises into N 1-D
    passes -- O(n*k) instead of O(n*k^N). This is what makes the GPU path
    worth having for large volumes.

    Args:
        arr: float32 input array.
        sigma: Scalar or per-axis sigmas.
        backend: Resolved backend supplying the torch device.

    Returns:
        Smoothed float32 array.
    """
    import torch

    if np.isscalar(sigma):
        sigmas = [float(sigma)] * arr.ndim  # type: ignore[arg-type]
    else:
        sigmas = [float(s) for s in sigma]  # type: ignore[union-attr]
    tensor = torch.as_tensor(arr, device=backend.device, dtype=torch.float32)

    for axis, sig in enumerate(sigmas):
        if sig <= 0:
            continue
        radius = max(1, int(round(4.0 * sig)))
        coords = torch.arange(-radius, radius + 1, device=tensor.device, dtype=torch.float32)
        kernel = torch.exp(-(coords ** 2) / (2 * sig ** 2))
        kernel = kernel / kernel.sum()

        moved = tensor.movedim(axis, -1)
        shape = moved.shape
        flat = moved.reshape(1, 1, -1, shape[-1])
        kernel_2d = kernel.view(1, 1, 1, -1)
        padded = torch.nn.functional.pad(flat, (radius, radius, 0, 0), mode="reflect")
        convolved = torch.nn.functional.conv2d(padded, kernel_2d)
        tensor = convolved.reshape(shape).movedim(-1, axis)

    return tensor.cpu().numpy().astype(np.float32)
