"""Whether this snap can drive an NVIDIA GPU: ``python3 -m myna.server.cuda_probe 13.0``
exits 0 when the host driver's libcuda.so.1 loads, initialises, sees a device
and supports CUDA 13.0 or newer, and 1 with the reason otherwise.

modelctl matches a GPU engine on the vendor id alone. Without this check a
machine whose driver the snap cannot use (none installed, a kernel module that
did not build, one older than the runtime) gets the GPU engine, downloads its
components, and serves on the CPU (#49). The install hooks run it first.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Sequence
from typing import Any

# snapd exposes the host driver here through the opengl interface; the hooks'
# LD_LIBRARY_PATH may not name it.
_LIBCUDA = ("libcuda.so.1", "/var/lib/snapd/lib/gl/libcuda.so.1")


def _load(name: str) -> Any:
    return ctypes.CDLL(name)


def _cuda_version(packed: int) -> str:
    return f"{packed // 1000}.{packed % 1000 // 10}"


def unusable_because(min_version: int, load: Callable[[str], Any] | None = None) -> str | None:
    """None when the driver is usable, else why not. ``min_version`` is CUDA's
    packed form, 1000 * major + 10 * minor."""
    load = load or _load
    cuda = None
    for name in _LIBCUDA:
        try:
            cuda = load(name)
            break
        except OSError:
            continue
    if cuda is None:
        return "libcuda.so.1 cannot be loaded: no NVIDIA driver reaches the snap"
    if (error := cuda.cuInit(0)) != 0:
        return f"cuInit failed with CUDA error {error}"
    count = ctypes.c_int(0)
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
        return "the driver sees no CUDA device"
    version = ctypes.c_int(0)
    if cuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
        return "the driver did not report its CUDA version"
    if version.value < min_version:
        return (
            f"the driver supports CUDA {_cuda_version(version.value)}; "
            f"this engine needs {_cuda_version(min_version)}"
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    major, _, minor = args[0].partition(".")
    reason = unusable_because(1000 * int(major) + 10 * int(minor or 0), _load)
    if reason is None:
        return 0
    print(f"no usable NVIDIA GPU: {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
