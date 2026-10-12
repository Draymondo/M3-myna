"""The install-time check that keeps a GPU engine off a machine whose driver
the snap cannot use (#49)."""

from __future__ import annotations

import pytest

from myna.server import cuda_probe


class _FakeCuda:
    """libcuda.so.1's three entry points the probe calls, scripted."""

    def __init__(self, *, init=0, devices=1, version=13000):
        self._init, self._devices, self._version = init, devices, version

    def cuInit(self, flags):  # noqa: N802 - the driver API's name
        return self._init

    def cuDeviceGetCount(self, count):  # noqa: N802
        count._obj.value = self._devices
        return 0

    def cuDriverGetVersion(self, version):  # noqa: N802
        version._obj.value = self._version
        return 0


def _loader(lib):
    def load(name):
        if lib is None:
            raise OSError(f"{name}: cannot open shared object file")
        return lib

    return load


def test_a_new_enough_driver_with_a_device_is_usable():
    assert cuda_probe.unusable_because(13000, _loader(_FakeCuda())) is None


@pytest.mark.parametrize(
    ("lib", "reason"),
    [
        (None, "libcuda.so.1 cannot be loaded"),
        (_FakeCuda(init=100), "cuInit failed with CUDA error 100"),
        (_FakeCuda(devices=0), "no CUDA device"),
        (_FakeCuda(version=12080), "the driver supports CUDA 12.8; this engine needs 13.0"),
    ],
)
def test_a_driver_the_snap_cannot_use_says_why(lib, reason):
    assert reason in cuda_probe.unusable_because(13000, _loader(lib))


def test_the_command_exits_by_the_verdict(monkeypatch, capsys):
    monkeypatch.setattr(cuda_probe, "_load", _loader(_FakeCuda(version=12080)))
    assert cuda_probe.main(["13.0"]) == 1
    assert "needs 13.0" in capsys.readouterr().out
    monkeypatch.setattr(cuda_probe, "_load", _loader(_FakeCuda()))
    assert cuda_probe.main(["13.0"]) == 0
