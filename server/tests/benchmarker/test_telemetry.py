"""Clock, thermal, power and energy telemetry.

Every source is faked at the boundary: a sysfs tree under ``tmp_path`` and an
``nvidia-smi`` script on PATH that answers like the real one (two GPUs, one of
them without a power reading, and a compute-apps list). The sampler process
itself runs for real in one test, so the parent/child hand-off is pinned too.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from myna.benchmarker import _telemetry
from myna.benchmarker._telemetry import (
    GpuReader,
    RaplMeter,
    TelemetrySampler,
    VramReader,
    cpu_freqs_mhz,
    cpu_throttle_count,
    package_temp_c,
    parse_gpu_line,
    reason_field,
    sample_loop,
    summarise,
    supported,
    throttle_names,
)

# ─── fakes ───────────────────────────────────────────────────────────────────


def put(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def fake_sysfs(root: Path) -> Path:
    """Two cores, a k10temp package sensor and one RAPL package zone."""
    put(root, "sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "3410235")
    put(root, "sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq", "1200000")
    put(root, "sys/devices/system/cpu/cpu10/cpufreq/scaling_cur_freq", "4000000")
    put(root, "sys/class/hwmon/hwmon0/name", "nvme")
    put(root, "sys/class/hwmon/hwmon0/temp1_input", "90000")
    put(root, "sys/class/hwmon/hwmon5/name", "k10temp")
    put(root, "sys/class/hwmon/hwmon5/temp1_label", "Tctl")
    put(root, "sys/class/hwmon/hwmon5/temp1_input", "61250")
    put(root, "sys/class/powercap/intel-rapl:0/name", "package-0")
    put(root, "sys/class/powercap/intel-rapl:0/energy_uj", "1000000")
    put(root, "sys/class/powercap/intel-rapl:0/max_energy_range_uj", "65532610987")
    put(root, "sys/class/powercap/intel-rapl:0:0/name", "core")
    put(root, "sys/class/powercap/intel-rapl:0:0/energy_uj", "5")
    return root


FAKE_NVIDIA_SMI = """#!{python}
import os, sys, time
args = sys.argv[1:]
query = next(a.split("=", 1)[1] for a in args if a.startswith("--query"))
interval = int(args[args.index("-lms") + 1]) / 1000 if "-lms" in args else None
if query.endswith("reasons.active") and interval is None:
    sys.exit(0 if os.environ.get("FAKE_REASONS", "clocks_event_reasons") in query else 2)
if query == "temperature.gpu.tlimit":
    sys.exit(2 if os.environ.get("FAKE_NO_MARGIN") else 0)
margin = "temperature.gpu.tlimit" in query
n = 0
while True:
    if args[0].startswith("--query-gpu"):
        print("0, 1800, 8000, 61, 55.20, 87, 60.00" + (", 26" if margin else "") + ", 0x4")
        print("1, 210, 405, 39, [N/A], 0, [N/A]" + (", [N/A]" if margin else "") + ", 0x1")
    else:
        print(f"2026/09/30 02:24:56.{{n:03d}}, {{os.environ['FAKE_PID']}}, 700")
        print(f"2026/09/30 02:24:56.{{n:03d}}, 999999, 4096")
    sys.stdout.flush()
    n += 1
    if interval is None:
        break
    time.sleep(interval)
"""


@pytest.fixture
def fake_nvidia(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    tool = bindir / "nvidia-smi"
    tool.write_text(FAKE_NVIDIA_SMI.format(python=sys.executable), encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_PID", str(os.getpid()))
    return tool


@pytest.fixture
def no_nvidia(tmp_path, monkeypatch):
    empty = tmp_path / "nobin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


# ─── sysfs readers ───────────────────────────────────────────────────────────


def test_cpu_frequencies_are_mhz_in_core_order(tmp_path):
    assert cpu_freqs_mhz(fake_sysfs(tmp_path)) == [3410.2, 1200.0, 4000.0]


def test_cpu_frequencies_are_empty_without_cpufreq(tmp_path):
    assert cpu_freqs_mhz(tmp_path) == []


def test_package_temperature_prefers_tdie_over_tctl(tmp_path):
    root = fake_sysfs(tmp_path)
    put(root, "sys/class/hwmon/hwmon5/temp2_label", "Tdie")
    put(root, "sys/class/hwmon/hwmon5/temp2_input", "51000")
    assert package_temp_c(root) == 51.0


def test_package_temperature_reads_k10temp_tctl(tmp_path):
    assert package_temp_c(fake_sysfs(tmp_path)) == 61.2


def test_package_temperature_takes_the_hottest_intel_package(tmp_path):
    put(tmp_path, "sys/class/hwmon/hwmon1/name", "coretemp")
    put(tmp_path, "sys/class/hwmon/hwmon1/temp1_label", "Package id 0")
    put(tmp_path, "sys/class/hwmon/hwmon1/temp1_input", "70000")
    put(tmp_path, "sys/class/hwmon/hwmon1/temp2_label", "Core 0")
    put(tmp_path, "sys/class/hwmon/hwmon1/temp2_input", "99000")
    put(tmp_path, "sys/class/hwmon/hwmon2/name", "coretemp")
    put(tmp_path, "sys/class/hwmon/hwmon2/temp1_label", "Package id 1")
    put(tmp_path, "sys/class/hwmon/hwmon2/temp1_input", "72500")
    assert package_temp_c(tmp_path) == 72.5


def test_package_temperature_falls_back_to_the_x86_thermal_zone(tmp_path):
    put(tmp_path, "sys/class/thermal/thermal_zone0/type", "acpitz")
    put(tmp_path, "sys/class/thermal/thermal_zone0/temp", "20000")
    put(tmp_path, "sys/class/thermal/thermal_zone1/type", "x86_pkg_temp")
    put(tmp_path, "sys/class/thermal/thermal_zone1/temp", "66000")
    assert package_temp_c(tmp_path) == 66.0


def test_package_temperature_falls_back_to_the_hottest_cpu_thermal_zone(tmp_path):
    put(tmp_path, "sys/class/thermal/thermal_zone0/type", "gpu-thermal")
    put(tmp_path, "sys/class/thermal/thermal_zone0/temp", "80000")
    put(tmp_path, "sys/class/thermal/thermal_zone1/type", "cpu-thermal")
    put(tmp_path, "sys/class/thermal/thermal_zone1/temp", "58300")
    put(tmp_path, "sys/class/thermal/thermal_zone2/type", "cpu1-top-thermal")
    put(tmp_path, "sys/class/thermal/thermal_zone2/temp", "61000")
    assert package_temp_c(tmp_path) == 61.0


def test_package_temperature_is_none_without_a_package_sensor(tmp_path):
    put(tmp_path, "sys/class/hwmon/hwmon0/name", "nvme")
    put(tmp_path, "sys/class/hwmon/hwmon0/temp1_input", "90000")
    put(tmp_path, "sys/class/hwmon/hwmon1/name", "k10temp")
    put(tmp_path, "sys/class/hwmon/hwmon1/temp1_label", "Tccd1")
    put(tmp_path, "sys/class/hwmon/hwmon1/temp1_input", "garbage")
    assert package_temp_c(tmp_path) is None


def test_cpu_throttle_count_sums_the_package_counters(tmp_path):
    base = "sys/devices/system/cpu/cpu{}/thermal_throttle/package_throttle_count"
    put(tmp_path, base.format(0), "3")
    put(tmp_path, base.format(1), "4")
    assert cpu_throttle_count(tmp_path) == 7


def test_cpu_throttle_count_is_unknown_where_the_cpu_exposes_none(tmp_path):
    assert cpu_throttle_count(fake_sysfs(tmp_path)) is None


# ─── RAPL ────────────────────────────────────────────────────────────────────


def test_rapl_reads_package_zones_only_as_joules_since_start(tmp_path):
    root = fake_sysfs(tmp_path)
    meter = RaplMeter(root)
    assert meter.read() == 0.0
    put(root, "sys/class/powercap/intel-rapl:0/energy_uj", "3500000")
    put(root, "sys/class/powercap/intel-rapl:0:0/energy_uj", "999999999")
    assert meter.read() == 2.5


def test_rapl_counts_across_a_counter_wrap(tmp_path):
    root = fake_sysfs(tmp_path)
    put(root, "sys/class/powercap/intel-rapl:0/max_energy_range_uj", "2000000")
    put(root, "sys/class/powercap/intel-rapl:0/energy_uj", "1900000")
    meter = RaplMeter(root)
    put(root, "sys/class/powercap/intel-rapl:0/energy_uj", "100000")
    assert meter.read() == 0.2


def test_rapl_sums_every_package(tmp_path):
    root = fake_sysfs(tmp_path)
    put(root, "sys/class/powercap/intel-rapl:1/name", "package-1")
    put(root, "sys/class/powercap/intel-rapl:1/energy_uj", "0")
    put(root, "sys/class/powercap/intel-rapl:1/max_energy_range_uj", "65532610987")
    meter = RaplMeter(root)
    put(root, "sys/class/powercap/intel-rapl:0/energy_uj", "2000000")
    put(root, "sys/class/powercap/intel-rapl:1/energy_uj", "1000000")
    assert meter.read() == 2.0


def test_rapl_is_unknown_when_the_counter_is_unreadable(tmp_path):
    # energy_uj is root-only since CVE-2020-8694; a directory stands in for a
    # file the open refuses, which chmod cannot fake when tests run as root.
    root = fake_sysfs(tmp_path)
    counter = root / "sys/class/powercap/intel-rapl:0/energy_uj"
    counter.unlink()
    counter.mkdir()
    meter = RaplMeter(root)
    assert meter.read() is None


def test_rapl_turns_unknown_for_good_when_the_counter_goes_away(tmp_path):
    root = fake_sysfs(tmp_path)
    meter = RaplMeter(root)
    counter = root / "sys/class/powercap/intel-rapl:0/energy_uj"
    counter.unlink()
    assert meter.read() is None
    put(root, "sys/class/powercap/intel-rapl:0/energy_uj", "2000000")
    assert meter.read() is None


def test_rapl_is_unknown_without_powercap(tmp_path):
    assert RaplMeter(tmp_path).read() is None


# ─── nvidia-smi ──────────────────────────────────────────────────────────────


def test_a_gpu_line_parses_into_named_fields():
    assert parse_gpu_line("0, 1800, 8000, 61, 55.20, 87, 100.00, 0x0000000000000004") == {
        "index": 0,
        "sm_mhz": 1800.0,
        "mem_mhz": 8000.0,
        "temp_c": 61.0,
        "power_w": 55.2,
        "util_pct": 87.0,
        "power_limit_w": 100.0,
        "thermal_margin_c": None,
        "throttle": 4,
    }


def test_a_gpu_line_carries_the_thermal_margin_when_queried():
    parsed = parse_gpu_line("0, 1800, 8000, 61, 55.20, 87, 100.00, 26, 0x4", margin=True)
    assert parsed is not None
    assert parsed["thermal_margin_c"] == 26.0 and parsed["power_limit_w"] == 100.0
    assert parsed["throttle"] == 4


def test_a_gpu_line_keeps_missing_readings_unknown():
    parsed = parse_gpu_line("1, [N/A], 405, N/A, [N/A], 0, [N/A], [N/A], [Not Supported]", True)
    assert parsed is not None
    assert parsed["sm_mhz"] is None and parsed["power_w"] is None
    assert parsed["temp_c"] is None and parsed["throttle"] is None
    assert parsed["power_limit_w"] is None and parsed["thermal_margin_c"] is None


@pytest.mark.parametrize(
    "line",
    ["", "No devices were found", "0, 1, 2", "x, 1, 2, 3, 4, 5, 6, 7", "0, 1, 2, 3, 4, 5, 6"],
)
def test_a_line_that_is_not_a_gpu_reading_is_dropped(line):
    assert parse_gpu_line(line) is None


def test_throttle_names_ignore_idle_and_clock_settings():
    assert throttle_names(0x1 | 0x2 | 0x10 | 0x100) == []
    assert throttle_names(0x4 | 0x40) == ["sw_power_cap", "hw_thermal"]
    assert throttle_names(None) == []


def test_the_reason_field_follows_the_driver(fake_nvidia, monkeypatch):
    assert reason_field() == "clocks_event_reasons.active"
    monkeypatch.setenv("FAKE_REASONS", "clocks_throttle_reasons")
    assert reason_field() == "clocks_throttle_reasons.active"


def test_the_reason_field_falls_back_when_the_probe_cannot_run(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    assert reason_field() == "clocks_throttle_reasons.active"


def test_the_thermal_margin_is_probed(fake_nvidia, monkeypatch):
    assert supported("temperature.gpu.tlimit")
    monkeypatch.setenv("FAKE_NO_MARGIN", "1")
    assert not supported("temperature.gpu.tlimit")


def test_the_gpu_reader_keeps_the_latest_line_per_gpu():
    reader = GpuReader(
        ["0, 1, 1, 1, 10, 1, 9, 0x0", "1, 2, 2, 2, 20, 2, 9, 0x0", "0, 3, 3, 3, 30, 3, 9, 0x0"]
    )
    reader.run()
    assert [g["power_w"] for g in reader.snapshot()] == [30.0, 20.0]


def test_the_gpu_reader_reads_the_margin_column_it_asked_for():
    reader = GpuReader(["0, 1, 1, 1, 10, 1, 9, 30, 0x0"], margin=True)
    reader.run()
    assert reader.snapshot()[0]["thermal_margin_c"] == 30.0


def test_the_vram_reader_sums_one_timestamp_for_the_tree():
    reader = VramReader(
        [
            "t1, 10, 100",
            "t1, 11, 50",
            "t2, 10, 300",
            "t2, 99, 4096",
            "t2, 11, 20",
            "t2, pid, 1",
            "garbage",
        ]
    )
    reader.run()
    assert reader.used_mb({10, 11}) == 320.0
    assert reader.used_mb({12}) is None


def test_the_vram_reader_forgets_a_pid_missing_from_the_next_poll():
    reader = VramReader(["t1, 10, 100", "t1, 11, 50", "t2, 10, 300"])
    reader.run()
    assert reader.used_mb({10, 11}) == 300.0


def test_the_vram_reader_reads_silence_as_the_gpu_let_go():
    # nvidia-smi prints nothing for a poll where no process holds the GPU.
    now = iter([100.0, 102.0, 104.0])
    reader = VramReader(["t1, 10, 100"], stale_after=3.0, clock=lambda: next(now))
    reader.run()
    assert reader.used_mb({10}) == 100.0
    assert reader.used_mb({10}) is None


def test_the_vram_reader_knows_nothing_before_the_first_line():
    assert VramReader([]).used_mb({1}) is None


# ─── sample_loop ─────────────────────────────────────────────────────────────


def run_loop(tmp_path, root, *, ticks=3, pid=None):
    out = tmp_path / "trace.jsonl"
    stop = threading.Event()
    thread = threading.Thread(
        target=sample_loop,
        kwargs={
            "pid": pid or os.getpid(),
            "out": out,
            "interval": 0.05,
            "root": root,
            "stop": stop,
        },
    )
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if out.exists() and len(out.read_text(encoding="utf-8").splitlines()) >= ticks:
            break
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    return _telemetry.read_trace(out)


def test_the_loop_samples_two_gpus_and_the_tree_vram(tmp_path, fake_nvidia):
    samples = run_loop(tmp_path, fake_sysfs(tmp_path), ticks=4)
    last = samples[-1]
    assert [g["index"] for g in last["gpus"]] == [0, 1]
    assert last["gpus"][0]["power_w"] == 55.2 and last["gpus"][1]["power_w"] is None
    assert last["gpus"][0]["power_limit_w"] == 60.0
    assert last["gpus"][0]["thermal_margin_c"] == 26.0
    assert last["vram_mb"] == 700.0
    assert last["cpu_mhz"] == [3410.2, 1200.0, 4000.0]
    assert last["cpu_temp_c"] == 61.2
    assert last["cpu_energy_j"] == 0.0
    assert last["rss_mb"] > 0
    assert samples[0]["t"] < last["t"]


def test_the_loop_leaves_the_margin_out_of_a_query_the_driver_rejects(
    tmp_path, fake_nvidia, monkeypatch
):
    monkeypatch.setenv("FAKE_NO_MARGIN", "1")
    last = run_loop(tmp_path, fake_sysfs(tmp_path), ticks=4)[-1]
    assert last["gpus"][0]["power_w"] == 55.2
    assert last["gpus"][0]["thermal_margin_c"] is None


def test_the_loop_without_a_gpu_records_none(tmp_path, no_nvidia):
    samples = run_loop(tmp_path, tmp_path)
    assert all(s["gpus"] == [] and s["vram_mb"] is None for s in samples)
    assert all(s["cpu_energy_j"] is None and s["cpu_temp_c"] is None for s in samples)


def test_the_loop_on_a_dead_pid_records_unknown_rss(tmp_path, no_nvidia):
    samples = run_loop(tmp_path, tmp_path, pid=999999)
    assert all(s["rss_mb"] is None for s in samples)
    assert summarise(samples, audio_seconds=1.0)["peak_rss_mb"] is None


def test_a_tree_whose_memory_cannot_be_read_has_unknown_rss(monkeypatch):
    import psutil

    def denied(self):
        raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(psutil.Process, "memory_info", denied)
    rss, pids = _telemetry._tree(os.getpid())
    assert rss is None and os.getpid() in pids


def test_the_loop_takes_one_last_sample_on_stop(tmp_path, no_nvidia):
    out = tmp_path / "trace.jsonl"
    stop = threading.Event()
    kwargs = {"pid": os.getpid(), "out": out, "interval": 60.0, "root": tmp_path, "stop": stop}
    thread = threading.Thread(target=sample_loop, kwargs=kwargs)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not _telemetry.read_trace(out):
        time.sleep(0.02)
    stop.set()
    thread.join(timeout=10)
    assert len(_telemetry.read_trace(out)) == 2


def test_the_loop_stops_when_its_parent_is_gone(tmp_path, no_nvidia, monkeypatch):
    parents = iter([1234, 1234, 1])
    monkeypatch.setattr(os, "getppid", lambda: next(parents, 1))
    out = tmp_path / "trace.jsonl"
    sample_loop(pid=os.getpid(), out=out, interval=0.01, root=tmp_path, stop=threading.Event())
    assert len(_telemetry.read_trace(out)) == 2


def test_main_samples_until_sigterm_and_restores_the_handler(tmp_path, no_nvidia):
    out = tmp_path / "trace.jsonl"
    before = signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(0.3, signal.raise_signal, (signal.SIGTERM,))
    timer.start()
    _telemetry.main(
        [
            f"--pid={os.getpid()}",
            f"--out={out}",
            "--interval=0.05",
            f"--root={tmp_path}",
            "--nice=0",
        ]
    )
    timer.join()
    assert signal.getsignal(signal.SIGTERM) is before
    assert len(_telemetry.read_trace(out)) >= 2


# ─── summarise ───────────────────────────────────────────────────────────────


def sample(t, *, cpu_j=None, gpus=(), rss=100.0, vram=None, count=None):
    return {
        "t": t,
        "cpu_mhz": [3000.0],
        "cpu_temp_c": 60.0 + t,
        "cpu_energy_j": cpu_j,
        "cpu_throttle_count": count,
        "rss_mb": rss,
        "vram_mb": vram,
        "gpus": list(gpus),
    }


def gpu(index, power, throttle=0x1, temp=50.0, util=50.0, limit=100.0, margin=30.0):
    return {
        "index": index,
        "power_w": power,
        "throttle": throttle,
        "temp_c": temp,
        "util_pct": util,
        "power_limit_w": limit,
        "thermal_margin_c": margin,
    }


def test_energy_adds_rapl_and_every_gpu_integrated_over_time():
    trace = [
        sample(0.0, cpu_j=0.0, gpus=[gpu(0, 10.0), gpu(1, 100.0)], vram=500.0),
        sample(1.0, cpu_j=20.0, gpus=[gpu(0, 30.0), gpu(1, 100.0)], rss=300.0, vram=900.0),
        sample(3.0, cpu_j=50.0, gpus=[gpu(0, 30.0), gpu(1, 50.0)], rss=200.0),
    ]
    cell = summarise(trace, audio_seconds=10.0)
    # gpu0: 20 J + 60 J; gpu1: 100 J + 150 J; rapl 50 J.
    assert cell["cpu_energy_j"] == 50.0
    assert cell["gpu_energy_j_by_index"] == {"0": 80.0, "1": 250.0}
    assert cell["gpu_energy_j"] == 330.0
    assert cell["energy_j"] == 380.0
    assert cell["j_per_audio_s"] == 38.0
    assert cell["energy_sources"] == ["rapl", "nvidia"]
    assert cell["peak_rss_mb"] == 300.0 and cell["peak_vram_mb"] == 900.0
    assert cell["telemetry_seconds"] == 3.0
    assert cell["max_cpu_temp_c"] == 63.0 and cell["max_gpu_temp_c"] == 50.0


def test_a_gpu_missing_any_power_reading_makes_all_energy_unknown():
    # nvidia-smi prints [N/A] for a GPU without a power sensor. Its energy is
    # not 0 J, so neither is the total it belongs to.
    trace = [
        sample(0.0, cpu_j=0.0, gpus=[gpu(0, 10.0), gpu(1, 100.0)]),
        sample(1.0, cpu_j=20.0, gpus=[gpu(0, 30.0), gpu(1, 100.0)]),
        sample(3.0, cpu_j=50.0, gpus=[gpu(0, 30.0), gpu(1, None)]),
    ]
    cell = summarise(trace, audio_seconds=10.0)
    assert cell["gpu_energy_j_by_index"] == {"0": 80.0, "1": None}
    assert cell["gpu_energy_j"] is None and cell["cpu_energy_j"] == 50.0
    assert cell["energy_j"] is None and cell["j_per_audio_s"] is None
    assert cell["energy_sources"] == []


def test_gpu_energy_without_rapl_is_no_total():
    trace = [sample(0.0, gpus=[gpu(0, 10.0)]), sample(1.0, gpus=[gpu(0, 30.0)])]
    cell = summarise(trace, audio_seconds=1.0)
    assert cell["gpu_energy_j"] == 20.0
    assert cell["energy_j"] is None and cell["energy_sources"] == []


def test_rapl_alone_is_the_total_on_a_machine_without_gpus():
    cell = summarise([sample(0.0, cpu_j=1.0), sample(2.0, cpu_j=11.0)], audio_seconds=5.0)
    assert cell["energy_j"] == 10.0 and cell["j_per_audio_s"] == 2.0
    assert cell["energy_sources"] == ["rapl"]


def test_gpu_energy_is_held_back_to_the_first_tick():
    # The first tick can land before nvidia-smi's first line; RAPL counts from
    # that tick, so the GPU's first power reading stands in for the gap.
    trace = [
        sample(0.0, cpu_j=0.0),
        sample(1.0, cpu_j=5.0, gpus=[gpu(0, 10.0)]),
        sample(3.0, cpu_j=15.0, gpus=[gpu(0, 30.0)]),
    ]
    cell = summarise(trace, audio_seconds=1.0)
    assert cell["gpu_energy_j"] == 50.0 and cell["energy_j"] == 65.0


def test_a_gpu_read_once_has_unknown_energy():
    # The first tick can land before nvidia-smi's first line.
    trace = [sample(0.0), sample(1.0, gpus=[gpu(0, 30.0)])]
    cell = summarise(trace, audio_seconds=1.0)
    assert cell["gpu_energy_j_by_index"] == {"0": None} and cell["gpu_energy_j"] is None


def test_a_cell_without_gpus_or_rapl_has_unknown_energy():
    cell = summarise([sample(0.0), sample(1.0)], audio_seconds=10.0)
    assert cell["energy_j"] is None and cell["j_per_audio_s"] is None
    assert cell["gpu_energy_j"] is None and cell["cpu_energy_j"] is None
    assert cell["energy_sources"] == []
    assert cell["peak_vram_mb"] is None and cell["max_gpu_temp_c"] is None
    assert cell["throttled"] == {"cpu": None, "gpu": None}


def test_energy_per_audio_second_needs_audio():
    trace = [sample(0.0, cpu_j=0.0), sample(1.0, cpu_j=5.0)]
    assert summarise(trace, audio_seconds=0.0)["j_per_audio_s"] is None


def test_gpu_throttling_is_flagged_with_its_reasons():
    trace = [
        sample(0.0, gpus=[gpu(0, 10.0, throttle=0x1)]),
        sample(1.0, gpus=[gpu(0, 95.0, throttle=0x4), gpu(1, 5.0, throttle=0x40 | 0x8)]),
        sample(2.0, gpus=[gpu(0, 60.0, throttle=0x20, margin=2.0)]),
    ]
    cell = summarise(trace, audio_seconds=1.0)
    assert cell["throttled"]["gpu"] is True
    assert cell["gpu_throttle_reasons"] == [
        "hw_slowdown",
        "hw_thermal",
        "sw_power_cap",
        "sw_thermal",
    ]
    assert cell["gpu_throttle_unverified"] == []


def test_an_idle_gpu_is_not_throttled():
    cell = summarise([sample(0.0, gpus=[gpu(0, 10.0, throttle=0x1)])], audio_seconds=1.0)
    assert cell["throttled"]["gpu"] is False and cell["gpu_throttle_reasons"] == []


@pytest.mark.parametrize("util", [0.0, None])
def test_caps_reported_by_an_unloaded_gpu_are_not_throttling(util):
    # zephyrus at rest: 0% util, 16-20 W, 43-46 C, and sw_power_cap | sw_thermal
    # set with the idle bit clear.
    trace = [
        sample(0.0, gpus=[gpu(0, 18.0, throttle=0x24, util=util, limit=None, margin=None)]),
        sample(1.0, gpus=[gpu(0, 60.0, throttle=0x0, util=52.0)]),
    ]
    cell = summarise(trace, audio_seconds=1.0)
    assert cell["throttled"]["gpu"] is False and cell["gpu_throttle_reasons"] == []


def test_caps_far_from_the_limit_are_not_throttling():
    # A busy GPU at 20.5 W of 100 W and 41 C from slowdown, still flagging both.
    trace = [sample(0.0, gpus=[gpu(0, 20.5, throttle=0x24, util=52.0, margin=41.0)])]
    cell = summarise(trace, audio_seconds=1.0)
    assert cell["throttled"]["gpu"] is False and cell["gpu_throttle_reasons"] == []


@pytest.mark.parametrize(
    ("power", "limit", "margin", "throttled"),
    [(90.0, 100.0, None, True), (89.9, 100.0, None, False), (95.0, None, 5.0, True)],
)
def test_the_cap_edges(power, limit, margin, throttled):
    mask = 0x4 if limit else 0x20
    trace = [sample(0.0, gpus=[gpu(0, power, throttle=mask, limit=limit, margin=margin)])]
    assert summarise(trace, audio_seconds=1.0)["throttled"]["gpu"] is throttled


@pytest.mark.parametrize(("mask", "name"), [(0x4, "sw_power_cap"), (0x20, "sw_thermal")])
def test_a_cap_with_an_unknown_limit_leaves_the_verdict_open(mask, name):
    trace = [sample(0.0, gpus=[gpu(0, 95.0, throttle=mask, limit=None, margin=None)])]
    cell = summarise(trace, audio_seconds=1.0)
    assert cell["throttled"]["gpu"] is None
    assert cell["gpu_throttle_reasons"] == [] and cell["gpu_throttle_unverified"] == [name]


def test_a_confirmed_reason_outweighs_an_unverified_one():
    trace = [
        sample(0.0, gpus=[gpu(0, 95.0, throttle=0x4, margin=None)]),
        sample(1.0, gpus=[gpu(0, 95.0, throttle=0x24, margin=None)]),
    ]
    cell = summarise(trace, audio_seconds=1.0)
    assert cell["throttled"]["gpu"] is True
    assert cell["gpu_throttle_reasons"] == ["sw_power_cap"]
    assert cell["gpu_throttle_unverified"] == ["sw_thermal"]


# GPU 0 in every sample of the streaming cell of zephyrus' 2026-09-30
# parakeet fp32 run (t, SM MHz, W, util %, C, mask): RTX 4080 Laptop, driver
# 595.91. sw_power_cap | sw_thermal shows at rest and in one warm-up sample at
# 52% util and 20.5 W. The trace predates the limit columns; the second test
# adds what nvidia-smi reads on that GPU: a 100 W enforced limit and slowdown
# at 87 C (T.Limit 47 at 40 C).
ZEPHYRUS_STREAMING = [
    (0.011, 1785, 21.59, 0, 46, 0x1),
    (1.029, 1785, 20.51, 52, 46, 0x24),
    (2.048, 2175, 33.26, 34, 48, 0x0),
    (3.065, 2175, 54.43, 27, 49, 0x0),
    (4.083, 2175, 53.71, 45, 50, 0x0),
    (5.101, 2175, 50.97, 43, 50, 0x0),
    (6.119, 2175, 51.05, 37, 50, 0x0),
    (7.137, 2175, 50.48, 22, 50, 0x0),
    (8.155, 2175, 48.79, 26, 50, 0x0),
    (9.173, 2175, 49.15, 30, 51, 0x0),
    (10.191, 2175, 50.29, 42, 52, 0x0),
    (11.21, 2175, 57.22, 38, 52, 0x0),
    (12.228, 2175, 54.66, 46, 52, 0x0),
    (13.245, 2175, 49.86, 44, 52, 0x0),
    (14.263, 1830, 43.01, 33, 51, 0x1),
    (15.281, 2010, 44.95, 27, 52, 0x1),
    (16.299, 2175, 48.09, 19, 52, 0x0),
    (17.317, 2175, 47.21, 31, 52, 0x0),
    (18.326, 1995, 48.04, 44, 53, 0x1),
    (19.343, 2160, 51.51, 41, 53, 0x1),
    (20.361, 2160, 50.03, 31, 53, 0x1),
    (21.37, 2175, 48.16, 34, 53, 0x0),
    (22.388, 2175, 54.72, 14, 53, 0x0),
    (23.406, 2175, 55.34, 38, 54, 0x0),
    (24.424, 2175, 55.4, 29, 54, 0x0),
    (25.441, 2175, 51.08, 26, 54, 0x0),
    (26.459, 2175, 49.23, 30, 54, 0x0),
    (27.477, 2175, 47.21, 31, 53, 0x0),
    (28.495, 2055, 47.8, 47, 54, 0x1),
    (29.513, 2175, 52.15, 41, 54, 0x0),
    (30.532, 2175, 49.35, 42, 54, 0x0),
    (31.55, 2175, 62.79, 51, 56, 0x0),
    (32.568, 2175, 69.55, 83, 55, 0x0),
    (33.586, 2025, 68.28, 25, 57, 0x0),
    (34.604, 2175, 61.67, 37, 55, 0x0),
    (35.622, 2145, 53.31, 34, 55, 0x1),
    (36.64, 2175, 55.14, 42, 55, 0x0),
    (37.657, 2175, 57.64, 31, 55, 0x0),
    (38.676, 2175, 55.58, 42, 55, 0x0),
    (39.693, 2175, 62.09, 48, 58, 0x0),
    (40.711, 2175, 68.12, 72, 55, 0x0),
    (41.729, 1860, 68.38, 28, 58, 0x0),
    (42.747, 2175, 64.24, 31, 55, 0x0),
    (43.765, 2175, 58.49, 52, 56, 0x0),
    (44.783, 2175, 63.82, 33, 56, 0x0),
    (45.801, 2175, 57.86, 33, 56, 0x0),
    (46.819, 2070, 59.94, 52, 57, 0x0),
    (47.837, 2175, 60.59, 50, 57, 0x0),
    (48.855, 2175, 50.7, 0, 54, 0x0),
    (49.874, 2175, 21.68, 0, 53, 0x24),
    (50.892, 1665, 20.4, 0, 52, 0x24),
    (51.909, 1665, 17.53, 0, 52, 0x24),
    (52.927, 1665, 17.46, 0, 52, 0x24),
    (53.575, 1665, 17.43, 0, 51, 0x24),
]


def zephyrus_trace(**limits):
    trace = []
    for t, sm, power, util, temp, mask in ZEPHYRUS_STREAMING:
        reading = {"index": 0, "sm_mhz": sm, "power_w": power, "util_pct": util}
        reading |= {"temp_c": temp, "throttle": mask}
        if limits:
            reading["power_limit_w"] = limits["power_limit_w"]
            reading["thermal_margin_c"] = limits["slowdown_c"] - temp
        trace.append(sample(t, gpus=[reading]))
    return trace


def test_a_real_warm_up_sample_under_the_laptop_caps_is_not_throttling():
    cell = summarise(zephyrus_trace(power_limit_w=100.0, slowdown_c=87.0), audio_seconds=1.0)
    assert cell["throttled"]["gpu"] is False
    assert cell["gpu_throttle_reasons"] == [] and cell["gpu_throttle_unverified"] == []


def test_the_real_trace_without_limits_is_undecided_not_throttled():
    cell = summarise(zephyrus_trace(), audio_seconds=1.0)
    assert cell["throttled"]["gpu"] is None
    assert cell["gpu_throttle_unverified"] == ["sw_power_cap", "sw_thermal"]


def test_cpu_throttling_is_a_counter_that_moved():
    moved = [sample(0.0, count=5), sample(1.0, count=5), sample(2.0, count=7)]
    still = [sample(0.0, count=5), sample(1.0, count=5)]
    assert summarise(moved, audio_seconds=1.0)["throttled"]["cpu"] is True
    assert summarise(still, audio_seconds=1.0)["throttled"]["cpu"] is False


def test_an_empty_trace_summarises_to_unknowns():
    cell = summarise([], audio_seconds=1.0)
    assert cell["peak_rss_mb"] is None and cell["energy_j"] is None
    assert cell["telemetry_seconds"] is None and cell["max_cpu_temp_c"] is None
    assert cell["throttled"] == {"cpu": None, "gpu": None}


# ─── TelemetrySampler (the real child process) ───────────────────────────────


def test_the_sampler_process_traces_two_gpus_niced(tmp_path, fake_nvidia):
    sampler = TelemetrySampler(os.getpid(), interval=0.05, root=fake_sysfs(tmp_path))
    sampler.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and len(_telemetry.read_trace(sampler.trace)) < 3:
        time.sleep(0.05)
    nice = os.getpriority(os.PRIO_PROCESS, sampler.pid)
    samples = sampler.stop()
    assert nice == os.getpriority(os.PRIO_PROCESS, 0) + _telemetry.NICE or nice == 19
    assert len(samples) >= 3
    assert [g["index"] for g in samples[-1]["gpus"]] == [0, 1]
    assert samples[-1]["vram_mb"] == 700.0
    assert not sampler.trace.exists()
    assert sampler.error is None


def test_stopping_a_sampler_that_never_started_is_empty():
    sampler = TelemetrySampler(os.getpid())
    assert sampler.stop() == [] and sampler.error is None


def fake_child(sampler, code):
    sampler.command = lambda: [sys.executable, "-c", code]


def test_a_sampler_that_died_early_is_reported(capsys):
    sampler = TelemetrySampler(os.getpid())
    fake_child(sampler, "raise SystemExit(3)")
    sampler.start()
    assert sampler._proc is not None
    sampler._proc.wait(timeout=10)
    assert sampler.stop() == []
    assert sampler.error == "telemetry sampler exited 3 before the cell ended"
    assert "WARNING: telemetry sampler exited 3" in capsys.readouterr().err


def test_a_sampler_that_fails_on_stop_is_reported():
    sampler = TelemetrySampler(os.getpid())
    fake_child(
        sampler,
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(4))\n"
        f"open({str(sampler.trace)!r}, 'a').write('{{\"t\": 0.0}}\\n')\n"
        "time.sleep(30)",
    )
    sampler.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not _telemetry.read_trace(sampler.trace):
        time.sleep(0.02)
    assert sampler.stop() == [{"t": 0.0}]
    assert sampler.error == "telemetry sampler exited 4 on stop"


def test_a_sampler_that_wrote_nothing_is_reported(tmp_path):
    ready = tmp_path / "ready"
    sampler = TelemetrySampler(os.getpid())
    fake_child(
        sampler,
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(30)",
    )
    sampler.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    assert sampler.stop() == []
    assert sampler.error == "telemetry sampler wrote no samples"


def test_a_sampler_that_ignores_sigterm_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(_telemetry, "STOP_GRACE_S", 0.2)
    sampler = TelemetrySampler(os.getpid())
    ready = tmp_path / "ready"
    fake_child(
        sampler,
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(30)",
    )
    sampler.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not ready.exists():
        time.sleep(0.02)
    started = time.monotonic()
    assert sampler.stop() == []
    assert time.monotonic() - started < 5
    assert sampler.error == "telemetry sampler ignored SIGTERM and was killed"


def test_read_trace_skips_a_torn_last_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"t": 0.0}\n{"t": 1.', encoding="utf-8")
    assert _telemetry.read_trace(path) == [{"t": 0.0}]
    assert _telemetry.read_trace(tmp_path / "absent") == []
