"""Clocks, thermals, power and energy for one benchmark cell.

A laptop that drifts 35% over an hour makes a slow row and a hot machine look
the same, so every cell carries a 1 Hz trace: per-core CPU frequency, package
temperature, RAPL package energy, the served process tree's RSS and VRAM, and
for every NVIDIA GPU its clocks, temperature, power, utilisation and throttle
reasons. The trace is integrated into the cell's energy and throttle verdict.

Sampling runs in its own niced process (``python -m myna.benchmarker._telemetry``)
so it never takes the GIL or a core from the client being timed. nvidia-smi
is one long-lived ``-lms`` process per query, not an exec per sample. The
child appends one JSON line per tick to a file the parent reads back once the
cell ends, and exits on SIGTERM or when its parent is gone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import IO, Any

Sample = dict[str, Any]

INTERVAL_S = 1.0
NICE = 10
STOP_GRACE_S = 5.0

# NVML clock event reason bits that mean the GPU ran slower than it wanted to.
# Idle (0x1), application clocks (0x2), sync boost (0x10) and display clock
# (0x100) are not throttling.
THROTTLE_BITS = {
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}

# A power-cap or thermal bit counts only when the reading shows the cap: power
# within 10% of the enforced limit, or the temperature within 5 C of the
# slowdown point. A laptop GPU can report both bits all day at 20 W and 46 C.
POWER_CAP_FRACTION = 0.9
THERMAL_MARGIN_C = 5.0

GPU_FIELDS = (
    "index",
    "clocks.sm",
    "clocks.mem",
    "temperature.gpu",
    "power.draw",
    "utilization.gpu",
    "enforced.power.limit",
)
GPU_KEYS = ("sm_mhz", "mem_mhz", "temp_c", "power_w", "util_pct", "power_limit_w")
# Degrees left before thermal slowdown; drivers before ~530 lack the field and
# reject the whole query, so it is probed first.
MARGIN_FIELD = "temperature.gpu.tlimit"
MARGIN_KEY = "thermal_margin_c"


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: Path) -> int | None:
    text = _read(path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _number(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    return int(match.group(1)) if match else -1


# ---------------------------------------------------------------------------
# sysfs
# ---------------------------------------------------------------------------


def cpu_freqs_mhz(root: Path) -> list[float]:
    """Current frequency of every online core, in core order."""
    cores = sorted(
        (root / "sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: _number(p.name)
    )
    freqs = []
    for core in cores:
        khz = _read_int(core / "cpufreq/scaling_cur_freq")
        if khz is not None:
            freqs.append(round(khz / 1000, 1))
    return freqs


def _hwmon_temps(chip: Path) -> dict[str, float]:
    temps = {}
    for label in chip.glob("temp*_label"):
        name = _read(label)
        value = _read_int(chip / label.name.replace("_label", "_input"))
        if name is not None and value is not None:
            temps[name] = value / 1000
    return temps


def package_temp_c(root: Path) -> float | None:
    """The hottest CPU package: k10temp/zenpower Tdie (else Tctl), coretemp
    "Package id N", else the ``x86_pkg_temp`` thermal zone, else the hottest
    zone with cpu in its type (arm64 boards: ``cpu-thermal``, ``cpu1-top-thermal``)."""
    found: list[float] = []
    for chip in (root / "sys/class/hwmon").glob("hwmon*"):
        name = _read(chip / "name")
        temps = _hwmon_temps(chip)
        if name in ("k10temp", "zenpower"):
            value = temps.get("Tdie", temps.get("Tctl"))
            if value is not None:
                found.append(value)
        elif name == "coretemp":
            found.extend(v for k, v in temps.items() if k.startswith("Package id"))
    if not found:
        for zone in (root / "sys/class/thermal").glob("thermal_zone*"):
            milli = _read_int(zone / "temp")
            if _read(zone / "type") == "x86_pkg_temp" and milli is not None:
                found.append(milli / 1000)
    if not found:
        for zone in (root / "sys/class/thermal").glob("thermal_zone*"):
            milli = _read_int(zone / "temp")
            if "cpu" in (_read(zone / "type") or "") and milli is not None:
                found.append(milli / 1000)
    return round(max(found), 1) if found else None


def cpu_throttle_count(root: Path) -> int | None:
    """Intel's package thermal-throttle counter; None where the CPU has none."""
    counts = [
        _read_int(p)
        for p in (root / "sys/devices/system/cpu").glob(
            "cpu[0-9]*/thermal_throttle/package_throttle_count"
        )
    ]
    known = [c for c in counts if c is not None]
    return sum(known) if known else None


class RaplMeter:
    """Joules drawn by every CPU package since construction.

    ``energy_uj`` wraps at ``max_energy_range_uj`` and is root-only since
    CVE-2020-8694; an unreadable counter makes the whole reading unknown
    rather than a partial sum.
    """

    def __init__(self, root: Path):
        self._zones: list[tuple[Path, int]] = []
        for zone in sorted((root / "sys/class/powercap").glob("intel-rapl:*")):
            if re.fullmatch(r"intel-rapl:\d+", zone.name) and (
                _read(zone / "name") or ""
            ).startswith("package"):
                self._zones.append(
                    (zone / "energy_uj", _read_int(zone / "max_energy_range_uj") or 0)
                )
        self._last = [_read_int(path) for path, _ in self._zones]
        self._total_uj = 0
        self._ok = bool(self._zones) and None not in self._last

    def read(self) -> float | None:
        if not self._ok:
            return None
        now = [_read_int(path) for path, _ in self._zones]
        if None in now:
            self._ok = False
            return None
        for i, ((_, span), before, after) in enumerate(
            zip(self._zones, self._last, now, strict=True)
        ):
            assert before is not None and after is not None
            delta = after - before
            self._total_uj += delta if delta >= 0 else delta + span
            self._last[i] = after
        return round(self._total_uj / 1e6, 6)


# ---------------------------------------------------------------------------
# nvidia-smi
# ---------------------------------------------------------------------------


def _value(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # [N/A], [Not Supported], ...


def parse_gpu_line(line: str, margin: bool = False) -> Sample | None:
    """One ``--query-gpu`` CSV line, or None for anything else.

    ``margin`` says the query carried ``MARGIN_FIELD`` before the reasons.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(GPU_FIELDS) + margin + 1:
        return None
    try:
        index = int(parts[0])
    except ValueError:
        return None
    reading: Sample = {"index": index}
    reading.update(zip(GPU_KEYS, map(_value, parts[1 : len(GPU_FIELDS)]), strict=True))
    reading[MARGIN_KEY] = _value(parts[-2]) if margin else None
    try:
        reading["throttle"] = int(parts[-1], 16)
    except ValueError:
        reading["throttle"] = None
    return reading


def throttle_names(mask: int | None) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if mask and mask & bit]


def supported(field: str) -> bool:
    """Whether this driver's nvidia-smi accepts ``--query-gpu=field``."""
    try:
        probe = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def reason_field() -> str:
    """The throttle-reason field this driver knows: renamed in driver 535."""
    if supported("clocks_event_reasons.active"):
        return "clocks_event_reasons.active"
    return "clocks_throttle_reasons.active"


class GpuReader:
    """The latest ``--query-gpu`` reading per GPU, from a line stream."""

    def __init__(self, lines: Iterable[str], margin: bool = False):
        self._lines = lines
        self._margin = margin
        self._latest: dict[int, Sample] = {}

    def run(self) -> None:
        for line in self._lines:
            reading = parse_gpu_line(line, self._margin)
            if reading is not None:
                self._latest[reading["index"]] = reading

    def snapshot(self) -> list[Sample]:
        return [dict(r) for _, r in sorted(dict(self._latest).items())]


class VramReader:
    """VRAM per pid from ``--query-compute-apps=timestamp,pid,used_memory``.

    One poll prints a line per process, all with the same timestamp, and
    nothing at all when no process holds the GPU; the timestamp is what
    groups lines into a poll, and a poll older than ``stale_after`` seconds
    means every process has let go.
    """

    def __init__(
        self,
        lines: Iterable[str],
        stale_after: float = 3 * INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._lines = lines
        self._stale_after = stale_after
        self._clock = clock
        self._stamp: str | None = None
        self._seen = 0.0
        self._latest: dict[int, int] = {}

    def run(self) -> None:
        for line in self._lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                pid, used = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            if parts[0] != self._stamp:
                self._stamp = parts[0]
                self._latest = {}
            self._latest[pid] = used
            self._seen = self._clock()

    def used_mb(self, pids: set[int]) -> float | None:
        if self._clock() - self._seen > self._stale_after:
            return None
        mine = [used for pid, used in dict(self._latest).items() if pid in pids]
        return float(sum(mine)) if mine else None


def _smi(query: str, interval: float) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["nvidia-smi", query, "--format=csv,noheader,nounits", "-lms", str(int(interval * 1000))],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


# ---------------------------------------------------------------------------
# The sampling loop (child process)
# ---------------------------------------------------------------------------


def _tree(pid: int) -> tuple[float | None, set[int]]:
    """RSS (MB) and pids of ``pid`` and its descendants; RSS is None when
    no process in the tree could be read."""
    import psutil

    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return None, set()
    rss = []
    for proc in procs:
        try:
            rss.append(proc.memory_info().rss)
        except psutil.Error:
            pass
    return (round(sum(rss) / 1e6, 1) if rss else None), {p.pid for p in procs}


def sample_loop(*, pid: int, out: Path, interval: float, root: Path, stop: threading.Event) -> None:
    """Append a sample to ``out`` every ``interval`` until ``stop``, then one last."""
    parent = os.getppid()
    rapl = RaplMeter(root)
    gpus = GpuReader([])
    vram = VramReader([])
    procs: list[subprocess.Popen[str]] = []
    if shutil.which("nvidia-smi"):
        margin = supported(MARGIN_FIELD)
        fields = ",".join((*GPU_FIELDS, *([MARGIN_FIELD] if margin else []), reason_field()))
        procs = [
            _smi(f"--query-gpu={fields}", interval),
            _smi("--query-compute-apps=timestamp,pid,used_memory", interval),
        ]
        gpus = GpuReader(procs[0].stdout or [], margin)
        vram = VramReader(procs[1].stdout or [], stale_after=3 * interval)
        for reader in (gpus, vram):
            threading.Thread(target=reader.run, daemon=True).start()
    start = time.monotonic()

    def tick(fp: IO[str]) -> None:
        rss, pids = _tree(pid)
        record = {
            "t": round(time.monotonic() - start, 3),
            "wall": round(time.time(), 3),
            "cpu_mhz": cpu_freqs_mhz(root),
            "cpu_temp_c": package_temp_c(root),
            "cpu_energy_j": rapl.read(),
            "cpu_throttle_count": cpu_throttle_count(root),
            "rss_mb": rss,
            "vram_mb": vram.used_mb(pids),
            "gpus": gpus.snapshot(),
        }
        fp.write(json.dumps(record) + "\n")
        fp.flush()

    try:
        with out.open("a", encoding="utf-8") as fp:
            while True:
                tick(fp)
                if stop.wait(interval):
                    tick(fp)
                    break
                if os.getppid() != parent:
                    break
    finally:
        for proc in procs:
            proc.terminate()
            proc.wait()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="myna.benchmarker._telemetry")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=INTERVAL_S)
    parser.add_argument("--root", type=Path, default=Path("/"))
    parser.add_argument("--nice", type=int, default=NICE)
    args = parser.parse_args(argv)
    os.nice(args.nice)
    stop = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        sample_loop(pid=args.pid, out=args.out, interval=args.interval, root=args.root, stop=stop)
    finally:
        signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------------------
# The parent's side
# ---------------------------------------------------------------------------


def read_trace(path: Path) -> list[Sample]:
    """Samples written so far; a line torn by a kill is dropped."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    samples = []
    for line in lines:
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return samples


class TelemetrySampler:
    """Runs the sampling loop in a niced child for the span of one cell."""

    def __init__(self, pid: int, interval: float = INTERVAL_S, root: Path = Path("/")):
        self.target = pid
        self.interval = interval
        self.root = root
        fd, name = tempfile.mkstemp(prefix="myna-telemetry-", suffix=".jsonl")
        os.close(fd)
        self.trace = Path(name)
        self._proc: subprocess.Popen[bytes] | None = None
        self.error: str | None = None

    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "myna.benchmarker._telemetry",
            f"--pid={self.target}",
            f"--out={self.trace}",
            f"--interval={self.interval}",
            f"--root={self.root}",
            f"--nice={NICE}",
        ]

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def start(self) -> None:
        # The child imports this package from wherever the parent did: the
        # source tree, or the directory shiv unpacked the zipapp into.
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        self._proc = subprocess.Popen(self.command(), env=env, stdout=subprocess.DEVNULL)

    def stop(self) -> list[Sample]:
        """End the trace and return it; the temporary file is removed.

        ``error`` then says why the trace cannot be trusted whole, if it
        cannot: the child died early, stopped badly, or wrote nothing.
        """
        self.error = None
        if self._proc is not None:
            early = self._proc.poll()
            if early is not None:
                self.error = f"telemetry sampler exited {early} before the cell ended"
            else:
                self._proc.terminate()
                try:
                    status = self._proc.wait(timeout=STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
                    self.error = "telemetry sampler ignored SIGTERM and was killed"
                else:
                    if status != 0:
                        self.error = f"telemetry sampler exited {status} on stop"
        samples = read_trace(self.trace)
        self.trace.unlink(missing_ok=True)
        if self._proc is not None and not samples and self.error is None:
            self.error = "telemetry sampler wrote no samples"
        if self.error is not None:
            print(f"WARNING: {self.error} ({len(samples)} sample(s))", file=sys.stderr)
        return samples


# ---------------------------------------------------------------------------
# Per-cell verdict
# ---------------------------------------------------------------------------


def _gpu_energy_j(samples: list[Sample]) -> dict[str, float | None]:
    """Each GPU's trapezoid of power over the whole trace.

    nvidia-smi's first line can land after the first tick; the power it
    reads is held back to that tick, so GPU energy spans what RAPL spans.
    A GPU with any reading lacking power, or only one reading, is None: its
    energy is unknown, not the part that happened to be read.
    """
    series: dict[int, list[tuple[float, float | None]]] = {}
    for sample in samples:
        for reading in sample.get("gpus", []):
            series.setdefault(reading["index"], []).append((sample["t"], reading.get("power_w")))
    energy: dict[str, float | None] = {}
    for index, points in sorted(series.items()):
        read = [(t, p) for t, p in points if p is not None]
        if len(points) < 2 or len(read) < len(points):
            energy[str(index)] = None
            continue
        spans = zip(read, read[1:], strict=False)
        joules = sum((p0 + p1) / 2 * (t1 - t0) for (t0, p0), (t1, p1) in spans)
        (first_t, first_p), (last_t, last_p) = read[0], read[-1]
        joules += first_p * (first_t - samples[0]["t"]) + last_p * (samples[-1]["t"] - last_t)
        energy[str(index)] = round(joules, 3)
    return energy


def _capped(name: str, reading: Sample) -> bool | None:
    """Whether the reading bears out throttle reason ``name``; None when the
    limit it would be checked against is unknown."""
    if name == "sw_power_cap":
        power, limit = reading.get("power_w"), reading.get("power_limit_w")
        return None if power is None or not limit else power >= POWER_CAP_FRACTION * limit
    if name == "sw_thermal":
        margin = reading.get(MARGIN_KEY)
        return None if margin is None else margin <= THERMAL_MARGIN_C
    return True


def summarise(samples: list[Sample], audio_seconds: float) -> Sample:
    """A cell's peaks, energy and throttle verdict from its trace.

    Energy covers the whole cell (cold sample, warmup and measured passes), so
    J per audio-second divides by all the audio fed in that span. Unknown
    stays None: ``energy_j`` is RAPL plus every GPU in the trace or nothing,
    never a sum that counts an unread device as 0 J.

    A GPU is throttled by a reason seen while it was busy and borne out by
    its own reading (``_capped``). A power or thermal bit whose limit is
    unknown leaves the verdict None and is named in
    ``gpu_throttle_unverified``.
    """
    rapl = [s["cpu_energy_j"] for s in samples if s.get("cpu_energy_j") is not None]
    cpu_j = round(rapl[-1] - rapl[0], 3) if len(rapl) > 1 else None
    per_gpu = _gpu_energy_j(samples)
    known = [j for j in per_gpu.values() if j is not None]
    gpu_j = round(sum(known), 3) if per_gpu and len(known) == len(per_gpu) else None
    whole = cpu_j is not None and (gpu_j is not None or not per_gpu)
    energy = round((cpu_j or 0.0) + (gpu_j or 0.0), 3) if whole else None
    sources = ["rapl", *(["nvidia"] if per_gpu else [])] if whole else []
    readings = [g for s in samples for g in s.get("gpus", [])]
    masks = [g for g in readings if g.get("throttle") is not None]
    busy = [g for g in masks if (g.get("util_pct") or 0) > 0]
    verdicts = [(name, _capped(name, g)) for g in busy for name in throttle_names(g["throttle"])]
    reasons = sorted({name for name, hit in verdicts if hit})
    unverified = sorted({name for name, hit in verdicts if hit is None} - set(reasons))
    counts = [s["cpu_throttle_count"] for s in samples if s.get("cpu_throttle_count") is not None]
    rss = [s["rss_mb"] for s in samples if s.get("rss_mb") is not None]
    vram = [s["vram_mb"] for s in samples if s.get("vram_mb") is not None]
    cpu_temps = [s["cpu_temp_c"] for s in samples if s.get("cpu_temp_c") is not None]
    gpu_temps = [g["temp_c"] for g in readings if g.get("temp_c") is not None]
    return {
        "peak_rss_mb": max(rss) if rss else None,
        "peak_vram_mb": max(vram) if vram else None,
        "telemetry_seconds": round(samples[-1]["t"] - samples[0]["t"], 3) if samples else None,
        "cpu_energy_j": cpu_j,
        "gpu_energy_j": gpu_j,
        "gpu_energy_j_by_index": per_gpu,
        "energy_j": energy,
        "energy_sources": sources,
        "j_per_audio_s": round(energy / audio_seconds, 3)
        if energy is not None and audio_seconds > 0
        else None,
        "max_cpu_temp_c": max(cpu_temps) if cpu_temps else None,
        "max_gpu_temp_c": max(gpu_temps) if gpu_temps else None,
        "throttled": {
            "cpu": max(counts) > min(counts) if counts else None,
            "gpu": True if reasons else None if not masks or unverified else False,
        },
        "gpu_throttle_reasons": reasons,
        "gpu_throttle_unverified": unverified,
    }


if __name__ == "__main__":
    main()
