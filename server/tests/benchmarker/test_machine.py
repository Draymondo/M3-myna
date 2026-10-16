"""Machine-summary header record.

Every submitted results file leads with this record, so recipients can group
and filter by hardware without trusting hand-annotated provenance. The whole
point is that it degrades to None rather than raising: a probe that throws on
a container, a non-Ubuntu host, or a machine with no nvidia-smi would take the
whole sweep down with it.
"""

from __future__ import annotations

import subprocess

import pytest

from myna.benchmarker import machine
from myna.benchmarker._summarize import SCHEMA_VERSION

CPUINFO = """\
processor\t: 0
model name\t: AMD Ryzen 7 7840U
processor\t: 1
model name\t: AMD Ryzen 7 7840U
"""

OS_RELEASE = 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 24.04.1 LTS"\nVERSION_ID="24.04"\n'


@pytest.fixture
def fake_proc_files(monkeypatch):
    """Route machine.py's Path reads to in-memory content, keyed by path."""
    contents: dict[str, str | OSError] = {}
    real_read = machine.Path.read_text

    def read_text(self, *args, **kwargs):
        entry = contents.get(str(self))
        if isinstance(entry, OSError):
            raise entry
        if entry is not None:
            return entry
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(machine.Path, "read_text", read_text)
    return contents


# ─── CPU ─────────────────────────────────────────────────────────────────────


def test_cpu_model_reads_the_first_model_name_line(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = CPUINFO
    assert machine._cpu_model() == "AMD Ryzen 7 7840U"


def test_cpu_model_is_none_when_cpuinfo_has_no_model_name(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = "processor\t: 0\n"
    assert machine._cpu_model() is None


def test_cpu_model_is_none_when_cpuinfo_is_unreadable(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = OSError("no /proc")
    assert machine._cpu_model() is None


def test_cpu_cores_counts_processor_lines(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = CPUINFO
    assert machine._cpu_cores() == 2


def test_cpu_cores_is_none_rather_than_zero_when_nothing_matches(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = "\n"
    assert machine._cpu_cores() is None


def test_cpu_cores_is_none_when_cpuinfo_is_unreadable(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = OSError("no /proc")
    assert machine._cpu_cores() is None


def arm_cpuinfo(*parts: tuple[str, str]) -> str:
    """An arm64 /proc/cpuinfo: no model name, one MIDR stanza per core."""
    return "\n".join(
        f"processor\t: {i}\nCPU implementer\t: {implementer}\nCPU part\t: {part}\n"
        for i, (implementer, part) in enumerate(parts)
    )


PI5 = arm_cpuinfo(*[("0x41", "0xd0b")] * 4)
BIG_LITTLE = arm_cpuinfo(*[("0x41", "0xd05")] * 4, *[("0x41", "0xd0b")] * 2)


@pytest.fixture
def no_board(fake_proc_files):
    for path in (
        "/proc/device-tree/model",
        "/sys/class/dmi/id/sys_vendor",
        "/sys/class/dmi/id/product_name",
    ):
        fake_proc_files[path] = OSError("absent")
    return fake_proc_files


def test_arm_cpu_model_names_the_board_and_its_cores(no_board):
    no_board["/proc/cpuinfo"] = PI5
    no_board["/proc/device-tree/model"] = "Raspberry Pi 5 Model B Rev 1.0\x00"
    assert machine._cpu_model() == "Raspberry Pi 5 Model B Rev 1.0 (4x Cortex-A76)"


def test_arm_cpu_model_falls_back_to_the_dmi_board(no_board):
    no_board["/proc/cpuinfo"] = arm_cpuinfo(*[("0x51", "0x001")] * 12)
    no_board["/sys/class/dmi/id/sys_vendor"] = "LENOVO\n"
    no_board["/sys/class/dmi/id/product_name"] = "21BX\n"
    assert machine._cpu_model() == "LENOVO 21BX (12x Oryon)"


def test_arm_cpu_model_lists_each_core_type_in_order(no_board):
    no_board["/proc/cpuinfo"] = BIG_LITTLE
    assert machine._cpu_model() == "4x Cortex-A55 + 2x Cortex-A76"


def test_an_unknown_arm_core_keeps_its_midr_numbers(no_board):
    no_board["/proc/cpuinfo"] = arm_cpuinfo(("0x41", "0xfff"), ("0x99", "0x123"))
    assert machine._cpu_model() == "1x Arm 0xfff + 1x 0x99 0x123"


def test_cpu_clusters_group_cores_by_type_and_top_clock(no_board, sys_cpu):
    no_board["/proc/cpuinfo"] = BIG_LITTLE
    for cpu in range(6):
        sys_cpu(f"cpu{cpu}/cpufreq/cpuinfo_max_freq", "1800000" if cpu < 4 else "2400000")
    assert machine._cpu_clusters() == [
        {"core": "Cortex-A55", "count": 4, "max_mhz": 1800},
        {"core": "Cortex-A76", "count": 2, "max_mhz": 2400},
    ]


def test_x86_hybrid_cores_split_on_their_top_clock(fake_proc_files, sys_cpu):
    fake_proc_files["/proc/cpuinfo"] = CPUINFO
    sys_cpu("cpu0/cpufreq/cpuinfo_max_freq", "5100000")
    sys_cpu("cpu1/cpufreq/cpuinfo_max_freq", "3800000")
    assert machine._cpu_clusters() == [
        {"core": None, "count": 1, "max_mhz": 5100},
        {"core": None, "count": 1, "max_mhz": 3800},
    ]


def test_cpu_clusters_are_empty_when_cpuinfo_is_unreadable(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = OSError("no /proc")
    assert machine._cpu_clusters() == []


# ─── RAM ─────────────────────────────────────────────────────────────────────


def test_ram_gb_is_rounded_to_one_decimal():
    assert isinstance(machine._ram_gb(), float)


def test_ram_gb_is_none_when_psutil_cannot_answer(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: (_ for _ in ()).throw(RuntimeError("no /proc/meminfo"))
    )
    assert machine._ram_gb() is None


# ─── GPU ─────────────────────────────────────────────────────────────────────


SMI_QUERY = (
    "0, NVIDIA GeForce RTX 4080 Laptop GPU, 595.91.07, Disabled, 210, 3105, 405, 9001, [N/A], 12282\n"
    "1, NVIDIA RTX A2000, 595.91.07, Enabled, 300, 1200, 405, 6000, Enabled, 4096\n"
)
SMI_Q = "Driver Version                 : 595.91.07\nCUDA Version                   : 13.2\n"


def fake_run(stdout="", raises=None):
    def run(cmd, **kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    return run


def fake_smi(query=SMI_QUERY, details=SMI_Q, raises=None):
    """nvidia-smi answering the --query-gpu table and the -q report."""

    def run(cmd, **kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(
            cmd, 0, stdout=details if "-q" in cmd else query, stderr=""
        )

    return run


def test_gpus_lists_every_device_with_driver_clocks_and_ecc(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_smi())

    gpus = machine._gpus()

    assert [g["name"] for g in gpus] == ["NVIDIA GeForce RTX 4080 Laptop GPU", "NVIDIA RTX A2000"]
    first = gpus[0]
    assert first["index"] == 0
    assert first["driver_version"] == "595.91.07"
    assert first["cuda_driver_version"] == "13.2"
    assert first["persistence_mode"] == "Disabled"
    assert (first["sm_clock_mhz"], first["sm_clock_max_mhz"]) == (210, 3105)
    assert (first["mem_clock_mhz"], first["mem_clock_max_mhz"]) == (405, 9001)
    assert first["memory_total_mib"] == 12282
    # "[N/A]" is nvidia-smi saying the field does not apply: unknown, not a value.
    assert first["ecc_mode"] is None
    assert gpus[1]["ecc_mode"] == "Enabled"


def test_gpus_is_empty_when_nvidia_smi_is_absent(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_smi(raises=FileNotFoundError("nvidia-smi")))
    assert machine._gpus() == []


def test_gpus_is_empty_when_nvidia_smi_reports_no_devices(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_smi(query=""))
    assert machine._gpus() == []


def test_gpus_skips_an_unparseable_row(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_smi(query="garbage without commas\n"))
    assert machine._gpus() == []


def test_a_cuda_version_nvidia_smi_will_not_report_is_unknown(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_smi(details="Driver Version : 1\n"))
    assert machine._gpus()[0]["cuda_driver_version"] is None


def test_a_failing_detail_report_keeps_the_devices(monkeypatch):
    def run(cmd, **kwargs):
        if "-q" in cmd:
            raise subprocess.TimeoutExpired(cmd, 5)
        return subprocess.CompletedProcess(cmd, 0, stdout=SMI_QUERY, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    gpus = machine._gpus()
    assert len(gpus) == 2 and gpus[0]["cuda_driver_version"] is None


# ─── OS ──────────────────────────────────────────────────────────────────────


def test_ubuntu_version_reads_pretty_name_unquoted(fake_proc_files):
    fake_proc_files["/etc/os-release"] = OS_RELEASE
    assert machine._ubuntu_version() == "Ubuntu 24.04.1 LTS"


def test_ubuntu_version_is_none_without_a_pretty_name(fake_proc_files):
    fake_proc_files["/etc/os-release"] = 'NAME="Alpine"\n'
    assert machine._ubuntu_version() is None


def test_ubuntu_version_is_none_when_os_release_is_missing(fake_proc_files):
    fake_proc_files["/etc/os-release"] = OSError("no /etc/os-release")
    assert machine._ubuntu_version() is None


# ─── OS state ────────────────────────────────────────────────────────────────


@pytest.fixture
def sys_cpu(tmp_path, monkeypatch):
    """A fake /sys/devices/system/cpu, empty until a test writes into it."""
    monkeypatch.setattr(machine, "SYS_CPU", tmp_path)

    def write(relative: str, value: str) -> None:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")

    return write


def test_governor_is_the_one_every_cpu_runs(sys_cpu):
    sys_cpu("cpu0/cpufreq/scaling_governor", "performance")
    sys_cpu("cpu1/cpufreq/scaling_governor", "performance")
    assert machine._governor() == "performance"


def test_mixed_governors_are_all_named(sys_cpu):
    sys_cpu("cpu0/cpufreq/scaling_governor", "performance")
    sys_cpu("cpu1/cpufreq/scaling_governor", "powersave")
    assert machine._governor() == "performance,powersave"


def test_governor_is_none_without_cpufreq(sys_cpu):
    assert machine._governor() is None


def test_boost_reads_the_cpufreq_switch(sys_cpu):
    sys_cpu("cpufreq/boost", "0")
    assert machine._boost() is False


def test_boost_reads_intel_pstate_inverted(sys_cpu):
    sys_cpu("intel_pstate/no_turbo", "0")
    assert machine._boost() is True


def test_boost_is_none_when_the_kernel_exposes_no_switch(sys_cpu):
    assert machine._boost() is None


def test_smt_reports_control_and_whether_it_is_active(sys_cpu):
    sys_cpu("smt/control", "on")
    sys_cpu("smt/active", "1")
    assert machine._smt() == {"control": "on", "active": True}


def test_smt_is_unknown_field_by_field(sys_cpu):
    assert machine._smt() == {"control": None, "active": None}


def test_microcode_is_the_first_cpus_revision(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = CPUINFO + "microcode\t: 0xb600037\n"
    assert machine._microcode() == "0xb600037"


def test_microcode_is_none_when_cpuinfo_does_not_say(fake_proc_files):
    fake_proc_files["/proc/cpuinfo"] = CPUINFO
    assert machine._microcode() is None


def test_kernel_cmdline_is_read_verbatim(fake_proc_files):
    fake_proc_files["/proc/cmdline"] = "BOOT_IMAGE=/vmlinuz root=UUID=x ro quiet\n"
    assert machine._kernel_cmdline() == "BOOT_IMAGE=/vmlinuz root=UUID=x ro quiet"


def test_kernel_cmdline_is_none_when_unreadable(fake_proc_files):
    fake_proc_files["/proc/cmdline"] = OSError("no /proc")
    assert machine._kernel_cmdline() is None


SNAP_VERSION = "snap    2.77.1+ubuntu26.10.1\nsnapd   2.77.1+ubuntu26.10.1\nseries  16\n"


def test_snapd_version_comes_from_snap_version(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run(SNAP_VERSION))
    assert machine._snapd_version() == "2.77.1+ubuntu26.10.1"


def test_snapd_version_is_none_without_snapd(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run(raises=FileNotFoundError("snap")))
    assert machine._snapd_version() is None


def test_snapd_version_is_none_when_snapd_is_unreachable(monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run("snap    2.77\nsnapd   unavailable\n"))
    assert machine._snapd_version() is None


# ─── harness ─────────────────────────────────────────────────────────────────


def test_harness_version_is_the_one_baked_in_at_build_time(monkeypatch):
    monkeypatch.setattr(machine, "_read_build_version", lambda: "0.1.0+git40.a8521e61\n")
    assert machine._harness()["version"] == "0.1.0+git40.a8521e61"


def test_harness_version_is_none_outside_a_built_zipapp(monkeypatch):
    monkeypatch.setattr(machine, "_read_build_version", lambda: None)
    assert machine._harness()["version"] is None


def test_the_build_version_file_is_absent_from_the_source_tree():
    assert machine._read_build_version() is None


def test_harness_hashes_the_zipapp_it_runs_from(tmp_path, monkeypatch):
    import hashlib
    import zipfile

    pyz = tmp_path / "myna-bench.pyz"
    with zipfile.ZipFile(pyz, "w") as zf:
        zf.writestr("__main__.py", "")
    monkeypatch.setattr(machine.sys, "argv", [str(pyz), "run"])

    harness = machine._harness()

    assert harness["pyz"] == "myna-bench.pyz"
    assert harness["pyz_sha256"] == hashlib.sha256(pyz.read_bytes()).hexdigest()


def test_harness_names_no_zipapp_when_run_from_source(tmp_path, monkeypatch):
    script = tmp_path / "__main__.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(machine.sys, "argv", [str(script)])
    harness = machine._harness()
    assert harness["pyz"] is None and harness["pyz_sha256"] is None


# ─── collect ─────────────────────────────────────────────────────────────────


def test_collect_emits_the_header_record_schema(fake_proc_files, sys_cpu, monkeypatch):
    fake_proc_files["/proc/cpuinfo"] = CPUINFO
    fake_proc_files["/etc/os-release"] = OS_RELEASE
    monkeypatch.setattr(subprocess, "run", fake_smi(query=SMI_QUERY.splitlines()[1] + "\n"))

    header = machine.collect()

    assert header["type"] == "machine"
    assert header["schema_version"] == SCHEMA_VERSION
    assert header["gpus"][0]["driver_version"] == "595.91.07"
    assert set(header["os"]) == {
        "kernel_release",
        "kernel_cmdline",
        "cpu_microcode",
        "cpu_governor",
        "cpu_boost",
        "smt",
        "snapd",
    }
    assert set(header["harness"]) == {"version", "pyz", "pyz_sha256"}
    assert header["cpu"] == "AMD Ryzen 7 7840U"
    assert header["cpu_cores"] == 2
    assert header["cpu_clusters"] == [{"core": None, "count": 2, "max_mhz": None}]
    assert header["gpu"] == "NVIDIA RTX A2000"
    assert header["gpu_vram_gb"] == 4.0
    assert header["ubuntu"] == "Ubuntu 24.04.1 LTS"
    assert header["hostname"] and header["kernel"]
    assert header["collected_at"].endswith("+00:00")


def test_collect_survives_a_host_where_every_probe_fails(fake_proc_files, monkeypatch):
    fake_proc_files["/proc/cpuinfo"] = OSError("no /proc")
    fake_proc_files["/etc/os-release"] = OSError("no /etc/os-release")
    monkeypatch.setattr(subprocess, "run", fake_run(raises=FileNotFoundError("nvidia-smi")))
    monkeypatch.setattr(machine, "_ram_gb", lambda: None)

    header = machine.collect()

    assert header["type"] == "machine"
    assert [header[k] for k in ("cpu", "cpu_cores", "ram_gb", "gpu", "ubuntu")] == [None] * 5
    assert header["gpus"] == [] and header["cpu_clusters"] == []
    assert header["os"]["snapd"] is None and header["os"]["cpu_microcode"] is None


def test_a_non_numeric_clock_is_unknown_not_a_crash(monkeypatch):
    row = "0, GPU, 1.0, Enabled, busy, 1200, 405, 6000, Enabled, 4096\n"
    monkeypatch.setattr(subprocess, "run", fake_smi(query=row))
    assert machine._gpus()[0]["sm_clock_mhz"] is None
