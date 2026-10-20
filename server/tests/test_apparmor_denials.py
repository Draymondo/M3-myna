"""The AppArmor denial gate (tests/lib/apparmor_denials.py) over real journal lines."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_tree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "lib"))

import apparmor_denials as ad  # noqa: E402

KERNEL = (
    "Oct 16 21:31:11.648432 framework kernel: audit: type=1400 audit(1792182671.646:82798): "
    'apparmor="DENIED" operation="open" class="file" profile="snap.myna-parakeet.server" '
    'name="/sys/fs/cgroup/system.slice/memory.high" pid=1234723 comm="python3" '
    'requested_mask="r" denied_mask="r" fsuid=0 ouid=0'
)
CAPABLE = (
    "Oct 16 21:30:02.100000 framework kernel: audit: type=1400 audit(1792182602.100:82700): "
    'apparmor="DENIED" operation="capable" class="cap" profile="snap.myna-whisper.hook.install" '
    'pid=810874 comm="install" capability=1  capname="dac_override"'
)
FOREIGN_NAMESPACE = (
    'Oct 16 21:30:02 framework kernel: audit: apparmor="DENIED" operation="capable" class="cap" '
    'namespace="root//lxd-workshop.charles_myna-754c33df_<var-snap-lxd-common-lxd>" '
    'profile="systemd-detect-virt" pid=1883892 comm="systemd-detect-" capability=1  '
    'capname="dac_override"'
)


def dbus(label: str, peer: str, unique: str = ":1.2752") -> str:
    return (
        "Oct 13 00:48:37.745916 framework dbus-broker[2604201]: "
        'apparmor="DENIED" operation="dbus_signal" bus="session" '
        'path="/com/canonical/Myna/Dictation" interface="org.freedesktop.DBus.Properties" '
        f'member="PropertiesChanged" mask="send" name="{unique}" pid=6562 '
        f'label="{label}" peer_pid=81266 peer_label="{peer}"'
    )


def test_kernel_denial_is_ours():
    d = ad.parse(KERNEL)
    assert d is not None
    assert d["subject"] == "snap.myna-parakeet.server"
    assert d["name"] == "/sys/fs/cgroup/system.slice/memory.high"


def test_capability_denial_keeps_the_capability():
    d = ad.parse(CAPABLE)
    assert d is not None
    assert "capname=dac_override" in ad.key(d)
    assert "comm=install" in ad.key(d)


@pytest.mark.parametrize(
    "line",
    [
        FOREIGN_NAMESPACE,
        KERNEL.replace("snap.myna-parakeet.server", "snap.firefox.firefox"),
        KERNEL.replace("snap.myna-parakeet.server", "snap.mynaother.app"),
        dbus("snap.firefox.firefox", "snap.myna.myna"),
        KERNEL.replace('apparmor="DENIED"', 'apparmor="ALLOWED"'),
        "Oct 13 00:48:37 framework systemd[1]: Started snap.myna.myna.service.",
    ],
)
def test_other_lines_are_ignored(line):
    assert ad.parse(line) is None


def test_unique_bus_names_do_not_split_a_denial():
    unexpected, _ = ad.check(
        [dbus("snap.myna.hud", "unconfined", ":1.1"), dbus("snap.myna.hud", "unconfined", ":1.9")]
    )
    assert list(unexpected.values()) == [2]


def test_named_unconfined_peer_is_allowed_on_our_path():
    unexpected, allowed = ad.check([dbus("snap.myna.myna", "desktop-icons-ng")])
    assert not unexpected
    assert sum(allowed.values()) == 1


@pytest.mark.parametrize(
    "line",
    [
        dbus("snap.myna.myna", "unconfined"),
        dbus("snap.myna.myna", "snap.myna.hud"),
        dbus("snap.myna.hud", "desktop-icons-ng"),
        dbus("snap.myna.myna", "desktop-icons-ng").replace(
            "/com/canonical/Myna/Dictation", "/org/freedesktop/portal/desktop"
        ),
    ],
)
def test_allowlist_does_not_reach_past_its_case(line):
    unexpected, allowed = ad.check([line])
    assert sum(unexpected.values()) == 1
    assert not allowed


def test_main_fails_on_an_unexpected_denial(tmp_path, capsys):
    journal = tmp_path / "journal.log"
    journal.write_text("\n".join([KERNEL, KERNEL, dbus("snap.myna.myna", "desktop-icons-ng")]))
    assert ad.main([str(journal)]) == 1
    out = capsys.readouterr().out
    assert "DENIED x2: subject=snap.myna-parakeet.server comm=python3 operation=open" in out
    assert "allowed x1: " in out
    assert "FAIL: 1 unexpected" in out


def test_main_passes_on_a_clean_journal(tmp_path, capsys):
    journal = tmp_path / "journal.log"
    journal.write_text("\n".join([FOREIGN_NAMESPACE, dbus("snap.myna.myna", "desktop-icons-ng")]))
    assert ad.main([str(journal)]) == 0
    assert "FAIL" not in capsys.readouterr().out
