#!/usr/bin/env bash
# snapshot: installed
# Choosing a model snapd is still installing (issue #51): its slot lists
# before its model has arrived, and a switch to it used to disconnect
# Parakeet, then fail the connect on snapd's "changes in progress", leaving
# no model at all. The download is slowed at the VM's NIC so the install is
# still running when Settings opens; Settings must show it installing, keep
# Parakeet connected whatever is clicked, and let it be chosen once done.
set -uo pipefail
# shellcheck source=e2e-tests/suites/lib.sh
source "$(dirname "$0")/lib.sh"

S=shots/install-switch

assert_on "precondition: backend is parakeet" \
    'snap connections myna | grep -q "myna:backend.*myna-parakeet:provider"'
on 'sudo snap remove --purge myna-whisper' >/dev/null \
    || { echo "could not remove whisper" >&2; exit 1; }

# A few Mbit/s turns the model's download into minutes.
lxc config device override "$VM" eth0 limits.ingress=4Mbit >/dev/null \
    || { echo "could not slow eth0" >&2; exit 1; }
on 'sudo snap install --edge --no-wait myna-whisper' >/dev/null \
    || { echo "could not start the install" >&2; exit 1; }
# The slot shows before the install is done: that is the window of the bug.
on "$(poll 'snap list myna-whisper && snap changes | grep -q "Doing.*Install \"myna-whisper\""' | sed 's/^sh://')" \
    || { echo "whisper never listed while installing" >&2; exit 1; }

# The click is a negative check: the wait is how long a switch's disconnect
# took to land when the row still took it.
shot --polkit allow --monitor install-switch \
    waitfor:General "waitfor:Whisper@radio button" "waitfor:Installing" \
    shot:$S/01-installing.png \
    click:Whisper wait:5 shot:$S/02-clicked.png \
    || { echo "Whisper never showed installing" >&2; FAILED=1; }
assert_on "parakeet still connected while whisper installs" \
    'snap connections myna | grep -q "myna:backend.*myna-parakeet:provider"'
assert_on "whisper not connected while it installs" \
    '! snap connections myna | grep -q "myna:backend.*myna-whisper:provider"'

lxc config device unset "$VM" eth0 limits.ingress || true
on 'sudo snap watch --last=install' >/dev/null \
    || { echo "the install did not finish" >&2; exit 1; }

# Once done it is a model like any other.
shot --polkit allow \
    waitfor:General \
    click:Whisper "checked:Whisper@radio button" \
    "$(poll 'snap connections myna | grep -q "myna:backend.*myna-whisper:provider"')" \
    shot:$S/03-switched.png \
    || { echo "switch after install failed" >&2; FAILED=1; }
assert_on "backend connected to whisper once installed" \
    'snap connections myna | grep -q "myna:backend.*myna-whisper:provider"'
suite_status
