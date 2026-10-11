#!/usr/bin/env bash
# snapshot: localized
# Myna Settings reads a set-up machine as set up whatever the desktop
# language, though snap translates its own output there.
set -uo pipefail
# shellcheck source=e2e-tests/suites/lib.sh
source "$(dirname "$0")/lib.sh"

L=sv_SE.UTF-8
P=localized-$REL

# Positive control: without translated snap output this suite proves nothing.
assert_on "snap prints translated headers" \
    "LANG=$L snap list | head -1 | grep -qv '^Name'"
if shot --lang "$L" waitfor:Diagnostics shot:"$P"-settings.png; then
    echo "PASS: the settings window opened"
else
    echo "FAIL: the settings window did not open"; FAILED=1
fi
assert_on "onboarding found Myna and the model" \
    "grep 'onboarding assessment:' myna-shot/out/app.log | tail -1 | grep -q 'Myna found, Model found'"
suite_status
