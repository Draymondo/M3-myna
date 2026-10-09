#!/bin/bash
# run-suite.sh - every extension test, in one entrypoint.
#
#     test/run-suite.sh                  (from extensions/myna-shell/)
#
# exits 0 when everything that could run passed, non-zero otherwise.
#
# The suite is the host's pure GJS contract tests (placement, resolution,
# respawn, presence, host-composed logic), which need nothing but gjs. The
# live compositor behaviour the host drives (dock typing, focus safety,
# click-through, repositioning) escapes unit tests by design — it is
# verified on hardware (T125 / R28), so there is no headless-Shell harness
# here anymore.
#
# One entrypoint because three callers run this same list - the myna-shell
# workshop's `gjs-test` action, test/next-shell.sh, and anyone in a checkout.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$(dirname "$HERE")" || exit 1

# A bus of its own per test: some of them talk to a stand-in daemon on it, and
# a missing one would make them skip rather than fail.
for t in test/*.test.js; do
    dbus-run-session -- gjs -m "$t" || exit 1
done