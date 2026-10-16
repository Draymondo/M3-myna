#!/bin/bash
# Benchmark the Myna inference snaps from the store's edge channel and pack the
# results to send back. For a tester without a checkout, next to
# myna-bench.pyz and bench-edge.yaml:
#
#   ./bench-edge.sh
#
# It installs, measures and purges myna-parakeet, myna-whisper and myna-funasr,
# so a backend already set up for dictation is removed and needs reinstalling
# from Myna Settings afterwards. The CPU governor is set to performance for the
# run, as myna-bench requires, and restored after.
set -euo pipefail

[ "$(id -u)" = 0 ] || exec sudo bash "$0" "$@"
cd "$(dirname "$0")"

for tool in python3 ffmpeg snap; do
    if ! command -v "$tool" >/dev/null; then
        echo "bench-edge: $tool is missing; install it with apt first" >&2
        exit 1
    fi
done

installed=$(snap list 2>/dev/null | awk '$1 ~ /^myna-(parakeet|whisper|funasr)$/ {print $1}' | xargs)
if [ -n "$installed" ]; then
    echo "This removes $installed, settings included, and leaves it uninstalled."
    read -r -p "Continue? [y/N] " answer
    [[ $answer == [yY] ]] || exit 1
fi

[ -f corpus/manifest.json ] ||
    python3 myna-bench.pyz download-corpus --out corpus --select balanced -n 40

saved=()
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -e "$governor" ] && saved+=("$governor=$(cat "$governor")")
done
restore() {
    for entry in "${saved[@]}"; do
        echo "${entry#*=}" >"${entry%%=*}"
    done
}
trap restore EXIT
for entry in "${saved[@]}"; do
    echo performance >"${entry%%=*}"
done

python3 myna-bench.pyz run --config bench-edge.yaml 2>&1 | tee run.log ||
    echo "bench-edge: the sweep failed; packing what it wrote" >&2
python3 myna-bench.pyz summarize --in results.jsonl --no-ci 2>&1 | tee summary.txt || true

out="myna-bench-$(hostname)-$(date +%Y%m%d-%H%M).tar.gz"
tar -czf "$out" results.jsonl run.log summary.txt
[ -z "${SUDO_USER:-}" ] || chown "$SUDO_USER:" "$out" results.jsonl run.log summary.txt
echo
echo "Done. Please send $PWD/$out to the Myna team."
