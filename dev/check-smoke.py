#!/usr/bin/env python3
"""Fail unless a myna-bench results file shows every row finished and transcribed.

    python3 dev/check-smoke.py results.jsonl [--max-wer 0.35]

The arm64 snap workflow's gate: a build that installs but never produces
words, or produces nonsense, must not reach the store.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results")
    parser.add_argument("--max-wer", type=float, default=0.35)
    args = parser.parse_args()

    statuses: dict[str, str] = {}
    clips: dict[str, list[dict]] = defaultdict(list)
    with open(args.results, encoding="utf-8") as fp:
        for line in fp:
            record = json.loads(line)
            if "status" in record and "clip" not in record:
                statuses[record["label"]] = f"{record['status']} {record.get('reason') or ''}"
            elif "clip" in record and record.get("phase") != "warmup":
                clips[record["label"]].append(record)

    failures = [
        f"{label}: {status}" for label, status in statuses.items() if not status.startswith("ok")
    ]
    if not statuses:
        failures.append("no row finished")
    for label, records in clips.items():
        errors = [r["error"] for r in records if r.get("error")]
        wer = sum(r["wer"] for r in records) / len(records)
        rtfs = [r["rtf"] for r in records if r.get("rtf") is not None]
        rtf = f"{sum(rtfs) / len(rtfs):.3f}" if rtfs else "-"
        print(f"{label}: {len(records)} clips, mean WER {wer:.3f}, mean RTF {rtf}")
        if errors:
            failures.append(f"{label}: {len(errors)} clip errors, first: {errors[0]}")
        if wer > args.max_wer:
            failures.append(f"{label}: mean WER {wer:.3f} over {args.max_wer}")
    for label in statuses.keys() - clips.keys():
        if statuses[label].startswith("ok"):
            failures.append(f"{label}: ok with no clips scored")

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
