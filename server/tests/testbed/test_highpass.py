"""The measurement high-pass (myna.testbed.streaming.highpass)."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy", reason="adapter extras not installed")

from myna.testbed.streaming.highpass import HIGH_PASS_HZ, HighPass

RATE = 16_000


def _reference(x: np.ndarray) -> np.ndarray:
    pole = math.exp(-2.0 * math.pi * HIGH_PASS_HZ / RATE)
    out, prev_in, prev_out = [], float(x[0]), 0.0
    for v in x.astype(np.float64):
        prev_out = pole * (prev_out + v - prev_in)
        prev_in = v
        out.append(prev_out)
    return np.array(out)


def test_the_block_solution_matches_the_recurrence_across_calls():
    x = (np.random.default_rng(0).standard_normal(5000) * 0.1 - 0.2).astype(np.float32)
    f = HighPass()
    got = np.concatenate([f(x[:1000]), f(x[1000:1001]), f(x[1001:1001]), f(x[1001:])])
    assert np.max(np.abs(got - _reference(x))) < 1e-6


def test_an_offset_from_the_first_sample_is_silence():
    assert np.max(np.abs(HighPass()(np.full(RATE, -0.219, dtype=np.float32)))) < 1e-7


def test_speech_frequencies_pass_and_rumble_does_not():
    t = np.arange(RATE, dtype=np.float32) / RATE

    def gain(hz: float) -> float:
        y = HighPass()(np.sin(2 * np.pi * hz * t).astype(np.float32))[RATE // 2 :]
        return float(np.sqrt(2 * np.mean(y * y)))

    assert gain(1_000.0) > 0.97
    assert gain(300.0) > 0.94
    assert gain(15.0) < 0.2
