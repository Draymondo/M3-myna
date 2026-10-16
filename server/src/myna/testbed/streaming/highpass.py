"""The high-pass every level measurement looks through.

Speech carries next to nothing below HIGH_PASS_HZ, while a microphone's DC
offset and the slow drift of its bias would otherwise read as one endless
word: issue #64's internal microphone sat at -13 dBFS of offset, so the pause
detector never saw a pause and the coverage check never saw speech. The
client's voice tracker measures through the same filter (myna-audio voice.rs).
Only measurements are filtered; the audio a model decodes is not.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

HIGH_PASS_HZ = 80.0
_RATE = 16_000
# One-pole y[n] = R * (y[n-1] + x[n] - x[n-1]), solved in closed form per
# block; R**-_BLOCK stays a few thousand, well inside float64.
_POLE = math.exp(-2.0 * math.pi * HIGH_PASS_HZ / _RATE)
_BLOCK = 256
_GROWTH = _POLE ** np.arange(1, _BLOCK + 1, dtype=np.float64)


class HighPass:
    """Stateful one-pole high-pass over consecutive 16 kHz mono samples.

    Starts from its first sample, so an offset present from the start never
    rings through as a step."""

    def __init__(self) -> None:
        self._prev_in: float | None = None
        self._prev_out = 0.0

    def __call__(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        x = np.asarray(samples, dtype=np.float64)
        if not len(x):
            return x.astype(np.float32)
        if self._prev_in is None:
            self._prev_in = float(x[0])
        d = np.diff(x, prepend=self._prev_in)
        y = np.empty_like(x)
        prev = self._prev_out
        for lo in range(0, len(x), _BLOCK):
            block = d[lo : lo + _BLOCK]
            growth = _GROWTH[: len(block)]
            out = growth * (prev + np.cumsum(block / growth * _POLE))
            y[lo : lo + len(block)] = out
            prev = float(out[-1])
        self._prev_in = float(x[-1])
        self._prev_out = prev
        return y.astype(np.float32)
