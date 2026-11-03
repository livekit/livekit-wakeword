"""Streaming resampling to 16 kHz (`WakeWordModel(sample_rate=...)`)."""

from __future__ import annotations

import numpy as np
import pytest

from livekit.wakeword._resample import Resampler


def tone(rate: int, hz: float, seconds: float = 2.0) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * hz * np.arange(int(rate * seconds)) / rate)).astype(np.float32)


@pytest.mark.parametrize("rate", [8000, 22050, 24000, 44100, 48000, 96000])
def test_tones_pass_unchanged(rate: int) -> None:
    for hz in (300.0, 1000.0, 3500.0):
        y = Resampler(rate)(tone(rate, hz))
        assert abs(len(y) - 32000) <= 1
        t = np.arange(4000, 28000) / 16000  # steady state, past the filter delay
        basis = np.stack([np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t)], 1)
        fit, *_ = np.linalg.lstsq(basis, y[4000:28000], rcond=None)
        residual = y[4000:28000] - basis @ fit
        assert np.hypot(*fit) == pytest.approx(0.5, rel=1e-3)
        assert residual.std() < 1e-4  # better than 70 dB


@pytest.mark.parametrize("rate", [44100, 48000])
def test_frequencies_above_8khz_are_removed(rate: int) -> None:
    y = Resampler(rate)(tone(rate, 9000.0))
    assert np.abs(y[4000:]).max() < 0.5 * 10 ** (-80 / 20)


def test_chunking_does_not_change_the_output() -> None:
    x = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    whole = Resampler(44100)(x)
    r, parts, pos = Resampler(44100), [], 0
    for n in [1, 7, 441, 1000, 3333] * 20:
        parts.append(r(x[pos : pos + n]))
        pos += n
    parts.append(r(x[pos:]))
    np.testing.assert_allclose(np.concatenate(parts), whole, atol=1e-6)
    r.reset()
    np.testing.assert_allclose(r(x), whole, atol=1e-6)


def test_bad_rate() -> None:
    with pytest.raises(ValueError):
        Resampler(0)
