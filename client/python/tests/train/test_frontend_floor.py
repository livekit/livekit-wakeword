"""The log-mel floor travels with the encoder weights; robust augmentation covers quiet input."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

from livekit.wakeword_train.augment import AugConfig, _party, augment, robust_aug  # noqa: E402
from livekit.wakeword_train.micro import host_mel  # noqa: E402
from livekit.wakeword_train.nn import MEL_DEFAULT, MEL_LOW_FLOOR, LogMel  # noqa: E402


def test_old_checkpoints_keep_the_original_floor() -> None:
    old = {k: v for k, v in LogMel().state_dict().items() if not k.startswith("mel_")}
    fe = LogMel(params=MEL_LOW_FLOOR)
    fe.load_state_dict(old)
    assert fe.params == pytest.approx(MEL_DEFAULT)


def test_floor_round_trips_through_the_state_dict() -> None:
    fe = LogMel()
    fe.load_state_dict(LogMel(params=MEL_LOW_FLOOR).state_dict())
    assert fe.params == pytest.approx(MEL_LOW_FLOOR)


def test_low_floor_resolves_quiet_input() -> None:
    g = torch.Generator().manual_seed(0)
    quiet = torch.randn(1, 16000, generator=g) * 10 ** (-60 / 20)
    silent = torch.zeros(1, 16000)
    for params, resolves in ((MEL_DEFAULT, False), (MEL_LOW_FLOOR, True)):
        fe = LogMel(params=params)
        gap = float((fe(quiet) - fe(silent)).mean())
        assert (gap > 0.5) is resolves


def test_host_mel_uses_the_given_floor() -> None:
    a = np.zeros(16000, np.float32)
    assert host_mel(a).min() == pytest.approx((np.log(1e-6) + 6) / 4, abs=1e-4)
    lo = (np.log(MEL_LOW_FLOOR[0]) + MEL_LOW_FLOOR[1]) / MEL_LOW_FLOOR[2]
    assert host_mel(a, MEL_LOW_FLOOR).min() == pytest.approx(lo, abs=1e-4)


def test_robust_aug_quantizes_and_reaches_quiet_levels() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 16000, generator=g) * 0.1
    out, _ = augment(x, robust_aug(p_quiet=1.0, p_codec=0.0), None, None, gen=g)
    assert torch.allclose(out * 32767, torch.round(out * 32767), atol=1e-3)
    peak_db = 20 * torch.log10(out.abs().amax(1))
    assert float(peak_db.max()) < -38  # every row from the quiet range (+ hiss)


def test_party_adds_talkers_at_their_level_and_skips_noise_rows() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 16000, generator=g) * 0.1
    b = torch.randn(32, 16000, generator=g) * 0.1
    only = torch.zeros(32, dtype=torch.bool)
    only[:4] = True
    cfg = AugConfig(
        p_rir=0,
        p_noise=0,
        p_eq=0,
        p_bandlimit=0,
        p_clip=0,
        p_party=1.0,
        party_talkers=(2, 2),
        party_sir_db=(10.0, 10.0),
        level_dbfs=(-6.0, -6.0),
    )
    y = _party(x.clone(), x, b, cfg, g, only)
    assert torch.equal(y[:4], x[:4])  # noise-only rows untouched
    inter = (y[4:] - x[4:]).pow(2).mean(1)
    want = 2 * x[4:].pow(2).mean(1) / 10  # two talkers, each 10 dB below
    assert torch.allclose(inter, want, rtol=0.15)
