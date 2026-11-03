"""Batched GPU augmentation: room, noise, talkers, EQ, band limits, level, clipping, codecs and the
device side (mic high-pass, quiet input, self-noise, int16 quantization)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

SR = 16000


@dataclass
class AugConfig:
    p_rir: float = 0.5
    p_noise: float = 0.85
    snr_db: tuple[float, float] = (-5.0, 25.0)
    p_babble: float = 0.15  # other speech as the noise
    p_eq: float = 0.3
    p_bandlimit: float = 0.35
    p_clip: float = 0.05
    level_dbfs: tuple[float, float] = (-40.0, -2.0)  # peak level of the final mix
    p_codec: float = 0.0  # mu-law telephone or perceptual-codec artifacts
    p_highpass: float = 0.0  # small-mic low-frequency roll-off
    highpass_hz: tuple[float, float] = (80.0, 400.0)
    p_quiet: float = 0.0  # peak level from `quiet_dbfs` instead
    quiet_dbfs: tuple[float, float] = (-65.0, -40.0)
    p_device_noise: float = 0.0  # self-noise floor added after the level (rms dBFS)
    device_noise_dbfs: tuple[float, float] = (-95.0, -70.0)
    quantize: bool = False  # int16 steps, as an ADC delivers
    p_party: float = 0.0  # overlapping talkers, each `party_sir_db` below the speaker
    party_talkers: tuple[int, int] = (2, 4)
    party_sir_db: tuple[float, float] = (5.0, 15.0)


def robust_aug(**kw: float | bool | tuple[float, float]) -> AugConfig:
    """Augmentation for encoders with the low log-mel floor (`nn.MEL_LOW_FLOOR`)."""
    base: dict[str, object] = dict(
        p_codec=0.25, p_highpass=0.5, p_quiet=0.25, p_device_noise=0.9, quantize=True
    )
    return AugConfig(**{**base, **kw})  # type: ignore[arg-type]


class GpuNoise:
    """Noise pool on the GPU (float16), sampled as random crops."""

    def __init__(self, noise: np.ndarray, device: str = "cuda") -> None:
        with warnings.catch_warnings():  # read-only pools (memory-mapped) are never written to
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            self.x = torch.from_numpy(np.ascontiguousarray(noise)).to(device)

    def sample(self, n: int, length: int, gen: torch.Generator | None = None) -> torch.Tensor:
        starts = torch.randint(0, len(self.x) - length, (n,), device=self.x.device, generator=gen)
        idx = starts[:, None] + torch.arange(length, device=self.x.device)[None]
        return self.x[idx].float()


def fft_conv(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal convolution of each row of x (B, T) with h (B, K), truncated to T."""
    T, K = x.shape[1], h.shape[1]
    n = 1 << int(np.ceil(np.log2(T + K)))
    y: torch.Tensor = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(h, n), n)
    return y[:, :T]


def active_power(x: torch.Tensor, frame: int = 400) -> torch.Tensor:
    """Mean power of the louder half of frames (speech power of padded clips)."""
    B, T = x.shape
    n = T // frame
    e = x[:, : n * frame].reshape(B, n, frame).pow(2).mean(-1)
    top = e.topk(max(1, n // 2), dim=1).values
    return top.mean(1) + 1e-10


def random_eq(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Smooth random spectral tilt/bumps (+-6 dB), via rfft multiplication."""
    B, T = x.shape
    X = torch.fft.rfft(x)
    F = X.shape[1]
    knots = torch.rand(B, 8, device=x.device, generator=gen) * 12 - 6  # dB
    curve = torch.nn.functional.interpolate(
        knots[:, None], size=F, mode="linear", align_corners=True
    )[:, 0]
    y: torch.Tensor = torch.fft.irfft(X * (10 ** (curve / 20)), T)
    return y


def bandlimit(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    B, T = x.shape
    X = torch.fft.rfft(x)
    freqs = torch.fft.rfftfreq(T, 1 / SR).to(x.device)
    lo = torch.rand(B, 1, device=x.device, generator=gen) * 250 + 50
    hi = torch.rand(B, 1, device=x.device, generator=gen) * 5600 + 2000
    mask = ((freqs[None] >= lo) & (freqs[None] <= hi)).float()
    y: torch.Tensor = torch.fft.irfft(X * mask, T)
    return y


def mulaw_8k(x: torch.Tensor) -> torch.Tensor:
    """Telephone path: 16 kHz -> 8 kHz -> 8-bit mu-law -> 16 kHz."""
    import torchaudio.functional as AF

    y: torch.Tensor = AF.resample(x, SR, 8000)
    mu = 255.0
    q = torch.sign(y) * torch.log1p(mu * y.abs().clamp(max=1)) / np.log1p(mu)
    q = torch.round((q + 1) / 2 * mu) / mu * 2 - 1
    y = torch.sign(q) * ((1 + mu) ** q.abs() - 1) / mu
    out: torch.Tensor = AF.resample(y, 8000, SR)
    return out[:, : x.shape[1]]


def perceptual_codec(x: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Opus/MP3-like artifacts: band limit, coarse log-magnitude steps and spectral holes."""
    B, T = x.shape
    win = torch.hann_window(512, device=x.device)
    S = torch.stft(x, 512, 128, window=win, return_complex=True)
    mag, ph = S.abs(), torch.angle(S)
    F_ = mag.shape[1]
    cutoff = (torch.rand(B, 1, 1, device=x.device, generator=gen) * 0.5 + 0.45) * F_  # 3.6..7.6 kHz
    band = (torch.arange(F_, device=x.device)[None, :, None] < cutoff).float()
    db = 20 * torch.log10(mag + 1e-7)
    step = torch.rand(B, 1, 1, device=x.device, generator=gen) * 4 + 1.5  # dB quantization step
    db = torch.round(db / step) * step
    floor = db.amax(dim=(1, 2), keepdim=True) - (
        torch.rand(B, 1, 1, device=x.device, generator=gen) * 25 + 35
    )
    keep = (db > floor).float()
    mag = (10 ** (db / 20)) * band * keep
    return torch.istft(torch.polar(mag, ph), 512, 128, window=win, length=T)


def highpass(
    x: torch.Tensor, lo: float, hi: float, gen: torch.Generator | None = None
) -> torch.Tensor:
    """Random 2nd-order high-pass per row."""
    B, T = x.shape
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(T, 1 / SR, device=x.device)[None]
    fc = torch.exp(
        torch.rand(B, 1, device=x.device, generator=gen) * (np.log(hi) - np.log(lo)) + np.log(lo)
    )
    r = (f / fc) ** 2
    y: torch.Tensor = torch.fft.irfft(X * (r / torch.sqrt(1 + r * r)), T)
    return y


def _party(
    out: torch.Tensor,
    x: torch.Tensor,
    babble: torch.Tensor,
    cfg: AugConfig,
    gen: torch.Generator | None,
    noise_only: torch.Tensor | None,
) -> torch.Tensor:
    """Add `party_talkers` other rows of `babble` to a share `p_party` of the rows, each at its
    own level `party_sir_db` below the row's speech."""
    B, T = x.shape
    dev = x.device
    m = _bern(cfg.p_party, B, dev, gen)
    if noise_only is not None:
        m = m & ~noise_only
    if not m.any():
        return out
    lo_k, hi_k = cfg.party_talkers
    k = torch.randint(lo_k, hi_k + 1, (B,), device=dev, generator=gen)
    unit = babble / (babble.pow(2).mean(1, keepdim=True).sqrt() + 1e-6)
    p_s = active_power(x)
    lo, hi = cfg.party_sir_db
    inter = torch.zeros_like(x)
    for j in range(1, hi_k + 1):
        sir = torch.rand(B, device=dev, generator=gen) * (hi - lo) + lo
        g = torch.sqrt(p_s / 10 ** (sir / 10)) * (j <= k).float()
        inter = inter + g[:, None] * torch.roll(unit, j, 0)
    return torch.where(m[:, None], out + inter, out)


def _bern(p: float, n: int, device: torch.device, gen: torch.Generator | None) -> torch.Tensor:
    return torch.rand(n, device=device, generator=gen) < p


def augment(
    speech: torch.Tensor,
    cfg: AugConfig,
    noise: GpuNoise | None,
    rirs: torch.Tensor | None,
    babble: torch.Tensor | None = None,
    gen: torch.Generator | None = None,
    noise_only: torch.Tensor | None = None,
    snr_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Augment clean speech windows (B, T) on the GPU; returns (mix, reverberant clean).

    `noise_only` rows become pure noise; `babble` (B, T) is other speech for interference."""
    B, T = speech.shape
    dev = speech.device
    x = speech
    if rirs is not None and cfg.p_rir > 0:
        m = _bern(cfg.p_rir, B, dev, gen)
        if m.any():
            h = rirs[torch.randint(0, len(rirs), (int(m.sum()),), device=dev, generator=gen)]
            wet = fft_conv(x[m], h)
            wet = wet * torch.sqrt(active_power(x[m]) / active_power(wet))[:, None]
            x = x.clone()
            x[m] = wet
    clean = x
    out = x
    if noise is not None and cfg.p_noise > 0:
        m = _bern(cfg.p_noise, B, dev, gen)
        if noise_only is not None:
            m = m | noise_only
        n = noise.sample(B, T, gen)
        if babble is not None and cfg.p_babble > 0:
            bm = _bern(cfg.p_babble, B, dev, gen)
            n = torch.where(
                bm[:, None], babble / (babble.pow(2).mean(1, keepdim=True).sqrt() + 1e-6) * 0.1, n
            )
        lo, hi = cfg.snr_db
        snr = torch.rand(B, device=dev, generator=gen) * (hi - lo) + lo
        if snr_override is not None:
            snr = snr_override
        p_s = active_power(x)
        p_n = n.pow(2).mean(1) + 1e-10
        scale = torch.sqrt(p_s / (p_n * 10 ** (snr / 10)))
        out = torch.where(m[:, None], x + scale[:, None] * n, x)
        if noise_only is not None and noise_only.any():
            out = torch.where(noise_only[:, None], n, out)
    if babble is not None and cfg.p_party > 0 and B > 1:
        out = _party(out, x, babble, cfg, gen, noise_only)
    if cfg.p_eq > 0:
        m = _bern(cfg.p_eq, B, dev, gen)
        if m.any():
            out = out.clone()
            out[m] = random_eq(out[m], gen)
    if cfg.p_bandlimit > 0:
        m = _bern(cfg.p_bandlimit, B, dev, gen)
        if m.any():
            out = out.clone()
            out[m] = bandlimit(out[m], gen)
    if cfg.p_highpass > 0:
        m = _bern(cfg.p_highpass, B, dev, gen)
        if m.any():
            out = out.clone()
            out[m] = highpass(out[m], *cfg.highpass_hz, gen=gen)
    lo, hi = cfg.level_dbfs
    db = torch.rand(B, device=dev, generator=gen) * (hi - lo) + lo
    if cfg.p_quiet > 0:
        qlo, qhi = cfg.quiet_dbfs
        q = torch.rand(B, device=dev, generator=gen) * (qhi - qlo) + qlo
        db = torch.where(_bern(cfg.p_quiet, B, dev, gen), q, db)
    lvl = 10 ** (db / 20)
    pk = out.abs().amax(1) + 1e-6
    g = lvl / pk
    out = out * g[:, None]
    clean = clean * g[:, None]
    if cfg.p_device_noise > 0:
        m = _bern(cfg.p_device_noise, B, dev, gen)
        nlo, nhi = cfg.device_noise_dbfs
        rms = 10 ** ((torch.rand(B, device=dev, generator=gen) * (nhi - nlo) + nlo) / 20)
        hiss = torch.randn(B, T, device=dev, generator=gen) * (rms * m)[:, None]
        out = out + hiss
    if cfg.p_clip > 0:
        m = _bern(cfg.p_clip, B, dev, gen)
        drive = 1 + torch.rand(B, device=dev, generator=gen) * 4
        out = torch.where(m[:, None], torch.tanh(out * drive[:, None] * 3) / 3, out)
    if cfg.p_codec > 0:
        m = _bern(cfg.p_codec, B, dev, gen)
        if m.any():
            out = out.clone()
            idx = torch.nonzero(m).flatten()
            tel = torch.rand(len(idx), device=dev, generator=gen) < 0.3
            if tel.any():
                out[idx[tel]] = mulaw_8k(out[idx[tel]])
            if (~tel).any():
                out[idx[~tel]] = perceptual_codec(out[idx[~tel]], gen)
    out = out.clamp(-1, 1)
    if cfg.quantize:
        out = torch.round(out * 32767) / 32767
    return out, clean
