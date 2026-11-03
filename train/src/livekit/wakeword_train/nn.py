"""Networks: causal log-mel, the causal Conformer encoder (one frame per 80 ms) and the head that
scores the last 16 frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .micro import MicroEncoder

SR = 16000
FRAME = 1280  # samples per encoder frame (80 ms)
N_CTX = 16  # encoder frames seen by the head


# (floor, offset, scale): log-mel = (log(mel + floor) + offset) / scale, stored with the weights
MEL_DEFAULT = (1e-6, 6.0, 4.0)  # floor near -50 dBFS
MEL_LOW_FLOOR = (1e-9, 9.0, 6.0)  # near -80 dBFS: quiet speech and small mics


class LogMel(nn.Module):
    """Causal log-mel: frame k covers samples [160k-352, 160k+160). STFT as a conv1d."""

    basis: torch.Tensor
    fb: torch.Tensor
    mel_floor: torch.Tensor
    mel_offset: torch.Tensor
    mel_scale: torch.Tensor

    def __init__(
        self,
        n_mels: int = 40,
        n_fft: int = 512,
        win: int = 400,
        hop: int = 160,
        params: tuple[float, float, float] = MEL_DEFAULT,
    ) -> None:
        super().__init__()
        import librosa

        self.n_fft, self.hop, self.n_mels = n_fft, hop, n_mels
        w = np.zeros(n_fft, np.float32)
        w[n_fft - win :] = np.hanning(win + 2)[1:-1]  # right-aligned window: no lookahead
        k = np.arange(n_fft // 2 + 1)[:, None]
        t = np.arange(n_fft)[None]
        re = (np.cos(2 * np.pi * k * t / n_fft) * w).astype(np.float32)
        im = (-np.sin(2 * np.pi * k * t / n_fft) * w).astype(np.float32)
        self.register_buffer("basis", torch.from_numpy(np.concatenate([re, im])[:, None]))
        fb = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=40, fmax=7600)
        self.register_buffer("fb", torch.from_numpy(fb.T.astype(np.float32)))
        self.set_params(params)

    def set_params(self, params: tuple[float, float, float]) -> None:
        for name, v in zip(("mel_floor", "mel_offset", "mel_scale"), params, strict=True):
            self.register_buffer(name, torch.tensor(float(v)))

    @property
    def params(self) -> tuple[float, float, float]:
        return (float(self.mel_floor), float(self.mel_offset), float(self.mel_scale))

    def _load_from_state_dict(  # type: ignore[no-untyped-def]
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        # checkpoints without the buffers use the original front end
        for name, v in zip(("mel_floor", "mel_offset", "mel_scale"), MEL_DEFAULT, strict=True):
            state_dict.setdefault(prefix + name, torch.tensor(v))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def compress(self, mel: torch.Tensor) -> torch.Tensor:
        return (torch.log(mel + self.mel_floor) + self.mel_offset) / self.mel_scale

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = F.pad(wav[:, None], (self.n_fft - self.hop, 0))
        s = F.conv1d(x, self.basis, stride=self.hop)
        nb = self.n_fft // 2 + 1
        p = s[:, :nb].pow(2) + s[:, nb:].pow(2)
        mel = p.transpose(1, 2) @ self.fb
        return self.compress(mel)


class CausalConv1d(nn.Conv1d):
    def __init__(self, cin: int, cout: int, k: int, groups: int = 1) -> None:
        super().__init__(cin, cout, k, groups=groups)
        self.lpad = k - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.lpad, 0)))


class CausalConv2dT(nn.Conv2d):
    """2-D conv, causal along time (last dim), 'same' along frequency."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__(cin, cout, k, stride=stride, groups=groups)
        self.fpad = (k[0] - 1) // 2
        self.tpad = k[1] - stride[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.tpad, 0, self.fpad, self.fpad)))


class RoPE(nn.Module):
    inv: torch.Tensor

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.register_buffer(
            "inv", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)), persistent=False
        )

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:  # (B, H, T, Dh)
        t = torch.arange(offset, offset + x.shape[2], device=x.device, dtype=torch.float32)
        f = torch.einsum("t,d->td", t, self.inv)
        cos, sin = f.cos()[None, None], f.sin()[None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


def _ff(d: int, mult: int, drop: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d),
        nn.Linear(d, mult * d),
        nn.SiLU(),
        nn.Dropout(drop),
        nn.Linear(mult * d, d),
        nn.Dropout(drop),
    )


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, k: int, left: int, drop: float = 0.1) -> None:
        super().__init__()
        self.ff1, self.ff2 = _ff(d, ff, drop), _ff(d, ff, drop)
        self.ln_att = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.heads, self.left = heads, left
        self.rope = RoPE(d // heads)
        self.ln_conv = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = CausalConv1d(d, d, k, groups=d)
        self.conv_norm = nn.LayerNorm(d)
        self.pw2 = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ln_out = nn.LayerNorm(d)

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.rope(q), self.rope(k)
        i = torch.arange(T, device=x.device)
        mask = (i[None] <= i[:, None]) & (i[:, None] - i[None] <= self.left)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out: torch.Tensor = self.o(y.transpose(1, 2).reshape(B, T, D))
        return out

    def conv(self, x: torch.Tensor) -> torch.Tensor:
        y = F.glu(self.pw1(x), dim=-1).transpose(1, 2)
        y = self.dw(y).transpose(1, 2)
        out: torch.Tensor = self.pw2(F.silu(self.conv_norm(y)))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.drop(self.attn(self.ln_att(x)))
        x = x + self.drop(self.conv(self.ln_conv(x)))
        x = x + 0.5 * self.ff2(x)
        out: torch.Tensor = self.ln_out(x)
        return out


class Encoder(nn.Module):
    """Causal Conformer: log-mel -> 8x depthwise-separable conv subsampling -> conformer blocks."""

    def __init__(
        self,
        dim: int = 96,
        n_mels: int = 40,
        d: int = 144,
        layers: int = 4,
        heads: int = 4,
        ff: int = 2,
        k: int = 15,
        left: int = 32,
        sub_c: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.frontend = LogMel(n_mels)
        self.sub = nn.Sequential(
            CausalConv2dT(1, sub_c, (3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            CausalConv2dT(sub_c, sub_c, (3, 3), stride=(2, 2), groups=sub_c),
            nn.Conv2d(sub_c, sub_c, 1),
            nn.ReLU(inplace=True),
            CausalConv2dT(sub_c, sub_c, (3, 3), stride=(2, 2), groups=sub_c),
            nn.Conv2d(sub_c, sub_c, 1),
            nn.ReLU(inplace=True),
        )
        fq = n_mels
        for _ in range(3):
            fq = (fq - 1) // 2 + 1
        self.lin = nn.Linear(sub_c * fq, d)
        self.blocks = nn.ModuleList([ConformerBlock(d, heads, ff, k, left) for _ in range(layers)])
        self.out = nn.Linear(d, dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T // 1280, dim)
        n = wav.shape[1] // FRAME
        mel = self.frontend(wav)[:, : n * 8]
        x = self.sub(mel.transpose(1, 2)[:, None])
        B, C, Fq, T = x.shape
        x = self.lin(x.permute(0, 3, 1, 2).reshape(B, T, C * Fq))
        for b in self.blocks:
            x = b(x)
        out: torch.Tensor = self.out(x)
        return out[:, :n]

    @staticmethod
    def frame_end_samples(n_frames: int) -> np.ndarray:
        return (np.arange(n_frames) + 1) * FRAME


class Head(nn.Module):
    """Conv-attention classifier over the last 16 encoder frames (returns logits)."""

    def __init__(
        self, dim: int = 96, layer_dim: int = 128, n_blocks: int = 2, n_heads: int = 4
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(dim, layer_dim, 3, padding=1),
            nn.LayerNorm([layer_dim, N_CTX]),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers += [
                nn.Conv1d(layer_dim, layer_dim, 3, padding=1),
                nn.LayerNorm([layer_dim, N_CTX]),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        self.attention = nn.MultiheadAttention(layer_dim, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(layer_dim)
        self.out = nn.Linear(layer_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, 16, dim) -> (N,)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        a, _ = self.attention(x, x, x, need_weights=False)
        x = self.attn_norm(x + a).mean(1)
        out: torch.Tensor = self.out(x)
        return out.squeeze(-1)


class Detector(nn.Module):
    def __init__(
        self, encoder: Encoder | MicroEncoder, head: nn.Module | None = None, norm: bool = True
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head or Head(encoder.dim)
        # micro goes without: LayerNorm does not quantize well
        self.norm: nn.Module = nn.LayerNorm(encoder.dim) if norm else nn.Identity()

    def head_windows(self, f: torch.Tensor, last: int | None = None) -> torch.Tensor:
        """f (B, n, D) -> logits (B, n - 15) for every 16-frame window (or only the `last` k)."""
        f = self.norm(f)
        if last is not None:
            f = f[:, f.shape[1] - (last + N_CTX - 1) :]
        B, n, D = f.shape
        w = f.unfold(1, N_CTX, 1).permute(0, 1, 3, 2)
        k = w.shape[1]
        logits: torch.Tensor = self.head(w.reshape(B * k, N_CTX, D))
        return logits.reshape(B, k)

    def forward(self, wav: torch.Tensor, last: int | None = None) -> torch.Tensor:
        return self.head_windows(self.encoder(wav), last)
