"""Export a trained detector as one stateful streaming ONNX file (cache-aware, no recomputation).

Each call consumes 80 ms of new audio (1280 samples at 16 kHz) plus the state tensors and returns
the wake word probability for the newest 16-frame window and the updated state. It is numerically
equivalent to the full-sequence model (`tests/test_train_v2.py`):

* log-mel: the last 352 samples of audio are carried over (causal STFT window),
* conv subsampling: each causal stride-2 conv keeps the last input frame of its input,
* attention: un-rotated K/V of the last `left` frames per block; RoPE is applied with
  window-relative positions, which is exact because RoPE scores only depend on relative offsets
  (and avoids position drift over days-long streams),
* depthwise conv: the last k-1 frames of its input per block,
* head: the last 15 (normalized) encoder frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import N_CTX, CausalConv2dT, ConformerBlock, Detector, Encoder

CHUNK = 1280


class StreamingConformerDetector(nn.Module):
    def __init__(self, det: Detector) -> None:
        super().__init__()
        enc = det.encoder
        assert isinstance(enc, Encoder)
        self.enc, self.head, self.norm = enc, det.head, det.norm
        self.blocks = cast("list[ConformerBlock]", list(enc.blocks))
        b0 = self.blocks[0]
        self.left, self.heads = b0.left, b0.heads
        self.d = enc.lin.out_features
        self.dh = self.d // self.heads
        self.kconv = b0.dw.kernel_size[0]
        self.fe = enc.frontend
        self.tail = self.fe.n_fft - self.fe.hop

    def initial_state(self) -> dict[str, torch.Tensor]:
        c = cast(CausalConv2dT, self.enc.sub[0]).out_channels
        f0 = self.fe.n_mels
        f1 = (f0 + 2 - 3) // 2 + 1
        f2 = (f1 + 2 - 3) // 2 + 1
        L, H, Dh, d = len(self.blocks), self.heads, self.dh, self.d
        return {
            "audio": torch.zeros(1, self.tail),
            "sub0": torch.zeros(1, 1, f0, 1),
            "sub1": torch.zeros(1, c, f1, 1),
            "sub2": torch.zeros(1, c, f2, 1),
            "k": torch.zeros(L, 1, H, self.left, Dh),
            "v": torch.zeros(L, 1, H, self.left, Dh),
            "conv": torch.zeros(L, 1, d, self.kconv - 1),
            "ctx": torch.zeros(1, N_CTX - 1, self.enc.dim),
            "step": torch.zeros(1, dtype=torch.long),
        }

    def _mel(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, tail + 1280) -> (1, 8, F)
        fe = self.fe
        s = F.conv1d(x[:, None], fe.basis, stride=fe.hop)
        nb = fe.n_fft // 2 + 1
        p = s[:, :nb].pow(2) + s[:, nb:].pow(2)
        mel = p.transpose(1, 2) @ fe.fb
        return fe.compress(mel)

    @staticmethod
    def _cconv(
        conv: CausalConv2dT, x: torch.Tensor, cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal stride-2 conv along time (last dim) with 1 cached input frame.

        'same' pad along freq."""
        xc = torch.cat([cache, x], dim=-1)
        # the conv pads freq itself: F.pad exports as a Pad with computed amounts, which tract
        # (the Rust runtime) cannot analyse
        y = F.conv2d(
            xc, conv.weight, conv.bias, conv.stride, (conv.fpad, 0), conv.dilation, conv.groups
        )
        return y, xc[..., -1:]

    def _block(
        self,
        b: ConformerBlock,
        x: torch.Tensor,
        k_c: torch.Tensor,
        v_c: torch.Tensor,
        conv_c: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + 0.5 * b.ff1(x)
        # attention over [cache (left frames), current]
        h = b.ln_att(x)
        q, k, v = (
            b.qkv(h).reshape(1, 1, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        )  # (1, H, 1, Dh)
        K = torch.cat([k_c, k], dim=2)  # (1, H, left+1, Dh)
        V = torch.cat([v_c, v], dim=2)
        Kr = b.rope(K)  # positions 0..left
        qr = b.rope(q, offset=self.left)
        y = F.scaled_dot_product_attention(qr, Kr, V, attn_mask=valid[None, None, None, :])
        x = x + b.o(y.transpose(1, 2).reshape(1, 1, self.d))
        # causal depthwise conv module
        h = b.ln_conv(x)
        g = F.glu(b.pw1(h), dim=-1).transpose(1, 2)  # (1, d, 1)
        gc = torch.cat([conv_c, g], dim=-1)  # (1, d, k)
        y = F.conv1d(gc, b.dw.weight, b.dw.bias, groups=self.d).transpose(1, 2)
        x = x + b.pw2(F.silu(b.conv_norm(y)))
        x = x + 0.5 * b.ff2(x)
        return b.ln_out(x), K[:, :, 1:], V[:, :, 1:], gc[..., 1:]

    def forward(
        self,
        chunk: torch.Tensor,
        audio: torch.Tensor,
        sub0: torch.Tensor,
        sub1: torch.Tensor,
        sub2: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        conv: torch.Tensor,
        ctx: torch.Tensor,
        step: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x = torch.cat([audio, chunk], dim=1)
        mel = self._mel(x)  # (1, 8, F)
        h = mel.transpose(1, 2)[:, None]  # (1, 1, F, 8)
        s = self.enc.sub
        h, sub0 = self._cconv(cast(CausalConv2dT, s[0]), h, sub0)
        h = F.relu(h)
        h, sub1 = self._cconv(cast(CausalConv2dT, s[2]), h, sub1)
        h = F.relu(s[3](h))
        h, sub2 = self._cconv(cast(CausalConv2dT, s[5]), h, sub2)
        h = F.relu(s[6](h))  # (1, C, F', 1)
        B, C, Fq, T = h.shape
        z = self.enc.lin(h.permute(0, 3, 1, 2).reshape(1, 1, C * Fq))
        # keys before the stream start are masked out
        pos = torch.arange(self.left + 1)
        valid = pos >= (self.left - step)
        ks, vs, cs = [], [], []
        for i, b in enumerate(self.blocks):
            z, ki, vi, ci = self._block(b, z, k[i], v[i], conv[i], valid)
            ks.append(ki)
            vs.append(vi)
            cs.append(ci)
        frame = self.norm(self.enc.out(z))  # (1, 1, dim)
        win = torch.cat([ctx, frame], dim=1)  # (1, 16, dim)
        score = torch.sigmoid(self.head(win))
        new = (
            x[:, -self.tail :],
            sub0,
            sub1,
            sub2,
            torch.stack(ks),
            torch.stack(vs),
            torch.stack(cs),
            win[:, 1:],
            step + 1,
        )
        return (score, *new)


STATE_NAMES = ["audio", "sub0", "sub1", "sub2", "k", "v", "conv", "ctx", "step"]
WARMUP_S = 2.0  # runtimes warm the zero state on this much silence


@torch.no_grad()
def warm_state(m: StreamingConformerDetector, seconds: float = WARMUP_S) -> dict[str, torch.Tensor]:
    """The state after `seconds` of digital silence.

    Training windows always contain real encoder frames (of audio or silence) in front of the wake
    word, never all-zero state, so a stream that starts from zeros scores phrases spoken in its
    first ~1.3 s poorly (-0.07 on the hey-livekit bench). Starting from this state is equivalent
    to the listener having heard silence before the stream began."""
    st = m.initial_state()
    zeros = torch.zeros(1, CHUNK)
    for _ in range(int(round(seconds * 16000 / CHUNK))):
        out = m(zeros, *[st[n] for n in STATE_NAMES])
        st = dict(zip(STATE_NAMES, out[1:]))
    return st


def calibrate(det: Detector, logit_at_target: float, temperature: float = 2.0) -> Detector:
    """Fold a score calibration into the head: score = sigmoid((logit - c) / T).

    With `c` set to the logit that gives the target false-alarm rate on calibration audio, a
    threshold of 0.5 hits that target, and `T` spreads the scores so other thresholds stay
    meaningful."""
    with torch.no_grad():
        out = cast(nn.Linear, det.head.out)  # Head and MicroHead both end in a Linear `out`
        out.weight /= temperature
        out.bias.sub_(logit_at_target).div_(temperature)
    return det


def export_streaming(
    det: Detector, path: str | Path, opset: int = 17, metadata: dict[str, str] | None = None
) -> Path:
    """Export a stateful ONNX model.

    Inputs `chunk` (1, 1280) float32 + state tensors, outputs `score` + new state tensors (named
    `<state>_out`). Runtimes start every state tensor at zero (`step` is int64, the rest float32)
    and feed `WARMUP_S` of silence before the first real chunk, which gives `warm_state`. The state
    is plain inputs (no initializer defaults) so that tract, the Rust runtime, can feed it."""
    import onnx

    path = Path(path)
    m = StreamingConformerDetector(det.cpu().eval()).eval()
    st = m.initial_state()
    args = (torch.zeros(1, CHUNK), *[st[n] for n in STATE_NAMES])
    # the fused MHA fast path (taken when no parameter requires grad) has no ONNX export
    prev = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        torch.onnx.export(
            m,
            args,
            str(path),
            input_names=["chunk", *STATE_NAMES],
            output_names=["score", *[f"{n}_out" for n in STATE_NAMES]],
            opset_version=opset,
            dynamo=False,
        )
    finally:
        torch.backends.mha.set_fastpath_enabled(prev)
    if metadata:
        model = onnx.load(str(path))
        for k, v in metadata.items():
            model.metadata_props.add(key=k, value=v)
        onnx.save(model, str(path))
    return path
