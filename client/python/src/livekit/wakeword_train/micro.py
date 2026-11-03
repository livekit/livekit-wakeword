"""Micro tier: BC-ResNet encoder + conv head for microcontrollers. The device computes the log-mel
front end (`client/esp32/livekit_wakeword/src/lkww_frontend.c`); the model takes 8 log-mel
frames + its state per 80 ms step and ships as int8 TFLite (int8 weights, activations, inputs and
outputs) for TFLite Micro."""

from __future__ import annotations

import copy
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import FRAME, MEL_DEFAULT, N_CTX, Detector, LogMel

logger = logging.getLogger(__name__)


class CausalConv2dT(nn.Conv2d):
    """2-D conv, causal along time (last dim), 'same' along frequency."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(cin, cout, k, stride=stride, dilation=dilation, groups=groups, bias=bias)
        kf, kt = k
        self.fpad = dilation[0] * (kf - 1) // 2
        self.tpad = dilation[1] * (kt - 1) + 1 - stride[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _CTX is not None:
            return _streaming_conv2d(self, x)
        out: torch.Tensor = super().forward(F.pad(x, (self.tpad, 0, self.fpad, self.fpad)))
        return out


class SubSpectralNorm(nn.Module):
    def __init__(self, c: int, s: int = 5) -> None:
        super().__init__()
        self.s = s
        self.bn = nn.BatchNorm2d(c * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, F, T)
        B, C, Fq, T = x.shape
        if Fq % self.s:
            return x
        y: torch.Tensor = self.bn(x.reshape(B, C * self.s, Fq // self.s, T))
        return y.reshape(B, C, Fq, T)


class BCBlock(nn.Module):
    def __init__(
        self, cin: int, cout: int, fstride: int, tstride: int, dil: int, drop: float = 0.1
    ) -> None:
        super().__init__()
        self.trans = cin != cout or fstride > 1 or tstride > 1
        self.tstride = tstride
        self.proj: nn.Module = (
            nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
            if cin != cout
            else nn.Identity()
        )
        self.f2 = nn.Sequential(
            CausalConv2dT(cout, cout, (3, 1), stride=(fstride, 1), groups=cout, bias=False),
            SubSpectralNorm(cout),
        )
        self.f1 = nn.Sequential(
            CausalConv2dT(
                cout, cout, (1, 3), stride=(1, tstride), dilation=(1, dil), groups=cout, bias=False
            ),
            nn.BatchNorm2d(cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 1, bias=False),
            nn.Dropout2d(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x2: torch.Tensor = self.f2(x)
        y: torch.Tensor = self.f1(x2.mean(2, keepdim=True))  # (B, C, 1, T')
        if self.trans:
            if self.tstride > 1:
                x2 = x2[..., self.tstride - 1 :: self.tstride]
            n = min(y.shape[-1], x2.shape[-1])
            return F.relu(x2[..., :n] + y[..., :n])
        return F.relu(x + x2 + y)


class MicroEncoder(nn.Module):
    """BC-ResNet: 16 kHz audio -> one `dim`-d frame per 80 ms, strictly causal."""

    def __init__(
        self,
        dim: int = 64,
        n_mels: int = 40,
        mult: float = 5.0,
        depth: tuple[int, ...] = (2, 2, 4, 4),
    ) -> None:
        super().__init__()
        self.dim = dim
        self.frontend = LogMel(n_mels)
        base = [int(round(c * mult)) for c in (8, 12, 16, 20)]
        c0 = int(round(16 * mult))
        self.stem = nn.Sequential(
            CausalConv2dT(1, c0, (5, 5), stride=(2, 2), bias=False), nn.BatchNorm2d(c0), nn.ReLU()
        )
        blocks: list[nn.Module] = []
        cin = c0
        tstrides, fstrides = [1, 2, 2, 1], [1, 2, 2, 1]  # stem 2 -> 8x in time overall
        for i, (c, d) in enumerate(zip(base, depth)):
            for j in range(d):
                first = j == 0
                dil = 2 ** min(j, 3) if i >= 2 else 1
                blocks.append(
                    BCBlock(cin, c, fstrides[i] if first else 1, tstrides[i] if first else 1, dil)
                )
                cin = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            CausalConv2dT(cin, cin, (5, 5), groups=cin, bias=False),
            nn.Conv2d(cin, 2 * cin, 1, bias=False),
            nn.BatchNorm2d(2 * cin),
            nn.ReLU(),
        )
        self.out = nn.Linear(2 * cin, dim)

    def body(self, mel: torch.Tensor) -> torch.Tensor:  # (B, Tm, F) -> (B, Tm // 8, dim)
        x = self.blocks(self.stem(mel.transpose(1, 2)[:, None]))
        h: torch.Tensor = self.head(x).mean(2)
        out: torch.Tensor = self.out(h.transpose(1, 2))
        return out

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T // 1280, dim)
        n = wav.shape[1] // FRAME
        return self.body(self.frontend(wav)[:, : n * 8])[:, :n]

    @staticmethod
    def frame_end_samples(n_frames: int) -> np.ndarray:
        return (np.arange(n_frames) + 1) * FRAME


class MicroHead(nn.Module):
    """Two 1-D convs over the 16-frame window, mean-pool, linear (TFLite Micro friendly)."""

    def __init__(self, dim: int = 64, ch: int = 32, n_ctx: int = N_CTX) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Linear(ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, 16, dim) -> (N,)
        h: torch.Tensor = self.conv(x.transpose(1, 2)).mean(-1)
        out: torch.Tensor = self.out(h)
        return out.squeeze(-1)


def micro_detector(encoder: MicroEncoder | None = None, head_ch: int = 32) -> Detector:
    enc = encoder or MicroEncoder()  # weights: livekit-wakeword-encoder-micro (`encoders`)
    return Detector(enc, MicroHead(enc.dim, ch=head_ch), norm=False)


# ---- streaming: every causal conv keeps its last `tpad` input frames as state ----
class _Ctx:
    def __init__(self, caches: list[torch.Tensor] | None) -> None:
        self.caches = caches  # None = discovery (zeros created on the fly)
        self.i = 0
        self.out: list[torch.Tensor] = []
        self.shapes: list[tuple[int, ...]] = []


_CTX: _Ctx | None = None


def _streaming_conv2d(conv: CausalConv2dT, x: torch.Tensor) -> torch.Tensor:
    assert _CTX is not None
    p = conv.tpad
    if p > 0:
        shape = tuple(x.shape[:-1]) + (p,)
        if _CTX.caches is None:
            _CTX.shapes.append(shape)
            cache = torch.zeros(shape, dtype=x.dtype)
        else:
            cache = _CTX.caches[_CTX.i]
            _CTX.i += 1
        x = torch.cat([cache, x], dim=-1)
        _CTX.out.append(x[..., -p:])
    out: torch.Tensor = nn.Conv2d.forward(conv, F.pad(x, (0, 0, conv.fpad, conv.fpad)))
    return out


@contextmanager
def _streaming(ctx: _Ctx) -> Any:
    global _CTX
    _CTX = ctx
    try:
        yield ctx
    finally:
        _CTX = None


class StreamingMicro(nn.Module):
    """(mel (1, 8, n_mels), s0 .. sK) -> (score (1,), s0_out .. sK_out), one 80 ms step.

    Every conv cache and the 15-frame head context is its own flat (1, n) state tensor: 2-D
    tensors are never transposed by converters, each gets its own quantization scale, and no wide
    concat is needed (TFLite Micro caps CONCATENATION at 10 inputs). A runtime copies each
    `s<i>_out` into `s<i>` after every step."""

    def __init__(self, det: Detector) -> None:
        super().__init__()
        enc = det.encoder
        assert isinstance(enc, MicroEncoder)
        self.det = det.eval()  # registered once: shared submodules break ONNX constant folding
        self.n_mels = enc.frontend.n_mels
        self.mel_params = enc.frontend.params
        with torch.no_grad(), _streaming(_Ctx(None)) as ctx:
            enc.body(torch.zeros(1, 8, self.n_mels))
        self.shapes = [*ctx.shapes, (1, N_CTX - 1, enc.dim)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.names = [f"s{i}" for i in range(len(self.sizes))]

    @property
    def enc(self) -> MicroEncoder:
        enc = self.det.encoder
        assert isinstance(enc, MicroEncoder)
        return enc

    def zero_state(self) -> list[torch.Tensor]:
        return [torch.zeros(1, n) for n in self.sizes]

    def forward(self, mel: torch.Tensor, *flat: torch.Tensor) -> tuple[torch.Tensor, ...]:
        states = [p.reshape(s) for p, s in zip(flat, self.shapes)]
        with _streaming(_Ctx(states[:-1])) as ctx:
            frame = self.det.norm(self.enc.body(mel))
        win = torch.cat([states[-1], frame], dim=1)
        score = torch.sigmoid(self.det.head(win))
        new = [*ctx.out, win[:, 1:]]
        return (score.reshape(1), *[t.reshape(1, -1) for t in new])


def host_mel(audio: np.ndarray, params: tuple[float, float, float] = MEL_DEFAULT) -> np.ndarray:
    """Reference host front end (same as the ESP32 `lkww_frontend.c`): audio -> (T // 1280, 8, 40).
    `params` (floor, offset, scale) come with the encoder (`LogMel.params`)."""
    with torch.no_grad():
        mel = LogMel(params=params)(torch.from_numpy(audio.astype(np.float32))[None])[0]
    n = mel.shape[0] // 8
    out: np.ndarray = mel[: n * 8].reshape(n, 8, -1).numpy()
    return out


@torch.no_grad()
def warm_micro_state(m: StreamingMicro, seconds: float = 2.0) -> list[torch.Tensor]:
    """State after `seconds` of silence (see `export.warm_state`): runtimes reach it by zeroing the
    state and running that much silence through the model."""
    state = m.zero_state()
    for mel in host_mel(np.zeros(int(seconds * 16000), np.float32), m.mel_params):
        state = list(m(torch.from_numpy(mel)[None], *state)[1:])
    return state


class _SubSpectralConv(nn.Module):
    """A trained SubSpectralNorm as reshape -> per-channel 1x1 grouped conv -> reshape: the same
    function, in the form converters quantize well. As a BatchNorm on the reshaped tensor it
    becomes MUL / ADD by constant tensors, quantized per tensor, and its per-subband gains span
    signs and orders of magnitude (unusable in int8); conv weights are quantized per channel."""

    def __init__(self, ssn: SubSpectralNorm) -> None:
        super().__init__()
        bn, self.s = ssn.bn, ssn.s
        c = bn.num_features
        assert bn.running_mean is not None and bn.running_var is not None
        mean, var = bn.running_mean, bn.running_var
        scale = bn.weight / torch.sqrt(var + bn.eps)
        self.conv = nn.Conv2d(c, c, 1, groups=c, bias=True)
        with torch.no_grad():
            self.conv.weight.copy_(scale.view(c, 1, 1, 1))
            cast(torch.Tensor, self.conv.bias).copy_(bn.bias - mean * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Fq, T = x.shape
        if Fq % self.s:
            return x
        y: torch.Tensor = self.conv(x.reshape(B, C * self.s, Fq // self.s, T))
        return y.reshape(B, C, Fq, T)


def _export_form(det: Detector) -> Detector:
    """A copy of `det` with every SubSpectralNorm as a `_SubSpectralConv` (same outputs)."""
    det = copy.deepcopy(det).cpu().eval()
    for mod in list(det.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, SubSpectralNorm):
                setattr(mod, name, _SubSpectralConv(child))
    return det


def export_micro(
    det: Detector,
    out_dir: str | Path,
    name: str,
    tflite: bool | None = None,
    dataset: str | list[str] | None = None,
    opset: int = 17,
) -> dict[str, Path]:
    """Export `<name>_micro.onnx` (inputs `mel` + states `s0..sK`, outputs `score` + `s<i>_out`)
    and, with the `tflite` extra, `<name>_micro.tflite` (int8, see `convert_micro_tflite`).
    Runtimes zero the states and warm up on 2 s of silence (`warm_micro_state`). `dataset` adds
    wake word and confusable clips to the quantization calibration set."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = StreamingMicro(_export_form(det)).eval()
    onnx_path = out_dir / f"{name}_micro.onnx"
    torch.onnx.export(
        m,
        (torch.zeros(1, 8, m.n_mels), *m.zero_state()),
        str(onnx_path),
        input_names=["mel", *m.names],
        output_names=["score", *[f"{n}_out" for n in m.names]],
        opset_version=opset,
        dynamo=False,
    )
    _simplify(onnx_path)
    out = {"onnx": onnx_path}
    # the host front end must use the encoder's log-mel floor (C: lkww_frontend_init)
    side = out_dir / f"{name}_micro.json"
    side.write_text(json.dumps({"mel_params": list(m.mel_params)}) + "\n")
    out["frontend"] = side
    if tflite is None:
        import importlib.util

        tflite = importlib.util.find_spec("onnx2tf") is not None
    if not tflite:
        return out
    try:
        tfl = out_dir / f"{name}_micro.tflite"
        out["tflite"] = convert_micro_tflite(m, onnx_path, tfl, dataset)
    except Exception as e:  # keep the ONNX model; TFLite is an extra artifact
        logger.warning("TFLite conversion failed (%s); %s is still usable", e, onnx_path)
    return out


def _simplify(path: Path) -> None:
    """Fold the exporter's computed pad amounts into constants (onnx-simplifier). onnx2tf
    mis-lays-out the unfolded `Pad` chains, and the simplified graph is ~5x smaller."""
    try:
        import onnx
        import onnxsim
    except ImportError:
        return
    model, ok = onnxsim.simplify(onnx.load(str(path)))
    if ok:
        onnx.save(model, str(path))


def _with_phrases(
    a: np.ndarray, sets: dict[str, Any], rng: np.random.Generator, k: int
) -> np.ndarray:
    """Splice wake word and confusable clips into a 20 s calibration stream."""
    for j in range(6):
        cs = sets["positive" if (j + k) % 2 == 0 else "adversarial"]
        if not len(cs.lengths):
            continue
        c = cs.clip(int(rng.integers(len(cs.lengths)))).astype(np.float32) / 32767
        at = int(rng.integers(0, max(1, len(a) - len(c))))
        a[at : at + len(c)] = c[: len(a) - at]
    return a


def _calibration_steps(
    m: StreamingMicro, dataset: str | list[str] | None = None, n: int = 800
) -> dict[str, np.ndarray]:
    """Representative inputs for quantization: streamed held-out speech (clean, noisy, quiet),
    with the phrase's own wake word and confusable clips spliced in when `dataset` is given (so
    activation ranges also cover detections, not just background)."""
    from . import data as D

    rng = np.random.default_rng(0)
    files = [(r, f.replace("train", "validation")) for r, f in D.negative_speech_files()]
    pool = D.StreamPool([x for x in files if "fleurs" in x[0]], buffer_s=240, min_fill_s=200)
    noise = np.asarray(D.noise_pool(), np.float32)
    sets = D.load_clipsets(dataset) if dataset else None
    warm = warm_micro_state(m)
    rows: list[list[np.ndarray]] = []
    for k in range(16):
        a = pool.crop(rng, 16000 * 20).astype(np.float32) / 32767
        if sets is not None and k % 4 != 3:
            a = _with_phrases(a, sets, rng, k)
        if k % 3 == 1:
            nz = noise[k * 320000 : k * 320000 + len(a)]
            a = a + nz * np.sqrt((a**2).mean() / ((nz**2).mean() * 10 + 1e-12))
        if k % 3 == 2:
            a = a * 0.05
        state = [w.clone() for w in warm]
        with torch.no_grad():
            for mel in host_mel(a, m.mel_params):
                x = torch.from_numpy(mel)[None]
                rows.append([x.numpy(), *[t.numpy() for t in state]])
                state = list(m(x, *state)[1:])
    pool.stop(join=True)
    idx = rng.choice(len(rows), min(n, len(rows)), replace=False)
    names = ["mel", *m.names]
    return {nm: np.concatenate([rows[i][j] for i in idx]) for j, nm in enumerate(names)}


def convert_micro_tflite(
    m: StreamingMicro,
    onnx_path: Path,
    out: Path,
    dataset: str | list[str] | None = None,
) -> Path:
    """ONNX -> int8 TFLite with builtin ops only (int8 weights, activations, inputs and outputs:
    the state passes between steps without quantize / dequantize ops; runtimes quantize the mel
    input and dequantize the score)."""
    import tempfile

    import onnx2tf
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # conversion is CPU work; leave the GPU to torch
    calib = _calibration_steps(m, dataset)
    names = ["mel", *m.names]
    with tempfile.TemporaryDirectory() as tmp:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=tmp,
            keep_shape_absolutely_input_names=names,
            copy_onnx_input_output_names_to_tflite=True,
            output_signaturedefs=True,
            non_verbose=True,
        )
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)

        def rep() -> Any:
            for i in range(len(calib["mel"])):
                yield {k: calib[k][i : i + 1].astype(np.float32) for k in names}

        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        out.write_bytes(conv.convert())
    return out


def _quantize(x: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    """Float -> the tensor's type (integer I/O models take quantized inputs)."""
    dtype = detail["dtype"]
    if dtype == np.float32:
        f: np.ndarray = x.astype(np.float32)
        return f
    scale, zero = detail["quantization"]
    info = np.iinfo(dtype)
    q = np.round(x / scale) + zero
    out: np.ndarray = np.clip(q, info.min, info.max).astype(dtype)
    return out


def _dequantize(x: Any, detail: dict[str, Any]) -> np.ndarray:
    a: np.ndarray = np.asarray(x)
    if a.dtype == np.float32:
        return a
    scale, zero = detail["quantization"]
    out: np.ndarray = ((a.astype(np.float32) - zero) * scale).astype(np.float32)
    return out


class MicroRuntime:
    """Run `<name>_micro.onnx` (ONNX Runtime) or `<name>_micro.tflite` (LiteRT) on 16 kHz audio:
    host log-mel per 80 ms, states carried between steps, warm start on 2 s of silence. Also a
    livekit-wakeword-bench detector (`detect`)."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.name = f"micro:{path.name}"
        side = path.with_suffix(".json")
        self.mel_params: tuple[float, float, float] = MEL_DEFAULT
        if side.exists():
            p = json.loads(side.read_text())["mel_params"]
            self.mel_params = (float(p[0]), float(p[1]), float(p[2]))
        if path.suffix == ".tflite":
            from ai_edge_litert.interpreter import Interpreter

            self._runner = Interpreter(model_path=str(path)).get_signature_runner()
            ins = self._runner.get_input_details()
            self._in = ins
            self._out = self._runner.get_output_details()
            self._names = sorted((k for k in ins if k != "mel"), key=lambda k: int(k[1:]))
            sizes = [int(ins[n]["shape"][1]) for n in self._names]
            self._tflite = True
        else:
            import onnxruntime as ort

            self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._names = [i.name for i in self._sess.get_inputs()][1:]
            sizes = [int(i.shape[1]) for i in self._sess.get_inputs()[1:]]
            self._tflite = False
        zero = {n: np.zeros((1, k), np.float32) for n, k in zip(self._names, sizes)}
        self._state = zero
        for mel in host_mel(np.zeros(32000, np.float32), self.mel_params):
            self.step(mel)
        self._warm = {k: v.copy() for k, v in self._state.items()}

    def step(self, mel: np.ndarray) -> float:
        """One 80 ms step from (8, 40) log-mel frames; returns the score."""
        x = mel[None].astype(np.float32)
        if self._tflite:
            feed = {k: _quantize(v, self._in[k]) for k, v in {"mel": x, **self._state}.items()}
            o = self._runner(**feed)
            n = len(self._names)
            self._state = {
                f"s{k}": _dequantize(o[f"output_{k + 1}"], self._out[f"output_{k + 1}"])
                for k in range(n)
            }
            return float(_dequantize(o["output_0"], self._out["output_0"]).reshape(-1)[0])
        out = self._sess.run(None, {"mel": x, **self._state})
        self._state = dict(zip(self._names, out[1:]))
        return float(np.asarray(out[0]).reshape(-1)[0])

    def reset(self) -> None:
        self._state = {k: v.copy() for k, v in self._warm.items()}

    def scores(self, audio: np.ndarray) -> np.ndarray:
        """Scores for every 80 ms step of a clip (float in [-1, 1]), from a fresh (warm) state."""
        self.reset()
        return np.array(
            [self.step(m) for m in host_mel(audio.astype(np.float32), self.mel_params)], np.float32
        )

    def detect(self, audio: np.ndarray, sample_rate: int) -> Any:
        from livekit_wakeword_bench import Prediction

        assert sample_rate == 16000, "resample to 16 kHz first"
        s = self.scores(audio)
        return Prediction(((np.arange(len(s)) + 1) * 0.08).astype(np.float32), s)
