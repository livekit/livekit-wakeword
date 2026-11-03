"""Micro tier: the streaming model equals the full-sequence model, and its ONNX export (mel +
per-cache states in, score + states out) runs in ONNX Runtime with the same scores."""

from pathlib import Path

import numpy as np
import pytest
import torch

from livekit.wakeword_train.micro import (
    MicroEncoder,
    StreamingMicro,
    export_micro,
    host_mel,
    micro_detector,
    warm_micro_state,
)


@pytest.fixture(scope="module")
def det() -> torch.nn.Module:
    torch.manual_seed(0)
    d = micro_detector(MicroEncoder(dim=16, mult=1.0))
    return d.eval()


def _audio(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(16000 * seconds)) * 0.1).astype(np.float32)


def test_micro_detector_has_no_layernorm(det: torch.nn.Module) -> None:
    assert isinstance(det.norm, torch.nn.Identity)


def test_streaming_equals_full_sequence(det: torch.nn.Module) -> None:
    m = StreamingMicro(det).eval()
    x = _audio(3.0)
    state = warm_micro_state(m)
    stream = []
    with torch.no_grad():
        for mel in host_mel(x):
            out = m(torch.from_numpy(mel)[None], *state)
            stream.append(float(out[0]))
            state = list(out[1:])
        lead = np.zeros(32000, np.float32)
        full = torch.sigmoid(det(torch.from_numpy(np.concatenate([lead, x]))[None]))
    ref = full[0].numpy()
    np.testing.assert_allclose(stream, ref[-len(stream) :], atol=1e-5)


def test_onnx_export_matches(det: torch.nn.Module, tmp_path: Path) -> None:
    import onnxruntime as ort

    files = export_micro(det, tmp_path, "ww", tflite=False)
    sess = ort.InferenceSession(str(files["onnx"]), providers=["CPUExecutionProvider"])
    m = StreamingMicro(det).eval()
    names = [i.name for i in sess.get_inputs()][1:]
    assert names == m.names
    # a runtime starts by zeroing the states and running 2 s of silence through the model
    state = {n: np.zeros((1, k), np.float32) for n, k in zip(m.names, m.sizes)}

    def step(mel: np.ndarray) -> float:
        nonlocal state
        out = sess.run(None, {"mel": mel[None], **state})
        state = dict(zip(m.names, out[1:]))
        return float(out[0][0])

    for mel in host_mel(np.zeros(32000, np.float32)):
        step(mel)
    for got, want in zip(state.values(), warm_micro_state(m)):
        np.testing.assert_allclose(got, want.numpy(), atol=1e-5)
    tstate = warm_micro_state(m)
    out, ref = [], []
    with torch.no_grad():
        for mel in host_mel(_audio(2.0, seed=1)):
            out.append(step(mel))
            r = m(torch.from_numpy(mel)[None], *tstate)
            ref.append(float(r[0]))
            tstate = list(r[1:])
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_micro_runtime_matches_streaming_model(det: torch.nn.Module, tmp_path: Path) -> None:
    from livekit.wakeword_train.micro import MicroRuntime

    files = export_micro(det, tmp_path, "ww", tflite=False)
    rt = MicroRuntime(files["onnx"])
    x = _audio(2.0, seed=2)
    m = StreamingMicro(det).eval()
    st = warm_micro_state(m)
    ref = []
    with torch.no_grad():
        for mel in host_mel(x):
            r = m(torch.from_numpy(mel)[None], *st)
            ref.append(float(r[0]))
            st = list(r[1:])
    np.testing.assert_allclose(rt.scores(x), ref, atol=1e-5)
    np.testing.assert_allclose(rt.scores(x), ref, atol=1e-5)  # reset() restores the warm state
