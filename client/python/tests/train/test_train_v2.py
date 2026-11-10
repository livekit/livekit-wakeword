"""v2 training modules: shapes/causality, calibration folding, streaming export round trip."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

from livekit.wakeword import WakeWordModel  # noqa: E402
from livekit.wakeword_train.export import calibrate, export_streaming  # noqa: E402
from livekit.wakeword_train.nn import FRAME, Detector, Encoder, Head  # noqa: E402


def _tiny() -> Detector:
    torch.manual_seed(0)
    enc = Encoder(dim=16, d=32, layers=1, heads=2, ff=2, k=5, left=8, sub_c=8)
    return Detector(enc, Head(16, 16, 1, n_heads=2)).eval()


def test_encoder_shape_and_causality() -> None:
    det = _tiny()
    x = torch.randn(1, FRAME * 30) * 0.05
    with torch.no_grad():
        f = det.encoder(x)
        assert f.shape == (1, 30, 16)
        x2 = x.clone()
        x2[:, FRAME * 20 :] = torch.randn(1, FRAME * 10)
        f2 = det.encoder(x2)
    assert torch.allclose(f[:, :20], f2[:, :20], atol=1e-5)  # the future cannot change the past


def test_calibration_shifts_logits() -> None:
    det = _tiny()
    x = torch.randn(1, FRAME * 24) * 0.05
    with torch.no_grad():
        before = det(x)
        calibrate(det, logit_at_target=1.5, temperature=2.0)
        after = det(x)
    assert torch.allclose(after, (before - 1.5) / 2.0, atol=1e-5)


def test_export_runs_in_runtime(tmp_path) -> None:
    det = _tiny()
    path = export_streaming(det, tmp_path / "ww.onnx", metadata={"wake_word": "test"})
    import onnx

    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert meta["wake_word"] == "test"
    x = (np.random.default_rng(0).standard_normal(FRAME * 30) * 0.05).astype(np.float32)
    model = WakeWordModel(models=[path])
    streamed = [model.predict(x[i : i + FRAME])["ww"] for i in range(0, len(x), FRAME)]
    # the default state is "2 s of silence heard", so every step matches the full-sequence model
    # run on silence + x (not only the steps after the first 16 frames)
    lead = np.zeros(int(16000 * 2.0), np.float32)
    with torch.no_grad():
        ref = torch.sigmoid(det(torch.from_numpy(np.concatenate([lead, x]))[None]))[0].numpy()
    np.testing.assert_allclose(streamed, ref[-len(streamed) :], atol=1e-4)


def test_runtime_warm_up_reaches_the_warm_state(tmp_path) -> None:
    """State is plain inputs; zeros + 2 s of silence (what every runtime does) is `warm_state`."""
    import onnx

    from livekit.wakeword_train.export import STATE_NAMES, StreamingConformerDetector, warm_state

    det = _tiny()
    path = export_streaming(det, tmp_path / "ww.onnx")
    graph = onnx.load(str(path)).graph
    assert [i.name for i in graph.input] == ["chunk", *STATE_NAMES]
    assert not {i.name for i in graph.initializer} & set(STATE_NAMES)
    assert not [n for n in graph.node if n.op_type == "Pad"]  # tract cannot analyse them
    warm = warm_state(StreamingConformerDetector(det).eval())
    model = WakeWordModel(models=[path])
    state = model._models["ww"]._warm
    for name in STATE_NAMES:
        np.testing.assert_allclose(state[name], warm[name].numpy(), atol=1e-5)


def test_export_runs_in_tract(tmp_path) -> None:
    """tract (the Rust SDK's pure-Rust runtime) loads and streams the export like onnxruntime."""
    tract = pytest.importorskip("tract")
    det = _tiny()
    path = export_streaming(det, tmp_path / "ww.onnx")
    plan = tract.onnx().model_for_path(str(path)).into_optimized().into_runnable()
    model = WakeWordModel(models=[path])
    x = (np.random.default_rng(1).standard_normal(FRAME * 20) * 0.05).astype(np.float32)
    want = [model.predict(x[i * FRAME : (i + 1) * FRAME])["ww"] for i in range(20)]
    state = list(model._models["ww"]._warm.values())
    got = []
    for i in range(20):
        out = plan.run([x[None, i * FRAME : (i + 1) * FRAME], *state])
        # the Python bindings read every output as float32, so carry the int64 `step` ourselves
        arrays = [o.to_numpy() for o in out[:-1]]
        got.append(float(arrays[0].reshape(-1)[0]))
        state = [*arrays[1:], state[-1] + 1]
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_natural_positives_only_for_dictionary_words() -> None:
    from livekit.wakeword_train.pipeline import natural_positives_help

    assert natural_positives_help("hey jarvis")
    assert natural_positives_help("Computer")
    assert not natural_positives_help("hey livekit")
    assert natural_positives_help("hola livekit", language="es")


@pytest.mark.parametrize("with_sentences", [True, False])
def test_stream_without_confusable_clips_still_batches(
    monkeypatch: pytest.MonkeyPatch, with_sentences: bool
) -> None:
    """Languages with no confusable phrases generate no adversarial clips (#119); the stream
    must fill those rows from sentence clips or background speech instead of failing."""
    import numpy as np

    from livekit.wakeword_train import data as D
    from livekit.wakeword_train.loop import TrainConfig, W, _Stream

    class Clips:
        def __init__(self, n: int) -> None:
            self.ends = np.full(n, 8000)
            self._n = n

        def __len__(self) -> int:
            return self._n

        def clip(self, i: int) -> np.ndarray:
            return np.ones(16000, np.int16)

    class Pool:
        def crop(self, rng: object, n: int) -> np.ndarray:
            return np.full(n, 7, np.int16)

    sets = {
        "positive": Clips(2),
        "adversarial": Clips(0),
        "speech": Clips(2 if with_sentences else 0),
    }
    monkeypatch.setattr(D, "load_clipsets", lambda _: sets)
    monkeypatch.setattr(D, "negative_speech_files", lambda: ["x"])
    monkeypatch.setattr(D, "StreamPool", lambda *a, **k: Pool())
    cfg = TrainConfig(n_pos=1, n_adv=3, n_speech=1, n_noise=0)
    x, pend, _ = next(iter(_Stream("stub", cfg)))
    assert x.shape == (5, W)
    assert (pend[1:] == -1).all()
    assert (x[1:4] != 0).any(dim=1).all()
