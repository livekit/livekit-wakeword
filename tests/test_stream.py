"""Tests for cached prediction over overlapping wake-word windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from livekit.wakeword import WakeWordModel, WakeWordStream


class FakeMelFrontend:
    def __init__(self, outputs: list[np.ndarray]):
        self._outputs = iter(outputs)
        self.call_count = 0

    def __call__(self, _audio: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return next(self._outputs)[np.newaxis, :, :]


class FakeSpeechEmbedding:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(windows.shape[0])
        values = windows[:, 0, 0].astype(np.float32)
        return np.repeat(values[:, np.newaxis], 96, axis=1)


@dataclass
class FakeInput:
    name: str = "features"


class FakeClassifier:
    def __init__(self) -> None:
        self.call_count = 0

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.call_count += 1
        score = float(feeds["features"].mean())
        return [np.array([[score]], dtype=np.float32)]


def mel_frames() -> np.ndarray:
    return np.arange(197 * 32, dtype=np.float32).reshape(197, 32)


def shifted_mel(previous: np.ndarray) -> np.ndarray:
    new_frames = previous[-8:] + 100_000
    return np.vstack((previous[8:], new_frames))


def fake_model(mels: list[np.ndarray]) -> tuple[WakeWordModel, FakeSpeechEmbedding, FakeClassifier]:
    model = WakeWordModel.__new__(WakeWordModel)
    model._mel_frontend = FakeMelFrontend(mels)
    embedding = FakeSpeechEmbedding()
    model._speech_embedding = embedding
    classifier = FakeClassifier()
    model._classifiers = {"test": (classifier, "features")}
    return model, embedding, classifier


def test_first_stream_prediction_embeds_all_windows_in_one_batch():
    model, embedding, classifier = fake_model([mel_frames()])

    stream = model.create_stream()
    scores = stream.predict(np.zeros(32_000, dtype=np.int16))

    assert isinstance(stream, WakeWordStream)
    assert embedding.batch_sizes == [16]
    assert classifier.call_count == 1
    assert set(scores) == {"test"}


def test_shifted_prediction_reuses_fifteen_embeddings():
    first = mel_frames()
    model, embedding, classifier = fake_model([first, shifted_mel(first)])
    stream = model.create_stream()

    stream.predict(np.zeros(32_000, dtype=np.int16))
    stream.predict(np.zeros(32_000, dtype=np.int16))

    assert model._mel_frontend.call_count == 2
    assert embedding.batch_sizes == [16, 1]
    assert classifier.call_count == 2


def test_changed_overlapping_window_is_recomputed():
    first = mel_frames()
    second = shifted_mel(first)
    second[0, 0] += 1
    model, embedding, _classifier = fake_model([first, second])
    stream = model.create_stream()

    stream.predict(np.zeros(32_000, dtype=np.int16))
    stream.predict(np.zeros(32_000, dtype=np.int16))

    # The modified first window and the newly arrived final window both change.
    assert embedding.batch_sizes == [16, 2]


def test_streams_from_one_model_do_not_share_cache_state():
    first = mel_frames()
    model, embedding, _classifier = fake_model([first, first])

    model.create_stream().predict(np.zeros(32_000, dtype=np.int16))
    model.create_stream().predict(np.zeros(32_000, dtype=np.int16))

    assert embedding.batch_sizes == [16, 16]


def test_reset_forces_the_next_prediction_to_rebuild_all_embeddings():
    first = mel_frames()
    second = shifted_mel(first)
    model, embedding, _classifier = fake_model([first, second, second])
    stream = model.create_stream()

    stream.predict(np.zeros(32_000, dtype=np.int16))
    stream.predict(np.zeros(32_000, dtype=np.int16))
    stream.reset()
    stream.predict(np.zeros(32_000, dtype=np.int16))

    assert embedding.batch_sizes == [16, 1, 16]


def test_stateless_prediction_keeps_individual_embedding_calls():
    model, embedding, classifier = fake_model([mel_frames()])

    model.predict(np.zeros(32_000, dtype=np.int16))

    assert embedding.batch_sizes == [1] * 16
    assert classifier.call_count == 1


def test_stream_matches_stateless_scores_with_real_feature_models():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    model = WakeWordModel(sess_options=options)
    classifier = FakeClassifier()
    model._classifiers = {"test": (classifier, "features")}
    stream = model.create_stream()

    rng = np.random.default_rng(2820)
    audio = np.zeros(32_000 + 4 * 1_280, dtype=np.int16)
    audio[30_000:34_000] = rng.integers(-20_000, 20_000, 4_000, dtype=np.int16)
    chunks = [audio[start : start + 32_000] for start in range(0, 4 * 1_280, 1_280)]

    for chunk in chunks:
        assert stream.predict(chunk) == model.predict(chunk)
