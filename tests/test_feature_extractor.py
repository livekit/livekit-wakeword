"""Tests for MelSpectrogramFrontend and SpeechEmbedding."""

from __future__ import annotations

import numpy as np
import pytest

from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend
from livekit.wakeword.resources import get_mel_model_path


class TestMelSpectrogramFrontend:
    """Test the ONNX mel spectrogram frontend."""

    def test_output_shape(self):
        frontend = MelSpectrogramFrontend(onnx_path=get_mel_model_path())
        audio = np.random.randn(2, 32000).astype(np.float32)
        mel = frontend(audio)
        batch, time_frames, n_mels = mel.shape
        assert batch == 2
        assert n_mels == 32
        assert time_frames > 0

    def test_single_sample(self):
        frontend = MelSpectrogramFrontend(onnx_path=get_mel_model_path())
        audio = np.random.randn(1, 16000).astype(np.float32)  # 1 second
        mel = frontend(audio)
        assert mel.shape[0] == 1
        assert mel.shape[2] == 32
        assert mel.shape[1] > 90

    def test_1d_input(self):
        """1D input should be auto-batched."""
        frontend = MelSpectrogramFrontend(onnx_path=get_mel_model_path())
        audio = np.random.randn(16000).astype(np.float32)
        mel = frontend(audio)
        assert mel.ndim == 3
        assert mel.shape[0] == 1
        assert mel.shape[2] == 32

    def test_deterministic(self):
        frontend = MelSpectrogramFrontend(onnx_path=get_mel_model_path())
        audio = np.random.randn(1, 16000).astype(np.float32)
        mel1 = frontend(audio)
        mel2 = frontend(audio)
        np.testing.assert_allclose(mel1, mel2)

    def test_nonexistent_onnx_raises(self, tmp_path):
        """Non-existent ONNX path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MelSpectrogramFrontend(onnx_path=tmp_path / "nonexistent.onnx")
