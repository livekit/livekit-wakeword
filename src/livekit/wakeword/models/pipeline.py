"""End-to-end wake word detection pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import WakeWordConfig
from ..resources import get_embedding_model_path, get_mel_model_path
from .classifier import build_classifier
from .feature_extractor import MelSpectrogramFrontend, SpeechEmbedding


class WakeWordClassifier(nn.Module):
    """Classifier-only module operating on pre-extracted embeddings.

    Used during training when features are pre-computed.
    Input: (batch, 16, 96) → Output: (batch, 1)
    """

    def __init__(self, config: WakeWordConfig):
        super().__init__()
        self.classifier = build_classifier(
            model_type=config.model.model_type,
            model_size=config.model.model_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class WakeWordPipeline:
    """Full pipeline: waveform → score.

    Chains MelSpectrogramFrontend (ONNX) → SpeechEmbedding (ONNX) → Classifier (PyTorch).
    Used for inference. Not an nn.Module since the first two stages are ONNX.
    """

    def __init__(self, config: WakeWordConfig):
        self.mel_frontend = MelSpectrogramFrontend(
            onnx_path=get_mel_model_path(),
        )
        self.speech_embedding = SpeechEmbedding(
            onnx_path=get_embedding_model_path(),
        )
        self.classifier = build_classifier(
            model_type=config.model.model_type,
            model_size=config.model.model_size,
        )
        self.classifier.eval()

    def load_classifier_weights(self, path: str | Path) -> None:
        """Load trained classifier weights."""
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        # Handle WakeWordClassifier wrapper (keys prefixed with 'classifier.')
        self.classifier.load_state_dict(
            {k.removeprefix("classifier."): v for k, v in state.items()}
        )

    @torch.no_grad()
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Run full pipeline from raw audio to wake word score.

        Args:
            audio: (batch, samples) or (samples,) float32 raw 16kHz audio

        Returns:
            (batch, 1) confidence scores [0, 1]
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Stage 1: mel spectrogram (ONNX) → (batch, time_frames, 32)
        mel = self.mel_frontend(audio)

        # Stage 2: speech embedding (ONNX) → (batch, n_windows, 96)
        embeddings = self.speech_embedding.extract_embeddings(mel)

        # Take last 16 timesteps for classifier, pad if needed
        n_windows = embeddings.shape[1]
        if n_windows > 16:
            embeddings = embeddings[:, -16:, :]
        elif n_windows < 16:
            pad = np.zeros(
                (embeddings.shape[0], 16 - n_windows, 96),
                dtype=np.float32,
            )
            embeddings = np.concatenate([pad, embeddings], axis=1)

        # Stage 3: classifier (PyTorch) → (batch, 1)
        emb_tensor = torch.from_numpy(embeddings)
        scores = self.classifier(emb_tensor)
        return scores.numpy()
