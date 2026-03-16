"""Mel-spectrogram frontend and speech embedding via ONNX runtime.

The original openWakeWord pipeline uses two frozen ONNX models:
1. melspectrogram.onnx — torchlibrosa-based mel spectrogram (power_to_db)
2. embedding_model.onnx — Google's speech_embedding CNN (~330k params)

We use ONNX runtime for both to guarantee output compatibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

class MelSpectrogramFrontend:
    """Stage 1: Raw audio → mel-spectrogram features.

    Primary: ONNX runtime with the original melspectrogram.onnx model.
    Fallback: torchaudio with matched parameters (close but not exact).

    The ONNX model produces power_to_db mel features. Post-processing
    applies x/10 + 2 to match Google's expected input range.

    Output: (batch, time_frames, 32)
    """

    def __init__(self, onnx_path: str | Path):
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"Mel ONNX model not found: {onnx_path}\n"
                "This should not happen - please reinstall livekit-wakeword."
            )
        self._init_onnx(onnx_path)

    def _init_onnx(self, onnx_path: str | Path) -> None:
        import onnxruntime as ort

        self._onnx_session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._onnx_session.get_inputs()[0].name
        logger.info(f"Loaded mel ONNX model from {onnx_path}")

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram features.

        Args:
            audio: (batch, samples) float32 audio at 16kHz

        Returns:
            (batch, time_frames, 32) normalized mel features
        """
        return self._forward_onnx(audio)

    def _forward_onnx(self, audio: np.ndarray) -> np.ndarray:
        """Run ONNX mel model + post-processing normalization."""
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # ONNX model expects float32, outputs (batch, time_frames, 32) in dB scale
        results = []
        for i in range(audio.shape[0]):
            # Model input: (1, samples)
            inp = audio[i : i + 1].astype(np.float32)
            out = self._onnx_session.run(None, {self._input_name: inp})
            results.append(out[0])  # (1, time_frames, 32)

        mel = np.concatenate(results, axis=0)
        # ONNX model outputs (batch, 1, time_frames, 32) — squeeze channel dim
        if mel.ndim == 4:
            mel = mel[:, 0, :, :]  # → (batch, time_frames, 32)
        # Post-processing: x/10 + 2 (matches openWakeWord's melspec_transform)
        mel = mel / 10.0 + 2.0
        return mel

class SpeechEmbedding:
    """Stage 2: Google's speech_embedding CNN via ONNX runtime.

    The original model was reconstructed from TFHub's google/speech_embedding/1
    by inspecting the TFLite graph. It's a 5-block CNN (~330k params) that maps
    76-frame mel windows to 96-dim embedding vectors.

    ONNX input:  (batch, 76, 32, 1)  — channels-last mel spectrogram window
    ONNX output: (batch, 1, 1, 96)   — 96-dim embedding
    """

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"Embedding ONNX model not found: {onnx_path}\n"
                "This should not happen - please reinstall livekit-wakeword."
            )

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"Loaded embedding ONNX model from {onnx_path}")

    def __call__(self, mel_windows: np.ndarray) -> np.ndarray:
        """Compute embeddings for mel spectrogram windows.

        Args:
            mel_windows: (batch, 76, 32) mel spectrogram windows

        Returns:
            (batch, 96) embedding vectors
        """
        # ONNX model expects (batch, 76, 32, 1) channels-last
        if mel_windows.ndim == 3:
            mel_windows = mel_windows[..., np.newaxis]

        mel_windows = mel_windows.astype(np.float32)
        outputs = self._session.run(None, {self._input_name: mel_windows})
        # Output is (batch, 1, 1, 96) → squeeze to (batch, 96)
        return outputs[0].squeeze(axis=(1, 2))

    def extract_embeddings(
        self,
        mel_features: np.ndarray,
        window_size: int = 76,
        stride: int = 8,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Extract embeddings from a full mel-spectrogram using sliding window.

        Args:
            mel_features: (batch, time_frames, 32) from MelSpectrogramFrontend
            window_size: Number of mel frames per window (default 76)
            stride: Hop between windows (default 8)
            batch_size: Max windows to process at once through ONNX

        Returns:
            (batch, n_windows, 96) embedding sequence
        """
        if mel_features.ndim == 2:
            mel_features = mel_features[np.newaxis, :]

        batch, time_frames, n_mels = mel_features.shape
        n_windows = max(0, (time_frames - window_size) // stride + 1)

        if n_windows == 0:
            return np.zeros((batch, 0, 96), dtype=np.float32)

        all_embeddings = []
        for b in range(batch):
            # Build all windows for this sample
            windows = np.stack(
                [
                    mel_features[b, i * stride : i * stride + window_size, :]
                    for i in range(n_windows)
                ],
                axis=0,
            )  # (n_windows, 76, 32)

            # Process in batches through ONNX
            embs = []
            for ws in range(0, n_windows, batch_size):
                chunk = windows[ws : ws + batch_size]
                embs.append(self(chunk))  # (chunk_size, 96)
            all_embeddings.append(np.concatenate(embs, axis=0))  # (n_windows, 96)

        return np.stack(all_embeddings, axis=0)  # (batch, n_windows, 96)
