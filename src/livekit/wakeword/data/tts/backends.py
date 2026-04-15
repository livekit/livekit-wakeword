"""TTS backend registry and protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ...config import TtsBackend, WakeWordConfig
from .piper_backend import PiperVitsBackend
from .voxcpm_backend import VoxCpmBackend


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Contract for engines that write training clips under ``run_generate``.

    Implementations apply **voice/speaker diversification** appropriate to the
    engine (e.g. Piper SLERP, VoxCPM voice design). Orchestration only passes
    phrases, paths, counts, and batching.
    """

    def validate_artifacts(self) -> None:
        """Raise if required models or files are missing."""
        ...

    def synthesize_clips(
        self,
        phrases: list[str],
        output_dir: Path,
        n_samples: int,
        *,
        start_index: int = 0,
        batch_size: int = 50,
    ) -> list[Path]:
        """Write ``clip_%06d.wav`` at 16 kHz; honor *start_index* for resume."""
        ...


def get_tts_backend(config: WakeWordConfig) -> SpeechSynthesizer:
    """Construct the configured TTS backend."""
    if config.tts_backend is TtsBackend.piper_vits:
        return PiperVitsBackend.from_config(config)
    if config.tts_backend is TtsBackend.voxcpm:
        return VoxCpmBackend.from_config(config)
    raise ValueError(f"Unsupported tts_backend: {config.tts_backend!r}")
