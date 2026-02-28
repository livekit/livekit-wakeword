"""Audio augmentation pipeline."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

from ..config import WakeWordConfig
from .features import extract_features_for_config

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """Augmentation pipeline for wake word clips.

    Applies per-sample and batched augmentations, RIR convolution,
    and background noise mixing.
    """

    def __init__(
        self,
        background_paths: list[Path],
        rir_paths: list[Path],
        sample_rate: int = 16000,
    ):
        self.sample_rate = sample_rate
        self.background_files = self._collect_wavs(background_paths)
        self.rir_files = self._collect_wavs(rir_paths)
        self._per_sample_aug = None
        self._batch_aug = None

    @staticmethod
    def _collect_wavs(dirs: list[Path]) -> list[Path]:
        wavs: list[Path] = []
        for d in dirs:
            if d.exists():
                wavs.extend(d.glob("**/*.wav"))
        return wavs

    def _get_per_sample_augmentations(self):  # type: ignore[no-untyped-def]
        """Lazy-load audiomentations transforms."""
        if self._per_sample_aug is None:
            from audiomentations import Compose, SevenBandParametricEQ, TanhDistortion

            self._per_sample_aug = Compose(
                [
                    SevenBandParametricEQ(p=0.25),
                    TanhDistortion(p=0.25),
                ]
            )
        return self._per_sample_aug

    def _get_batch_augmentations(self):  # type: ignore[no-untyped-def]
        """Lazy-load torch-audiomentations transforms."""
        if self._batch_aug is None:
            from torch_audiomentations import (
                AddBackgroundNoise,
                AddColoredNoise,
                BandStopFilter,
                Compose,
                Gain,
                PitchShift,
            )

            transforms = [
                PitchShift(
                    min_transpose_semitones=-3.0,
                    max_transpose_semitones=3.0,
                    sample_rate=self.sample_rate,
                    p=0.25,
                ),
                BandStopFilter(p=0.25),
                AddColoredNoise(p=0.25),
                Gain(max_gain_in_db=0.0, p=1.0),
            ]
            if self.background_files:
                # Collect all unique parent directories containing background audio
                bg_dirs = list({str(p.parent) for p in self.background_files})
                transforms.insert(
                    3,
                    AddBackgroundNoise(
                        background_paths=bg_dirs,
                        min_snr_in_db=-10.0,
                        max_snr_in_db=15.0,
                        sample_rate=self.sample_rate,
                        p=0.75,
                    ),
                )
            self._batch_aug = Compose(transforms)
        return self._batch_aug

    def apply_rir(self, audio: np.ndarray, p: float = 0.5) -> np.ndarray:
        """Convolve audio with a random room impulse response."""
        if random.random() > p or not self.rir_files:
            return audio
        import soundfile as sf
        from scipy.signal import fftconvolve

        rir_path = random.choice(self.rir_files)
        rir, sr = sf.read(str(rir_path))
        if rir.ndim > 1:
            rir = rir[:, 0]
        # Normalize RIR
        rir = rir / (np.max(np.abs(rir)) + 1e-8)
        convolved = fftconvolve(audio, rir, mode="full")[: len(audio)]
        return convolved.astype(np.float32)

    def augment_clip(self, audio: np.ndarray) -> np.ndarray:
        """Apply per-sample augmentations to a single clip."""
        aug = self._get_per_sample_augmentations()
        return aug(samples=audio, sample_rate=self.sample_rate)

    def augment_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply batched augmentations.

        Args:
            batch: (batch, 1, samples) tensor

        Returns:
            (batch, 1, samples) augmented tensor
        """
        aug = self._get_batch_augmentations()
        return aug(batch, sample_rate=self.sample_rate).samples

    def mix_with_background(
        self,
        audio: np.ndarray,
        snr_db_range: tuple[float, float] = (5.0, 15.0),
    ) -> np.ndarray:
        """Mix audio with random background noise at given SNR."""
        if not self.background_files:
            return audio
        import soundfile as sf

        bg_path = random.choice(self.background_files)
        bg, sr = sf.read(str(bg_path))
        if bg.ndim > 1:
            bg = bg[:, 0]

        # Loop or crop background to match audio length
        if len(bg) < len(audio):
            repeats = (len(audio) // len(bg)) + 1
            bg = np.tile(bg, repeats)
        start = random.randint(0, max(0, len(bg) - len(audio)))
        bg = bg[start : start + len(audio)]

        # Compute SNR mixing
        snr_db = random.uniform(*snr_db_range)
        audio_power = np.mean(audio**2) + 1e-8
        bg_power = np.mean(bg**2) + 1e-8
        scale = np.sqrt(audio_power / (bg_power * 10 ** (snr_db / 10)))
        mixed = audio + scale * bg
        return mixed.astype(np.float32)


def align_clip_to_end(
    audio: np.ndarray,
    target_length: int,
    jitter_samples: int = 3200,  # 200ms at 16kHz
) -> np.ndarray:
    """Align a clip to the END of the target window with random jitter.

    Positive clips are placed at the end of the window with 0-200ms jitter.
    """
    result = np.zeros(target_length, dtype=np.float32)
    jitter = random.randint(0, jitter_samples)
    end_pos = target_length - jitter
    start_pos = max(0, end_pos - len(audio))
    clip_start = max(0, len(audio) - (end_pos - start_pos))
    result[start_pos:end_pos] = audio[clip_start : clip_start + (end_pos - start_pos)]
    return result


def run_augment(config: WakeWordConfig) -> None:
    """Run augmentation pipeline on generated clips.

    Augments clips and then extracts features through the frozen pipeline.
    """
    target_duration = config.augmentation.clip_duration

    model_dir = config.model_output_dir
    augmentor = AudioAugmentor(
        background_paths=[Path(p) for p in config.augmentation.background_paths],
        rir_paths=[Path(p) for p in config.augmentation.rir_paths],
    )

    for round_idx in range(config.augmentation.rounds):
        logger.info(f"Augmentation round {round_idx + 1}/{config.augmentation.rounds}")
        for split in ["positive_train", "positive_test", "negative_train", "negative_test"]:
            clip_dir = model_dir / split
            if not clip_dir.exists():
                logger.warning(f"Skipping {split}: directory not found")
                continue
            _augment_directory(
                clip_dir,
                augmentor,
                is_positive="positive" in split,
                round_idx=round_idx,
                target_duration_s=target_duration,
            )

    # Extract features through frozen pipeline
    logger.info("Extracting features through frozen embedding pipeline...")
    extract_features_for_config(config)


def _augment_directory(
    clip_dir: Path,
    augmentor: AudioAugmentor,
    is_positive: bool,
    target_duration_s: float = 2.0,
    sample_rate: int = 16000,
    round_idx: int = 0,
) -> None:
    """Augment all WAV files in a directory.

    Round 0 overwrites originals in-place. Subsequent rounds write to new
    files (e.g. ``clip_000000_r1.wav``) but still read from the round-0
    (already augmented) files, not the raw TTS output.
    """
    import soundfile as sf
    from tqdm import tqdm

    target_length = int(target_duration_s * sample_rate)
    # Only read original clips (no _r<N> suffix) to avoid compounding
    wav_files = sorted(p for p in clip_dir.glob("*.wav") if "_r" not in p.stem.split("_")[-1])

    for wav_path in tqdm(wav_files, desc=f"Augmenting {clip_dir.name}", unit="clip"):
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)

        # Apply per-sample augmentations
        audio = augmentor.augment_clip(audio)

        # Apply RIR
        audio = augmentor.apply_rir(audio)

        # Mix with background
        audio = augmentor.mix_with_background(audio)

        # Align positive clips to end of window
        if is_positive:
            audio = align_clip_to_end(audio, target_length)
        else:
            # Center-pad or crop negatives
            if len(audio) < target_length:
                padded = np.zeros(target_length, dtype=np.float32)
                start = (target_length - len(audio)) // 2
                padded[start : start + len(audio)] = audio
                audio = padded
            elif len(audio) > target_length:
                start = (len(audio) - target_length) // 2
                audio = audio[start : start + target_length]

        if round_idx == 0:
            out_path = wav_path
        else:
            out_path = wav_path.with_name(f"{wav_path.stem}_r{round_idx}.wav")
        sf.write(str(out_path), audio, sample_rate)
