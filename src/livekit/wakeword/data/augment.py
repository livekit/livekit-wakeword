"""Audio augmentation pipeline."""

from __future__ import annotations

import logging
import os
import random
import sys
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from ..config import WakeWordConfig

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """Augmentation pipeline for wake word clips.

    Applies per-sample augmentations, RIR convolution,
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

    @staticmethod
    def _collect_wavs(dirs: list[Path]) -> list[Path]:
        wavs: list[Path] = []
        for d in dirs:
            if d.exists():
                wavs.extend(d.glob("**/*.wav"))
        return wavs

    def _get_per_sample_augmentations(self) -> Any:
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


_ALL_SPLITS = [
    "positive_train", "positive_test",
    "negative_train", "negative_test",
    "background_train", "background_test",
]


def run_augment(config: WakeWordConfig) -> None:
    """Run augmentation pipeline on generated clips."""
    import re

    target_duration = config.augmentation.clip_duration

    model_dir = config.model_output_dir

    # Clean up old augmented files before starting fresh augmentation.
    # This prevents stale _rN.wav files from previous runs piling up.
    _aug_re = re.compile(r"^clip_\d{6}_r\d+\.wav$")
    for split in _ALL_SPLITS:
        clip_dir = model_dir / split
        if not clip_dir.exists():
            continue
        old_augs = [p for p in clip_dir.glob("*.wav") if _aug_re.match(p.name)]
        if old_augs:
            logger.info(f"Cleaning {len(old_augs)} old augmented files from {split}")
            for p in old_augs:
                p.unlink()

    background_paths = [Path(p) for p in config.augmentation.background_paths]
    rir_paths = [Path(p) for p in config.augmentation.rir_paths]
    augmentor = AudioAugmentor(
        background_paths=background_paths,
        rir_paths=rir_paths,
    )

    n_workers = config.augmentation.n_workers
    mp_context = config.augmentation.mp_context

    for round_idx in range(config.augmentation.rounds):
        logger.info(f"Augmentation round {round_idx + 1}/{config.augmentation.rounds}")
        for split in _ALL_SPLITS:
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
                n_workers=n_workers,
                mp_context=mp_context,
                background_paths=background_paths,
                rir_paths=rir_paths,
            )


def _process_one(
    wav_path: Path,
    augmentor: AudioAugmentor,
    is_positive: bool,
    round_idx: int,
    target_length: int,
    sample_rate: int,
) -> None:
    """Augment a single WAV file in place and write its ``_rN.wav`` output.

    Identical to the body of the single-threaded loop — kept as a standalone
    function so both the serial and parallel paths share exactly one
    implementation of the per-clip pipeline.
    """
    import re

    import soundfile as sf

    audio, _sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    audio = augmentor.augment_clip(audio)
    audio = augmentor.apply_rir(audio)
    audio = augmentor.mix_with_background(audio)

    if round_idx == 0:
        if is_positive:
            audio = align_clip_to_end(audio, target_length)
        else:
            if len(audio) < target_length:
                padded = np.zeros(target_length, dtype=np.float32)
                start = (target_length - len(audio)) // 2
                padded[start : start + len(audio)] = audio
                audio = padded
            elif len(audio) > target_length:
                start = (len(audio) - target_length) // 2
                audio = audio[start : start + target_length]

    orig_stem = re.sub(r"_r\d+$", "", wav_path.stem)
    out_path = wav_path.with_name(f"{orig_stem}_r{round_idx}.wav")
    sf.write(str(out_path), audio, sample_rate)


# --- Multiprocessing support -------------------------------------------------
#
# Worker processes each build their own ``AudioAugmentor`` via ``_init_worker``.
# The parent's instance is never pickled: ``AudioAugmentor._per_sample_aug`` is
# lazily initialised to an ``audiomentations.Compose`` whose members include
# unpicklable SciPy state, so round-tripping it through ``Pool.map`` is
# fragile. Sending only the source paths and re-constructing is robust.

_WORKER_AUGMENTOR: AudioAugmentor | None = None


def _init_worker(
    background_paths: list[Path],
    rir_paths: list[Path],
    sample_rate: int,
    seed: int | None,
) -> None:
    global _WORKER_AUGMENTOR
    _WORKER_AUGMENTOR = AudioAugmentor(
        background_paths=background_paths,
        rir_paths=rir_paths,
        sample_rate=sample_rate,
    )
    # Give each worker a distinct random state so RIR/background picks and
    # audiomentations probabilities aren't identical across processes.
    worker_seed = (seed or 0) ^ (os.getpid() & 0xFFFFFFFF)
    random.seed(worker_seed)
    np.random.seed(worker_seed & 0xFFFFFFFF)


def _augment_one(args: tuple[Path, bool, int, int, int]) -> None:
    wav_path, is_positive, round_idx, target_length, sample_rate = args
    assert _WORKER_AUGMENTOR is not None, "worker not initialised"
    _process_one(
        wav_path=wav_path,
        augmentor=_WORKER_AUGMENTOR,
        is_positive=is_positive,
        round_idx=round_idx,
        target_length=target_length,
        sample_rate=sample_rate,
    )


def _pick_context(user_choice: str):
    if user_choice != "auto":
        return get_context(user_choice)
    return get_context("spawn" if sys.platform == "win32" else "fork")


def _parallel_augment_directory(
    wav_files: list[Path],
    is_positive: bool,
    round_idx: int,
    target_length: int,
    sample_rate: int,
    background_paths: list[Path],
    rir_paths: list[Path],
    n_workers: int,
    mp_context: str,
    desc: str,
) -> None:
    from tqdm import tqdm

    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(wav_files)))

    ctx = _pick_context(mp_context)
    chunksize = max(1, len(wav_files) // (n_workers * 16))

    tasks = [
        (p, is_positive, round_idx, target_length, sample_rate) for p in wav_files
    ]

    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(background_paths, rir_paths, sample_rate, round_idx),
    ) as pool:
        for _ in tqdm(
            pool.imap_unordered(_augment_one, tasks, chunksize=chunksize),
            total=len(tasks),
            desc=desc,
            unit="clip",
        ):
            pass


def _augment_directory(
    clip_dir: Path,
    augmentor: AudioAugmentor,
    is_positive: bool,
    target_duration_s: float = 2.0,
    sample_rate: int = 16000,
    round_idx: int = 0,
    n_workers: int = 1,
    mp_context: str = "auto",
    background_paths: list[Path] | None = None,
    rir_paths: list[Path] | None = None,
) -> None:
    """Augment all WAV files in a directory.

    Round 0 reads the original TTS clips (``clip_000000.wav``).
    Subsequent rounds read the previous round's output so that
    augmentation compounds (stacks) progressively.  Every round
    writes to its own file (``clip_000000_r0.wav``, ``_r1.wav``, …)
    so the originals are always preserved.

    When ``n_workers != 1`` the per-clip loop is parallelised across a
    ``multiprocessing.Pool``. Each worker builds its own ``AudioAugmentor``
    from ``background_paths`` / ``rir_paths`` so the parent's lazy-loaded
    audiomentations instance does not need to be pickled.
    """
    import re

    from tqdm import tqdm

    target_length = int(target_duration_s * sample_rate)

    if round_idx == 0:
        # Round 0: read original TTS clips
        _src_re = re.compile(r"^clip_\d{6}\.wav$")
    else:
        # Round N: read previous round's output
        _src_re = re.compile(rf"^clip_\d{{6}}_r{round_idx - 1}\.wav$")

    wav_files = sorted(p for p in clip_dir.glob("*.wav") if _src_re.match(p.name))

    if not wav_files:
        return

    desc = f"Augmenting {clip_dir.name} r{round_idx}"

    if n_workers != 1:
        _parallel_augment_directory(
            wav_files=wav_files,
            is_positive=is_positive,
            round_idx=round_idx,
            target_length=target_length,
            sample_rate=sample_rate,
            background_paths=background_paths or [],
            rir_paths=rir_paths or [],
            n_workers=n_workers,
            mp_context=mp_context,
            desc=desc,
        )
        return

    # Single-threaded path — unchanged.
    for wav_path in tqdm(wav_files, desc=desc, unit="clip"):
        _process_one(
            wav_path=wav_path,
            augmentor=augmentor,
            is_positive=is_positive,
            round_idx=round_idx,
            target_length=target_length,
            sample_rate=sample_rate,
        )
