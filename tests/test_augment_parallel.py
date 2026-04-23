"""Parity tests for the parallel (multiprocessing) augmentation path.

These round-trip a small synthetic dataset through both the single-threaded
and the ``multiprocessing.Pool`` code paths in ``_augment_directory`` and
assert that the output files match in count, shape, and duration.

Audio content is NOT expected to be bit-identical: each worker has its own
``random.seed`` / ``np.random.seed`` state, so RIR choice, background
selection, SNR draws, and per-sample augmentation probabilities will differ.
The contract is "same number of outputs, same shape/duration, same file
naming scheme" — which is what the downstream feature extractor actually
cares about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from livekit.wakeword.data.augment import AudioAugmentor, _augment_directory


SAMPLE_RATE = 16000
CLIP_DURATION_S = 2.0
N_CLIPS = 20


def _make_synthetic_clips(clip_dir: Path, n: int, duration_s: float = 1.0) -> None:
    """Write ``n`` tiny synthetic WAVs named clip_000000.wav .. clip_{n-1:06d}.wav."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    length = int(duration_s * SAMPLE_RATE)
    rng = np.random.default_rng(42)
    for i in range(n):
        audio = rng.standard_normal(length).astype(np.float32) * 0.1
        sf.write(str(clip_dir / f"clip_{i:06d}.wav"), audio, SAMPLE_RATE)


def _augmentor(tmp_path: Path) -> AudioAugmentor:
    """Augmentor with no backgrounds or RIRs — keeps the test hermetic.

    With empty background/RIR pools the DSP becomes deterministic up to the
    audiomentations per-sample transforms. The worker random seeds still
    diverge, which is what the parity assertion below accounts for.
    """
    empty = tmp_path / "empty_backgrounds"
    empty.mkdir(exist_ok=True)
    return AudioAugmentor(background_paths=[empty], rir_paths=[empty])


def _list_round_outputs(clip_dir: Path, round_idx: int) -> list[Path]:
    return sorted(clip_dir.glob(f"clip_*_r{round_idx}.wav"))


def test_parallel_matches_singlethreaded(tmp_path: Path) -> None:
    """Single-threaded and 4-worker parallel paths produce the same output set."""
    target_length = int(CLIP_DURATION_S * SAMPLE_RATE)
    empty = tmp_path / "empty_backgrounds"

    clip_dir_serial = tmp_path / "serial"
    clip_dir_parallel = tmp_path / "parallel"
    _make_synthetic_clips(clip_dir_serial, N_CLIPS)
    _make_synthetic_clips(clip_dir_parallel, N_CLIPS)

    # --- single-threaded path ---
    _augment_directory(
        clip_dir=clip_dir_serial,
        augmentor=_augmentor(tmp_path),
        is_positive=True,
        target_duration_s=CLIP_DURATION_S,
        sample_rate=SAMPLE_RATE,
        round_idx=0,
        n_workers=1,
    )
    serial_outputs = _list_round_outputs(clip_dir_serial, 0)

    # --- parallel path ---
    _augment_directory(
        clip_dir=clip_dir_parallel,
        augmentor=_augmentor(tmp_path),
        is_positive=True,
        target_duration_s=CLIP_DURATION_S,
        sample_rate=SAMPLE_RATE,
        round_idx=0,
        n_workers=4,
        mp_context="auto",
        background_paths=[empty],
        rir_paths=[empty],
    )
    parallel_outputs = _list_round_outputs(clip_dir_parallel, 0)

    # File count and naming must match exactly.
    assert len(serial_outputs) == N_CLIPS
    assert len(parallel_outputs) == N_CLIPS
    assert [p.name for p in serial_outputs] == [p.name for p in parallel_outputs]

    # Every output must have the aligned target length (round 0, positive).
    for p in serial_outputs + parallel_outputs:
        audio, sr = sf.read(str(p))
        assert sr == SAMPLE_RATE
        assert audio.shape == (target_length,), (
            f"{p} has shape {audio.shape}, expected ({target_length},)"
        )
        assert audio.dtype in (np.float32, np.float64)


def test_parallel_negative_shape(tmp_path: Path) -> None:
    """Negative clips (center-padded) also come out at target length from both paths."""
    target_length = int(CLIP_DURATION_S * SAMPLE_RATE)
    empty = tmp_path / "empty_backgrounds"

    clip_dir = tmp_path / "neg_parallel"
    _make_synthetic_clips(clip_dir, N_CLIPS, duration_s=0.5)

    _augment_directory(
        clip_dir=clip_dir,
        augmentor=_augmentor(tmp_path),
        is_positive=False,
        target_duration_s=CLIP_DURATION_S,
        sample_rate=SAMPLE_RATE,
        round_idx=0,
        n_workers=3,
        mp_context="auto",
        background_paths=[empty],
        rir_paths=[empty],
    )
    outputs = _list_round_outputs(clip_dir, 0)
    assert len(outputs) == N_CLIPS
    for p in outputs:
        audio, _ = sf.read(str(p))
        assert audio.shape == (target_length,)


def test_n_workers_auto(tmp_path: Path) -> None:
    """``n_workers=0`` (auto) uses all available cores without crashing."""
    empty = tmp_path / "empty_backgrounds"
    clip_dir = tmp_path / "auto"
    _make_synthetic_clips(clip_dir, N_CLIPS)

    _augment_directory(
        clip_dir=clip_dir,
        augmentor=_augmentor(tmp_path),
        is_positive=True,
        target_duration_s=CLIP_DURATION_S,
        sample_rate=SAMPLE_RATE,
        round_idx=0,
        n_workers=0,
        mp_context="auto",
        background_paths=[empty],
        rir_paths=[empty],
    )
    assert len(_list_round_outputs(clip_dir, 0)) == N_CLIPS


def test_round_1_reads_r0(tmp_path: Path) -> None:
    """Round 1 in the parallel path correctly reads the _r0 outputs of round 0."""
    empty = tmp_path / "empty_backgrounds"
    clip_dir = tmp_path / "multi_round"
    _make_synthetic_clips(clip_dir, N_CLIPS)

    for round_idx in (0, 1):
        _augment_directory(
            clip_dir=clip_dir,
            augmentor=_augmentor(tmp_path),
            is_positive=True,
            target_duration_s=CLIP_DURATION_S,
            sample_rate=SAMPLE_RATE,
            round_idx=round_idx,
            n_workers=2,
            mp_context="auto",
            background_paths=[empty],
            rir_paths=[empty],
        )

    assert len(_list_round_outputs(clip_dir, 0)) == N_CLIPS
    assert len(_list_round_outputs(clip_dir, 1)) == N_CLIPS
    # Originals preserved.
    assert len(sorted(clip_dir.glob("clip_[0-9]*.wav"))) >= N_CLIPS
