"""Parity tests for the parallel (multiprocessing) feature-extraction path.

Round-trips a small synthetic dataset of ``_rN.wav`` clips through both the
single-threaded and the ``multiprocessing.Pool`` code paths in
``extract_features_from_directory`` and asserts that the output tensors
match in shape, dtype, and per-clip order — the contract downstream
training actually depends on.

Feature values are also asserted to be numerically close: ONNX runtime is
deterministic for a fixed thread count, and the parallel path pins each
worker to 1 intra-/inter-op thread, so outputs should be ``np.allclose``
within a tiny tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from livekit.wakeword.data.features import (
    N_EMBEDDING_TIMESTEPS,
    extract_features_from_directory,
)
from livekit.wakeword.models.feature_extractor import (
    MelSpectrogramFrontend,
    SpeechEmbedding,
)
from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path


SAMPLE_RATE = 16000
N_CLIPS = 12


def _make_synthetic_r0_clips(clip_dir: Path, n: int, duration_s: float = 2.0) -> None:
    """Write ``n`` synthetic ``clip_NNNNNN_r0.wav`` files (the shape the extractor looks for)."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    length = int(duration_s * SAMPLE_RATE)
    rng = np.random.default_rng(7)
    for i in range(n):
        audio = rng.standard_normal(length).astype(np.float32) * 0.1
        sf.write(str(clip_dir / f"clip_{i:06d}_r0.wav"), audio, SAMPLE_RATE)


def test_parallel_matches_singlethreaded(tmp_path: Path) -> None:
    """Serial and parallel extraction produce equivalent (N, 16, 96) tensors."""
    clip_dir = tmp_path / "feats"
    _make_synthetic_r0_clips(clip_dir, N_CLIPS)

    mel_path = get_mel_model_path()
    emb_path = get_embedding_model_path()

    mel_frontend = MelSpectrogramFrontend(onnx_path=mel_path)
    speech_embedding = SpeechEmbedding(onnx_path=emb_path)

    serial = extract_features_from_directory(
        clip_dir=clip_dir,
        mel_frontend=mel_frontend,
        speech_embedding=speech_embedding,
        n_workers=1,
    )

    parallel = extract_features_from_directory(
        clip_dir=clip_dir,
        mel_frontend=mel_frontend,
        speech_embedding=speech_embedding,
        n_workers=3,
        mp_context="auto",
        mel_path=mel_path,
        embedding_path=emb_path,
        execution_providers=["CPUExecutionProvider"],
    )

    # Shape + dtype contract.
    assert serial.shape == (N_CLIPS, N_EMBEDDING_TIMESTEPS, 96)
    assert parallel.shape == serial.shape
    assert parallel.dtype == serial.dtype

    # Per-clip ordering + numerical equivalence. ORT with intra_op=inter_op=1
    # is deterministic, so values should match to within tight float tolerance.
    np.testing.assert_allclose(parallel, serial, rtol=1e-4, atol=1e-5)


def test_n_workers_auto(tmp_path: Path) -> None:
    """``n_workers=0`` (auto) runs to completion and returns the right shape."""
    clip_dir = tmp_path / "auto_feats"
    _make_synthetic_r0_clips(clip_dir, N_CLIPS)

    mel_path = get_mel_model_path()
    emb_path = get_embedding_model_path()

    mel_frontend = MelSpectrogramFrontend(onnx_path=mel_path)
    speech_embedding = SpeechEmbedding(onnx_path=emb_path)

    features = extract_features_from_directory(
        clip_dir=clip_dir,
        mel_frontend=mel_frontend,
        speech_embedding=speech_embedding,
        n_workers=0,
        mp_context="auto",
        mel_path=mel_path,
        embedding_path=emb_path,
        execution_providers=["CPUExecutionProvider"],
    )
    assert features.shape == (N_CLIPS, N_EMBEDDING_TIMESTEPS, 96)


def test_parallel_requires_model_paths(tmp_path: Path) -> None:
    """Parallel mode errors clearly when model paths aren't supplied."""
    import pytest

    clip_dir = tmp_path / "bad_feats"
    _make_synthetic_r0_clips(clip_dir, 3)

    mel_frontend = MelSpectrogramFrontend(onnx_path=get_mel_model_path())
    speech_embedding = SpeechEmbedding(onnx_path=get_embedding_model_path())

    with pytest.raises(ValueError, match="mel_path"):
        extract_features_from_directory(
            clip_dir=clip_dir,
            mel_frontend=mel_frontend,
            speech_embedding=speech_embedding,
            n_workers=2,
        )
