"""Feature extraction: audio → mel-spectrogram → speech embeddings → .npy."""

from __future__ import annotations

import logging
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ..config import WakeWordConfig
from ..models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from ..resources import get_embedding_model_path, get_mel_model_path

logger = logging.getLogger(__name__)

# Target: 16 embedding timesteps per training example
N_EMBEDDING_TIMESTEPS = 16


def _pad_or_truncate(embeddings: np.ndarray) -> np.ndarray:
    """Take last N_EMBEDDING_TIMESTEPS or left-pad a (n_windows, 96) embedding."""
    if embeddings.shape[0] >= N_EMBEDDING_TIMESTEPS:
        return embeddings[-N_EMBEDDING_TIMESTEPS:]
    pad = np.zeros(
        (N_EMBEDDING_TIMESTEPS - embeddings.shape[0], 96),
        dtype=np.float32,
    )
    return np.concatenate([pad, embeddings], axis=0)


def _extract_one(
    wav_path: Path,
    mel_frontend: MelSpectrogramFrontend,
    speech_embedding: SpeechEmbedding,
) -> np.ndarray:
    """Read one WAV and return its (16, 96) feature tensor."""
    import soundfile as sf

    audio, _sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)

    mel = mel_frontend(audio)
    embeddings = speech_embedding.extract_embeddings(mel)
    return _pad_or_truncate(embeddings[0])


# --- Multiprocessing support -------------------------------------------------
#
# ONNX Runtime inference sessions are not pickle-friendly, so each worker
# constructs its own mel + embedding sessions via the pool's ``initializer``.
#
# When running under a Pool we pin each session to a single intra-/inter-op
# thread — otherwise N workers × M ORT threads thread-explode and either
# crash or thrash. The single-threaded (n_workers=1) path keeps ORT's default
# thread pool so existing behavior is unchanged.

_WORKER_MEL: MelSpectrogramFrontend | None = None
_WORKER_EMB: SpeechEmbedding | None = None


def _init_feature_worker(
    mel_path: str,
    embedding_path: str,
    execution_providers: list[str],
) -> None:
    """Per-worker initialiser: build mel + embedding ONNX sessions once."""
    global _WORKER_MEL, _WORKER_EMB
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    # Single-threaded per worker — see module docstring above.
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1

    _WORKER_MEL = MelSpectrogramFrontend(
        onnx_path=mel_path,
        execution_providers=execution_providers,
        session_options=sess_opts,
    )
    _WORKER_EMB = SpeechEmbedding(
        onnx_path=embedding_path,
        execution_providers=execution_providers,
        session_options=sess_opts,
    )


def _feature_worker_task(wav_path: Path) -> np.ndarray:
    assert _WORKER_MEL is not None and _WORKER_EMB is not None, "worker not initialised"
    return _extract_one(wav_path, _WORKER_MEL, _WORKER_EMB)


def _pick_context(user_choice: str):
    if user_choice != "auto":
        return get_context(user_choice)
    return get_context("spawn" if sys.platform == "win32" else "fork")


def _parallel_extract_features_from_directory(
    wav_files: list[Path],
    mel_path: str,
    embedding_path: str,
    execution_providers: list[str],
    n_workers: int,
    mp_context: str,
    desc: str,
) -> list[np.ndarray]:
    from tqdm import tqdm

    if n_workers == 0:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(wav_files)))

    ctx = _pick_context(mp_context)
    chunksize = max(1, len(wav_files) // (n_workers * 16))

    all_features: list[np.ndarray] = []
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_feature_worker,
        initargs=(mel_path, embedding_path, execution_providers),
    ) as pool:
        # Ordered map (imap, not imap_unordered): downstream training expects
        # deterministic per-clip order within a split.
        for feat in tqdm(
            pool.imap(_feature_worker_task, wav_files, chunksize=chunksize),
            total=len(wav_files),
            desc=desc,
            unit="clip",
        ):
            all_features.append(feat)
    return all_features


def extract_features_from_directory(
    clip_dir: Path,
    mel_frontend: MelSpectrogramFrontend,
    speech_embedding: SpeechEmbedding,
    n_workers: int = 1,
    mp_context: str = "auto",
    mel_path: str | Path | None = None,
    embedding_path: str | Path | None = None,
    execution_providers: list[str] | None = None,
) -> np.ndarray:
    """Extract (N_clips, 16, 96) features from a directory of WAV files.

    Processes clips through MelSpectrogramFrontend → SpeechEmbedding,
    then takes last 16 embedding timesteps per clip.

    When ``n_workers != 1`` the per-clip loop runs under a
    ``multiprocessing.Pool``; each worker constructs its own mel + embedding
    ONNX session via the pool's initializer (ORT sessions aren't pickle-safe).
    ``mel_path`` / ``embedding_path`` / ``execution_providers`` are required in
    that case so workers can rebuild the models. The single-threaded path
    keeps the existing behavior — the ``mel_frontend`` / ``speech_embedding``
    arguments are used directly.
    """
    import re

    from tqdm import tqdm

    _aug_re = re.compile(r"^clip_\d{6}_r\d+\.wav$")
    wav_files = sorted(p for p in clip_dir.glob("*.wav") if _aug_re.match(p.name))
    if not wav_files:
        logger.warning(f"No WAV files in {clip_dir}")
        return np.zeros((0, N_EMBEDDING_TIMESTEPS, 96), dtype=np.float32)

    desc = f"Features {clip_dir.name}"

    if n_workers != 1:
        if mel_path is None or embedding_path is None:
            raise ValueError(
                "Parallel feature extraction requires mel_path and embedding_path "
                "(workers cannot pickle an ONNX InferenceSession and must re-open "
                "the model files)."
            )
        all_features = _parallel_extract_features_from_directory(
            wav_files=wav_files,
            mel_path=str(mel_path),
            embedding_path=str(embedding_path),
            execution_providers=execution_providers or ["CPUExecutionProvider"],
            n_workers=n_workers,
            mp_context=mp_context,
            desc=desc,
        )
    else:
        all_features = []
        for wav_path in tqdm(wav_files, desc=desc, unit="clip"):
            all_features.append(_extract_one(wav_path, mel_frontend, speech_embedding))

    if not all_features:
        return np.zeros((0, N_EMBEDDING_TIMESTEPS, 96), dtype=np.float32)

    return np.stack(all_features, axis=0)  # (N_clips, 16, 96)


def run_extraction(config: WakeWordConfig) -> None:
    """Extract and save features for all splits of a wake word config."""
    mel_path = get_mel_model_path()
    embedding_path = get_embedding_model_path()
    providers = config.feature_extraction.execution_providers

    mel_frontend = MelSpectrogramFrontend(
        onnx_path=mel_path,
        execution_providers=providers,
    )
    speech_embedding = SpeechEmbedding(
        onnx_path=embedding_path,
        execution_providers=providers,
    )

    model_dir = config.model_output_dir
    splits = [
        ("positive_train", "positive_features_train.npy"),
        ("positive_test", "positive_features_test.npy"),
        ("negative_train", "negative_features_train.npy"),
        ("negative_test", "negative_features_test.npy"),
        ("background_train", "background_noise_features_train.npy"),
        ("background_test", "background_noise_features_test.npy"),
    ]

    for clip_subdir, feature_filename in splits:
        clip_dir = model_dir / clip_subdir
        if not clip_dir.exists():
            logger.warning(f"Skipping feature extraction for {clip_subdir}: not found")
            continue

        logger.info(f"Extracting features from {clip_dir}...")
        features = extract_features_from_directory(
            clip_dir=clip_dir,
            mel_frontend=mel_frontend,
            speech_embedding=speech_embedding,
            n_workers=config.feature_extraction.n_workers,
            mp_context=config.feature_extraction.mp_context,
            mel_path=mel_path,
            embedding_path=embedding_path,
            execution_providers=providers,
        )

        out_path = model_dir / feature_filename
        np.save(str(out_path), features)
        logger.info(f"Saved {features.shape} features to {out_path}")
