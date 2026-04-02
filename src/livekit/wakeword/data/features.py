"""Feature extraction: audio → mel-spectrogram → speech embeddings → .npy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import WakeWordConfig
from ..models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from ..resources import get_embedding_model_path, get_mel_model_path

logger = logging.getLogger(__name__)

# Target: 16 embedding timesteps per training example
N_EMBEDDING_TIMESTEPS = 16


def extract_features_from_directory(
    clip_dir: Path,
    mel_frontend: MelSpectrogramFrontend,
    speech_embedding: SpeechEmbedding,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract (N_clips, 16, 96) features from a directory of WAV files.

    Processes clips through MelSpectrogramFrontend → SpeechEmbedding,
    then takes last 16 embedding timesteps per clip.
    """
    import soundfile as sf
    from tqdm import tqdm

    wav_files = sorted(clip_dir.glob("*.wav"))
    if not wav_files:
        logger.warning(f"No WAV files in {clip_dir}")
        return np.zeros((0, N_EMBEDDING_TIMESTEPS, 96), dtype=np.float32)

    all_features: list[np.ndarray] = []

    for wav_path in tqdm(wav_files, desc=f"Features {clip_dir.name}", unit="clip"):
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)

        # Stage 1: mel spectrogram — (1, time_frames, 32)
        mel = mel_frontend(audio)

        # Stage 2: speech embeddings — (1, n_windows, 96)
        embeddings = speech_embedding.extract_embeddings(mel)
        clip_emb = embeddings[0]  # (n_windows, 96)

        # Take last N_EMBEDDING_TIMESTEPS or pad on the left
        if clip_emb.shape[0] >= N_EMBEDDING_TIMESTEPS:
            clip_emb = clip_emb[-N_EMBEDDING_TIMESTEPS:]
        else:
            pad = np.zeros(
                (N_EMBEDDING_TIMESTEPS - clip_emb.shape[0], 96),
                dtype=np.float32,
            )
            clip_emb = np.concatenate([pad, clip_emb], axis=0)

        all_features.append(clip_emb)

    if not all_features:
        return np.zeros((0, N_EMBEDDING_TIMESTEPS, 96), dtype=np.float32)

    return np.stack(all_features, axis=0)  # (N_clips, 16, 96)


def extract_features_from_long_audio(
    audio_paths: list[Path],
    mel_frontend: MelSpectrogramFrontend,
    speech_embedding: SpeechEmbedding,
    clip_duration: float = 2.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Extract (N_clips, 16, 96) features from long audio files by chunking.

    Slices each audio file into non-overlapping clips of ``clip_duration``
    seconds, then extracts features from each chunk.  Useful for turning
    background-noise recordings into standalone training negatives.
    """
    import soundfile as sf
    from tqdm import tqdm

    chunk_samples = int(clip_duration * sample_rate)
    all_features: list[np.ndarray] = []

    for audio_path in tqdm(audio_paths, desc="Features (background)", unit="file"):
        audio, sr = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)

        # Slice into non-overlapping chunks
        n_chunks = len(audio) // chunk_samples
        for i in range(n_chunks):
            chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]

            mel = mel_frontend(chunk)
            embeddings = speech_embedding.extract_embeddings(mel)
            clip_emb = embeddings[0]  # (n_windows, 96)

            if clip_emb.shape[0] >= N_EMBEDDING_TIMESTEPS:
                clip_emb = clip_emb[-N_EMBEDDING_TIMESTEPS:]
            else:
                pad = np.zeros(
                    (N_EMBEDDING_TIMESTEPS - clip_emb.shape[0], 96),
                    dtype=np.float32,
                )
                clip_emb = np.concatenate([pad, clip_emb], axis=0)

            all_features.append(clip_emb)

    if not all_features:
        return np.zeros((0, N_EMBEDDING_TIMESTEPS, 96), dtype=np.float32)

    return np.stack(all_features, axis=0)


def run_extraction(config: WakeWordConfig) -> None:
    """Extract and save features for all splits of a wake word config."""
    mel_frontend = MelSpectrogramFrontend(
        onnx_path=get_mel_model_path(),
    )
    speech_embedding = SpeechEmbedding(
        onnx_path=get_embedding_model_path(),
    )

    model_dir = config.model_output_dir
    splits = [
        ("positive_train", "positive_features_train.npy"),
        ("positive_test", "positive_features_test.npy"),
        ("negative_train", "negative_features_train.npy"),
        ("negative_test", "negative_features_test.npy"),
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
            batch_size=config.augmentation.batch_size,
        )

        out_path = model_dir / feature_filename
        np.save(str(out_path), features)
        logger.info(f"Saved {features.shape} features to {out_path}")

    # Extract features from background noise as standalone negatives
    bg_paths: list[Path] = []
    for bg_dir in config.augmentation.background_paths:
        d = Path(bg_dir)
        if d.exists():
            bg_paths.extend(d.glob("**/*.wav"))
    if bg_paths:
        logger.info(f"Extracting background noise features from {len(bg_paths)} files...")
        bg_features = extract_features_from_long_audio(
            audio_paths=bg_paths,
            mel_frontend=mel_frontend,
            speech_embedding=speech_embedding,
            clip_duration=config.augmentation.clip_duration,
        )
        out_path = model_dir / "background_noise_features.npy"
        np.save(str(out_path), bg_features)
        logger.info(f"Saved {bg_features.shape} background noise features to {out_path}")
    else:
        logger.info("No background noise files found, skipping background feature extraction")
