"""Synthetic data generation: pluggable TTS + adversarial negatives (default: Piper VITS)."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

from ..config import CustomPositiveSource, WakeWordConfig
from .piper.text import expand_unknown_words, get_cmudict
from .tts import get_tts_backend
from .tts.piper_backend import PiperVitsBackend

logger = logging.getLogger(__name__)

# Matches original clips (clip_000000.wav) but NOT augmented variants (clip_000000_r1.wav)
_ORIGINAL_CLIP_RE = re.compile(r"^clip_\d{6}\.wav$")

# ARPAbet vowel phonemes (used to add optional stress markers in regex patterns)
_VOWEL_PHONES = frozenset(
    [
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AX",
        "AXR",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IX",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
        "UX",
    ]
)


def _count_original_clips(directory: Path) -> int:
    """Count ``clip_######.wav`` files in *directory*, excluding augmented variants."""
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if _ORIGINAL_CLIP_RE.match(f.name))


def _copy_custom_positives(
    split_dir: Path,
    sources: list[CustomPositiveSource],
    start_index: int,
    expected_sample_rate: int = 16000,
) -> int:
    """Append user-supplied ``.wav`` files to ``split_dir`` as ``clip_NNNNNN.wav``.

    Each file in each source is copied ``source.multiplier`` times.  Output
    indexing starts at *start_index* and increases monotonically across sources.
    Existing output files are skipped so repeated calls are idempotent
    (resume-safe after interruption).

    Input files must match *expected_sample_rate* and be mono — mismatches raise
    rather than silently resampling, so configuration errors surface early.

    Returns:
        Number of new files written on this call.

    Raises:
        FileNotFoundError: If a source ``path`` does not exist.
        NotADirectoryError: If a source ``path`` exists but is not a directory.
        ValueError: If any ``.wav`` has the wrong sample rate or is not mono.
    """
    import shutil

    import soundfile as sf

    if not sources:
        return 0

    split_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    next_index = start_index

    for source in sources:
        src_path = Path(source.path)
        if not src_path.exists():
            raise FileNotFoundError(f"Custom positive source path does not exist: {src_path}")
        if not src_path.is_dir():
            raise NotADirectoryError(
                f"Custom positive source path must be a directory, got: {src_path}"
            )

        wav_files = sorted(src_path.glob("*.wav"))
        non_wav = [p.name for p in src_path.iterdir() if p.is_file() and p.suffix.lower() != ".wav"]
        if non_wav:
            sample = ", ".join(non_wav[:3]) + (" ..." if len(non_wav) > 3 else "")
            logger.warning(
                "Ignoring %d non-.wav file(s) in %s: %s",
                len(non_wav),
                src_path,
                sample,
            )

        if not wav_files:
            logger.warning("No .wav files found in custom positive source: %s", src_path)
            continue

        # Validate sample rate and channels up front so we don't partially
        # copy before discovering a bad file deep in the list.
        for wav in wav_files:
            info = sf.info(str(wav))
            if info.samplerate != expected_sample_rate:
                raise ValueError(
                    f"Custom positive {wav} has sample rate {info.samplerate}, "
                    f"expected {expected_sample_rate}. Pre-convert the file "
                    f"(e.g. `sox in.wav -r 16000 -c 1 out.wav`)."
                )
            if info.channels != 1:
                raise ValueError(
                    f"Custom positive {wav} has {info.channels} channels, "
                    f"expected 1 (mono). Pre-convert the file "
                    f"(e.g. `sox in.wav -r 16000 -c 1 out.wav`)."
                )

        logger.info(
            "Injecting %d file(s) from %s × multiplier %d = %d copies",
            len(wav_files),
            src_path,
            source.multiplier,
            len(wav_files) * source.multiplier,
        )

        for wav in wav_files:
            for _ in range(source.multiplier):
                out_path = split_dir / f"clip_{next_index:06d}.wav"
                next_index += 1
                if out_path.exists():
                    continue
                shutil.copy2(wav, out_path)
                written += 1

    if written:
        logger.info(
            "Custom positive injection: wrote %d new clip(s) (directory now contains clips 0..%d)",
            written,
            next_index - 1,
        )
    else:
        logger.info("Custom positive injection: all clips already present, nothing to do")
    return written


def _phoneme_replacements(
    phones: list[str],
    max_replace: int | None = None,
) -> list[str]:
    """Generate regex patterns by replacing 1..max_replace phonemes with a wildcard.

    Each replaced position becomes ``(.){1,3}`` which matches any 1-3
    phoneme characters in the CMU pronunciation string.  This is the same
    approach used by openWakeWord — broad enough to catch phonetically
    similar words that wouldn't be found via a hand-curated substitution map.
    """
    from itertools import combinations

    if max_replace is None:
        max_replace = max(0, len(phones) - 2)
    max_replace = min(max_replace, len(phones))

    wildcard = "(.){1,3}"
    results: list[str] = []
    for r in range(1, max_replace + 1):
        for positions in combinations(range(len(phones)), r):
            parts = phones.copy()
            for pos in positions:
                parts[pos] = wildcard
            results.append(" ".join(parts))
    return results


def _get_word_phonemes(word: str) -> list[str]:
    """Get stress-stripped phonemes for a word, with all stress variants on vowels.

    Returns phoneme strings with regex stress wildcards (e.g. ``"AA[0|1|2]"``)
    so that ``pronouncing.search()`` matches any stress variant.
    """
    import pronouncing

    raw = pronouncing.phones_for_word(word)
    if not raw:
        return []
    # Strip existing stress, then add optional stress for vowels
    phones = [re.sub(r"\d+", "", p) for p in raw[0].split()]
    return [p + "[0|1|2]" if p in _VOWEL_PHONES else p for p in phones]


def generate_adversarial_phrases(
    target_phrases: list[str],
    n_phrases: int | None = None,
    include_partial_phrase: float = 1.0,
    include_input_words: float = 0.2,
    max_replace: int | None = None,
) -> list[str]:
    """Generate phonetically similar phrases to the target using CMUDict regex search.

    For each word in the target phrase, generates regex patterns by replacing
    1 to ``max_replace`` phonemes with a broad wildcard, then searches CMUDict
    for matching words.  This catches both close and moderately-distant phonetic
    neighbors without requiring a hand-curated substitution map.

    Unknown words (not in CMUDict) are split into known subwords when possible,
    e.g. "livekit" → "live" + "kit", allowing substitutions on both parts.

    When *n_phrases* is ``None`` (the default) all unique adversarial phrases
    are returned — no cap is applied.

    Args:
        target_phrases: Target wake word phrases.
        n_phrases: Maximum number of adversarial phrases to return
            (``None`` = no cap).
        include_partial_phrase: Probability of generating partial phrases
            (each word removed in turn).
        include_input_words: Probability of including individual original
            words as adversarial entries.
        max_replace: Maximum number of phonemes to replace per word
            (``None`` = ``len(phones) - 2``).
    """
    import pronouncing

    cmu = get_cmudict()
    adversarial: list[str] = []

    for phrase in target_phrases:
        raw_words = phrase.lower().split()
        words = expand_unknown_words(raw_words, cmu)

        # Get phonemes (with stress wildcards) for each word
        word_phonemes = [_get_word_phonemes(w) for w in words]

        # Generate substitutions for each word position
        for word_idx, (word, phones) in enumerate(zip(words, word_phonemes)):
            if not phones:
                continue

            # Build regex patterns by replacing phonemes with wildcards
            patterns = _phoneme_replacements(phones, max_replace)

            # For short words (≤2 phonemes), also include the base pattern
            if len(phones) <= 2:
                patterns.append(" ".join(phones))

            adversarial_words: list[str] = []
            for pattern in patterns:
                try:
                    matches = pronouncing.search(pattern)
                except re.error:
                    continue
                # Exclude homophones (same pronunciation = not adversarial)
                for match in matches:
                    if match.lower() != word:
                        match_phones = pronouncing.phones_for_word(match)
                        if (
                            match_phones
                            and match_phones[0] != (pronouncing.phones_for_word(word) or [""])[0]
                        ):
                            adversarial_words.append(match)

            # Build adversarial phrases by replacing the current word
            for replacement in adversarial_words:
                new_words = words.copy()
                new_words[word_idx] = replacement
                adversarial.append(" ".join(new_words))

        # Partial phrase adversarials
        if (
            include_partial_phrase > 0
            and len(words) > 1
            and random.random() < include_partial_phrase
        ):
            for i in range(len(words)):
                partial = " ".join(words[:i] + words[i + 1 :])
                if partial:
                    adversarial.append(partial)

        # Include original words individually
        if include_input_words > 0:
            for word in words:
                if random.random() < include_input_words:
                    adversarial.append(word)

    # Deduplicate and remove the original target phrases
    target_set = {p.lower() for p in target_phrases}
    adversarial = [p for p in set(adversarial) if p not in target_set]
    random.shuffle(adversarial)
    if n_phrases is not None:
        adversarial = adversarial[:n_phrases]
    return adversarial


def synthesize_clips(
    phrases: list[str],
    output_dir: Path,
    n_samples: int,
    vits_model_path: Path | None = None,
    noise_scales: list[float] | None = None,
    noise_scale_ws: list[float] | None = None,
    length_scales: list[float] | None = None,
    slerp_weights: list[float] | None = None,
    max_speakers: int | None = None,
    batch_size: int = 50,
    start_index: int = 0,
) -> list[Path]:
    """Synthesize speech clips using Piper VITS + SLERP (library / test helper).

    Returns list of paths to generated .wav files.
    """
    if vits_model_path is None or not vits_model_path.exists():
        raise FileNotFoundError(
            f"VITS model not found at {vits_model_path}. "
            "Cannot generate audio — refusing to produce silent placeholders. "
            "Download the model first: livekit-wakeword setup --config <your.yaml>"
        )

    backend = PiperVitsBackend(
        model_path=vits_model_path,
        noise_scales=noise_scales if noise_scales is not None else [0.98],
        noise_scale_ws=noise_scale_ws if noise_scale_ws is not None else [0.98],
        length_scales=length_scales if length_scales is not None else [0.75, 1.0, 1.25],
        slerp_weights=slerp_weights if slerp_weights is not None else [0.2, 0.35, 0.5, 0.65, 0.8],
        max_speakers=max_speakers,
    )
    return backend.synthesize_clips(
        phrases=phrases,
        output_dir=output_dir,
        n_samples=n_samples,
        start_index=start_index,
        batch_size=batch_size,
    )


def _generate_background_clips(
    config: WakeWordConfig,
    split_name: str,
    n_samples: int,
) -> None:
    """Generate background noise clips by randomly sampling from background audio.

    Short source files are tiled with random offsets and occasional reversal
    to avoid audible periodicity.  Always produces exactly *n_samples* clips
    regardless of how much source audio is available.
    """
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm

    bg_paths: list[Path] = []
    for bg_dir in config.augmentation.background_paths:
        d = Path(bg_dir)
        if d.exists():
            bg_paths.extend(d.glob("**/*.wav"))

    if not bg_paths:
        logger.info("No background noise files found, skipping %s", split_name)
        return

    sample_rate = 16000
    chunk_samples = int(config.augmentation.clip_duration * sample_rate)

    out_dir = config.model_output_dir / split_name
    existing = _count_original_clips(out_dir)
    if existing >= n_samples:
        logger.info(
            "Split %s already complete (%d/%d clips), skipping",
            split_name,
            existing,
            n_samples,
        )
        return
    if existing > 0:
        logger.info("Resuming split %s from clip %d / %d", split_name, existing, n_samples)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load all background audio
    all_audio: list[np.ndarray] = []
    for bp in bg_paths:
        audio, sr = sf.read(str(bp))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        if sr != sample_rate:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        all_audio.append(audio)

    logger.info(
        "Generating %d %s clips from %d source files",
        n_samples - existing,
        split_name,
        len(all_audio),
    )

    for i in tqdm(range(existing, n_samples), desc=f"Background ({split_name})", unit="clip"):
        # Pick a random source file
        audio = random.choice(all_audio)

        # Tile short files with varied segments to fill clip duration
        if len(audio) < chunk_samples:
            segments: list[np.ndarray] = []
            n = len(audio)
            while sum(len(s) for s in segments) < chunk_samples:
                start = random.randint(0, n - 1)
                seg = np.roll(audio, -start)
                if random.random() < 0.5:
                    seg = seg[::-1]
                segments.append(seg)
            audio = np.concatenate(segments)

        # Random offset into the (possibly tiled) audio
        max_start = len(audio) - chunk_samples
        start = random.randint(0, max(0, max_start))
        clip = audio[start : start + chunk_samples]

        out_path = out_dir / f"clip_{i:06d}.wav"
        sf.write(str(out_path), clip, sample_rate)

    logger.info("Wrote %d background clips to %s", n_samples, out_dir)


def run_generate(config: WakeWordConfig) -> None:
    """Run the full generate pipeline for a wake word config.

    Supports resuming: counts existing ``clip_######.wav`` files in each split
    directory and skips completed splits or resumes partial ones from the
    existing count.
    """
    model_dir = config.model_output_dir
    tts = get_tts_backend(config)
    tts.validate_artifacts()

    # --- Positive splits ---
    splits: list[tuple[str, list[str], int]] = [
        ("positive_train", config.target_phrases, config.n_samples),
        ("positive_test", config.target_phrases, config.n_samples_val),
    ]

    for split_name, phrases, n_target in splits:
        split_dir = model_dir / split_name
        existing = _count_original_clips(split_dir)
        if existing >= n_target:
            logger.info(
                "Split %s already complete (%d/%d clips), skipping",
                split_name,
                existing,
                n_target,
            )
            continue
        if existing > 0:
            logger.info(
                "Resuming split %s from clip %d / %d",
                split_name,
                existing,
                n_target,
            )
        else:
            logger.info("Generating %d %s clips...", n_target, split_name)
        tts.synthesize_clips(
            phrases=phrases,
            output_dir=split_dir,
            n_samples=n_target,
            start_index=existing,
            batch_size=config.tts_batch_size,
        )

    # --- Custom positive samples (user-supplied recordings) ---
    # Appended after TTS so the combined directory keeps the clip_NNNNNN.wav
    # layout.  start_index is pinned to config.n_samples (not the live count)
    # so layout is deterministic even if TTS skipped clips due to OOM — gaps
    # are harmless to the augmentation regex.
    if config.custom_positive_samples:
        _copy_custom_positives(
            split_dir=model_dir / "positive_train",
            sources=config.custom_positive_samples,
            start_index=config.n_samples,
        )

    # --- Adversarial negative splits ---
    neg_train_dir = model_dir / "negative_train"
    neg_test_dir = model_dir / "negative_test"
    neg_train_existing = _count_original_clips(neg_train_dir)
    neg_test_existing = _count_original_clips(neg_test_dir)

    # Skip adversarial phrase generation entirely if both negative splits are complete
    if neg_train_existing >= config.n_samples and neg_test_existing >= config.n_samples_val:
        logger.info("Both negative splits already complete, skipping adversarial generation")
    else:
        logger.info("Generating adversarial negative phrases...")
        adv_phrases = generate_adversarial_phrases(
            target_phrases=config.target_phrases,
        )
        if config.custom_negative_phrases:
            adv_phrases.extend(config.custom_negative_phrases)

        if not adv_phrases:
            logger.warning(
                "No adversarial phrases generated; using common English filler phrases as fallback"
            )
            adv_phrases = ["hello", "okay", "hey", "stop", "go", "yes", "no"]

        neg_splits: list[tuple[str, Path, int, int]] = [
            ("negative_train", neg_train_dir, config.n_samples, neg_train_existing),
            ("negative_test", neg_test_dir, config.n_samples_val, neg_test_existing),
        ]

        for split_name, split_dir, n_target, existing in neg_splits:
            if existing >= n_target:
                logger.info(
                    "Split %s already complete (%d/%d clips), skipping",
                    split_name,
                    existing,
                    n_target,
                )
                continue
            if existing > 0:
                logger.info(
                    "Resuming split %s from clip %d / %d",
                    split_name,
                    existing,
                    n_target,
                )
            else:
                logger.info("Synthesizing %d %s clips...", n_target, split_name)
            tts.synthesize_clips(
                phrases=adv_phrases,
                output_dir=split_dir,
                n_samples=n_target,
                start_index=existing,
                batch_size=config.tts_batch_size,
            )

    # --- Background noise splits ---
    if config.n_background_samples > 0:
        _generate_background_clips(config, "background_train", config.n_background_samples)
    if config.n_background_samples_val > 0:
        _generate_background_clips(config, "background_test", config.n_background_samples_val)
