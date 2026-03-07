"""Synthetic data generation: VITS TTS with SLERP speaker blending + adversarial negatives."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any

from ..config import WakeWordConfig
from ._piper_generate import generate_samples

logger = logging.getLogger(__name__)

# Matches original clips (clip_000000.wav) but NOT augmented variants (clip_000000_r1.wav)
_ORIGINAL_CLIP_RE = re.compile(r"^clip_\d{6}\.wav$")

# ARPAbet vowel phonemes (used to add optional stress markers in regex patterns)
_VOWEL_PHONES = frozenset([
    "AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY",
    "EH", "ER", "EY", "IH", "IX", "IY",
    "OW", "OY", "UH", "UW", "UX",
])


def _get_cmudict() -> dict[str, list[str]]:
    """Load CMU Pronouncing Dictionary via nltk."""
    import nltk

    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict

    # cmudict.dict() returns {word: [pron1, pron2, ...]} where each pron is list[str]
    # Take the first pronunciation for each word
    return {word: prons[0] for word, prons in cmudict.dict().items()}


def _count_original_clips(directory: Path) -> int:
    """Count ``clip_######.wav`` files in *directory*, excluding augmented variants."""
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if _ORIGINAL_CLIP_RE.match(f.name))


def _expand_unknown_words(
    words: list[str],
    cmu: dict[str, list[str]],
) -> list[str]:
    """Expand words not in CMUDict by splitting into known subwords.

    For example, "livekit" → ["live", "kit"] since both are in CMUDict.
    Known words are kept as-is.
    """
    expanded: list[str] = []
    for word in words:
        if word in cmu:
            expanded.append(word)
            continue
        # Try all split points, prefer longest left match
        best_split: tuple[str, str] | None = None
        for i in range(2, len(word) - 1):
            left, right = word[:i], word[i:]
            if left in cmu and right in cmu:
                if best_split is None or len(left) > len(best_split[0]):
                    best_split = (left, right)
        if best_split is not None:
            logger.debug("Split unknown word %r → %r", word, best_split)
            expanded.extend(best_split)
        else:
            # Can't split — keep original (will be skipped in substitution)
            expanded.append(word)
    return expanded


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
    return [
        p + "[0|1|2]" if p in _VOWEL_PHONES else p
        for p in phones
    ]


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

    cmu = _get_cmudict()
    adversarial: list[str] = []

    for phrase in target_phrases:
        raw_words = phrase.lower().split()
        words = _expand_unknown_words(raw_words, cmu)

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
                        if match_phones and match_phones[0] != (pronouncing.phones_for_word(word) or [""])[0]:
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
    """Synthesize speech clips using VITS with SLERP speaker blending.

    Uses the vendored piper-sample-generator to produce diverse synthetic
    voices by interpolating between speaker embeddings (904 speakers in the
    libritts-high model).

    Returns list of paths to generated .wav files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmu = _get_cmudict()
    phrases = [
        " ".join(_expand_unknown_words(p.lower().split(), cmu)) for p in phrases
    ]

    if vits_model_path is None or not vits_model_path.exists():
        raise FileNotFoundError(
            f"VITS model not found at {vits_model_path}. "
            "Cannot generate audio — refusing to produce silent placeholders. "
            "Download the model first: python -m livekit.wakeword.data setup"
        )

    generated = generate_samples(
        text=phrases,
        output_dir=output_dir,
        max_samples=n_samples,
        model=vits_model_path,
        batch_size=batch_size,
        slerp_weights=slerp_weights,
        length_scales=length_scales,
        noise_scales=noise_scales,
        noise_scale_ws=noise_scale_ws,
        max_speakers=max_speakers,
        start_index=start_index,
    )
    logger.info(f"Generated {len(generated)} clips in {output_dir}")
    return generated


def run_generate(config: WakeWordConfig) -> None:
    """Run the full generate pipeline for a wake word config.

    Supports resuming: counts existing ``clip_######.wav`` files in each split
    directory and skips completed splits or resumes partial ones from the
    existing count.
    """
    model_dir = config.model_output_dir
    vits_path = config.data_path / "piper" / "en-us-libritts-high.pt"
    if not vits_path.exists():
        raise FileNotFoundError(
            f"VITS model not found at {vits_path}. "
            "Run setup first: python -m livekit.wakeword.data setup"
        )

    synth_kwargs: dict[str, Any] = {
        "noise_scales": config.noise_scales,
        "noise_scale_ws": config.noise_scale_ws,
        "length_scales": config.length_scales,
        "slerp_weights": config.slerp_weights,
        "max_speakers": config.max_speakers,
    }

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
                split_name, existing, n_target,
            )
            continue
        if existing > 0:
            logger.info(
                "Resuming split %s from clip %d / %d",
                split_name, existing, n_target,
            )
        else:
            logger.info("Generating %d %s clips...", n_target, split_name)
        synthesize_clips(
            phrases=phrases,
            output_dir=split_dir,
            n_samples=n_target,
            vits_model_path=vits_path,
            batch_size=config.tts_batch_size,
            start_index=existing,
            **synth_kwargs,
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
                    split_name, existing, n_target,
                )
                continue
            if existing > 0:
                logger.info(
                    "Resuming split %s from clip %d / %d",
                    split_name, existing, n_target,
                )
            else:
                logger.info("Synthesizing %d %s clips...", n_target, split_name)
            synthesize_clips(
                phrases=adv_phrases,
                output_dir=split_dir,
                n_samples=n_target,
                vits_model_path=vits_path,
                batch_size=config.tts_batch_size,
                start_index=existing,
                **synth_kwargs,
            )

