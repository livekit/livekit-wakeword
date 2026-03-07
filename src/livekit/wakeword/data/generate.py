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

# Phoneme substitution map for adversarial generation
SIMILAR_PHONEMES: dict[str, list[str]] = {
    "AA": ["AH", "AO", "AE"],
    "AE": ["EH", "AA", "AH"],
    "AH": ["AA", "AE", "ER"],
    "AO": ["AA", "AH", "OW"],
    "AW": ["OW", "AO"],
    "AY": ["EY", "OY"],
    "B": ["P", "D"],
    "CH": ["SH", "JH"],
    "D": ["T", "B", "G"],
    "DH": ["TH", "Z"],
    "EH": ["AE", "IH", "AH"],
    "ER": ["AH", "R"],
    "EY": ["AY", "EH"],
    "F": ["V", "TH"],
    "G": ["K", "D"],
    "HH": [""],
    "IH": ["EH", "IY", "AH"],
    "IY": ["IH", "EY"],
    "JH": ["CH", "ZH"],
    "K": ["G", "T"],
    "L": ["R", "W"],
    "M": ["N", "NG"],
    "N": ["M", "NG"],
    "NG": ["N", "M"],
    "OW": ["AO", "AW"],
    "OY": ["AY", "OW"],
    "P": ["B", "T"],
    "R": ["L", "W"],
    "S": ["Z", "SH"],
    "SH": ["S", "CH", "ZH"],
    "T": ["D", "K", "P"],
    "TH": ["DH", "F"],
    "UH": ["UW", "AH"],
    "UW": ["UH", "OW"],
    "V": ["F", "B"],
    "W": ["L", "R"],
    "Y": ["IY"],
    "Z": ["S", "ZH"],
    "ZH": ["SH", "Z"],
}


def _get_cmudict() -> dict[str, list[str]]:
    """Load CMU Pronouncing Dictionary via nltk."""
    import nltk

    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict

    # cmudict.dict() returns {word: [pron1, pron2, ...]} where each pron is list[str]
    # Take the first pronunciation for each word
    return {word: prons[0] for word, prons in cmudict.dict().items()}


def _build_reverse_phoneme_index(
    cmu: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build index from phoneme tuple → list of words."""
    from collections import defaultdict

    index: dict[str, list[str]] = defaultdict(list)
    for word, phones in cmu.items():
        # Strip stress markers
        stripped = tuple(p.rstrip("012") for p in phones)
        index[" ".join(stripped)].append(word)
    return dict(index)


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


def generate_adversarial_phrases(
    target_phrases: list[str],
    n_phrases: int = 200,
    include_partial_phrase: float = 1.0,
    include_input_words: float = 0.2,
) -> list[str]:
    """Generate phonetically similar phrases to the target using CMUDict.

    For each word in target phrases, find words with similar phonemes and
    combine them to create adversarial negative phrases. Unknown words
    (not in CMUDict) are split into known subwords when possible, e.g.
    "livekit" → "live" + "kit", allowing substitutions on both parts.
    """
    import pronouncing

    cmu = _get_cmudict()
    rev_index = _build_reverse_phoneme_index(cmu)
    adversarial: list[str] = []

    for phrase in target_phrases:
        raw_words = phrase.lower().split()
        words = _expand_unknown_words(raw_words, cmu)

        # Get phonemes for each word
        word_phonemes: list[list[str]] = []
        for word in words:
            phones = pronouncing.phones_for_word(word)
            if phones:
                word_phonemes.append([p.rstrip("012") for p in phones[0].split()])
            else:
                word_phonemes.append([])

        # Generate substitutions for each word position
        for word_idx, (word, phones) in enumerate(zip(words, word_phonemes)):
            if not phones:
                continue
            # Try substituting each phoneme
            for phone_idx, phone in enumerate(phones):
                subs = SIMILAR_PHONEMES.get(phone, [])
                for sub in subs:
                    if not sub:
                        continue
                    new_phones = phones.copy()
                    new_phones[phone_idx] = sub
                    key = " ".join(new_phones)
                    if key in rev_index:
                        for replacement in rev_index[key][:3]:
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

    # Deduplicate, remove the original target phrases, and limit
    target_set = {p.lower() for p in target_phrases}
    adversarial = [p for p in set(adversarial) if p not in target_set]
    random.shuffle(adversarial)
    return adversarial[:n_phrases]


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

