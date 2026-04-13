"""English phrase normalization for Piper VITS (CMUDict-based)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_cmudict() -> dict[str, list[str]]:
    """Load CMU Pronouncing Dictionary via nltk."""
    import nltk

    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict

    return {word: prons[0] for word, prons in cmudict.dict().items()}


def expand_unknown_words(
    words: list[str],
    cmu: dict[str, list[str]],
) -> list[str]:
    """Expand words not in CMUDict by splitting into known subwords."""
    expanded: list[str] = []
    for word in words:
        if word in cmu:
            expanded.append(word)
            continue
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
            expanded.append(word)
    return expanded


def normalize_phrases_for_piper(phrases: list[str]) -> list[str]:
    """Lowercase, split words, expand unknowns via CMUDict, rejoin."""
    cmu = get_cmudict()
    return [" ".join(expand_unknown_words(p.lower().split(), cmu)) for p in phrases]
