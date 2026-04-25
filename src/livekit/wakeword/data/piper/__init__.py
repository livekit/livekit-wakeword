"""Piper-style VITS TTS: SLERP speaker blending and sample generation."""

from __future__ import annotations

from .synthesis import generate_samples, get_phonemes, remove_silence
from .text import expand_unknown_words, get_cmudict, normalize_phrases_for_piper

__all__ = [
    "expand_unknown_words",
    "generate_samples",
    "get_cmudict",
    "get_phonemes",
    "normalize_phrases_for_piper",
    "remove_silence",
]
