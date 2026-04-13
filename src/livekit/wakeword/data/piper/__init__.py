"""Piper-style VITS TTS: SLERP speaker blending and sample generation."""

from __future__ import annotations

from livekit.wakeword.tts_constants import PIPER_DATA_SUBDIR

from .defaults import (
    DEFAULT_CHECKPOINT_RELPATH,
    DEFAULT_CHECKPOINT_STEM,
    DEFAULT_CONFIG_JSON_FILENAME,
    DEFAULT_RELEASE_BASE_URL,
    DEFAULT_RELEASE_TAG,
    DEFAULT_STATE_DICT_FILENAME,
    RELEASE_CONFIG_JSON_ASSET,
    RELEASE_STATE_DICT_ASSET,
    default_checkpoint_path,
)
from .synthesis import generate_samples, get_phonemes, remove_silence
from .text import expand_unknown_words, get_cmudict, normalize_phrases_for_piper

__all__ = [
    "DEFAULT_CHECKPOINT_RELPATH",
    "DEFAULT_CHECKPOINT_STEM",
    "DEFAULT_CONFIG_JSON_FILENAME",
    "DEFAULT_RELEASE_BASE_URL",
    "DEFAULT_RELEASE_TAG",
    "DEFAULT_STATE_DICT_FILENAME",
    "PIPER_DATA_SUBDIR",
    "RELEASE_CONFIG_JSON_ASSET",
    "RELEASE_STATE_DICT_ASSET",
    "default_checkpoint_path",
    "expand_unknown_words",
    "generate_samples",
    "get_cmudict",
    "get_phonemes",
    "normalize_phrases_for_piper",
    "remove_silence",
]
