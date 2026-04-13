"""Pluggable TTS backends for synthetic data generation."""

from __future__ import annotations

from .backends import SpeechSynthesizer, get_tts_backend

__all__ = ["SpeechSynthesizer", "get_tts_backend"]
