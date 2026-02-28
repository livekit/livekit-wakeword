"""Tests for augmentation utilities."""

from __future__ import annotations

import numpy as np

from livekit.wakeword.data.augment import align_clip_to_end


class TestAlignClipToEnd:
    def test_basic_alignment(self):
        audio = np.ones(8000, dtype=np.float32)  # 0.5s at 16kHz
        target_length = 32000  # 2s
        result = align_clip_to_end(audio, target_length, jitter_samples=0)
        assert result.shape == (target_length,)
        # Audio should be at the end
        assert np.sum(result[-8000:]) > 0
        assert np.sum(result[:16000]) == 0.0

    def test_output_length(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = align_clip_to_end(audio, 32000, jitter_samples=0)
        assert len(result) == 32000

    def test_longer_clip_than_target(self):
        audio = np.ones(48000, dtype=np.float32)  # 3s, longer than target
        result = align_clip_to_end(audio, 32000, jitter_samples=0)
        assert len(result) == 32000
