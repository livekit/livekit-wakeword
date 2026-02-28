"""Tests for inference pipeline components."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.models.pipeline import WakeWordClassifier


class TestWakeWordClassifier:
    def test_forward(self, sample_config: WakeWordConfig, sample_embeddings: torch.Tensor):
        model = WakeWordClassifier(sample_config)
        out = model(sample_embeddings)
        assert out.shape == (4, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_save_load(self, sample_config: WakeWordConfig, sample_embeddings: torch.Tensor, tmp_path):
        model = WakeWordClassifier(sample_config)
        out1 = model(sample_embeddings)

        # Save and reload
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        model2 = WakeWordClassifier(sample_config)
        model2.load_state_dict(torch.load(path, weights_only=True))
        out2 = model2(sample_embeddings)

        torch.testing.assert_close(out1, out2)
