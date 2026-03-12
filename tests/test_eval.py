"""Tests for the eval module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livekit.wakeword.eval.evaluate import (
    _compute_aut,
    _compute_det_curve,
    _predict_onnx,
)


class TestDetCurve:
    """Tests for DET curve computation."""

    def test_perfect_separation(self) -> None:
        """Perfect model: positives all 1.0, negatives all 0.0."""
        pos = np.ones(100)
        neg = np.zeros(100)
        thresholds, fpr, fnr = _compute_det_curve(pos, neg)

        # At threshold 0.5: FPR=0 (no negatives >= 0.5), FNR=0 (all positives >= 0.5)
        idx_05 = np.argmin(np.abs(thresholds - 0.5))
        assert fpr[idx_05] == 0.0
        assert fnr[idx_05] == 0.0

    def test_random_model(self) -> None:
        """Random scores should have non-trivial FPR and FNR."""
        rng = np.random.RandomState(42)
        pos = rng.rand(500)
        neg = rng.rand(500)
        thresholds, fpr, fnr = _compute_det_curve(pos, neg)

        # At threshold 0.5: roughly half should be misclassified
        idx_05 = np.argmin(np.abs(thresholds - 0.5))
        assert 0.3 < fpr[idx_05] < 0.7
        assert 0.3 < fnr[idx_05] < 0.7

    def test_shape(self) -> None:
        """Output arrays should all be the same length."""
        pos = np.random.rand(50)
        neg = np.random.rand(50)
        thresholds, fpr, fnr = _compute_det_curve(pos, neg)
        assert thresholds.shape == fpr.shape == fnr.shape


class TestAUT:
    """Tests for Area Under the DET curve."""

    def test_perfect_model_aut_near_zero(self) -> None:
        """Perfect separation should yield AUT near 0."""
        pos = np.ones(100)
        neg = np.zeros(100)
        _, fpr, fnr = _compute_det_curve(pos, neg)
        aut = _compute_aut(fpr, fnr)
        assert aut < 0.01

    def test_random_model_aut(self) -> None:
        """Random model should have AUT significantly above 0."""
        rng = np.random.RandomState(42)
        pos = rng.rand(500)
        neg = rng.rand(500)
        _, fpr, fnr = _compute_det_curve(pos, neg)
        aut = _compute_aut(fpr, fnr)
        assert 0.1 < aut < 0.5

    def test_aut_bounded(self) -> None:
        """AUT should be between 0 and 1."""
        pos = np.random.rand(100)
        neg = np.random.rand(100)
        _, fpr, fnr = _compute_det_curve(pos, neg)
        aut = _compute_aut(fpr, fnr)
        assert 0.0 <= aut <= 1.0


class TestOnnxPredict:
    """Tests for ONNX prediction helper."""

    def test_predict_with_mock_session(self) -> None:
        """Ensure _predict_onnx batches and concatenates correctly."""

        class _MockInput:
            name = "embeddings"

        class _MockSession:
            def get_inputs(self):
                return [_MockInput()]

            def run(self, _output_names, inputs):
                batch = inputs["embeddings"]
                # Mimic real ONNX output shape: (batch, 1)
                scores = batch.mean(axis=(1, 2))[:, np.newaxis]
                return [scores]

        features = np.random.randn(50, 16, 96).astype(np.float32)
        scores = _predict_onnx(_MockSession(), features, batch_size=16)  # type: ignore[arg-type]
        assert scores.shape == (50,)
