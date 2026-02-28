"""Tests for training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.training.metrics import (
    accuracy,
    evaluate_model,
    false_positives_per_hour,
    recall_at_threshold,
)


class TestMetrics:
    def test_fpph(self):
        preds = np.array([0.1, 0.6, 0.8, 0.3, 0.9])
        fpph = false_positives_per_hour(preds, threshold=0.5, total_hours=1.0)
        assert fpph == 3.0

    def test_fpph_zero_hours(self):
        preds = np.array([0.9])
        assert false_positives_per_hour(preds, 0.5, 0.0) == float("inf")

    def test_recall(self):
        preds = np.array([0.9, 0.8, 0.3, 0.95])
        recall = recall_at_threshold(preds, threshold=0.5)
        assert recall == 0.75

    def test_recall_empty(self):
        assert recall_at_threshold(np.array([]), 0.5) == 0.0

    def test_accuracy(self):
        pos = np.array([0.9, 0.8, 0.3])
        neg = np.array([0.1, 0.2, 0.6])
        acc = accuracy(pos, neg, threshold=0.5)
        # 2 TP + 2 TN = 4/6
        assert abs(acc - 4 / 6) < 1e-6

    def test_evaluate_model(self):
        pos = np.array([0.9, 0.8])
        neg = np.array([0.1, 0.2])
        result = evaluate_model(pos, neg, threshold=0.5, validation_hours=1.0)
        assert "fpph" in result
        assert "recall" in result
        assert "accuracy" in result
        assert result["recall"] == 1.0
        assert result["fpph"] == 0.0


class TestDataset:
    def test_mmap_batch_generator(self, sample_features_file: Path):
        from livekit.wakeword.data.dataset import mmap_batch_generator

        gen = mmap_batch_generator(
            data_files={"pos": sample_features_file, "neg": sample_features_file},
            n_per_class={"pos": 10, "neg": 20},
            label_funcs={"pos": lambda _: 1, "neg": lambda _: 0},
        )
        features, labels = next(gen)
        assert features.shape == (30, 16, 96)
        assert labels.shape == (30,)
        assert np.sum(labels == 1) == 10
        assert np.sum(labels == 0) == 20
