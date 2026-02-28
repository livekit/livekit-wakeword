"""Tests for DNN and RNN classifiers."""

from __future__ import annotations

import torch

from livekit.wakeword.config import ModelSize, ModelType
from livekit.wakeword.models.classifier import (
    DNNClassifier,
    RNNClassifier,
    build_classifier,
)


class TestDNNClassifier:
    def test_output_shape(self, sample_embeddings: torch.Tensor):
        model = DNNClassifier(layer_dim=32, n_blocks=1)
        out = model(sample_embeddings)
        assert out.shape == (4, 1)

    def test_output_range(self, sample_embeddings: torch.Tensor):
        model = DNNClassifier(layer_dim=32, n_blocks=1)
        out = model(sample_embeddings)
        assert (out >= 0).all() and (out <= 1).all()

    def test_sizes(self, sample_embeddings: torch.Tensor):
        for layer_dim, n_blocks in [(16, 1), (32, 1), (128, 2), (256, 3)]:
            model = DNNClassifier(layer_dim=layer_dim, n_blocks=n_blocks)
            out = model(sample_embeddings)
            assert out.shape == (4, 1)


class TestRNNClassifier:
    def test_output_shape(self, sample_embeddings: torch.Tensor):
        model = RNNClassifier()
        out = model(sample_embeddings)
        assert out.shape == (4, 1)

    def test_output_range(self, sample_embeddings: torch.Tensor):
        model = RNNClassifier()
        out = model(sample_embeddings)
        assert (out >= 0).all() and (out <= 1).all()


class TestBuildClassifier:
    def test_build_dnn(self, sample_embeddings: torch.Tensor):
        model = build_classifier(ModelType.dnn, ModelSize.small)
        out = model(sample_embeddings)
        assert out.shape == (4, 1)

    def test_build_rnn(self, sample_embeddings: torch.Tensor):
        model = build_classifier(ModelType.rnn, ModelSize.medium)
        out = model(sample_embeddings)
        assert out.shape == (4, 1)

    def test_all_sizes(self, sample_embeddings: torch.Tensor):
        for size in ModelSize:
            for mtype in ModelType:
                model = build_classifier(mtype, size)
                out = model(sample_embeddings)
                assert out.shape == (4, 1)
