"""End-to-end wake word detection pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import WakeWordConfig
from .classifier import build_classifier


class WakeWordClassifier(nn.Module):
    """Classifier-only module operating on pre-extracted embeddings.

    Used during training when features are pre-computed.
    Input: (batch, 16, 96) → Output: (batch, 1)
    """

    def __init__(self, config: WakeWordConfig):
        super().__init__()
        self.classifier = build_classifier(
            model_type=config.model.model_type,
            model_size=config.model.model_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
