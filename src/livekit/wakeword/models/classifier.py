"""Classification heads for wake word detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from livekit.wakeword.config import ModelSize, ModelType, MODEL_SIZE_PRESETS


class FCNBlock(nn.Module):
    """Fully-connected block: Linear → LayerNorm → ReLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DNNClassifier(nn.Module):
    """DNN classification head.

    Flatten(16×96=1536) → Linear(1536, layer_dim) → LayerNorm → ReLU
    → N×FCNBlock → Linear(layer_dim, 1) → Sigmoid
    """

    def __init__(
        self,
        n_timesteps: int = 16,
        embedding_dim: int = 96,
        layer_dim: int = 32,
        n_blocks: int = 1,
    ):
        super().__init__()
        input_dim = n_timesteps * embedding_dim
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(input_dim, layer_dim),
            nn.LayerNorm(layer_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers.append(FCNBlock(layer_dim))
        layers.append(nn.Linear(layer_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 16, 96) embedding sequence

        Returns:
            (batch, 1) confidence score [0, 1]
        """
        return self.net(x)


class RNNClassifier(nn.Module):
    """Bi-LSTM classification head.

    Bi-LSTM(96→64, 2 layers) → Linear(128, 1) → Sigmoid
    """

    def __init__(
        self,
        embedding_dim: int = 96,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 16, 96) embedding sequence

        Returns:
            (batch, 1) confidence score [0, 1]
        """
        output, _ = self.lstm(x)
        # Use last timestep
        last = output[:, -1, :]
        return self.head(last)


def build_classifier(
    model_type: ModelType = ModelType.dnn,
    model_size: ModelSize = ModelSize.small,
) -> nn.Module:
    """Factory function to build a classifier from config enums."""
    layer_dim, n_blocks = MODEL_SIZE_PRESETS[model_size]

    if model_type == ModelType.dnn:
        return DNNClassifier(layer_dim=layer_dim, n_blocks=n_blocks)
    elif model_type == ModelType.rnn:
        return RNNClassifier(hidden_dim=layer_dim, num_layers=max(1, n_blocks))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
