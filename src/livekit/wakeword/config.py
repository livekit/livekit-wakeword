"""Pydantic configuration models with YAML loading."""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, Field, model_validator

_logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    dnn = "dnn"
    rnn = "rnn"


class ModelSize(str, Enum):
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"


# Preset mapping: size -> (layer_dim, n_blocks)
MODEL_SIZE_PRESETS: dict[ModelSize, tuple[int, int]] = {
    ModelSize.tiny: (16, 1),
    ModelSize.small: (32, 1),
    ModelSize.medium: (128, 2),
    ModelSize.large: (256, 3),
}


class AugmentationConfig(BaseModel):
    clip_duration: float = 2.0
    batch_size: int = 16
    rounds: int = 1
    background_paths: list[str] = Field(default_factory=lambda: ["./data/backgrounds"])
    rir_paths: list[str] = Field(default_factory=lambda: ["./data/rirs"])


class ModelConfig(BaseModel):
    model_type: ModelType = ModelType.dnn
    model_size: ModelSize = ModelSize.small

    @property
    def layer_dim(self) -> int:
        return MODEL_SIZE_PRESETS[self.model_size][0]

    @property
    def n_blocks(self) -> int:
        return MODEL_SIZE_PRESETS[self.model_size][1]


class WakeWordConfig(BaseModel):
    """Top-level config for a wake word model."""

    model_name: str
    target_phrases: list[str]

    # Data generation
    n_samples: int = 10000
    n_samples_val: int = 2000
    tts_batch_size: int = 50
    custom_negative_phrases: list[str] = Field(default_factory=list)

    # TTS parameters (VITS + SLERP speaker blending)
    noise_scales: list[float] = Field(default_factory=lambda: [0.98])
    noise_scale_ws: list[float] = Field(default_factory=lambda: [0.98])
    length_scales: list[float] = Field(default_factory=lambda: [0.75, 1.0, 1.25])
    slerp_weights: list[float] = Field(default_factory=lambda: [0.5])
    max_speakers: int | None = None

    # Paths
    data_dir: Annotated[str, Field(description="Root data directory")] = "./data"
    output_dir: str = "./output"

    # Augmentation
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    # Model
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Training
    steps: int = 50000
    learning_rate: float = 1e-4
    max_negative_weight: float = 1500.0
    target_fp_per_hour: float = 0.2
    batch_n_per_class: dict[str, int] = Field(
        default_factory=lambda: {
            "positive": 50,
            "adversarial_negative": 50,
            "ACAV100M_sample": 1024,
        }
    )

    @model_validator(mode="after")
    def _warn_unknown_batch_keys(self) -> Self:
        known_keys = {"positive", "adversarial_negative", "ACAV100M_sample"}
        unknown = set(self.batch_n_per_class) - known_keys
        if unknown:
            _logger.warning(
                f"Unrecognized keys in batch_n_per_class: {unknown}. "
                f"Known keys: {sorted(known_keys)}"
            )
        return self

    @property
    def model_output_dir(self) -> Path:
        return Path(self.output_dir) / self.model_name

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


def load_config(path: str | Path) -> WakeWordConfig:
    """Load a WakeWordConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return WakeWordConfig(**data)
