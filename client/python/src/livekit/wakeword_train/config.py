"""Training config: only `wakeword` is required (`livekit-wakeword setup` writes <name>.yaml)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ModelType = Literal["standard", "micro"]
PLANS = ("build", "ship", "scale")  # LiveKit Cloud plans: 5 / 20 / 50 concurrent TTS requests
CACHE = Path.home() / ".cache" / "livekit-wakeword"


def slug(phrase: str) -> str:
    """File-safe model name for a phrase: "Hey LiveKit!" -> "hey_livekit"."""
    return re.sub(r"[^\w]+", "_", phrase.lower(), flags=re.UNICODE).strip("_") or "wakeword"


class WakeWordConfig(BaseModel):
    wakeword: str = Field(min_length=1, description='The wake phrase, e.g. "hey livekit"')
    model_name: str | None = Field(None, description="Output file stem (default: from the phrase)")
    language: str = Field("en", description="Language of the phrase (TTS voices)")
    pronunciations: dict[str, str] = Field(
        default_factory=dict,
        description='ARPAbet for words outside the CMU dictionary, e.g. {"zork": "Z AO1 R K"}',
    )
    output_dir: Path = Field(Path("."), description="Where models are written")
    data_dir: Path | None = Field(
        None, description="Where generated speech lives (default: the cache, per wake word)"
    )

    # dataset generation (LiveKit Inference TTS through livekit-wakeword-bench)
    plan: str = Field("build", description="LiveKit plan: build / ship / scale (TTS concurrency)")
    concurrency: int | None = Field(
        None, ge=1, description="Concurrent TTS requests; overrides `plan` (e.g. enterprise)"
    )
    n_positive: int = Field(
        500, ge=1, description="Wake word clips per dataset (x2 when there are no plain-text ones)"
    )
    n_confusable: int = Field(500, ge=0, description="Close-sounding confusable clips")
    n_sentence: int = Field(150, ge=0, description="Everyday sentences (negatives)")

    # training
    steps: int = Field(5000, ge=1, description="Fine-tuning steps, standard model")
    micro_steps: int = Field(5000, ge=1, description="Fine-tuning steps, micro model")
    target_fpph: float = Field(0.5, gt=0, description="False alarms per hour at score 0.5")

    @property
    def name(self) -> str:
        return self.model_name or slug(self.wakeword)

    @property
    def dataset_dir(self) -> Path:
        return Path(self.data_dir) if self.data_dir else CACHE / "datasets" / self.name

    @property
    def runs_dir(self) -> Path:
        return CACHE / "runs" / self.name

    def model_path(self, type: ModelType = "standard") -> Path:
        stem = self.name if type == "standard" else f"{self.name}_micro"
        return Path(self.output_dir) / f"{stem}.{'onnx' if type == 'standard' else 'tflite'}"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        data = self.model_dump(mode="json", exclude_none=True)
        if not data["pronunciations"]:
            del data["pronunciations"]
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return path


def load_config(path: str | Path) -> WakeWordConfig:
    """A config from YAML (`livekit-wakeword setup` writes one per wake word)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return WakeWordConfig(**data)


def default_config_path(wakeword: str) -> Path:
    return Path(f"{slug(wakeword)}.yaml")
