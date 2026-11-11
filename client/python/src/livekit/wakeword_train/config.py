"""Training config: only `wakeword` is required (`livekit-wakeword setup` writes <name>.yaml)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ModelType = Literal["standard", "micro"]
PLANS = ("build", "ship", "scale")  # LiveKit Cloud plans: 5 / 20 / 50 concurrent TTS requests
CACHE = Path.home() / ".cache" / "livekit-wakeword"


def slug(phrase: str) -> str:
    """File-safe model name for a phrase: "Hey LiveKit!" -> "hey_livekit"."""
    return re.sub(r"[^\w]+", "_", phrase.lower(), flags=re.UNICODE).strip("_") or "wakeword"


def is_english(language: str) -> bool:
    """Whether a language code is English ("en", "en-US", "en_GB", ...)."""
    return re.split(r"[-_]", language.strip().lower())[0] == "en"


# scripts written without spaces between words: Han, kana, Thai, Lao, Myanmar, Khmer
_UNSPACED = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff]"
)


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def says_phrase(text: str, phrase: str) -> bool:
    """Whether `text` contains `phrase`, ignoring case and punctuation: as whole words where
    words are spaced, as characters in scripts without spaces (Chinese, Japanese, Thai)."""
    t, p = _words(text), _words(phrase)
    if not p:
        return False
    if _UNSPACED.search(p):
        return p.replace(" ", "") in t.replace(" ", "")
    return f" {p} " in f" {t} "


class WakeWordConfig(BaseModel):
    # a misspelt key would otherwise fall back to its default without a word
    model_config = ConfigDict(extra="forbid")

    wakeword: str = Field(min_length=1, description='The wake phrase, e.g. "hey livekit"')
    model_name: str | None = Field(None, description="Output file stem (default: from the phrase)")
    language: str = Field("en", description="Language of the phrase (TTS voices)")
    pronunciations: dict[str, str] = Field(
        default_factory=dict,
        description='ARPAbet for words outside the CMU dictionary, e.g. {"zork": "Z AO1 R K"}',
    )
    hard_negatives: list[str] = Field(
        default_factory=list,
        description="Close-sounding phrases the model must not fire on (English adds its own; "
        "other languages have only these)",
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

    @field_validator("hard_negatives")
    @classmethod
    def _clean_hard_negatives(cls, v: list[str]) -> list[str]:
        return list(dict.fromkeys(p.strip() for p in v if p.strip()))

    @model_validator(mode="after")
    def _hard_negatives_are_not_the_phrase(self) -> WakeWordConfig:
        said = [p for p in self.hard_negatives if says_phrase(p, self.wakeword)]
        if said:  # they would teach the model not to fire on the wake word
            raise ValueError(f"hard_negatives must not say the wake word: {', '.join(said)}")
        return self

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
        if not data["hard_negatives"] and is_english(self.language):
            del data["hard_negatives"]  # listed (empty) for other languages, to fill in
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return path


def load_config(path: str | Path) -> WakeWordConfig:
    """A config from YAML (`livekit-wakeword setup` writes one per wake word)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return WakeWordConfig(**data)


def default_config_path(wakeword: str) -> Path:
    return Path(f"{slug(wakeword)}.yaml")
