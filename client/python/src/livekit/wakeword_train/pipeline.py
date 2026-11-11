"""The pipeline behind the CLI: `run_setup` (generate speech with livekit-wakeword-bench),
`run_train` (fine-tune the pretrained encoders, calibrate, export) and `run_eval`."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import ModelType, WakeWordConfig, is_english

logger = logging.getLogger(__name__)
# share of training clips with overlapping talkers (party scenes); calibration keeps the plain mix
STANDARD_P_PARTY = 0.3
MICRO_P_PARTY = 0.6


class SetupError(RuntimeError):
    """The environment cannot run the pipeline; the message says what to fix."""


def check_environment(config: WakeWordConfig, require_gpu: bool = False) -> list[str]:
    """Problems that would stop setup or training; empty when everything is in place."""
    problems = []
    if sys.version_info < (3, 12):
        problems.append("Python 3.12+ is required for dataset generation (livekit-wakeword-bench)")
    else:
        try:
            import livekit_wakeword_bench  # noqa: F401
        except ImportError:
            problems.append("livekit-wakeword-bench is missing: `uv sync --extra train`")
    missing = [
        k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.getenv(k)
    ]
    if missing:
        problems.append(
            f"set {', '.join(missing)} (LiveKit Inference TTS): `lk cloud auth`, then "
            "`lk app env -w` writes them to .env.local"
        )
    if config.concurrency is None and config.plan not in ("build", "ship", "scale"):
        problems.append(
            f"unknown plan {config.plan!r}: use build / ship / scale, or set concurrency"
        )
    try:
        import torch

        if require_gpu and not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            problems.append("no CUDA GPU or Apple silicon GPU found (training on CPU is very slow)")
    except ImportError:
        problems.append("torch is missing: `uv sync --extra train`")
    free = shutil.disk_usage(Path.home()).free / 2**30
    if free < 10:
        problems.append(f"only {free:.0f} GB free in your home directory (about 10 GB is needed)")
    return problems + phrase_problems(config)


CMUDICT = "http://www.speech.cs.cmu.edu/cgi-bin/cmudict"


def unknown_words(config: WakeWordConfig) -> list[str]:
    """Words of the phrase with no ARPAbet pronunciation: not in the CMU dictionary, not made of
    words in it ("livekit" = live + kit) and not in `config.pronunciations`."""
    try:
        from livekit_wakeword_bench.scaffold import guess_pronunciations
    except ImportError:
        return []
    given = {w.lower() for w in config.pronunciations}
    return [w for w in guess_pronunciations(config.wakeword)[1] if w not in given]


def phoneme_steered(config: WakeWordConfig) -> bool:
    """Whether TTS is steered with phonemes: English phrases whose words all have an ARPAbet
    pronunciation (`phrase_problems` asks for missing ones). ARPAbet is English, so phrases in
    other languages use plain-text TTS only."""
    return is_english(config.language) and not unknown_words(config)


def phrase_problems(config: WakeWordConfig) -> list[str]:
    """Pronunciations the phrase is missing, or that are not valid ARPAbet."""
    problems = []
    try:
        from livekit_wakeword_bench.phonemes import validate_arpabet
    except ImportError:
        return []
    for word, arpabet in config.pronunciations.items():
        if err := validate_arpabet(arpabet):
            problems.append(f"pronunciation for {word!r} is not ARPAbet ({err}), e.g. Z AO1 R K")
    missing = unknown_words(config)
    if missing and is_english(config.language):
        flags = " ".join(f'--pronounce "{w}=<ARPAbet>"' for w in missing)
        problems.append(
            f"no pronunciation for {', '.join(map(repr, missing))} (not in the CMU dictionary): "
            f'run setup again with {flags} (e.g. "zork=Z AO1 R K"), or add them under '
            f"`pronunciations:` in {config.name}.yaml; {CMUDICT} spells similar words"
        )
    hard = [w for w in unknown_hard_negative_words(config) if w not in missing]
    if hard and phoneme_steered(config):  # bench drops phoneme-steered phrases it cannot spell
        flags = " ".join(f'--pronounce "{w}=<ARPAbet>"' for w in hard)
        problems.append(
            f"no pronunciation for {', '.join(map(repr, hard))} in hard_negatives (not in the CMU "
            f"dictionary): run setup again with {flags}, or add them under `pronunciations:`"
        )
    return problems


def unknown_hard_negative_words(config: WakeWordConfig) -> list[str]:
    """Words of English hard negatives with no ARPAbet pronunciation (see `unknown_words`)."""
    if not is_english(config.language) or not config.hard_negatives:
        return []
    try:
        from livekit_wakeword_bench.scaffold import guess_pronunciations
    except ImportError:
        return []
    given = {w.lower() for w in config.pronunciations}
    words = (w for p in config.hard_negatives for w in guess_pronunciations(p)[1])
    return [w for w in dict.fromkeys(words) if w not in given]


def natural_positives_help(phrase: str, language: str = "en") -> bool:
    """Whether plain-text TTS positives help: only when TTS reads the words as users say them
    (dictionary words; brand names like "livekit" get varied readings)."""
    if not is_english(language):
        return True
    import pronouncing

    words = re.findall(r"[a-z']+", phrase.lower())
    return bool(words) and all(pronouncing.phones_for_word(w) for w in words)


def _generate(
    config: WakeWordConfig,
    out: Path,
    splits: list[str],
    overwrite: bool = False,
    hard_negatives: bool = True,
    **knobs: Any,
) -> None:
    """Generate `splits` into `out` with bench; `hard_negatives`: send the config's (sets without
    confusables leave them out, so changing them does not regenerate those)."""
    from livekit_wakeword_bench import generate

    if config.concurrency:
        knobs["tts__concurrency"] = config.concurrency
    for word, arpabet in config.pronunciations.items():  # JSON-quoted: bench parses knobs as YAML
        knobs[f"pronunciations__{word.lower()}"] = json.dumps(arpabet)
    if hard_negatives and config.hard_negatives:
        knobs["hard_negatives"] = json.dumps(config.hard_negatives, ensure_ascii=False)
    if not phoneme_steered(config):
        # plain-text TTS only; bench still wants an entry per word, unused without phonemes
        knobs["tts__phonemes"] = False
        for word in unknown_words(config):
            knobs.setdefault(f"pronunciations__{word}", json.dumps("AH0"))
    generate(
        config.wakeword,
        out,
        size="small",
        plan=config.plan,
        splits=splits,
        language=config.language,
        overwrite=overwrite,
        **knobs,
    )


def _done(d: Path, split: str) -> bool:
    return any((d / "data").glob(f"{split}-*.parquet"))


def _bench_config(d: Path) -> dict[str, Any] | None:
    """The bench config a generated dataset was made with."""
    import yaml

    try:
        cfg = yaml.safe_load((d / "bench_config.yaml").read_text())
    except (OSError, yaml.YAMLError):
        return None
    return cfg if isinstance(cfg, dict) else None


def _train_counts(d: Path) -> tuple[int, int, int] | None:
    """(positive, confusable, sentence) clip counts a generated training set was made with."""
    try:
        split = (_bench_config(d) or {})["splits"]["train"]
        return int(split["n_positive"]), int(split["n_adversarial"]), int(split["n_speech"])
    except (KeyError, TypeError):
        return None


def _config_change(d: Path, config: WakeWordConfig, confusables: bool = True) -> str | None:
    """What the dataset in `d` was generated for that the config no longer says: the phrase,
    language, pronunciations (the old clips say the phrase differently) or hard negatives (sets
    with `confusables` only); None when it matches. Datasets without a bench config match."""
    bench = _bench_config(d)
    if bench is None:
        return None
    made = {str(k).lower(): str(v) for k, v in (bench.get("pronunciations") or {}).items()}
    if str(bench.get("wake_word", config.wakeword)) != config.wakeword:
        return "wakeword changed"
    if str(bench.get("language", config.language)) != config.language:
        return "language changed"
    if any(made.get(w.lower()) != a for w, a in config.pronunciations.items()):
        return "pronunciations changed"
    if confusables and sorted(bench.get("hard_negatives") or []) != sorted(config.hard_negatives):
        return "hard_negatives changed"
    return None


def _wanted(config: WakeWordConfig) -> dict[str, tuple[int, int, int]]:
    """The training sets for the phrase: (positive, confusable, sentence) clips per set."""
    n_pos, n_conf, n_sent = config.n_positive, config.n_confusable, config.n_sentence
    if not phoneme_steered(config):  # plain text only: one set with everything
        return {"natural": (2 * n_pos, n_conf, n_sent)}
    if not natural_positives_help(config.wakeword, config.language):
        # steered clips make up the whole positive share when there are no plain-text ones
        return {"steered": (2 * n_pos, n_conf, n_sent)}
    return {"steered": (n_pos, n_conf, n_sent), "natural": (n_pos, 0, 0)}


def _stale(out: Path, config: WakeWordConfig, counts: tuple[int, int, int]) -> str | None:
    """Why the training set in `out` does not match the config (missing, clip counts, phrase,
    language, pronunciations or hard negatives); None when it is up to date."""
    if not _done(out, "train"):
        return "missing"
    if _train_counts(out) != counts:
        return "clip counts changed"
    return _config_change(out, config, confusables=counts[1] > 0)


def train_datasets(config: WakeWordConfig) -> list[str]:
    """The phrase's training sets (generated by `run_setup`)."""
    d = config.dataset_dir
    return [str(d / k) for k in _wanted(config) if _done(d / k, "train")]


def run_setup(config: WakeWordConfig, eval: bool = False) -> list[str]:
    """Check the environment and generate the training speech (cached). With `eval`, also the
    held-out test set `run_eval` scores on."""
    problems = check_environment(config)
    if problems:
        raise SetupError("cannot set up:\n  - " + "\n  - ".join(problems))
    from .encoders import encoder_dir

    try:  # fetched now (once), so `train` never stops for it
        logger.info("pretrained encoders: %s", encoder_dir("standard").parent)
    except RuntimeError as e:
        raise SetupError(str(e)) from None
    d = config.dataset_dir
    if not phoneme_steered(config):
        logger.info("%r: generating with plain-text TTS only", config.wakeword)
    if not is_english(config.language) and config.n_confusable and not config.hard_negatives:
        logger.warning(
            "no confusables for %r: they are generated for English only. Give close-sounding "
            "phrases with `setup --hard-negative` or under `hard_negatives:` in %s.yaml so the "
            "model learns to reject them",
            config.wakeword,
            config.name,
        )
    for kind, (n_pos, n_conf, n_sent) in _wanted(config).items():
        out = d / kind
        reason = _stale(out, config, (n_pos, n_conf, n_sent))
        if reason is None:
            continue
        done = reason != "missing"
        if done:  # the old checkpoint learned from other speech
            logger.info("%s: regenerating %s", reason, out)
            (config.runs_dir / "standard.pt").unlink(missing_ok=True)
        logger.info("generating %s training speech for %r", kind, config.wakeword)
        _generate(
            config,
            out,
            ["train"],
            overwrite=done,
            hard_negatives=n_conf > 0,
            **({"tts__phonemes": False} if kind == "natural" else {}),
            splits__train__n_positive=n_pos,
            splits__train__n_adversarial=n_conf,
            splits__train__n_speech=n_sent,
        )
    if eval:
        test_dataset(config)
    return train_datasets(config)


def test_dataset(config: WakeWordConfig) -> Path:
    """The held-out test set (unseen voices, noise and rooms; cached)."""
    d = config.dataset_dir / "test"
    change = _config_change(d, config) if _done(d, "test") else None
    stale = change is not None
    if stale:
        logger.info("%s: regenerating %s", change, d)
    if not _done(d, "test") or stale:
        problems = check_environment(config)
        if problems:
            raise SetupError("cannot generate the test set:\n  - " + "\n  - ".join(problems))
        logger.info("generating the test set for %r", config.wakeword)
        _generate(config, d, ["test"], overwrite=stale)
    return d


def run_train(
    config: WakeWordConfig, type: ModelType = "standard", device: str | None = None
) -> Path:
    """Train, calibrate and export; returns the model path. standard: `<name>.onnx`. micro:
    `<name>_micro.tflite` (needs the `tflite` extra, else `<name>_micro.onnx`), distilled from
    the phrase's standard model, which is trained first if needed."""
    import torch

    from .encoders import load_pretrained
    from .export import calibrate, export_streaming
    from .loop import TrainConfig, calibration_logit, default_aug, train

    data = train_datasets(config)
    if not data:
        data = run_setup(config)
    config.runs_dir.mkdir(parents=True, exist_ok=True)
    std_ckpt = config.runs_dir / "standard.pt"
    if type == "standard" or not std_ckpt.exists():
        start = load_pretrained("standard").bake(config.wakeword)
        aug = default_aug(start)
        train_aug = replace(aug, p_party=STANDARD_P_PARTY)
        det = train(start, data, TrainConfig(steps=config.steps, aug=train_aug), device=device)
        torch.save(det.state_dict(), std_ckpt)  # micro's teacher
        c = calibration_logit(det, target_fpph=config.target_fpph, device=device, aug=aug)
        logger.info(
            "calibration: score 0.5 = %.2f false alarms / hour (logit %.2f)", config.target_fpph, c
        )
        out = config.model_path("standard")
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = {"wake_word": config.wakeword, "language": config.language, "threshold": "0.5"}
        path = export_streaming(
            calibrate(det.cpu(), c), out, metadata={**meta, "format": "livekit-wakeword-v2"}
        )
        if type == "standard":
            return path
    return _train_micro(config, data, std_ckpt, device)


def _train_micro(
    config: WakeWordConfig, data: list[str], std_ckpt: Path, device: str | None
) -> Path:
    import torch

    from .encoders import load_pretrained
    from .export import calibrate
    from .loop import TrainConfig, calibration_logit, default_aug, train
    from .micro import export_micro

    teacher = load_pretrained("standard").bake(config.wakeword)
    teacher.load_state_dict(torch.load(std_ckpt, map_location="cpu"))
    start = load_pretrained("micro").bake(config.wakeword)
    aug = default_aug(start)
    # party scenes (overlapping talkers) in training only; calibration stays on the plain mix
    train_aug = replace(aug, p_party=MICRO_P_PARTY)
    cfg = TrainConfig(steps=config.micro_steps, ft_top_blocks=None, kd_weight=2.0, aug=train_aug)
    mdet = train(start, data, cfg, device=device, teacher=teacher)
    c = calibration_logit(mdet, target_fpph=config.target_fpph, device=device, aug=aug)
    # micro scores are sharper; temperature 4 keeps useful thresholds clear of 1.0
    mdet = calibrate(mdet.cpu(), c, temperature=4.0)
    torch.save(mdet.state_dict(), config.runs_dir / "micro.pt")
    out = config.model_path("micro")
    files = export_micro(mdet, out.parent, config.name, dataset=data)
    logger.info("micro model: %s", ", ".join(str(p) for p in files.values()))
    return files.get("tflite", files["onnx"])


def _is_streaming_onnx(path: Path) -> bool:
    """A standard model takes raw audio (`chunk`); a micro `.onnx` takes log-mels (`mel`)."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    return any(i.name == "chunk" for i in graph.input)


def run_eval(
    config: WakeWordConfig,
    model_path: str | Path | None = None,
    media: str | Path | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """The livekit-wakeword-bench summary on the phrase's test set: `score` (recall at 0.5 false
    alarms per hour), metrics at threshold 0.5 and, with `media` (`lkwb media`), false alarms per
    hour on long-form TV, music and podcasts."""
    from livekit_wakeword_bench import benchmark

    path = Path(model_path or config.model_path("standard"))
    if not path.exists():
        raise SetupError(f"no model at {path}: train it first (`livekit-wakeword train`)")
    if media is not None and not Path(media).is_dir():
        raise SetupError(f"no media set at {media}: build it once with `lkwb media -o {media}`")
    if path.suffix == ".onnx" and _is_streaming_onnx(path):
        spec = f"livekit:{path}"
    else:  # micro: the host reference runtime, same numerics as the device
        spec = f"livekit.wakeword_train.micro:MicroRuntime?path={path}"
    res = benchmark(spec, test_dataset(config), workers=workers, media=media)
    res.save(config.runs_dir / f"eval_{path.stem}")
    summary: dict[str, Any] = res.summary()
    summary.pop("curve", None)
    return summary
