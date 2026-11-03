"""`livekit-wakeword` CLI: setup, train, eval, listen."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from .config import WakeWordConfig

app = typer.Typer(
    name="livekit-wakeword",
    help="Type a wake word, get a streaming model.",
    no_args_is_help=True,
)
console = Console()
# logs go to stderr so `--json` output on stdout stays machine-readable
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True)],
)
for _noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger("livekit.wakeword_train")

WAKEWORD = typer.Option(None, "--wakeword", "-w", help='The wake phrase, e.g. "hey livekit"')
CONFIG = typer.Option(
    None, "--config", "-c", hidden=True, help="Config YAML (default: <name>.yaml)"
)


def _config(wakeword: str | None, config_path: Path | None, **overrides: object) -> WakeWordConfig:
    """--config, else <name>.yaml if present, else the defaults; plus CLI overrides."""
    from .config import WakeWordConfig, default_config_path, load_config

    if config_path is None and wakeword and default_config_path(wakeword).exists():
        config_path = default_config_path(wakeword)
    if config_path is not None:
        cfg = load_config(config_path)
        if wakeword and wakeword != cfg.wakeword:
            raise typer.BadParameter(f"{config_path} is for {cfg.wakeword!r}, not {wakeword!r}")
    elif wakeword:
        cfg = WakeWordConfig(wakeword=wakeword)
    else:
        raise typer.BadParameter("give --wakeword (or --config)")
    changes = {k: v for k, v in overrides.items() if v is not None}
    return cfg.model_copy(update=changes) if changes else cfg


@app.command()
def setup(
    wakeword: str | None = WAKEWORD,
    config_path: Path | None = CONFIG,
    language: str | None = typer.Option(None, "--language", "-l", help="Language of the phrase"),
    plan: str | None = typer.Option(
        None,
        "--plan",
        help="LiveKit plan, sets TTS concurrency: build (5) / ship (20) / scale (50)",
    ),
    concurrency: int | None = typer.Option(
        None, "--concurrency", help="Concurrent TTS requests (overrides --plan, e.g. enterprise)"
    ),
    clips: float | None = typer.Option(
        None,
        "--clips",
        min=0.1,
        help="Generate N times the default clips (more speech, better calibration)",
    ),
    eval: bool = typer.Option(False, "--eval", help="Also generate the test set for `eval`"),
) -> None:
    """Check the environment, write the config (<name>.yaml) and generate the datasets."""
    from .config import WakeWordConfig, default_config_path
    from .pipeline import SetupError, run_setup

    counts: dict[str, int] = {}
    if clips is not None:  # a multiple of the default clip counts
        fields = WakeWordConfig.model_fields
        counts = {
            k: max(1, round(int(fields[k].default) * clips))
            for k in ("n_positive", "n_confusable", "n_sentence")
        }
    cfg = _config(
        wakeword, config_path, language=language, plan=plan, concurrency=concurrency, **counts
    )
    path = config_path or default_config_path(cfg.wakeword)
    if not Path(path).exists() or any(v is not None for v in (language, plan, concurrency, clips)):
        cfg.save(path)
        logger.info("config: %s", path)
    try:
        sets = run_setup(cfg, eval=eval)
    except SetupError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    logger.info("datasets ready: %s", ", ".join(sets))


@app.command()
def train(
    wakeword: str | None = WAKEWORD,
    type: str = typer.Option(
        "standard", "--type", "-t", help="standard (ONNX) or micro (ESP32 TFLite)"
    ),
    config_path: Path | None = CONFIG,
    steps: int | None = typer.Option(
        None, "--steps", help="Fine-tuning steps (default from config)"
    ),
    out: Path | None = typer.Option(None, "--out", "-o", help="Directory to write the model to"),
) -> None:
    """Train, calibrate and export a model (runs setup first if the datasets are missing)."""
    from .pipeline import SetupError, run_train

    if type not in ("standard", "micro"):
        raise typer.BadParameter("--type must be standard or micro")
    key = "steps" if type == "standard" else "micro_steps"
    cfg = _config(wakeword, config_path, **{key: steps, "output_dir": out})
    try:
        path = run_train(cfg, type=type)  # type: ignore[arg-type]
    except SetupError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    hint = f"  (try it: livekit-wakeword listen {path})" if type == "standard" else ""
    logger.info("Done: %s%s", path, hint)


@app.command("eval")
def eval_cmd(
    wakeword: str | None = WAKEWORD,
    type: str = typer.Option("standard", "--type", "-t", help="standard or micro"),
    model: Path | None = typer.Option(
        None, "--model", "-m", help="Model file (default: from config)"
    ),
    config_path: Path | None = CONFIG,
    media: Path | None = typer.Option(
        None, "--media", help="Long-form media negatives (`lkwb media`): false alarms per hour"
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the full summary as JSON"),
) -> None:
    """Score a model on the phrase's held-out test set (livekit-wakeword-bench)."""
    from .pipeline import run_eval

    cfg = _config(wakeword, config_path)
    path = model or cfg.model_path("micro" if type == "micro" else "standard")
    s = run_eval(cfg, path, media=media)
    if as_json:
        print(json.dumps(s, indent=2))
        return
    a = s["at_threshold"]
    console.print(
        f"[bold]{path.name}[/] on {s['n_positive']} wake words, {s['fpph_hours']} h of negatives"
    )
    console.print(f"  score (recall at 0.5 false alarms / h): [bold]{s['score']:.3f}[/]")
    console.print(
        f"  at threshold {a['threshold']:.2f}: recall {a['recall']:.3f}, "
        f"{a['fpph']:.2f} false alarms / h, "
        f"confusables accepted {a['adversarial_false_accept_rate']:.1%}"
    )
    if s.get("media"):
        m = s["media"]
        console.print(
            f"  media ({m['hours']} h of TV, music, podcasts): "
            f"{m['at_threshold']:.2f} false alarms / h"
        )


@app.command()
def listen(
    models: list[Path] = typer.Argument(..., help="Wake word .onnx model(s)"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Detection threshold"),
) -> None:
    """Listen on the microphone and print detections."""
    import asyncio

    from livekit.wakeword.listener import WakeWordListener
    from livekit.wakeword.model import WakeWordModel

    model = WakeWordModel(models=list(models))

    async def _run() -> None:
        async with WakeWordListener(model, threshold=threshold) as listener:
            logger.info(f"Listening for {', '.join(model.model_names)} (Ctrl+C to stop)")
            while True:
                d = await listener.wait_for_detection()
                logger.info(f"Detected {d.name} ({d.confidence:.2f})")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
