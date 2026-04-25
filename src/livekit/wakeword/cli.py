"""Typer CLI — replaces all notebook functionality with scripted commands."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from .config import TtsBackend, WakeWordConfig, load_config

app = typer.Typer(
    name="livekit-wakeword",
    help="Simplified pure-PyTorch wake word detection.",
    no_args_is_help=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("livekit.wakeword")


@app.command()
def setup(
    data_dir: str = typer.Option(
        "./data",
        help="Root data directory when --config is omitted",
    ),
    config_path: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "Wake word YAML: data_dir from config; Piper weights if "
            "tts_backend is piper_vits; VoxCPM HF snapshot if tts_backend is voxcpm"
        ),
    ),
    skip_acav: bool = typer.Option(
        False, "--skip-acav", help="Skip downloading ACAV100M features (~16 GB)"
    ),
) -> None:
    """Download external dependencies: optional Piper VITS, ACAV100M features, RIRs, backgrounds."""
    from .defaults import piper as piper_defaults

    if config_path is not None:
        cfg = load_config(config_path)
        data_path = cfg.data_path.resolve()
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info("Setting up data dependencies (from %s)...", config_path)
        if cfg.tts_backend is TtsBackend.piper_vits:
            pt_dest = cfg.piper_checkpoint_path
            pt_dest.parent.mkdir(parents=True, exist_ok=True)
            _download_piper_checkpoint(pt_dest)
        elif cfg.tts_backend is TtsBackend.voxcpm:
            _download_voxcpm_model(cfg)
        else:
            logger.info(
                "Skipping TTS weight download (tts_backend=%s).",
                cfg.tts_backend.value,
            )
    else:
        data_path = Path(data_dir).resolve()
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info("Setting up livekit-wakeword data dependencies...")
        pt_dest = piper_defaults.checkpoint_path(data_path)
        pt_dest.parent.mkdir(parents=True, exist_ok=True)
        _download_piper_checkpoint(pt_dest)

    # Download ACAV100M features
    features_dir = data_path / "features"
    features_dir.mkdir(exist_ok=True)
    if skip_acav:
        logger.info("Skipping ACAV100M features (--skip-acav). Downloading validation only...")
        _download_validation_features(features_dir)
    else:
        _download_features(features_dir)

    # Download RIRs
    rir_dir = data_path / "rirs"
    rir_dir.mkdir(exist_ok=True)
    _download_rirs(rir_dir)

    # Download background noise (MUSAN noise subset)
    bg_dir = data_path / "backgrounds"
    bg_dir.mkdir(exist_ok=True)
    _download_musan_noise(bg_dir)

    logger.info("Setup complete!")


def _download_piper_checkpoint(pt_dest: Path) -> None:
    """Download bundled Piper VITS state_dict and JSON config next to *pt_dest*."""
    import urllib.request

    from rich.progress import Progress

    from .defaults import piper as piper_defaults

    base_url = piper_defaults.RELEASE_BASE_URL

    pt_url = f"{base_url}/{piper_defaults.RELEASE_STATE_DICT_ASSET}"
    pt_name = pt_dest.name
    if not pt_dest.exists():
        logger.info("Downloading %s (~166 MB)...", pt_name)
        try:
            with Progress() as progress:
                task = progress.add_task(f"[cyan]{pt_name}", total=None)
                tmp_path = pt_dest.with_suffix(".tmp")

                def _reporthook(block_num: int, block_size: int, total: int) -> None:
                    if total > 0:
                        progress.update(task, total=total, completed=block_num * block_size)

                urllib.request.urlretrieve(pt_url, str(tmp_path), reporthook=_reporthook)
                tmp_path.rename(pt_dest)
            logger.info("Downloaded %s", pt_name)
        except Exception as e:
            logger.warning(f"Failed to download VITS checkpoint: {e}")
            tmp_path = pt_dest.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        logger.info(f"VITS checkpoint already exists: {pt_dest}")

    json_url = f"{base_url}/{piper_defaults.RELEASE_CONFIG_JSON_ASSET}"
    json_dest = pt_dest.with_suffix(".json")
    if not json_dest.exists():
        logger.info("Downloading VITS config JSON...")
        try:
            urllib.request.urlretrieve(json_url, str(json_dest))
            logger.info("Downloaded VITS config JSON")
        except Exception as e:
            logger.warning(f"Failed to download VITS config JSON: {e}")
    else:
        logger.info(f"VITS config already exists: {json_dest}")


def _download_voxcpm_model(cfg: WakeWordConfig) -> None:
    """Fetch the VoxCPM HF snapshot during ``setup`` (same idea as Piper: download if missing).

    Uses ``voxcpm_tts.model_id`` and ``voxcpm_local_model_path`` from config. Only runs
    ``snapshot_download`` if that directory is missing or empty, so re-running setup does
    not redownload gigabytes.
    """
    dest = cfg.voxcpm_local_model_path
    if dest.is_dir() and any(dest.iterdir()):
        logger.info("VoxCPM weights already present at %s", dest)
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning(
            "huggingface-hub not installed; cannot download VoxCPM. "
            "Install train extras or: uv pip install huggingface-hub"
        )
        return
    dest.mkdir(parents=True, exist_ok=True)
    repo = cfg.voxcpm_tts.model_id
    logger.info("Downloading VoxCPM snapshot %s → %s (large; may take a while)...", repo, dest)
    snapshot_download(repo_id=repo, local_dir=str(dest))


def _download_validation_features(features_dir: Path) -> None:
    """Download only the validation features (~176 MB)."""
    try:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading validation features (~176 MB, ~11hrs)...")
        hf_hub_download(
            repo_id="binhpham/livekit_wakeword_features",
            filename="validation_set_features.npy",
            local_dir=str(features_dir),
            repo_type="dataset",
        )
        logger.info("Validation features downloaded.")
    except ImportError:
        logger.warning(
            "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
        )
    except Exception as e:
        logger.warning(f"Failed to download validation features: {e}")


def _download_features(features_dir: Path) -> None:
    """Download pre-computed ACAV100M and validation features."""
    try:
        from huggingface_hub import hf_hub_download

        logger.info("Downloading ACAV100M features (~16 GB, ~2000hrs)...")
        hf_hub_download(
            repo_id="binhpham/livekit_wakeword_features",
            filename="openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
            local_dir=str(features_dir),
            repo_type="dataset",
        )
        logger.info("ACAV100M features downloaded.")
    except ImportError:
        logger.warning(
            "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
        )
    except Exception as e:
        logger.warning(f"Failed to download ACAV100M features: {e}")

    _download_validation_features(features_dir)


def _download_musan_noise(bg_dir: Path) -> None:
    """Download MUSAN noise subset (~6 hrs of background noise)."""
    # Check if noise files already present
    existing = list(bg_dir.glob("**/*.wav"))
    if existing:
        logger.info(f"Background noise already present: {len(existing)} files in {bg_dir}")
        return

    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading MUSAN background audio from HuggingFace (~1.1 GB)...")
        snapshot_download(
            repo_id="FluidInference/musan",
            repo_type="dataset",
            allow_patterns="**/*.wav",
            local_dir=str(bg_dir),
        )
        extracted = list(bg_dir.glob("**/*.wav"))
        logger.info(f"Downloaded {len(extracted)} background noise files")
    except ImportError:
        logger.warning(
            "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
        )
    except Exception as e:
        logger.warning(f"Failed to download MUSAN noise: {e}")


def _download_rirs(rir_dir: Path) -> None:
    """Download MIT room impulse responses (~8 MB, 270 WAV files)."""
    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading MIT room impulse responses (~8 MB)...")
        snapshot_download(
            repo_id="davidscripka/MIT_environmental_impulse_responses",
            repo_type="dataset",
            allow_patterns="16khz/*.wav",
            local_dir=str(rir_dir),
        )
        logger.info("RIRs downloaded.")
    except ImportError:
        logger.warning(
            "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
        )
    except Exception as e:
        logger.warning(f"Failed to download RIRs: {e}")


@app.command()
def generate(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
) -> None:
    """Generate synthetic speech clips (positive + adversarial negative)."""
    config = load_config(config_path)

    logger.info(f"Generating data for '{config.model_name}'...")
    logger.info(f"Target phrases: {config.target_phrases}")

    from .data.generate import run_generate

    run_generate(config)
    logger.info("Generation complete!")


@app.command()
def augment(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
) -> None:
    """Augment clips and extract features through frozen pipeline."""
    config = load_config(config_path)

    logger.info(f"Augmenting data for '{config.model_name}'...")

    from .data.augment import run_augment
    from .data.features import run_extraction

    run_augment(config)
    logger.info("Augmentation complete!")

    logger.info("Extracting features through frozen embedding pipeline...")
    run_extraction(config)
    logger.info("Feature extraction complete!")


@app.command()
def train(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
) -> None:
    """Train classifier on extracted features (3-phase adaptive training)."""
    config = load_config(config_path)

    logger.info(f"Training '{config.model_name}' model...")
    logger.info(f"Model: {config.model.model_type.value} ({config.model.model_size.value})")
    logger.info(f"Steps: {config.steps}")

    from .training.trainer import run_train

    model_path = run_train(config)
    logger.info(f"Training complete! Model saved to {model_path}")


@app.command()
def export(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
    quantize: bool = typer.Option(False, "--quantize", help="Apply INT8 quantization"),
) -> None:
    """Export trained model to ONNX (optionally quantize for embedded)."""
    config = load_config(config_path)

    logger.info(f"Exporting '{config.model_name}' to ONNX...")

    from .export.onnx import run_export

    onnx_path = run_export(config, quantize=quantize)
    logger.info(f"Export complete! ONNX model at {onnx_path}")


@app.command()
def eval(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
    model_path: str = typer.Option(
        None, "--model", "-m", help="Path to ONNX model (default: <output_dir>/<model_name>.onnx)"
    ),
) -> None:
    """Evaluate model on validation set: DET curve, AUT, FPPH, recall."""
    from pathlib import Path

    config = load_config(config_path)

    if model_path is None:
        resolved_model = config.model_output_dir / f"{config.model_name}.onnx"
    else:
        resolved_model = Path(model_path)

    logger.info(f"Evaluating '{config.model_name}' with model {resolved_model}...")

    from .eval.evaluate import run_eval

    results = run_eval(config, resolved_model)

    logger.info(
        f"AUT={results['aut']:.4f}  FPPH={results['fpph']:.2f}  "
        f"Recall={results['recall']:.1%}  Threshold={results['threshold']:.2f}"
    )
    logger.info(f"DET curve: {config.model_output_dir / f'{config.model_name}_det.png'}")


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
) -> None:
    """Run entire pipeline end-to-end: generate → augment → train → export."""
    config = load_config(config_path)

    logger.info(f"Running full pipeline for '{config.model_name}'...")

    from .data.augment import run_augment
    from .data.features import run_extraction
    from .data.generate import run_generate
    from .eval.evaluate import run_eval
    from .export.onnx import run_export
    from .training.trainer import run_train

    logger.info("Step 1/6: Generate synthetic data")
    run_generate(config)

    logger.info("Step 2/6: Augment clips")
    run_augment(config)

    logger.info("Step 3/6: Extract features")
    run_extraction(config)

    logger.info("Step 4/6: Train classifier")
    run_train(config)

    logger.info("Step 5/6: Export to ONNX")
    onnx_path = run_export(config)

    logger.info("Step 6/6: Evaluate model")
    results = run_eval(config, onnx_path)
    logger.info(
        f"Eval: AUT={results['aut']:.4f}  FPPH={results['fpph']:.2f}  "
        f"Recall={results['recall']:.1%}"
    )

    logger.info("Full pipeline complete!")
