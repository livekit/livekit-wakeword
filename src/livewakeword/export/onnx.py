"""ONNX export and INT8 quantization for wake word models."""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
import torch

from livewakeword.config import WakeWordConfig
from livewakeword.models.pipeline import WakeWordClassifier

logger = logging.getLogger(__name__)


def export_classifier(
    config: WakeWordConfig,
    model_path: Path,
    output_path: Path,
    opset_version: int = 18,
) -> Path:
    """Export classifier head to ONNX.

    Input shape: (1, 16, 96) — pre-extracted embeddings
    Output shape: (1, 1) — confidence score
    """
    model = WakeWordClassifier(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 16, 96)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["embeddings"],
        output_names=["score"],
        dynamic_axes={
            "embeddings": {0: "batch"},
            "score": {0: "batch"},
        },
    )

    # Bundle external data into a single ONNX file
    onnx_model = onnx.load(str(output_path), load_external_data=True)
    onnx.save(onnx_model, str(output_path), save_as_external_data=False)

    # Remove leftover external data file if it exists
    external_data_path = output_path.with_suffix(".onnx.data")
    if external_data_path.exists():
        external_data_path.unlink()

    logger.info(f"Exported classifier ONNX to {output_path}")
    return output_path


def export_full_pipeline(
    config: WakeWordConfig,
    model_path: Path,
    output_path: Path,
    opset_version: int = 18,
) -> Path:
    """Export classifier ONNX alongside the frozen mel/embedding ONNX models.

    The full pipeline (waveform → score) cannot be a single ONNX graph because
    the mel-spectrogram and speech-embedding stages are already separate ONNX
    models (melspectrogram.onnx, embedding_model.onnx). This function exports
    the classifier head and copies the frozen models into the output directory
    so that all three ONNX files are co-located for deployment.

    Output directory will contain:
        - <model_name>.onnx (classifier: embeddings → score)
        - melspectrogram.onnx (copy from data/models/)
        - embedding_model.onnx (copy from data/models/)
    """
    import shutil

    from livewakeword.models.feature_extractor import (
        EMBEDDING_ONNX_FILENAME,
        MEL_ONNX_FILENAME,
    )

    # Export the classifier head
    classifier_path = export_classifier(config, model_path, output_path, opset_version)

    # Copy frozen ONNX models alongside the classifier
    models_dir = config.data_path / "models"
    out_dir = output_path.parent
    for onnx_name in [MEL_ONNX_FILENAME, EMBEDDING_ONNX_FILENAME]:
        src = models_dir / onnx_name
        dst = out_dir / onnx_name
        if src.exists() and src != dst:
            shutil.copy2(src, dst)
            logger.info(f"Copied {onnx_name} to {out_dir}")
        elif not src.exists():
            logger.warning(f"Frozen model not found: {src}")

    logger.info(f"Exported full pipeline (3 ONNX files) to {out_dir}")
    return classifier_path


def quantize_onnx(input_path: Path, output_path: Path | None = None) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if output_path is None:
        output_path = input_path.with_suffix(".int8.onnx")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    logger.info(f"Quantized ONNX model to {output_path}")
    return output_path


def run_export(config: WakeWordConfig, quantize: bool = False) -> Path:
    """Export trained model to ONNX."""
    model_dir = config.model_output_dir
    model_path = model_dir / f"{config.model_name}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    # Export classifier head
    onnx_path = model_dir / f"{config.model_name}.onnx"
    export_classifier(config, model_path, onnx_path)

    # Optionally quantize
    if quantize:
        quantize_onnx(onnx_path)

    return onnx_path
