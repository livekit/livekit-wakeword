"""ONNX export and INT8 quantization for wake word models."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import onnx
import torch

from ..config import ExportFormat, WakeWordConfig
from ..models.pipeline import WakeWordClassifier

logger = logging.getLogger(__name__)


def export_onnx(
    config: WakeWordConfig,
    model_path: Path,
    output_path: Path,
    opset_version: int = 18,
) -> Path:
    """Export classifier head to ONNX.

    Input shape: (batch, 16, 96) — pre-extracted embeddings
    Output shape: (batch, 1) — confidence score
    """
    model = WakeWordClassifier(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(2, 16, 96)
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


def quantize_onnx(input_path: Path, output_path: Path | None = None) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    if output_path is None:
        output_path = input_path.with_suffix(".int8.onnx")

    # The torch dynamo ONNX exporter emits value_info entries describing the
    # weight initializers (e.g. a Gemm B of shape [out, in]). When the dynamic
    # quantizer rewrites Gemm->MatMul it transposes those weights in place but
    # leaves the value_info stale, so its strict shape-inference pass then fails
    # with "Inferred shape and existing shape differ". Dropping initializer
    # value_info (which is redundant — shapes are inferred from the tensors)
    # avoids the conflict. We do this on a temp copy so the input model on disk
    # is left untouched.
    model = onnx.load(str(input_path))
    init_names = {init.name for init in model.graph.initializer}
    kept = [vi for vi in model.graph.value_info if vi.name not in init_names]

    if len(kept) == len(model.graph.value_info):
        # Nothing to strip — quantize the input directly.
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
    else:
        del model.graph.value_info[:]
        model.graph.value_info.extend(kept)
        with tempfile.TemporaryDirectory() as tmp_dir:
            cleaned = Path(tmp_dir) / "cleaned.onnx"
            onnx.save(model, str(cleaned))
            quantize_dynamic(
                model_input=str(cleaned),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
            )

    logger.info(f"Quantized ONNX model to {output_path}")
    return output_path


def run_export(
    config: WakeWordConfig,
    quantize: bool = False,
    format: ExportFormat | str | None = None,
) -> Path:
    """Export the trained classifier head.

    Args:
        config: Wake word config.
        quantize: Apply INT8 quantization to the exported artifact.
        format: Output format (``onnx`` or ``tflite``). Defaults to
            ``config.output_format`` when ``None``.

    Returns:
        Path to the primary exported artifact for the chosen format. ONNX is
        always produced as well, since TFLite is converted from it.
    """
    fmt = ExportFormat(format) if format is not None else config.output_format

    # Fail fast on unsupported (head, format) combinations before doing any work.
    if fmt == ExportFormat.tflite:
        from .tflite import ensure_tflite_supported

        ensure_tflite_supported(config.model.model_type)

    model_dir = config.model_output_dir
    model_path = model_dir / f"{config.model_name}.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    # Export classifier head to ONNX (also the conversion source for TFLite).
    onnx_path = model_dir / f"{config.model_name}.onnx"
    export_onnx(config, model_path, onnx_path)

    if fmt == ExportFormat.onnx:
        if quantize:
            quantize_onnx(onnx_path)
        return onnx_path

    if fmt == ExportFormat.tflite:
        # TFLite quantization is applied by the TF converter, not the ONNX path.
        from .tflite import export_tflite

        tflite_path = model_dir / f"{config.model_name}.tflite"
        return export_tflite(onnx_path, tflite_path, quantize=quantize)

    raise ValueError(f"Unsupported export format: {fmt}")
