"""Tests for model export (ONNX + TFLite) and quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from livekit.wakeword.config import ExportFormat, ModelType, WakeWordConfig
from livekit.wakeword.export import export_tflite, run_export
from livekit.wakeword.export.tflite import TFLITE_SUPPORTED_HEADS, ensure_tflite_supported
from livekit.wakeword.models.pipeline import WakeWordClassifier


def _make_checkpoint(tmp_path: Path, model_type: ModelType) -> WakeWordConfig:
    """Build a config + a randomly-initialized .pt checkpoint at the expected path."""
    config = WakeWordConfig(
        model_name="ww",
        target_phrases=["hey test"],
        output_dir=str(tmp_path / "output"),
    )
    config.model.model_type = model_type
    model = WakeWordClassifier(config)
    model.eval()
    config.model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.model_output_dir / f"{config.model_name}.pt")
    return config


def test_run_export_onnx_io_contract(tmp_path: Path):
    config = _make_checkpoint(tmp_path, ModelType.dnn)
    out = run_export(config)  # default format == onnx
    assert out.suffix == ".onnx" and out.exists()

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "embeddings"
    x = np.random.randn(1, 16, 96).astype(np.float32)
    y = session.run(None, {"embeddings": x})[0]
    assert y.shape == (1, 1)
    assert 0.0 <= float(y[0, 0]) <= 1.0


def test_run_export_format_accepts_enum_and_string(tmp_path: Path):
    config = _make_checkpoint(tmp_path, ModelType.dnn)
    assert run_export(config, format="onnx").suffix == ".onnx"
    assert run_export(config, format=ExportFormat.onnx).suffix == ".onnx"


def test_quantize_onnx_dnn(tmp_path: Path):
    """Regression: dynamic quantization of a dynamo-exported Gemm head.

    The torch dynamo exporter emits value_info for weight initializers; the ORT
    dynamic quantizer transposes those weights (Gemm->MatMul) without updating it,
    which previously crashed quantization with a ShapeInferenceError.
    """
    config = _make_checkpoint(tmp_path, ModelType.dnn)
    out = run_export(config, quantize=True, format="onnx")
    int8 = config.model_output_dir / "ww.int8.onnx"
    assert int8.exists()

    session = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    y = session.run(None, {"embeddings": np.random.randn(1, 16, 96).astype(np.float32)})[0]
    assert y.shape == (1, 1)

    # The source ONNX must not be mutated by quantization, and must still run.
    src = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert src.get_inputs()[0].name == "embeddings"


def test_tflite_supported_heads():
    assert ModelType.dnn in TFLITE_SUPPORTED_HEADS
    ensure_tflite_supported(ModelType.dnn)  # does not raise


@pytest.mark.parametrize("model_type", [ModelType.conv_attention, ModelType.rnn])
def test_tflite_rejects_unsupported_heads(tmp_path: Path, model_type: ModelType):
    """run_export(format=tflite) must fail fast for unsupported heads.

    The guard fires before any checkpoint load or conversion, so this does not
    require the optional tflite extra.
    """
    config = _make_checkpoint(tmp_path, model_type)
    with pytest.raises(NotImplementedError, match="TFLite export is not supported"):
        run_export(config, format="tflite")


def test_tflite_export_dnn(tmp_path: Path):
    """Full ONNX->TFLite conversion for the dnn head (skipped without the extra)."""
    pytest.importorskip("onnx2tf")
    tf = pytest.importorskip("tensorflow")

    config = _make_checkpoint(tmp_path, ModelType.dnn)
    out = run_export(config, format="tflite")
    assert out.suffix == ".tflite" and out.exists()

    interp = tf.lite.Interpreter(model_path=str(out))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    # openWakeWord contract: static (1, 16, 96) input, (1, 1) output.
    assert tuple(in_det["shape"]) == (1, 16, 96)
    assert tuple(out_det["shape"]) == (1, 1)

    # Numerical parity with the ONNX source.
    x = np.random.randn(1, 16, 96).astype(np.float32)
    interp.set_tensor(in_det["index"], x)
    interp.invoke()
    y_tflite = float(interp.get_tensor(out_det["index"])[0, 0])
    session = ort.InferenceSession(
        str(config.model_output_dir / "ww.onnx"), providers=["CPUExecutionProvider"]
    )
    y_onnx = float(session.run(None, {"embeddings": x})[0][0, 0])
    assert abs(y_tflite - y_onnx) < 1e-4


def test_tflite_export_dnn_quantized(tmp_path: Path):
    """--quantize --format tflite must still satisfy the openWakeWord contract."""
    pytest.importorskip("onnx2tf")
    tf = pytest.importorskip("tensorflow")

    config = _make_checkpoint(tmp_path, ModelType.dnn)
    out = run_export(config, quantize=True, format="tflite")
    assert out.suffix == ".tflite" and out.exists()

    interp = tf.lite.Interpreter(model_path=str(out))
    interp.allocate_tensors()
    assert tuple(interp.get_input_details()[0]["shape"]) == (1, 16, 96)
    assert tuple(interp.get_output_details()[0]["shape"]) == (1, 1)


def test_export_tflite_missing_extra_message(tmp_path: Path):
    """If onnx2tf is unavailable, export_tflite raises a clear install hint."""
    try:
        import onnx2tf  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="tflite"):
            export_tflite(tmp_path / "missing.onnx", tmp_path / "out.tflite")
    else:
        pytest.skip("tflite extra installed; ImportError path not exercised")
