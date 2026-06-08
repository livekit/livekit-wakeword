"""TFLite export for wake word classifier heads (openWakeWord-compatible).

The classifier is first exported to ONNX (see ``onnx.py``), then converted
ONNX → TF SavedModel → TFLite via ``onnx2tf``, which handles the NCHW↔NHWC
layout transposes that a naive converter would get wrong.

openWakeWord compatibility contract
-----------------------------------
openWakeWord loads classifier models with ``ai_edge_litert.interpreter`` (the
LiteRT successor to ``tflite_runtime``) and runs them like so::

    interp.set_tensor(input_index, x)   # x is (1, 16, 96) float32
    interp.invoke()
    score = interp.get_tensor(output_index)[0][0]   # (1, 1) -> scalar

That imposes three hard requirements on the artifact we emit:

1. **Static input shape ``(1, 16, 96)`` float32.** openWakeWord never calls
   ``resize_tensor_input``, so a dynamic/None batch axis would break it. We
   force the shape via ``overwrite_input_shape`` during conversion.
2. **Single output of shape ``(1, 1)``** (the sigmoid confidence).
3. **Builtin TFLite ops only** — the LiteRT interpreter is created without the
   Flex/SELECT_TF delegate. ``onnx2tf`` targets builtin ops and fails the
   conversion rather than emitting Flex, so a successful export is loadable.

Head support (verified via onnx2tf 1.28 / TF 2.x):

- ``dnn``: converts cleanly and is **bit-exact** with the ONNX/PyTorch model.
- ``conv_attention`` / ``rnn``: do **not** currently convert through onnx2tf
  (the attention emits an unsupported constant; the LSTM lowers to ``TensorList``
  ops that require the Flex delegate openWakeWord cannot load). Use ``dnn`` for
  openWakeWord-compatible TFLite, or deploy those heads via ONNX.

Requires the optional ``tflite`` extra: ``uv sync --extra tflite``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ..config import ModelType

logger = logging.getLogger(__name__)

# Fixed classifier I/O contract shared with openWakeWord.
_INPUT_NAME = "embeddings"
_INPUT_SHAPE = "1,16,96"

# Heads whose ONNX graph converts to builtin-op TFLite that openWakeWord can load.
# conv_attention and rnn currently do not (see module docstring).
TFLITE_SUPPORTED_HEADS = frozenset({ModelType.dnn})


def ensure_tflite_supported(model_type: ModelType) -> None:
    """Raise if ``model_type`` cannot be exported to openWakeWord-compatible TFLite.

    Fails fast with an actionable message instead of letting the onnx2tf
    conversion crash deep in its graph rewriter.
    """
    if model_type not in TFLITE_SUPPORTED_HEADS:
        supported = ", ".join(sorted(h.value for h in TFLITE_SUPPORTED_HEADS))
        raise NotImplementedError(
            f"TFLite export is not supported for the '{model_type.value}' head. "
            "It cannot be converted to builtin-op TFLite that openWakeWord can load "
            "(the attention block emits an unsupported constant; the LSTM requires the "
            "Flex delegate, which openWakeWord's LiteRT interpreter does not enable).\n"
            f"Supported heads: {supported}.\n"
            "Use '--format onnx' for this head, or set model.model_type to 'dnn'."
        )


def export_tflite(
    onnx_path: Path,
    output_path: Path,
    quantize: bool = False,
) -> Path:
    """Convert an exported classifier ONNX model to openWakeWord-compatible TFLite.

    Args:
        onnx_path: Source ``.onnx`` produced by ``export_onnx``.
        output_path: Destination ``.tflite`` path.
        quantize: Apply default (dynamic-range INT8) TFLite optimizations.

    Returns:
        Path to the written ``.tflite`` file with a static ``(1, 16, 96)`` input.

    Raises:
        ImportError: The optional ``tflite`` extra is not installed.
        FileNotFoundError: ``onnx_path`` does not exist.
        RuntimeError: onnx2tf/TFLite could not convert the graph (e.g. the head
            needs ops outside the builtin TFLite set).
    """
    try:
        import onnx2tf
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TFLite export requires the optional 'tflite' extra. Install with:\n"
            "  uv sync --extra tflite\n"
            "or: pip install 'livekit-wakeword[tflite]'"
        ) from e

    if not onnx_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {onnx_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # onnx2tf writes a TF SavedModel into tmp_dir. We pin the input to a
            # static (1, 16, 96) so the LiteRT interpreter (no resize) can feed it.
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=tmp_dir,
                overwrite_input_shape=[f"{_INPUT_NAME}:{_INPUT_SHAPE}"],
                # Keep the input as (1, 16, 96) exactly -- without this, onnx2tf's
                # NCHW->NHWC pass transposes it to (1, 96, 16) and breaks openWakeWord.
                keep_shape_absolutely_input_names=[_INPUT_NAME],
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=True,
            )

            converter = tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
            # Builtin ops only -- openWakeWord's interpreter has no Flex delegate.
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_bytes = converter.convert()
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert {onnx_path.name} to TFLite. The classifier head likely "
            "uses ops that cannot be mapped to the builtin TFLite set that openWakeWord "
            "requires (it loads models without the Flex delegate). Only the 'dnn' head is "
            f"currently supported for TFLite export.\nUnderlying error: {e}"
        ) from e

    output_path.write_bytes(tflite_bytes)
    logger.info(f"Exported openWakeWord-compatible TFLite to {output_path}")
    return output_path
