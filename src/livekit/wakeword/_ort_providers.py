"""Runtime selection of ONNX Runtime execution providers.

Central helper used everywhere an ``ort.InferenceSession`` is created, so the
provider list stays consistent across training (feature extraction), eval, and
inference (``WakeWordModel``).

Default behavior: prefer CUDA, fall back to CPU — driven by whatever ONNX
Runtime distribution is installed.  A backend is no longer a base dependency;
install the ``cpu`` extra (``onnxruntime``) or the ``gpu`` extra
(``onnxruntime-gpu``) to get one — see README for the switch.

Override via the ``LIVEKIT_WAKEWORD_ORT_PROVIDERS`` env var (comma-separated
provider names) — handy for forcing CPU for reproducibility, or opting into
less-common providers like CoreML / DirectML / ROCm / TensorRT.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_logger = logging.getLogger(__name__)

_DEFAULT_PREFERENCE: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
_ENV_VAR = "LIVEKIT_WAKEWORD_ORT_PROVIDERS"


def import_ort() -> ModuleType:
    """Import ``onnxruntime`` with an actionable error if no backend is installed.

    livekit-wakeword does not bundle an ONNX Runtime backend — exactly one of the
    ``cpu`` / ``gpu`` extras must be installed.
    """
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "No ONNX Runtime backend is installed. livekit-wakeword does not bundle "
            "one — install exactly one of:\n"
            "  pip install 'livekit-wakeword[cpu]'   # CPU-only\n"
            "  pip install 'livekit-wakeword[gpu]'   # CUDA (see README)"
        ) from exc
    return ort


def get_providers() -> list[str]:
    """Return the ONNX Runtime provider list to pass to ``InferenceSession``.

    Resolution order:

    1. If ``LIVEKIT_WAKEWORD_ORT_PROVIDERS`` is set and non-empty, parse it
       as a comma-separated list and return verbatim.  ORT itself will reject
       unknown providers at session-creation time with a clear error.
    2. Otherwise intersect ``onnxruntime.get_available_providers()`` with
       ``("CUDAExecutionProvider", "CPUExecutionProvider")`` in that order.
    3. If neither preferred provider is available (unusual — e.g. a non-CPU
       build without CUDA), return whatever ORT reports as available so
       session creation still succeeds.
    """
    ort = import_ort()

    override = os.environ.get(_ENV_VAR, "").strip()
    if override:
        providers = [p.strip() for p in override.split(",") if p.strip()]
        _logger.info("ORT providers from %s: %s", _ENV_VAR, providers)
        return providers

    available = set(ort.get_available_providers())
    providers = [p for p in _DEFAULT_PREFERENCE if p in available]
    if not providers:
        providers = list(ort.get_available_providers())
    _logger.info("ORT providers (auto-selected): %s", providers)
    return providers
