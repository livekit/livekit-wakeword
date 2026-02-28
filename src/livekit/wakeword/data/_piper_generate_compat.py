"""Compatibility shim for piper_train module.

The VITS .pt checkpoint is pickled with references to ``piper_train.vits.*``
classes.  The ``monotonic_align`` Cython extension often fails to compile and
is only needed for VITS *training*, not inference.  This module stubs it out
so ``torch.load`` can unpickle the model without the Cython binary.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types

logger = logging.getLogger(__name__)

_patched = False


def ensure_piper_train() -> None:
    """Make sure ``piper_train`` is importable for ``torch.load``."""
    global _patched  # noqa: PLW0603
    if _patched:
        return

    # Stub monotonic_align BEFORE piper_train.vits.models tries to import it
    ma_name = "piper_train.vits.monotonic_align"
    if ma_name not in sys.modules:
        stub = types.ModuleType(ma_name)
        stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules[ma_name] = stub
        # Also stub the nested .core module
        core_name = f"{ma_name}.core"
        sys.modules[core_name] = types.ModuleType(core_name)
        core_inner = f"{ma_name}.monotonic_align"
        sys.modules[core_inner] = types.ModuleType(core_inner)
        core_inner2 = f"{core_inner}.core"
        sys.modules[core_inner2] = types.ModuleType(core_inner2)

    importlib.import_module("piper_train")
    _patched = True
