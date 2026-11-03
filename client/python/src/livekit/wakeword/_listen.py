"""`livekit-wakeword listen`: microphone detection with the client alone (no `train` extra), so
`uvx --from 'livekit-wakeword[listener]' livekit-wakeword listen model.onnx` works on 3.11+."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path


def run(models: Sequence[Path], threshold: float = 0.5) -> None:
    """Listen on the microphone and print detections until Ctrl+C."""
    from .listener import PYAUDIO_HINT, WakeWordListener
    from .model import WakeWordModel

    try:
        import pyaudio  # noqa: F401
    except ImportError:
        sys.exit(f"livekit-wakeword: listen needs PyAudio: {PYAUDIO_HINT}")
    try:
        model = WakeWordModel(models=list(models))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"livekit-wakeword: {e}")

    async def _run() -> None:
        async with WakeWordListener(model, threshold=threshold) as listener:
            print(f"Listening for {', '.join(model.model_names)} (Ctrl+C to stop)", flush=True)
            while True:
                d = await listener.wait_for_detection()
                print(f"Detected {d.name} ({d.confidence:.2f})", flush=True)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def main(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="livekit-wakeword listen",
        description="Listen on the microphone and print detections.",
    )
    parser.add_argument("models", nargs="+", type=Path, help="Wake word .onnx model(s)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    args = parser.parse_args(argv)
    run(args.models, args.threshold)
