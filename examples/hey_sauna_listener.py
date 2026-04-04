"""Real-time "Hey Sauna" wake word detection from microphone.

Usage:
  uv run examples/hey_sauna_listener.py
  uv run examples/hey_sauna_listener.py --threshold 0.3   # more sensitive
  uv run examples/hey_sauna_listener.py --threshold 0.7   # fewer false positives

Requires the listener extra:
  pip install livekit-wakeword[listener]
"""

import argparse
import asyncio
from pathlib import Path

from livekit.wakeword import WakeWordModel
from livekit.wakeword.inference import WakeWordListener

MODEL_PATH = Path(__file__).parent / "resources" / "hey_sauna.onnx"


async def main():
    parser = argparse.ArgumentParser(description="Hey Sauna wake word listener")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5, lower = more sensitive)",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=2.0,
        help="Seconds to wait between detections (default: 2.0)",
    )
    args = parser.parse_args()

    model = WakeWordModel(models=[MODEL_PATH])
    async with WakeWordListener(
        model, threshold=args.threshold, debounce=args.debounce
    ) as listener:
        print(f"Listening for 'Hey Sauna' (threshold={args.threshold})...")
        print("Press Ctrl+C to stop.\n")
        while True:
            detection = await listener.wait_for_detection()
            print(
                f"  Detected '{detection.name}'! (confidence={detection.confidence:.2f})"
            )


if __name__ == "__main__":
    asyncio.run(main())
