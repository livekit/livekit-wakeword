"""The `livekit-wakeword` command: the trainer's CLI, with a clear message when the `train`
extra (or Python 3.12+) is missing."""

import sys


def main() -> None:
    if sys.version_info < (3, 12):
        sys.exit("livekit-wakeword: training needs Python 3.12+ (detection works on 3.11)")
    try:
        from livekit.wakeword_train.cli import app
    except ImportError as e:
        sys.exit(
            f"livekit-wakeword: training needs the train extra ({e.name} is missing): "
            "pip install 'livekit-wakeword[train]'"
        )
    app()
