"""The `livekit-wakeword` command: the trainer's CLI, with a clear message when the `train`
extra (or Python 3.12+) is missing. `listen` needs only the client (and PyAudio), so it runs
without the trainer."""

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["listen"]:
        from livekit.wakeword._listen import main as listen

        listen(args[1:])
        return
    if sys.version_info < (3, 12):
        sys.exit("livekit-wakeword: training needs Python 3.12+ (detection works on 3.11)")
    try:
        from livekit.wakeword_train.cli import app
    except ImportError as e:
        sys.exit(
            f"livekit-wakeword: training needs the train extra ({e.name} is missing): "
            "pip install 'livekit-wakeword[train]'"
        )
    app(args=args, prog_name="livekit-wakeword")
