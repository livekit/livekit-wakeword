# Contributing

Thanks for your interest in contributing!

## Before writing code

If you'd like to contribute code, it's recommended to first discuss your idea on the
[LiveKit Developer Community](https://community.livekit.io/c/robotics). This helps keep changes aligned with the LiveKit roadmap and avoids duplicated or unnecessary work.

## Before you open a pull request

Please make sure your change passes all of the following:

```sh
uv sync --extra tflite
uv run ruff check . && uv run ruff format --check .           # lints and formatting
uv run mypy -p livekit.wakeword -p livekit.wakeword_train     # types
uv run pytest                                                 # client + trainer tests
swift test                                                    # Swift client (macOS)
(cd client/js && pnpm install && pnpm lint && pnpm typecheck && pnpm test)  # JS client
```

Changes to the ESP32 component (`client/esp32`) should build both examples for the ESP32-S3 and
ESP32-P4 with ESP-IDF 5.5 (see [client/esp32/README.md](client/esp32/README.md#examples)).

## Pull requests

- Open PRs against the `main` branch
- Describe *what* the change does and *why*
- Link any related issues (e.g. `Closes #123`)
- Keep commits focused, and keep PRs limited to a single, logical change.
- For larger changes, consider using [stacked PRs](https://docs.github.com/en/pull-requests/how-tos/stacked-pull-requests).

## Reporting bugs

The issue tracker is for bugs or suspected bugs only. Bug reports must use the "Bug Report" template; issues that don't will be closed automatically.

## License

By contributing, you agree that your contributions will be licensed under the same terms as this project (see the `LICENSE` file)
