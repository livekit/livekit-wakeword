# Contributing

Thanks for your interest in contributing!

## Before writing code

If you'd like to contribute code, it's recommended to first discuss your idea on the
[LiveKit Developer Community](https://community.livekit.io/c/robotics). This helps keep changes aligned with the LiveKit roadmap and avoids duplicated or unnecessary work.

## Before you open a pull request

Please make sure your change passes all of the following:

```sh
uv run ruff check src/ tests/                # lints
uv run ruff format --check src/ tests/       # formatting
uv run mypy src/livekit/wakeword/            # types
uv run pytest tests/                         # tests
```

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
