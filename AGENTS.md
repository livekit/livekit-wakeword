# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

livekit-wakeword v2: type a wake word, get a streaming model. Training adapts pretrained encoders
(standard and micro; downloaded from the GitHub release by `setup`, or `LIVEKIT_WAKEWORD_ENCODERS`) to the phrase and
fine-tunes them on speech generated with livekit-wakeword-bench. Two tiers: standard (streaming
ONNX) and micro (int8 TFLite).

## Layout

- One Python package, `livekit-wakeword` (root `pyproject.toml`), from `client/python/src`:
  `livekit.wakeword` (detection) and `livekit.wakeword_train` (training, the `train` extra).
- **`client/python/`**: `src/livekit/wakeword` (Python 3.11+, numpy + onnxruntime only):
  `WakeWordModel` (`predict()` streams, `predict_clip()`), `WakeWordListener`, and the
  `livekit-wakeword` CLI entry point (`_cli.py`, which explains a missing extra).
  `src/livekit/wakeword_train` (the `train` extra, Python 3.12+): `config.py` (`WakeWordConfig`),
  `pipeline.py` (`run_setup`, `run_train`, `run_eval`), `cli.py`; internals: `encoders.py` (load,
  `bake(phrase)` -> plain `Detector`), `nn.py`, `loop.py`, `data.py`, `augment.py`, `export.py`
  (streaming ONNX; state is plain inputs, runtimes zero it and warm up on 2 s of silence),
  `micro.py` (BC-ResNet, int8 TFLite, `MicroRuntime` host reference). Tests: `tests/` (client)
  and `tests/train/` (trainer). Guides: `README.md` (client), `TRAINING.md` (trainer).
- **`client/swift/`**: Swift package sources and tests (manifest: root `Package.swift`), `Example/`
  SwiftUI demo.
- **`client/js/`**: `@livekit/wakeword` npm package (TypeScript, pnpm): `src/model.ts` (runtime-agnostic
  `WakeWordModel` on any ONNX Runtime JS package), `web.ts` (onnxruntime-web/wasm, plus the
  AudioWorklet `WakeWordListener` in `listener.ts`), `node.ts` (onnxruntime-node), `core.ts`,
  `resample.ts` (port of the Python resampler); `example/` Vite demo; tests check scores step by
  step against the Python client (`scripts/make_parity.py` -> `test/fixtures/parity.json`).
- **`client/esp32/`**: ESP-IDF component `livekit_wakeword/` (the only copy of the C front end and
  TFLM wrapper), `examples/`, `tools/` (host test, front-end CLI, table generator).
- **`models/`**: hey livekit standard + micro, used by tests and examples.
- Root `pyproject.toml` is the package plus the dev tools and ruff/mypy/pytest config.

## Commands

**Always use `uv` for package management. Always use `git` for version separation (branches, commits).**

```bash
uv sync --extra tflite                           # everything (tflite includes train)
uv run pytest                                    # client + trainer tests
uv run ruff check . && uv run ruff format .
uv run mypy -p livekit.wakeword -p livekit.wakeword_train
swift test                                       # Swift client
cd client/js && pnpm install && pnpm lint && pnpm typecheck && pnpm test   # JS client
(cd client/js && pnpm test:browser)              # JS listener in headless Chromium
uv run livekit-wakeword setup|train|eval|listen  # the CLI
```

## Key design decisions

- Exported models are plain detectors: the phrase conditioning is folded into weights at `bake`,
  so runtimes never see the text encoder.
- Data generation goes only through livekit-wakeword-bench (no in-repo TTS).
- Formats: standard is ONNX (onnxruntime in Python/Swift/JS, pure-Rust tract in the Rust SDK; the
  trainer's tests check exports load in tract), micro is int8 TFLite (TFLite Micro).
- No v1 compatibility: clients load streaming models only.
- Encoder pretraining and experiments live outside this repo; keep recipe details (data,
  objectives, ablations) out of public code, docs and model cards.

## Release process

- Versions are cut with a PR, then a tag
  - Run `scripts/update_version.sh X.Y.Z` (or `X.Y.Z-rc.N`): it bumps the Python package, the
    ESP32 component, the npm package (`client/js/package.json`) and `uv.lock`
  - Open a PR (e.g., "Release v2.1.0") with the bumped version and merge it
  - Publish a GitHub release on the merge commit with tag `vX.Y.Z` (mark `-rc.N` tags as
    pre-releases), include the automatically generated changelog, and attach the ready-made
    models (`*.onnx`, `*_micro.tflite`) and `NOTICE`
  - The tag triggers `.github/workflows/publish.yml`: it checks the tag matches the package
    version and publishes `livekit-wakeword` to PyPI and `@livekit/wakeword` to npm (`-rc.N` to
    the `next` dist-tag); Swift Package Manager picks up the tag
- The pretrained encoders (`encoders.tar.gz`) live on the release named in
  `encoders.ENCODERS_RELEASE`; only a new encoder needs a new asset there (and a new sha256),
  and the tarball must include `NOTICE`
- See `CONTRIBUTING.md` for the PR checklist

## Code Style

- Line length 100; ruff (E, F, I, UP); mypy strict with the pydantic plugin.
- Build system: hatchling, src layout in each package.
