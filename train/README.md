# Training (`livekit-wakeword-train`)

Setup (credentials, install) is in the [main README](../README.md#train-a-wake-word).

```bash
uv run livekit-wakeword setup --wakeword "hey livekit" --eval
uv run livekit-wakeword train --wakeword "hey livekit" --type standard   # or --type micro
uv run livekit-wakeword eval --wakeword "hey livekit"
```

or, in Python, `run_setup(config)`, `run_train(config, type=...)`, `run_eval(config, path)` from
`livekit.wakeword_train`, with `config = WakeWordConfig(wakeword="hey livekit")`.

## What each step does

1. **`setup`** checks the environment (Python 3.12+, LiveKit credentials, disk), downloads the
   pretrained encoders once (37 MB, checksum verified, cached under
   `~/.cache/livekit-wakeword/encoders`), writes the default config `<name>.yaml` if there is
   none, and generates speech for the phrase with
   [livekit-wakeword-bench](https://github.com/livekit/wakeword-bench) through LiveKit Inference
   TTS: the phrase in many voices, close-sounding confusables and everyday sentences. It is
   cached under `~/.cache/livekit-wakeword/datasets/<name>` (or `data_dir`). Later runs reuse it
   when clip counts match; changing the counts regenerates the affected training sets.
   `--eval` also generates the held-out test set.
2. **`train --type standard`** adapts the pretrained encoder `livekit-wakeword-encoder-standard`
   to the phrase, fine-tunes it on the generated speech (background speech streamed from public
   datasets, on-the-fly noise, reverberation, codec and microphone augmentation, quiet input
   and party scenes with several people talking included), calibrates it on held-out speech and on speech-free noise so a score of 0.5 means
   about `target_fpph` (default 0.5) false alarms per hour in both, and exports `<name>.onnx`: one
   stateful streaming model for the [clients](../client).
3. **`train --type micro`** does the same from `livekit-wakeword-encoder-micro`, with the phrase's
   standard model as its teacher (trained first if it does not exist), and exports
   `<name>_micro.tflite` (int8; needs the `tflite` extra) plus
   `<name>_micro.onnx`, for the [ESP32 client](../client/esp32).
4. **`eval`** scores a model on the phrase's held-out test set (unseen voices, noise and rooms,
   confusables, real conversational speech) and prints the bench score (recall at 0.5 false alarms
   per hour) and the numbers at the 0.5 threshold. `--media DIR` adds false alarms per hour on
   long-form TV, music and podcasts (`lkwb media -o DIR` builds that set once).

## Flags

Use `--wakeword` (or `-w`) to select the phrase for `setup`, `train` and `eval`. Run these commands
from the same directory so they can find the settings saved by `setup`. Each command has
`--help`, for example `uv run livekit-wakeword train --help`.

| Command | Option | Behavior |
|---|---|---|
| `setup` | `--language`, `-l` | Language of the phrase; defaults to `en` for a new config. |
| `setup` | `--plan` | TTS concurrency preset: `build` (5 requests), `ship` (20), or `scale` (50). Defaults to `build` for a new config. |
| `setup` | `--concurrency` | Concurrent TTS requests; overrides the plan's preset. |
| `setup` | `--clips N` | Multiply the default training clip counts by `N`; accepts decimals of at least `0.1`. |
| `setup` | `--eval` | Also generate the held-out test set. |
| `train` | `--type`, `-t` | `standard` (default) exports ONNX; `micro` exports int8 TFLite and a host ONNX model. |
| `train` | `--steps` | Override fine-tuning steps for the selected model type. Both types default to 5,000. |
| `train` | `--out`, `-o` | Directory for exported models; defaults to the saved `output_dir`, or the current directory for a new config. |
| `eval` | `--type`, `-t` | Select the standard (default) or micro model path from the config. |
| `eval` | `--model`, `-m` | Evaluate a specific model file instead of the path selected by `--type`. |
| `eval` | `--media DIR` | Also measure false alarms on long-form media prepared with `lkwb media`. |
| `eval` | `--json` | Print the full evaluation summary as JSON. |
| `listen` | Model file paths | Listen for one or more standard ONNX models. |
| `listen` | `--threshold`, `-t` | Detection score threshold; defaults to `0.5`. |

For example, generate four times the default training speech and write the model to `./models`:

```bash
uv run livekit-wakeword setup --wakeword "hey robot" --clips 4 --eval
uv run livekit-wakeword train --wakeword "hey robot" --out ./models
uv run livekit-wakeword eval --wakeword "hey robot" --model ./models/hey_robot.onnx
uv run --with pyaudio livekit-wakeword listen ./models/hey_robot.onnx
```

`--clips 4` sets `n_positive` to 2,000, `n_confusable` to 2,000 and `n_sentence` to 600, four times
the defaults of 500, 500 and 150. Positive clips come from two sets, or one doubled set when
plain-text generation is unavailable, giving 4,000 wake-word clips in total. The multiplier
always applies to the defaults, not the previous run's counts. It does not change the test set.

When the requested counts differ from the cached data, `setup` regenerates the affected training
sets and removes the saved standard training checkpoint, so micro training can build a new
teacher. Repeating `setup` with unchanged counts reuses the cached speech; omitting `--clips`
keeps the saved counts. More clips require more speech generation; use `eval` to measure their
effect on detection and false alarms.

`--out` and `--steps` apply only to that training command; they do not update the saved YAML.
After using `--out`, pass the exported file to `eval --model` and `listen`, as above.

## The config file

`setup` writes settings to `<name>.yaml` in the current directory, such as `hey_robot.yaml`.
Later commands load that file when you pass the same phrase. You can use the CLI without writing
YAML yourself. To load another file, use `--config PATH` (or `-c`) with `setup`, `train` or `eval`;
this option is supported but hidden from `--help`.

Edit the file to set options without a flag, such as `target_fpph` or `data_dir`, or to save defaults
for later commands. For example, setting `output_dir` here lets training and evaluation use the
same directory without additional flags.

| field | default | |
|---|---|---|
| `language` | `en` | TTS voices; phrases can be in any language and script |
| `plan` / `concurrency` | `build` / none | LiveKit plan sets concurrent TTS requests (build 5, ship 20, scale 50); `concurrency` overrides it |
| `n_positive`, `n_confusable`, `n_sentence` | 500, 500, 150 | generated clips per category (`--clips` scales all three; `n_positive` doubles for phrases with no plain-text positives, e.g. brand names) |
| `steps`, `micro_steps` | 5000, 5000 | fine-tuning steps |
| `target_fpph` | 0.5 | calibration target in false alarms per hour at score 0.5; measured rates can differ on other audio |
| `output_dir`, `data_dir` | `.`, the cache | where models and datasets go |

## Time

On one data-center GPU a new phrase takes about 11 minutes to a standard model: 6 minutes of
speech generation (at the `build` plan's 5 concurrent requests) and 5 of training, calibration
and export. A micro model adds about 7 minutes. The first run also downloads about 4 GB of noise,
room and speech sources once.

On an Apple silicon Mac, training runs on the GPU (MPS). On an M4 Pro a standard model takes
about 36 minutes after the same speech generation, and a micro model about 1.5 hours more
(0.87 s per training step, plus about 6 minutes for the TFLite export).

## Requirements

- A CUDA GPU or an Apple silicon Mac, and LiveKit credentials (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`;
  see [Get LiveKit credentials](../README.md#train-a-wake-word)).
- Python 3.12+ for dataset generation (`livekit-wakeword-bench`).
- Network access for the one-time encoder download from the GitHub release. To use your own copy
  instead, point `LIVEKIT_WAKEWORD_ENCODERS` at a directory with `standard/` and `micro/`.
