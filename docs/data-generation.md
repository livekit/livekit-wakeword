# Data Generation Pipeline

The generation stage synthesizes positive and negative audio clips using a pluggable TTS backend (default: **Piper VITS** with SLERP speaker blending, or **VoxCPM2** voice design) and phoneme-based adversarial phrase generation. Select the engine with `tts_backend` in the YAML config (`piper_vits` or `voxcpm`). **Multilingual wake words require `tts_backend: voxcpm`**. Piper follows a single checkpoint locale (bundled default is English–US). See `configs/test_voxcpm.yaml` and `configs/prod_voxcpm.yaml` for Chinese (`你好 livekit`) examples.

**Source:** `src/livekit/wakeword/data/generate.py`, `src/livekit/wakeword/data/piper/synthesis.py`, `src/livekit/wakeword/data/tts/`
**CLI:** `livekit-wakeword generate <config>`

**System dependency:** `espeak-ng` must be installed for phonemization (`brew install espeak-ng` on macOS, `apt install espeak-ng` on Linux).

## Overview

```
Target phrases (e.g., "hey jarvis")
    │
    ├──► TTS backend ──► Positive clips (.wav)
    │    (Piper VITS+SLERP or VoxCPM2 voice design)
    │
    ├──► Adversarial phrase generation
    │    (phoneme substitution via CMU dict)
    │         │
    │         └──► TTS backend ──► Negative clips (.wav)
    │
    └──► Background noise sampling ──► Background clips (.wav)
         (random slicing + tiling from background_paths)
```

### Default Sample Counts

| Split | Count | Config Field |
|-------|-------|-------------|
| Positive train | 10,000 | `n_samples` |
| Positive test | 2,000 | `n_samples_val` |
| Negative train | 10,000 | `n_samples` |
| Negative test | 2,000 | `n_samples_val` |
| Background train | 200 | `n_background_samples` |
| Background test | 40 | `n_background_samples_val` |

## Setup and TTS artifacts

`livekit-wakeword setup` installs shared data (features, RIRs, backgrounds) under a root `data_dir`. Backend-specific TTS weights (Piper checkpoint or VoxCPM snapshot) are downloaded only for the backend your config selects.

### With `--config` / `-c`

Pass your wake word YAML so setup aligns with generation:

- **`data_dir`** from the config is the root for all downloads (features, RIRs, backgrounds, Piper checkpoint, VoxCPM snapshot).
- **Piper** is downloaded **only if** `tts_backend` is `piper_vits`. **VoxCPM** weights are fetched with `snapshot_download` **only if** `tts_backend` is `voxcpm` and the configured target directory is missing or empty (see **VoxCPM2** below). Otherwise TTS-specific downloads are skipped.
- **Checkpoint path:** `piper_tts.checkpoint_relpath` is the path to the `.pt` state_dict **relative to `data_dir`**. The matching JSON config is always `same_basename.json` next to that file. Defaults to `piper/en-us-libritts-high.pt`.

Example:

```bash
livekit-wakeword setup --config configs/prod.yaml
```

### Without `--config`

If you omit `--config`, setup uses `--data-dir` (default `./data`) and **always** downloads the default Piper bundle to `piper/en-us-libritts-high.pt` under that root. This is handy when you do not have a YAML yet. Prefer `--config` for projects that use a non-default `data_dir`, a custom `piper_tts.checkpoint_relpath`, or **VoxCPM** (VoxCPM is not downloaded without `--config`).

## Piper VITS TTS with SLERP Speaker Blending (`tts_backend: piper_vits`)

`synthesize_clips()` generates speech clips using a VITS model with SLERP (Spherical Linear Interpolation) across 904 speaker embeddings. This approach, adapted from [dscripka/piper-sample-generator](https://github.com/dscripka/piper-sample-generator), creates thousands of unique synthetic voices from speaker pairs.

### How Speaker Blending Works

The VITS model contains 904 speaker embeddings. For each batch:

1. **Speaker pairs** are cycled through all `(i, j)` combinations of speaker IDs
2. **SLERP** interpolates the two speaker embeddings at each configured weight (e.g., 0.2 = close to speaker 1, 0.5 = midpoint, 0.8 = close to speaker 2)
3. The blended embedding is used for inference, producing a voice that sounds like neither original speaker. Multiple weights generate more diverse voices from each pair
4. Audio is resampled from 22050 Hz to 16000 Hz and silence-trimmed via WebRTC VAD

With 904 speakers, there are ~409,000 unique speaker pairs, each producing distinct vocal characteristics.

SLERP uses per-element fallback: when a speaker pair's embeddings are nearly parallel (dot product > 0.9995), only that pair falls back to linear interpolation; other pairs in the batch still use true SLERP. Dot products are clamped to `[-1, 1]` to prevent NaN from `acos` on float precision overflow.

### Parameters

Each clip is synthesized with a combination of:

| Parameter | Default Values | Description |
|-----------|---------------|-------------|
| `noise_scales` | `[0.98]` | Overall speech variability |
| `noise_scale_ws` | `[0.98]` | Phoneme duration variability |
| `length_scales` | `[0.75, 1.0, 1.25]` | Speaking rate (slow/normal/fast) |
| `slerp_weights` | `[0.2, 0.35, 0.5, 0.65, 0.8]` | Speaker interpolation weights (0=speaker1, 1=speaker2) |
| `max_speakers` | `null` | Cap on speaker IDs (null = all 904) |

The Cartesian product of `slerp_weights`, `length_scales`, `noise_scales`, and `noise_scale_ws` creates multiple prosody variations. Speaker pairs and settings are cycled until `n_samples` clips are generated.

### Model artifacts

The default `en-us-libritts-high.pt` VITS checkpoint (~166 MB) and its `.json` config are downloaded by `livekit-wakeword setup` (see **Setup and TTS artifacts**). Paths follow `piper_tts.checkpoint_relpath` under `data_dir`. The model **must** be present for `piper_vits`. Generation raises `FileNotFoundError` if the checkpoint is missing instead of emitting silent placeholders.

### Silence Trimming

Generated audio is silence-trimmed via WebRTC VAD. If the VAD strips too aggressively (result shorter than 150ms of content beyond the leading samples), the original untrimmed audio is kept instead.

## VoxCPM2 (`tts_backend: voxcpm`)

[VoxCPM2](https://github.com/OpenBMB/VoxCPM) is used in **voice design** mode only: each clip uses a text persona description plus the wake phrase (no reference-audio cloning in this integration). The Python package is optional: install with `uv sync --extra train --extra voxcpm` (upstream recommends **PyTorch ≥ 2.5**; see their docs for CUDA).

**Weights on disk:** `livekit-wakeword setup --config your.yaml` runs `snapshot_download(repo_id=voxcpm_tts.model_id, ...)` into `voxcpm_local_model_path`. By default this is `data_dir/voxcpm/VoxCPM2` (`voxcpm_tts.model_cache_relpath`), or `voxcpm_tts.local_model_path` if set (relative to `data_dir` or absolute). If that directory is already non-empty, setup skips the download (e.g. you prefetched or copied weights there).

**Diversification:** Defaults cover many `voice_design_prompts` × `cfg_values` × `inference_timesteps_list` (see `VoxCpmTtsConfig` in `config.py`). Clip *i* cycles through that Cartesian product so resumes stay aligned with `start_index`. Output is **16 kHz** `clip_%06d.wav` (model native rate is resampled with librosa).

## Adversarial Phrase Generation

`generate_adversarial_phrases()` creates phonetically similar but incorrect phrases to train the model to reject near-misses. The resulting phrases are fed back through the active TTS backend to produce negative clips.

### Algorithm

1. Load the CMU Pronouncing Dictionary via NLTK
2. For each target phrase:
   - **Expand unknown words:** Words not in CMUDict are split into known subwords (e.g., `"livekit"` → `["live", "kit"]`). The split tries all positions and prefers the longest left match. This enables phoneme substitutions on made-up/compound words that CMUDict doesn't contain.
   - Get the phoneme sequence for each word (with regex stress wildcards on vowels)
   - Generate regex patterns by replacing 1 to `max_replace` phonemes (default: `len(phones) - 2`) with a wildcard `(.){1,3}`
   - Search CMUDict with each regex pattern via `pronouncing.search()` to find phonetically similar words
   - Exclude homophones (same pronunciation = not adversarial)
   - With probability `include_partial_phrase` (default: 1.0), generate all partial phrases (each word removed in turn)
   - Include individual words with probability `include_input_words` (default: 0.2)
3. **Remove exact target phrases** from the adversarial list (safety filter)
4. Deduplicate and shuffle. When `n_phrases` is `None` (default), all unique phrases are returned with no cap.

### Regex Phoneme Replacement

For each word, phonemes are replaced with a broad regex wildcard `(.){1,3}` that matches any 1-3 phoneme characters. All combinations of 1 to `max_replace` replacement positions are tried, generating patterns that range from single-phoneme swaps (close neighbors) to multi-phoneme replacements (more distant matches). This is the same approach used by openWakeWord: broad enough to catch phonetically similar words without requiring a hand-curated substitution map.

### Custom Negatives

Additional negative phrases can be specified via `custom_negative_phrases` in the config. These are appended to the auto-generated adversarial phrases before synthesis.

## Background Noise Clip Generation

`_generate_background_clips()` samples clips from the raw background audio in `augmentation.background_paths` and writes them as `.wav` files into `background_train/` and `background_test/`.

Each clip is generated by:

1. **Pick a random source file** from the background audio pool
2. **Tile short files**: if the source is shorter than `clip_duration`, it is extended by concatenating segments with random roll offsets and 50% chance of reversal per segment, breaking audible periodicity
3. **Random slice**: pick a random start offset into the (possibly tiled) audio and extract exactly `clip_duration` samples

This ensures every clip sounds different even from the same source file. Set `n_background_samples: 0` and `n_background_samples_val: 0` to disable background noise training entirely.

Background clips go through the same augmentation pipeline as TTS clips (RIR, EQ, distortion, and additional background noise mixing), so the model sees realistic acoustic conditions on pure noise.

## Error Handling

Errors during TTS synthesis are handled **per-batch**, not per-pipeline. If a batch fails (e.g., espeak-ng can't phonemize a phrase), that batch is skipped with a warning and generation continues. The pipeline raises `RuntimeError` if:

- **5 consecutive batches** fail (indicates a systemic problem)
- **Zero clips** are generated (total failure)

## Output Directory Structure

After generation, the output directory contains:

```
output/<model_name>/
├── positive_train/      # Positive training clips (.wav)
├── positive_test/       # Positive validation clips (.wav)
├── negative_train/      # Adversarial negative training clips (.wav)
├── negative_test/       # Adversarial negative validation clips (.wav)
├── background_train/    # Background noise training clips (.wav)
└── background_test/     # Background noise validation clips (.wav)
```

## Adding a new TTS backend

Synthetic speech is pluggable so you can swap Piper VITS for another engine (e.g. cloud or on-device models). Follow this checklist.

### 1. Config

- Add a new member to **`TtsBackend`** in [`config.py`](../src/livekit/wakeword/config.py) (e.g. `qwen_tts = "qwen_tts"`).
- Add any engine-specific fields: either nested models on `WakeWordConfig` (pattern: `PiperTtsConfig` + `piper_tts`) or top-level keys, and document them in YAML.
- Put shared path strings that **`config.py` must import** in [`tts_constants.py`](../src/livekit/wakeword/tts_constants.py), not under `livekit.wakeword.data`, so you avoid a circular import (`config` → `data` package → modules that import `config`).

### 2. Implementation class

- Add a module under [`data/`](../src/livekit/wakeword/data/) (e.g. `data/qwen_tts/`) with a class that satisfies **`SpeechSynthesizer`** in [`tts/backends.py`](../src/livekit/wakeword/data/tts/backends.py):

  - **`validate_artifacts()`**: ensure required weights, credentials, or binaries exist; raise `FileNotFoundError` with a clear message if not.
  - **`synthesize_clips(phrases, output_dir, n_samples, *, start_index, batch_size)`**: write **`clip_%06d.wav`** at **16 kHz**, honoring **`start_index`** for resume the same way Piper does.

- Put **text normalization** and **voice / speaker diversification** inside the backend (not in `run_generate`). Piper uses CMUDict + SLERP; another engine should apply its own diversity strategy so training clips are not single-timbre.

### 3. Registry

- In **`get_tts_backend()`** in [`tts/backends.py`](../src/livekit/wakeword/data/tts/backends.py), map your `TtsBackend` value to your class (typically `YourBackend.from_config(config)` so hyperparameters are captured at construction time).

### 4. Setup (optional)

- If the engine needs downloaded assets, extend **`setup`** in [`cli.py`](../src/livekit/wakeword/cli.py): when `--config` is passed, branch on `config.tts_backend` and download only what that backend needs, using paths from the config (same pattern as Piper + `piper_checkpoint_path`).

### 5. Tests and docs

- Add unit tests for config loading and, where feasible, artifact validation.
- Document YAML keys and setup steps in this file.
