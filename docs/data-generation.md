# Data Generation Pipeline

The generation stage synthesizes positive and negative audio clips using VITS TTS with SLERP speaker blending and phoneme-based adversarial phrase generation.

**Source:** `src/livekit/wakeword/data/generate.py`, `src/livekit/wakeword/data/_piper_generate.py`
**CLI:** `livekit-wakeword generate <config>`

**System dependency:** `espeak-ng` must be installed for phonemization (`brew install espeak-ng` on macOS, `apt install espeak-ng` on Linux).

## Overview

```
Target phrases (e.g., "hey jarvis")
    │
    ├──► VITS TTS + SLERP ──► Positive clips (.wav)
    │    (904 speakers × prosody variations)
    │
    └──► Adversarial phrase generation
         (phoneme substitution via CMU dict)
              │
              └──► VITS TTS + SLERP ──► Negative clips (.wav)
```

## VITS TTS Synthesis with SLERP Speaker Blending

`synthesize_clips()` generates speech clips using a VITS model with SLERP (Spherical Linear Interpolation) across 904 speaker embeddings. This approach, adapted from [dscripka/piper-sample-generator](https://github.com/dscripka/piper-sample-generator), creates thousands of unique synthetic voices from speaker pairs.

### How Speaker Blending Works

The VITS model contains 904 speaker embeddings. For each batch:

1. **Speaker pairs** are cycled through all `(i, j)` combinations of speaker IDs
2. **SLERP** interpolates the two speaker embeddings at each configured weight (e.g., 0.2 = close to speaker 1, 0.5 = midpoint, 0.8 = close to speaker 2)
3. The blended embedding is used for inference, producing a voice that sounds like neither original speaker. Multiple weights generate more diverse voices from each pair
4. Audio is resampled from 22050 Hz to 16000 Hz and silence-trimmed via WebRTC VAD

With 904 speakers, there are ~409,000 unique speaker pairs, each producing distinct vocal characteristics.

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

### TTS Model

The `en-us-libritts-high.pt` VITS checkpoint (~166 MB) and its `.json` config are downloaded during `livekit-wakeword setup` to `data/piper/`. The model **must** be present — generation will raise `FileNotFoundError` if the model is missing rather than producing silent placeholders.

### Error Handling

Errors during synthesis are handled **per-batch**, not per-pipeline. If a batch fails (e.g., espeak-ng can't phonemize a phrase), that batch is skipped with a warning and generation continues. The pipeline raises `RuntimeError` if:

- **5 consecutive batches** fail (indicates a systemic problem)
- **Zero clips** are generated (total failure)

### SLERP Speaker Blending

SLERP interpolation uses per-element fallback: when a speaker pair's embeddings are nearly parallel (dot product > 0.9995), only that pair falls back to linear interpolation — other pairs in the batch still use true SLERP. Dot products are clamped to `[-1, 1]` to prevent NaN from `acos` on float precision overflow.

### Silence Trimming

Generated audio is silence-trimmed via WebRTC VAD. If the VAD strips too aggressively (result shorter than 150ms of content beyond the leading samples), the original untrimmed audio is kept instead.

### Default Sample Counts

| Split | Count | Config Field |
|-------|-------|-------------|
| Positive train | 10,000 | `n_samples` |
| Positive test | 2,000 | `n_samples_val` |
| Negative train | 10,000 | `n_samples` |
| Negative test | 2,000 | `n_samples_val` |

## Adversarial Phrase Generation

`generate_adversarial_phrases()` creates phonetically similar but incorrect phrases to train the model to reject near-misses.

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

For each word, phonemes are replaced with a broad regex wildcard `(.){1,3}` that matches any 1-3 phoneme characters. All combinations of 1 to `max_replace` replacement positions are tried, generating patterns that range from single-phoneme swaps (close neighbors) to multi-phoneme replacements (more distant matches). This is the same approach used by openWakeWord — broad enough to catch phonetically similar words without requiring a hand-curated substitution map.

### Custom Negatives

Additional negative phrases can be specified via `custom_negative_phrases` in the config. These are appended to the auto-generated adversarial phrases before synthesis.

## Output Directory Structure

After generation, the output directory contains:

```
output/<model_name>/
├── positive_train/      # Positive training clips (.wav)
├── positive_test/       # Positive validation clips (.wav)
├── negative_train/      # Adversarial negative training clips (.wav)
└── negative_test/       # Adversarial negative validation clips (.wav)
```
