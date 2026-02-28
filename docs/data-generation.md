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
2. **SLERP** interpolates the two speaker embeddings at the configured weight (e.g., 0.5 = midpoint)
3. The blended embedding is used for inference, producing a voice that sounds like neither original speaker
4. Audio is resampled from 22050 Hz to 16000 Hz and silence-trimmed via WebRTC VAD

With 904 speakers, there are ~409,000 unique speaker pairs, each producing distinct vocal characteristics.

### Parameters

Each clip is synthesized with a combination of:

| Parameter | Default Values | Description |
|-----------|---------------|-------------|
| `noise_scales` | `[0.98]` | Overall speech variability |
| `noise_scale_ws` | `[0.98]` | Phoneme duration variability |
| `length_scales` | `[0.75, 1.0, 1.25]` | Speaking rate (slow/normal/fast) |
| `slerp_weights` | `[0.5]` | Speaker interpolation weights (0=speaker1, 1=speaker2) |
| `max_speakers` | `null` | Cap on speaker IDs (null = all 904) |

The Cartesian product of `slerp_weights`, `length_scales`, `noise_scales`, and `noise_scale_ws` creates multiple prosody variations. Speaker pairs and settings are cycled until `n_samples` clips are generated.

### TTS Model

The `en-us-libritts-high.pt` VITS checkpoint (~255 MB) and its `.json` config are downloaded during `livekit-wakeword setup` to `data/piper/`. If the model is missing or generation fails, 1-second silence placeholders are written and a warning is logged.

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
2. Build a reverse phoneme index: phoneme sequence → list of words
3. For each target phrase:
   - Look up the phoneme sequence for each word
   - For each phoneme, try substituting it with similar phonemes from `SIMILAR_PHONEMES`
   - Look up words matching the substituted phoneme sequence (up to 3 per substitution)
   - Generate partial phrases (one word removed at a time) with probability `include_partial_phrase` (default: 1.0)
   - Include individual words with probability `include_input_words` (default: 0.2)
4. Deduplicate, shuffle, and limit to `n_phrases` (default: 200)

### SIMILAR_PHONEMES Map

The `SIMILAR_PHONEMES` dictionary maps 43 ARPAbet phonemes to phonetically similar alternatives. Examples:

| Phoneme | Similar |
|---------|---------|
| AA | AH, AO, AE |
| IY | IH, EY |
| T | D, TH |
| S | Z, SH |
| K | G, T |

This ensures adversarial phrases sound close to the target but are distinct words.

### Custom Negatives

Additional negative phrases can be specified via `custom_negative_phrases` in the config. These are appended to the auto-generated adversarial phrases before synthesis.

## Clip Duration Calculation

`compute_clip_duration()` determines the target audio window length for augmentation:

1. Sample up to 50 WAV files from the positive training directory
2. Compute the median duration
3. Return `max(2.0, median_duration + 0.75)`

The 750ms buffer ensures the wake word fits within the detection window with room for alignment jitter. The minimum of 2.0 seconds prevents windows that are too short.

## Output Directory Structure

After generation, the output directory contains:

```
output/<model_name>/
├── positive_train/      # Positive training clips (.wav)
├── positive_test/       # Positive validation clips (.wav)
├── negative_train/      # Adversarial negative training clips (.wav)
└── negative_test/       # Adversarial negative validation clips (.wav)
```
