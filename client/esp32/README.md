# ESP32

On-device wake word detection for ESP32 chips: the `livekit_wakeword` ESP-IDF component runs the
micro models that [the trainer](../python/TRAINING.md) makes from a phrase (or the ones in the
[release](../../README.md#models)):

```bash
livekit-wakeword train --wakeword "hey livekit" --type micro    # on a GPU machine -> hey_livekit_micro.tflite
```

```c
#include "livekit_wakeword.h"

static void on_wake(float score, void *ctx) { printf("wake word! %.2f\n", score); }

void app_main(void)
{
    lkww_config_t cfg = LKWW_DEFAULT_CONFIG();     // model embedded via menuconfig
    cfg.on_detect = on_wake;
    lkww_handle_t ww;
    ESP_ERROR_CHECK(lkww_create(&cfg, &ww));

    lkww_codec_dev_config_t dev = LKWW_CODEC_DEV_DEFAULT_CONFIG(get_record_handle());
    ESP_ERROR_CHECK(lkww_start_codec_dev(ww, &dev));
}
```

| | |
|---|---|
| Model | BC-ResNet, 149K parameters, int8 weights and activations, ~450 KB in flash |
| RAM | ~146 KB internal SRAM in all: the TensorFlow Lite Micro arena (~129 KB, sized to the model at `lkww_create`) and ~17 KB of front end and detector state |
| Compute | one model step per 80 ms of audio (1.1 M multiply-accumulates): 29 ms on an ESP32-P4 at 360 MHz (model in PSRAM) |
| Targets | ESP32-S3, ESP32-P4 (built with ESP-IDF 5.5 and 6.1) |

## Install

With the component manager, in your `main/idf_component.yml`:

```yaml
dependencies:
  livekit/livekit_wakeword:
    git: https://github.com/livekit/livekit-wakeword.git
    path: client/esp32/livekit_wakeword
    version: v2.0.0  # a release tag (or a branch / commit)
```

Then set the model in `idf.py menuconfig` → **livekit-wakeword** → *Wake word model (.tflite) to
embed* (relative to the project). `LKWW_DEFAULT_CONFIG()` uses it; to load a model another way,
set `cfg.model` / `cfg.model_size`.

## Microphones

All audio is 16 kHz mono 16-bit. Pick the path that fits your hardware:

| API | Use it for |
|---|---|
| `lkww_start_codec_dev()` | An `esp_codec_dev` record device: codec boards (`tempotian/codec_board`, `get_record_handle()`), BSPs, ES7210 / ES8311. The same device type the LiveKit ESP32 SDK records from. Multi-channel devices (TDM) work: set `channels`, `channel_mask` and `channel` (an index into the channels the mask keeps), the fields of the `esp_capture_audio_aec_src_cfg_t` you give the SDK. |
| `lkww_start_i2s()` | An I2S MEMS microphone (INMP441, ICS-43434, SPH0645) on three GPIOs. |
| `lkww_feed()` | Audio you already capture (another driver, a network stream). It only copies into a 1 s ring buffer, and detection runs on the component's task, so it is safe from a capture callback. |
| `lkww_process()` | Synchronous, from your own task. |

Only one consumer can read an `esp_codec_dev` device. With the LiveKit SDK, listen while idle and
call `lkww_stop()` (which closes the device) before joining a room; listen again when it ends, as
[examples/livekit_agent](examples/livekit_agent) does.

## Detection

`on_detect(score, ctx)` runs on the inference task when the score reaches `threshold` (0.5 is
calibrated to about 0.5 false alarms per hour), at most once per `refractory_ms` (2 s). Keep it
short. `lkww_reset()` forgets the audio heard so far; `lkww_last_score()`, `lkww_last_step_us()`
(must stay well under 80 000 µs) and `lkww_dropped_samples()` are for monitoring.

## Configuration (menuconfig → livekit-wakeword)

| Option | Default | |
|---|---|---|
| `LKWW_MODEL_PATH` | empty | `.tflite` to embed |
| `LKWW_ARENA_KB` | 0 | TensorFlow Lite Micro working memory; 0 measures what the model needs at `lkww_create` (briefly taking the largest free block, up to 256 KB) |
| `LKWW_ARENA_IN_PSRAM` | n | Arena in PSRAM: saves internal SRAM, slower |
| `LKWW_TASK_CORE` | 1 | Core for the inference task (-1: any) |
| `LKWW_TASK_PRIORITY` | 5 | Inference task priority |

## Examples

- [examples/i2s_mic](examples/i2s_mic): an INMP441 on GPIO 4/5/6, prints detections and timing.
- [examples/livekit_agent](examples/livekit_agent): on the wake word, join a LiveKit room and talk
  to your agent; listen again when the session ends (LiveKit ESP32 SDK). ESP32-S3 boards through
  `codec_board` (ESP32-S3-BOX-3 by default, ESP32-S3-Korvo-2 as `S3_Korvo_V2`) and the M5Stack Tab5
  (`idf.py set-target esp32p4`).

Both embed [`models/hey_livekit_micro.tflite`](../../models). Build with ESP-IDF 5.5 or later:

```bash
cd client/esp32/examples/i2s_mic
idf.py set-target esp32s3 && idf.py build flash monitor
```

## Other microcontrollers

The component's `src/` is plain C99 plus a small C++ TensorFlow Lite Micro wrapper, so it ports to
any TFLM target. Every 80 ms (1280 samples at 16 kHz):

1. `lkww_frontend_process(&fe, pcm, mel)` (`lkww_micro.h`, `lkww_frontend.c`) turns the new samples
   into `mel` (8 x 40 floats). It matches the training front end to 1e-4. Set it up once with
   `lkww_frontend_init(&fe, floor, offset, scale)`, using the three float32 values in the model's
   `livekit_wakeword.mel_params` metadata.
2. Run the model on `mel` (1, 8, 40) and the states `s0..sK`.
3. Copy each new state (signature output `output_{i+1}`) into `s<i>`; `output_0` is the score.

To start or reset a stream, zero the states, call `lkww_frontend_reset`, then run 2 s of silence
(25 steps). `LkwwDetector` in `lkww_tflm.h` does all of this:

```cpp
#include "lkww_tflm.h"

alignas(16) static uint8_t arena[160 * 1024];
static LkwwDetector det(g_hey_livekit_micro_tflite, arena, sizeof(arena));

void on_audio_80ms(const int16_t* pcm /* 1280 samples */) {
  if (det.Process(pcm) > 0.5f) { /* wake word */ }
}
```

`tools/` has the host test (`host_test.cc`: a model and mel frames in, scores out), the front end
as a command line tool (`frontend_cli.c`, used by the Python tests) and the table generator
(`gen_tables.py` writes `lkww_tables.h`).
