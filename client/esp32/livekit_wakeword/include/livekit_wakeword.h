/*
 * Copyright 2026 LiveKit, Inc. Licensed under the Apache License, Version 2.0.
 *
 * On-device wake word detection with livekit-wakeword micro models. Audio is 16 kHz mono int16;
 * see README.md for the microphone pipelines.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "driver/gpio.h"
#include "esp_codec_dev.h"
#include "esp_err.h"
#include "sdkconfig.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LKWW_SAMPLE_RATE 16000
#define LKWW_STEP_SAMPLES 1280 /*!< the model runs once per 80 ms of audio */

typedef struct lkww_detector *lkww_handle_t;

/** Called from the inference task when the wake word is detected. Keep it short. */
typedef void (*lkww_detect_cb_t)(float score, void *user_ctx);

typedef struct {
    const uint8_t *model;     /*!< <name>_micro.tflite in memory (flash is fine), must outlive the detector */
    size_t model_size;        /*!< bytes */
    float threshold;          /*!< detection threshold, 0..1 (0.5: about 0.5 false alarms / hour) */
    uint32_t refractory_ms;   /*!< ignore new detections for this long after one */
    size_t arena_bytes;       /*!< TensorFlow Lite Micro working memory, 0: measured at create */
    bool arena_in_psram;      /*!< put the arena in PSRAM (slower, saves internal SRAM) */
    lkww_detect_cb_t on_detect;
    void *user_ctx;
} lkww_config_t;

#if defined(LKWW_HAS_EMBEDDED_MODEL)
extern const uint8_t _binary_lkww_model_tflite_start[];
extern const uint8_t _binary_lkww_model_tflite_end[];
#define LKWW_EMBEDDED_MODEL _binary_lkww_model_tflite_start
#define LKWW_EMBEDDED_MODEL_SIZE ((size_t)(_binary_lkww_model_tflite_end - _binary_lkww_model_tflite_start))
#else
#define LKWW_EMBEDDED_MODEL NULL
#define LKWW_EMBEDDED_MODEL_SIZE 0
#endif

#ifdef CONFIG_LKWW_ARENA_IN_PSRAM
#define LKWW_DEFAULT_ARENA_IN_PSRAM true
#else
#define LKWW_DEFAULT_ARENA_IN_PSRAM false
#endif

/** Embedded model (menuconfig), threshold 0.5, 2 s refractory period. */
#define LKWW_DEFAULT_CONFIG() {                                \
    .model = LKWW_EMBEDDED_MODEL,                              \
    .model_size = LKWW_EMBEDDED_MODEL_SIZE,                    \
    .threshold = 0.5f,                                         \
    .refractory_ms = 2000,                                     \
    .arena_bytes = CONFIG_LKWW_ARENA_KB * 1024,                \
    .arena_in_psram = LKWW_DEFAULT_ARENA_IN_PSRAM,             \
    .on_detect = NULL,                                         \
    .user_ctx = NULL,                                          \
}

/** Load the model and allocate its memory. */
esp_err_t lkww_create(const lkww_config_t *cfg, lkww_handle_t *out);

/** Stop any pipeline and free everything. */
void lkww_destroy(lkww_handle_t ww);

/** Run detection synchronously; max_score (optional) gets the best score of the completed steps.
 * Do not mix with a running pipeline or lkww_feed(). */
esp_err_t lkww_process(lkww_handle_t ww, const int16_t *pcm, size_t samples, float *max_score);

/** Queue audio for the inference task (non-blocking, 1 s buffer; overflow is dropped and counted).
 * `channels` > 1 means interleaved input; `channel` is the one used. */
esp_err_t lkww_feed(lkww_handle_t ww, const int16_t *pcm, size_t frames, uint8_t channels,
                    uint8_t channel);

/** Forget everything heard so far. Safe while a pipeline runs (and from on_detect): the inference
 * task then resets before its next samples. */
void lkww_reset(lkww_handle_t ww);

/** Latest score (0..1) and the duration of the last 80 ms step. */
float lkww_last_score(lkww_handle_t ww);
uint32_t lkww_last_step_us(lkww_handle_t ww);

/** Samples lkww_feed() had to drop because inference fell behind. */
uint32_t lkww_dropped_samples(lkww_handle_t ww);

typedef struct {
    esp_codec_dev_handle_t handle; /*!< record device (ESP_CODEC_DEV_TYPE_IN) */
    uint8_t channels;              /*!< interleaved channels to open, e.g. 2 or 4 for an ES7210 in TDM */
    uint16_t channel_mask;         /*!< passed to esp_codec_dev_open(), 0 for all; the driver then
                                        returns only the selected channels */
    uint8_t channel;               /*!< channel to listen on: an index into the channels read, so
                                        into the selected ones when channel_mask is set */
    /* Same fields as esp_capture_audio_aec_src_cfg_t: copy record_handle, channel and channel_mask
     * from the config given to the LiveKit SDK, and set channel to the 'M' in its mic_layout. */
    bool skip_open;                /*!< true if you already opened the device at 16 kHz, 16-bit */
} lkww_codec_dev_config_t;

#define LKWW_CODEC_DEV_DEFAULT_CONFIG(record_handle) { \
    .handle = (record_handle),                         \
    .channels = 1,                                     \
    .channel_mask = 0,                                 \
    .channel = 0,                                      \
    .skip_open = false,                                \
}

/** Listen on an esp_codec_dev record device (exclusively; lkww_stop() releases it).
 * ESP_ERR_INVALID_STATE if a pipeline is already running: lkww_stop() it first. */
esp_err_t lkww_start_codec_dev(lkww_handle_t ww, const lkww_codec_dev_config_t *dev);

typedef struct {
    gpio_num_t bclk;    /*!< serial clock (SCK) */
    gpio_num_t ws;      /*!< word select (WS / LRCL) */
    gpio_num_t din;     /*!< data from the microphone (SD / DOUT) */
    bool right_slot;    /*!< true if the mic's L/R pin selects the right slot */
    int gain_shift;     /*!< left shift applied to the 24-bit sample, 0..8 (mic sensitivity) */
} lkww_i2s_config_t;

#define LKWW_I2S_DEFAULT_CONFIG(bclk_pin, ws_pin, din_pin) { \
    .bclk = (bclk_pin),                                    \
    .ws = (ws_pin),                                        \
    .din = (din_pin),                                      \
    .right_slot = false,                                   \
    .gain_shift = 2,                                       \
}

/** Listen on a standard-mode I2S MEMS microphone (32-bit slots, 24-bit samples).
 * ESP_ERR_INVALID_STATE if a pipeline is already running: lkww_stop() it first. */
esp_err_t lkww_start_i2s(lkww_handle_t ww, const lkww_i2s_config_t *mic);

/** Stop a running pipeline and close a record device opened by lkww_start_codec_dev(). */
esp_err_t lkww_stop(lkww_handle_t ww);

#ifdef __cplusplus
}
#endif
