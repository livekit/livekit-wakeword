// Copyright 2026 LiveKit, Inc. Licensed under the Apache License, Version 2.0.
#include "livekit_wakeword.h"

#include <new>
#include <string.h>

#include "driver/i2s_std.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_memory_utils.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/stream_buffer.h"
#include "freertos/task.h"
#include "lkww_tflm.h"

static const char *TAG = "livekit_wakeword";

namespace {
enum Mode { kNone, kFeed, kCodecDev, kI2s };
constexpr size_t kFeedBufferBytes = LKWW_SAMPLE_RATE * sizeof(int16_t);  // 1 s
constexpr uint32_t kTaskStack = 6 * 1024;
constexpr size_t kCodecFrameSamples = 320;  // 20 ms per esp_codec_dev_read()
}  // namespace

struct lkww_detector {
    lkww_config_t cfg;
    uint8_t *arena = nullptr;
    uint8_t *model_copy = nullptr;  // the model in PSRAM, owned
    LkwwDetector *det = nullptr;
    int16_t pending[LKWW_STEP_SAMPLES];
    size_t n_pending = 0;
    int64_t last_detect_us = 0;
    bool detected_once = false;
    volatile float last_score = 0;
    volatile uint32_t last_step_us = 0;
    volatile uint32_t dropped = 0;
    volatile bool reset_requested = false;  // lkww_reset() while the inference task runs

    Mode mode = kNone;
    volatile bool running = false;
    TaskHandle_t task = nullptr;
    SemaphoreHandle_t done = nullptr;
    StreamBufferHandle_t feed = nullptr;
    lkww_codec_dev_config_t dev = {};
    bool dev_opened = false;
    int16_t *dev_raw = nullptr;  // interleaved read buffer of the codec_dev task
    lkww_i2s_config_t mic = {};
    i2s_chan_handle_t rx = nullptr;
};

static void run_step(lkww_detector *ww)
{
    const int64_t t0 = esp_timer_get_time();
    const float score = ww->det->Process(ww->pending);
    const int64_t now = esp_timer_get_time();
    ww->last_step_us = (uint32_t)(now - t0);
    ww->last_score = score;
    if (score < ww->cfg.threshold) {
        return;
    }
    const int64_t refractory = (int64_t)ww->cfg.refractory_ms * 1000;
    if (ww->detected_once && now - ww->last_detect_us < refractory) {
        return;
    }
    ww->detected_once = true;
    ww->last_detect_us = now;
    if (ww->cfg.on_detect) {
        ww->cfg.on_detect(score, ww->cfg.user_ctx);
    }
}

/* Runs on the thread that steps the model, so the reset never lands mid-step. */
static void apply_reset(lkww_detector *ww)
{
    if (ww->reset_requested) {
        ww->reset_requested = false;
        ww->det->Reset();
        ww->n_pending = 0;
    }
}

static float push(lkww_detector *ww, const int16_t *pcm, size_t n)
{
    float best = 0;
    while (n > 0) {
        apply_reset(ww);
        size_t k = LKWW_STEP_SAMPLES - ww->n_pending;
        if (k > n) {
            k = n;
        }
        memcpy(ww->pending + ww->n_pending, pcm, k * sizeof(int16_t));
        ww->n_pending += k;
        pcm += k;
        n -= k;
        if (ww->n_pending == LKWW_STEP_SAMPLES) {
            run_step(ww);
            ww->n_pending = 0;
            if (ww->last_score > best) {
                best = ww->last_score;
            }
        }
    }
    return best;
}

/* Arena size for `model`: load it once in the largest arena available (up to kProbeArenaMax) and
 * read what TensorFlow Lite Micro used. Nothing is allocated in the arena after AllocateTensors,
 * so that much (plus alignment slack) is enough. 0 if the model does not load. */
static size_t measure_arena(const uint8_t *model, uint32_t caps)
{
    constexpr size_t kProbeArenaMax = 256 * 1024, kSlack = 256;
    size_t probe = heap_caps_get_largest_free_block(caps);
    probe = probe > kProbeArenaMax ? kProbeArenaMax : probe;
    probe = probe > 1024 ? (probe - 1024) & ~size_t{15} : 0;
    uint8_t *arena = probe ? (uint8_t *)heap_caps_aligned_alloc(16, probe, caps) : nullptr;
    if (arena == nullptr) {
        ESP_LOGE(TAG, "no memory to size the arena (largest free block %u bytes)",
                 (unsigned)heap_caps_get_largest_free_block(caps));
        return 0;
    }
    LkwwDetector *det = new (std::nothrow) LkwwDetector(model, arena, probe, /*warm=*/false);
    const size_t used = det != nullptr && det->ok() ? det->arena_used() : 0;
    delete det;
    heap_caps_free(arena);
    if (used == 0) {
        ESP_LOGE(TAG, "model failed to load in a %u byte arena (wrong file?)", (unsigned)probe);
        return 0;
    }
    return (used + kSlack + 15) & ~size_t{15};
}

esp_err_t lkww_create(const lkww_config_t *cfg, lkww_handle_t *out)
{
    if (cfg == nullptr || out == nullptr || cfg->model == nullptr || cfg->model_size == 0) {
        ESP_LOGE(TAG, "no model: set the model path in menuconfig (livekit-wakeword) or pass cfg.model");
        return ESP_ERR_INVALID_ARG;
    }
    lkww_detector *ww = new (std::nothrow) lkww_detector();
    if (ww == nullptr) {
        return ESP_ERR_NO_MEM;
    }
    ww->cfg = *cfg;
#if CONFIG_SPIRAM
    // The weights are read every step and PSRAM is faster than flash (about 25% shorter steps on
    // an ESP32-P4): a model in flash runs from a copy when there is PSRAM for it.
    if (!esp_ptr_external_ram(cfg->model) && !esp_ptr_internal(cfg->model)) {
        ww->model_copy = (uint8_t *)heap_caps_aligned_alloc(16, cfg->model_size, MALLOC_CAP_SPIRAM);
        if (ww->model_copy != nullptr) {
            memcpy(ww->model_copy, cfg->model, cfg->model_size);
            ww->cfg.model = ww->model_copy;
        }
    }
#endif
    cfg = &ww->cfg;
    const uint32_t caps = (cfg->arena_in_psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT;
    size_t arena_bytes = cfg->arena_bytes;
    if (arena_bytes == 0) {
        arena_bytes = measure_arena(cfg->model, caps);
        if (arena_bytes == 0) {
            heap_caps_free(ww->model_copy);
            delete ww;
            return ESP_ERR_NO_MEM;
        }
    }
    ww->cfg.arena_bytes = arena_bytes;
    ww->arena = (uint8_t *)heap_caps_aligned_alloc(16, arena_bytes, caps);
    if (ww->arena == nullptr) {
        ESP_LOGE(TAG, "cannot allocate a %u byte arena", (unsigned)arena_bytes);
        heap_caps_free(ww->model_copy);
        delete ww;
        return ESP_ERR_NO_MEM;
    }
    ww->det = new (std::nothrow) LkwwDetector(cfg->model, ww->arena, arena_bytes);
    if (ww->det == nullptr || !ww->det->ok()) {
        ESP_LOGE(TAG, "model failed to load (wrong file, or arena too small: try a larger LKWW_ARENA_KB, or 0)");
        delete ww->det;
        heap_caps_free(ww->arena);
        heap_caps_free(ww->model_copy);
        delete ww;
        return ESP_FAIL;
    }
    ww->done = xSemaphoreCreateBinary();
    ESP_LOGI(TAG, "model loaded: %u bytes, arena %u/%u bytes used, %d state tensors",
             (unsigned)cfg->model_size, (unsigned)ww->det->arena_used(), (unsigned)arena_bytes,
             ww->det->num_states());
    *out = ww;
    return ESP_OK;
}

void lkww_destroy(lkww_handle_t ww)
{
    if (ww == nullptr) {
        return;
    }
    lkww_stop(ww);
    if (ww->feed) {
        vStreamBufferDelete(ww->feed);
    }
    if (ww->done) {
        vSemaphoreDelete(ww->done);
    }
    delete ww->det;
    heap_caps_free(ww->arena);
    heap_caps_free(ww->model_copy);
    delete ww;
}

esp_err_t lkww_process(lkww_handle_t ww, const int16_t *pcm, size_t samples, float *max_score)
{
    if (ww == nullptr || (pcm == nullptr && samples > 0)) {
        return ESP_ERR_INVALID_ARG;
    }
    const float best = push(ww, pcm, samples);
    if (max_score) {
        *max_score = best;
    }
    return ESP_OK;
}

void lkww_reset(lkww_handle_t ww)
{
    if (ww == nullptr) {
        return;
    }
    ww->reset_requested = true;
    if (!ww->running) {
        apply_reset(ww);  // no inference task: reset now
    }  // else the inference task resets before its next samples (also from on_detect)
}

float lkww_last_score(lkww_handle_t ww) { return ww ? ww->last_score : 0.0f; }
uint32_t lkww_last_step_us(lkww_handle_t ww) { return ww ? ww->last_step_us : 0; }
uint32_t lkww_dropped_samples(lkww_handle_t ww) { return ww ? ww->dropped : 0; }

static void finish_task(lkww_detector *ww)
{
    xSemaphoreGive(ww->done);
    vTaskDelete(nullptr);
}

static void feed_task(void *arg)
{
    lkww_detector *ww = (lkww_detector *)arg;
    int16_t buf[LKWW_STEP_SAMPLES];
    while (ww->running) {
        size_t got = xStreamBufferReceive(ww->feed, buf, sizeof(buf), pdMS_TO_TICKS(100));
        if (got > 0) {
            push(ww, buf, got / sizeof(int16_t));
        }
    }
    finish_task(ww);
}

/* Interleaved channels esp_codec_dev_read() returns: with a mask, only the selected ones. */
static size_t read_channels(const lkww_codec_dev_config_t &dev)
{
    return dev.channel_mask ? (size_t)__builtin_popcount(dev.channel_mask) : dev.channels;
}

static void codec_dev_task(void *arg)
{
    lkww_detector *ww = (lkww_detector *)arg;
    const lkww_codec_dev_config_t &dev = ww->dev;
    const size_t frame_samples = kCodecFrameSamples;
    const size_t stride = read_channels(dev);  // the driver drops channels outside the mask
    int16_t *raw = ww->dev_raw;  // allocated by lkww_start_codec_dev: this task never exits early
    int16_t mono[frame_samples];
    while (ww->running) {
        int ret = esp_codec_dev_read(dev.handle, raw, frame_samples * stride * sizeof(int16_t));
        if (ret != ESP_CODEC_DEV_OK) {
            ESP_LOGW(TAG, "esp_codec_dev_read: %d", ret);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        for (size_t i = 0; i < frame_samples; i++) {
            mono[i] = raw[i * stride + dev.channel];
        }
        push(ww, mono, frame_samples);
    }
    finish_task(ww);
}

static void i2s_task(void *arg)
{
    lkww_detector *ww = (lkww_detector *)arg;
    const size_t frame_samples = 320;
    int32_t raw[frame_samples];
    int16_t pcm[frame_samples];
    const int shift = 16 - ww->mic.gain_shift;  // 24-bit sample in the top bits of a 32-bit slot
    while (ww->running) {
        size_t got = 0;
        if (i2s_channel_read(ww->rx, raw, sizeof(raw), &got, pdMS_TO_TICKS(200)) != ESP_OK) {
            continue;
        }
        const size_t n = got / sizeof(int32_t);
        for (size_t i = 0; i < n; i++) {
            int32_t s = raw[i] >> shift;
            pcm[i] = (int16_t)(s > 32767 ? 32767 : (s < -32768 ? -32768 : s));
        }
        push(ww, pcm, n);
    }
    finish_task(ww);
}

static esp_err_t start_task(lkww_detector *ww, Mode mode, TaskFunction_t fn, const char *name)
{
    if (ww->running) {
        return ESP_ERR_INVALID_STATE;
    }
    ww->mode = mode;
    ww->running = true;
    BaseType_t ok = CONFIG_LKWW_TASK_CORE < 0
        ? xTaskCreate(fn, name, kTaskStack, ww, CONFIG_LKWW_TASK_PRIORITY, &ww->task)
        : xTaskCreatePinnedToCore(fn, name, kTaskStack, ww, CONFIG_LKWW_TASK_PRIORITY, &ww->task,
                                  CONFIG_LKWW_TASK_CORE);
    if (ok != pdPASS) {
        ww->running = false;
        ww->mode = kNone;
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t lkww_feed(lkww_handle_t ww, const int16_t *pcm, size_t frames, uint8_t channels, uint8_t channel)
{
    if (ww == nullptr || channels == 0 || channel >= channels) {
        return ESP_ERR_INVALID_ARG;
    }
    if (ww->mode != kFeed) {
        if (ww->mode != kNone) {
            return ESP_ERR_INVALID_STATE;
        }
        if (ww->feed == nullptr) {
            ww->feed = xStreamBufferCreate(kFeedBufferBytes, sizeof(int16_t));
            if (ww->feed == nullptr) {
                return ESP_ERR_NO_MEM;
            }
        }
        esp_err_t err = start_task(ww, kFeed, feed_task, "lkww_feed");
        if (err != ESP_OK) {
            return err;
        }
    }
    int16_t tmp[160];
    while (frames > 0) {
        size_t k = frames < 160 ? frames : 160;
        const int16_t *src = pcm;
        if (channels > 1) {
            for (size_t i = 0; i < k; i++) {
                tmp[i] = pcm[i * channels + channel];
            }
            src = tmp;
        }
        size_t sent = xStreamBufferSend(ww->feed, src, k * sizeof(int16_t), 0);
        ww->dropped += (uint32_t)(k - sent / sizeof(int16_t));
        pcm += k * channels;
        frames -= k;
    }
    return ESP_OK;
}

esp_err_t lkww_start_codec_dev(lkww_handle_t ww, const lkww_codec_dev_config_t *dev)
{
    if (ww == nullptr || dev == nullptr || dev->handle == nullptr || dev->channels == 0 ||
        (dev->channel_mask >> dev->channels) != 0 || dev->channel >= read_channels(*dev)) {
        return ESP_ERR_INVALID_ARG;
    }
    if (ww->running) {  // before touching ww->dev, which the running task reads
        return ESP_ERR_INVALID_STATE;
    }
    int16_t *raw = (int16_t *)malloc(kCodecFrameSamples * read_channels(*dev) * sizeof(int16_t));
    if (raw == nullptr) {
        return ESP_ERR_NO_MEM;
    }
    if (!dev->skip_open) {
        esp_codec_dev_sample_info_t fs = {};
        fs.sample_rate = LKWW_SAMPLE_RATE;
        fs.channel = dev->channels;
        fs.bits_per_sample = 16;
        fs.channel_mask = dev->channel_mask;
        if (esp_codec_dev_open(dev->handle, &fs) != ESP_CODEC_DEV_OK) {
            ESP_LOGE(TAG, "esp_codec_dev_open failed");
            free(raw);
            return ESP_FAIL;
        }
    }
    ww->dev = *dev;
    ww->dev_opened = !dev->skip_open;
    ww->dev_raw = raw;
    esp_err_t err = start_task(ww, kCodecDev, codec_dev_task, "lkww_codec");
    if (err != ESP_OK) {
        if (ww->dev_opened) {
            esp_codec_dev_close(dev->handle);
            ww->dev_opened = false;
        }
        free(ww->dev_raw);
        ww->dev_raw = nullptr;
    }
    return err;
}

esp_err_t lkww_start_i2s(lkww_handle_t ww, const lkww_i2s_config_t *mic)
{
    if (ww == nullptr || mic == nullptr || mic->gain_shift < 0 || mic->gain_shift > 8) {
        return ESP_ERR_INVALID_ARG;
    }
    if (ww->running) {  // before creating a channel over the running one's ww->rx
        return ESP_ERR_INVALID_STATE;
    }
    i2s_chan_config_t chan = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan, nullptr, &ww->rx), TAG, "i2s_new_channel");
    i2s_std_config_t std = {};
    std.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(LKWW_SAMPLE_RATE);
    std.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
    std.slot_cfg.slot_mask = mic->right_slot ? I2S_STD_SLOT_RIGHT : I2S_STD_SLOT_LEFT;
    std.gpio_cfg.mclk = I2S_GPIO_UNUSED;
    std.gpio_cfg.bclk = mic->bclk;
    std.gpio_cfg.ws = mic->ws;
    std.gpio_cfg.dout = I2S_GPIO_UNUSED;
    std.gpio_cfg.din = mic->din;
    esp_err_t err = i2s_channel_init_std_mode(ww->rx, &std);
    if (err == ESP_OK) {
        err = i2s_channel_enable(ww->rx);
    }
    if (err != ESP_OK) {
        i2s_del_channel(ww->rx);
        ww->rx = nullptr;
        return err;
    }
    ww->mic = *mic;
    err = start_task(ww, kI2s, i2s_task, "lkww_i2s");
    if (err != ESP_OK) {
        i2s_channel_disable(ww->rx);
        i2s_del_channel(ww->rx);
        ww->rx = nullptr;
    }
    return err;
}

esp_err_t lkww_stop(lkww_handle_t ww)
{
    if (ww == nullptr) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!ww->running) {
        return ESP_OK;
    }
    ww->running = false;
    xSemaphoreTake(ww->done, portMAX_DELAY);
    if (ww->mode == kCodecDev) {
        if (ww->dev_opened) {
            esp_codec_dev_close(ww->dev.handle);
            ww->dev_opened = false;
        }
        free(ww->dev_raw);
        ww->dev_raw = nullptr;
    }
    if (ww->mode == kFeed && ww->feed) {
        xStreamBufferReset(ww->feed);  // no stale audio when lkww_feed() starts it again
    }
    if (ww->mode == kI2s && ww->rx) {
        i2s_channel_disable(ww->rx);
        i2s_del_channel(ww->rx);
        ww->rx = nullptr;
    }
    ww->task = nullptr;
    ww->mode = kNone;
    return ESP_OK;
}
