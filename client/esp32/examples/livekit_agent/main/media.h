
#pragma once

#include "esp_capture.h"
#include "esp_capture_defaults.h"
#include "av_render.h"

#ifdef __cplusplus
extern "C" {
#endif

int media_init(void);
esp_capture_handle_t media_get_capturer(void);
/// The microphone configuration the capturer was built with (record device, channels, layout).
const esp_capture_audio_aec_src_cfg_t *media_get_mic_config(void);
av_render_handle_t media_get_renderer(void);

#ifdef __cplusplus
}
#endif

