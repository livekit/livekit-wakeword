#pragma once

#include <stdint.h>

#include "esp_codec_dev.h"

#ifdef __cplusplus
extern "C" {
#endif

/// How the microphones come out of the record device (see esp_capture_audio_aec_src_cfg_t).
typedef struct {
    uint8_t channels;      ///< interleaved channels to open
    uint8_t channel_mask;  ///< channels kept (0: all)
    const char *layout;    ///< one letter per kept channel: M mic, R playback reference, N unused
} board_mic_t;

/// Bring up the audio codecs (and anything the board needs powered for them or the network).
void board_init(void);

esp_codec_dev_handle_t board_record_handle(void);
esp_codec_dev_handle_t board_playback_handle(void);
const board_mic_t *board_mic(void);

#ifdef __cplusplus
}
#endif
