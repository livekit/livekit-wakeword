/* Boards known to tempotian/codec_board (ESP32-S3-BOX-3, ESP32-S3-Korvo-2, ...), as in the LiveKit
 * ESP32 SDK "minimal" example. */
#include "sdkconfig.h"
#if CONFIG_LK_EXAMPLE_BOARD_CODEC_BOARD

#include "board.h"
#include "codec_board.h"
#include "codec_init.h"
#include "esp_log.h"

static const char *TAG = "board";

// ES7210 in TDM (menuconfig: the ESP32-S3-BOX-3 has a microphone and the playback reference first)
static const board_mic_t mic = {
    .channels = CONFIG_LK_EXAMPLE_MIC_CHANNELS,
    .channel_mask = CONFIG_LK_EXAMPLE_MIC_CHANNEL_MASK,
    .layout = CONFIG_LK_EXAMPLE_MIC_LAYOUT,
};

void board_init(void)
{
    ESP_LOGI(TAG, "Initializing board %s", CONFIG_LK_EXAMPLE_CODEC_BOARD_TYPE);
    set_codec_board_type(CONFIG_LK_EXAMPLE_CODEC_BOARD_TYPE);
    codec_init_cfg_t cfg = {
        .in_mode = CODEC_I2S_MODE_TDM,
        .in_use_tdm = true,
        .reuse_dev = false,
    };
    init_codec(&cfg);
}

esp_codec_dev_handle_t board_record_handle(void)
{
    return get_record_handle();
}

esp_codec_dev_handle_t board_playback_handle(void)
{
    return get_playback_handle();
}

const board_mic_t *board_mic(void)
{
    return &mic;
}

#endif
