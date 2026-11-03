/* M5Stack Tab5 (ESP32-P4): ES8388 speaker codec and ES7210 microphone ADC on one I2S port, set up
 * by hand like the LiveKit ESP32 SDK "custom_hardware" example (codec_board does not know the board,
 * and the Tab5 BSP pulls in the display, camera and LVGL stacks). The speaker amplifier and the
 * ESP32-C6 that provides Wi-Fi are switched on through PI4IOE5V6408 I/O expanders. */
#include "sdkconfig.h"
#if CONFIG_LK_EXAMPLE_BOARD_M5STACK_TAB5

#include "board.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "driver/i2s_tdm.h"
#include "esp_check.h"
#if __has_include("esp_hosted.h")  // Wi-Fi through the ESP32-C6 (esp_hosted); optional
#include "esp_hosted.h"
#define TAB5_HOSTED 1
#endif
#include "esp_codec_dev_defaults.h"
#include "esp_io_expander_pi4ioe5v6408.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "board";

#define TAB5_I2C_SDA GPIO_NUM_31
#define TAB5_I2C_SCL GPIO_NUM_32
#define TAB5_I2S_MCLK GPIO_NUM_30
#define TAB5_I2S_BCLK GPIO_NUM_27
#define TAB5_I2S_WS GPIO_NUM_29
#define TAB5_I2S_DOUT GPIO_NUM_26  // -> ES8388
#define TAB5_I2S_DIN GPIO_NUM_28   // <- ES7210
#define TAB5_SPEAKER_EN IO_EXPANDER_PIN_NUM_1  // expander 0x43
#define TAB5_WIFI_EN IO_EXPANDER_PIN_NUM_0     // expander 0x44

// ES7210 in TDM, four slots, measured: MIC1, the playback reference (speaker loopback), MIC2, unused.
static const board_mic_t mic = {.channels = 4, .channel_mask = 1 | 2, .layout = "MR"};

static i2c_master_bus_handle_t i2c_bus;
static i2s_chan_handle_t i2s_tx, i2s_rx;
static esp_codec_dev_handle_t play_dev, rec_dev;

static esp_err_t set_expander_pin(uint32_t addr, uint32_t pin, bool level)
{
    esp_io_expander_handle_t io;
    ESP_RETURN_ON_ERROR(esp_io_expander_new_i2c_pi4ioe5v6408(i2c_bus, addr, &io), TAG, "expander");
    ESP_RETURN_ON_ERROR(esp_io_expander_set_dir(io, pin, IO_EXPANDER_OUTPUT), TAG, "dir");
    ESP_RETURN_ON_ERROR(esp_io_expander_set_output_mode(io, pin, IO_EXPANDER_OUTPUT_MODE_PUSH_PULL),
                        TAG, "mode");
    return esp_io_expander_set_level(io, pin, level);
}

static esp_err_t init_i2s(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, &i2s_tx, &i2s_rx), TAG, "i2s channel");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(32, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {.mclk = TAB5_I2S_MCLK, .bclk = TAB5_I2S_BCLK, .ws = TAB5_I2S_WS,
                     .dout = TAB5_I2S_DOUT, .din = TAB5_I2S_DIN},
    };
    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(i2s_tx, &std_cfg), TAG, "i2s tx");

    i2s_tdm_slot_mask_t slots = I2S_TDM_SLOT0 | I2S_TDM_SLOT1 | I2S_TDM_SLOT2 | I2S_TDM_SLOT3;
    i2s_tdm_config_t tdm_cfg = {
        .clk_cfg = I2S_TDM_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = I2S_TDM_PHILIPS_SLOT_DEFAULT_CONFIG(32, I2S_SLOT_MODE_STEREO, slots),
        .gpio_cfg = {.mclk = TAB5_I2S_MCLK, .bclk = TAB5_I2S_BCLK, .ws = TAB5_I2S_WS,
                     .dout = TAB5_I2S_DOUT, .din = TAB5_I2S_DIN},
    };
    tdm_cfg.slot_cfg.total_slot = 4;
    ESP_RETURN_ON_ERROR(i2s_channel_init_tdm_mode(i2s_rx, &tdm_cfg), TAG, "i2s rx");

    // The codecs need the clocks running while they are configured
    i2s_channel_enable(i2s_tx);
    i2s_channel_enable(i2s_rx);
    return ESP_OK;
}

static esp_err_t init_codecs(void)
{
    audio_codec_i2s_cfg_t i2s_cfg = {.port = I2S_NUM_0, .rx_handle = i2s_rx, .tx_handle = i2s_tx};
    const audio_codec_data_if_t *data = audio_codec_new_i2s_data(&i2s_cfg);
    ESP_RETURN_ON_FALSE(data, ESP_FAIL, TAG, "i2s data");

    audio_codec_i2c_cfg_t es8388_i2c = {.port = I2C_NUM_0, .addr = ES8388_CODEC_DEFAULT_ADDR,
                                        .bus_handle = i2c_bus};
    es8388_codec_cfg_t es8388 = {
        .ctrl_if = audio_codec_new_i2c_ctrl(&es8388_i2c),
        .gpio_if = audio_codec_new_gpio(),
        .codec_mode = ESP_CODEC_DEV_WORK_MODE_DAC,
        .pa_pin = -1,  // the amplifier is on the I/O expander
        .hw_gain = {.pa_voltage = 5.0, .codec_dac_voltage = 3.3},
    };
    ESP_RETURN_ON_FALSE(es8388.ctrl_if, ESP_FAIL, TAG, "ES8388 i2c");
    esp_codec_dev_cfg_t out = {.dev_type = ESP_CODEC_DEV_TYPE_OUT,
                               .codec_if = es8388_codec_new(&es8388), .data_if = data};
    ESP_RETURN_ON_FALSE(out.codec_if, ESP_FAIL, TAG, "ES8388");
    play_dev = esp_codec_dev_new(&out);

    audio_codec_i2c_cfg_t es7210_i2c = {.port = I2C_NUM_0, .addr = ES7210_CODEC_DEFAULT_ADDR,
                                        .bus_handle = i2c_bus};
    es7210_codec_cfg_t es7210 = {
        .ctrl_if = audio_codec_new_i2c_ctrl(&es7210_i2c),
        .mic_selected = ES7210_SEL_MIC1 | ES7210_SEL_MIC2 | ES7210_SEL_MIC3 | ES7210_SEL_MIC4,
    };
    ESP_RETURN_ON_FALSE(es7210.ctrl_if, ESP_FAIL, TAG, "ES7210 i2c");
    esp_codec_dev_cfg_t in = {.dev_type = ESP_CODEC_DEV_TYPE_IN,
                              .codec_if = es7210_codec_new(&es7210), .data_if = data};
    ESP_RETURN_ON_FALSE(in.codec_if, ESP_FAIL, TAG, "ES7210");
    rec_dev = esp_codec_dev_new(&in);

    ESP_RETURN_ON_FALSE(play_dev && rec_dev, ESP_FAIL, TAG, "codec devices");
    esp_codec_dev_set_in_gain(rec_dev, CONFIG_LK_EXAMPLE_MIC_GAIN_DB);
    return ESP_OK;
}

void board_init(void)
{
    ESP_LOGI(TAG, "Initializing M5Stack Tab5");
    i2c_master_bus_config_t i2c_cfg = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = TAB5_I2C_SDA,
        .scl_io_num = TAB5_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&i2c_cfg, &i2c_bus));
    ESP_ERROR_CHECK(set_expander_pin(ESP_IO_EXPANDER_I2C_PI4IOE5V6408_ADDRESS_LOW, TAB5_SPEAKER_EN,
                                     true));
    ESP_ERROR_CHECK(set_expander_pin(ESP_IO_EXPANDER_I2C_PI4IOE5V6408_ADDRESS_HIGH, TAB5_WIFI_EN,
                                     true));
    vTaskDelay(pdMS_TO_TICKS(100));
#ifdef TAB5_HOSTED
    // the ESP32-C6 is powered now: start the Wi-Fi link to it (auto-init would run before app_main,
    // with the C6 still off)
    if (esp_hosted_init() != 0) {
        ESP_LOGE(TAG, "esp_hosted_init failed: no Wi-Fi");
    }
#endif
    ESP_ERROR_CHECK(init_i2s());
    ESP_ERROR_CHECK(init_codecs());
}

esp_codec_dev_handle_t board_record_handle(void)
{
    return rec_dev;
}

esp_codec_dev_handle_t board_playback_handle(void)
{
    return play_dev;
}

const board_mic_t *board_mic(void)
{
    return &mic;
}

#endif
