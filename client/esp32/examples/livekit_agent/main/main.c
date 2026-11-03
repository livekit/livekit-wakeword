/* Wake word -> LiveKit agent session -> listen again. Based on the LiveKit ESP32 SDK "minimal"
 * example (board.c, media.c). */
#include <string.h>

#include "board.h"
#include "esp_log.h"
#include "esp_netif_sntp.h"
#include "example.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "livekit.h"
#include "livekit_example_utils.h"
#include "livekit_wakeword.h"
#include "media.h"

static const char *TAG = "main";
static TaskHandle_t main_task;

static void on_wake(float score, void *ctx)
{
    ESP_LOGI(TAG, "wake word (score %.2f)", score);
    xTaskNotifyGive(main_task);
}

/* Listen to the same microphone, opened the same way, as the capturer the SDK publishes from. */
static void listen(lkww_handle_t ww)
{
    const esp_capture_audio_aec_src_cfg_t *mic = media_get_mic_config();
    lkww_codec_dev_config_t dev = LKWW_CODEC_DEV_DEFAULT_CONFIG(mic->record_handle);
    dev.channels = mic->channel;
    dev.channel_mask = mic->channel_mask;
    dev.channel = (uint8_t)(strchr(mic->mic_layout, 'M') - mic->mic_layout);
    lkww_reset(ww);
    ESP_ERROR_CHECK(lkww_start_codec_dev(ww, &dev));
    ESP_LOGI(TAG, "listening for the wake word");
}

void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_INFO);
    main_task = xTaskGetCurrentTaskHandle();

    livekit_system_init();
    board_init();
    media_init();
    esp_sntp_config_t sntp_config = ESP_NETIF_SNTP_DEFAULT_CONFIG_MULTIPLE(2,
        ESP_SNTP_SERVER_LIST("time.google.com", "pool.ntp.org"));
    esp_netif_sntp_init(&sntp_config);
    if (!lk_example_network_connect()) {
        ESP_LOGE(TAG, "no network");
        return;
    }

    lkww_config_t cfg = LKWW_DEFAULT_CONFIG();
    cfg.on_detect = on_wake;
    lkww_handle_t ww;
    ESP_ERROR_CHECK(lkww_create(&cfg, &ww));

    while (true) {
        listen(ww);
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        ESP_ERROR_CHECK(lkww_stop(ww));  // release the microphone for the SDK
        if (join_room()) {
            while (!room_ended()) {
                vTaskDelay(pdMS_TO_TICKS(200));
            }
        }
        leave_room();
        ulTaskNotifyTake(pdTRUE, 0);  // drop a detection that raced the session
    }
}
