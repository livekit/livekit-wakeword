/* Wake word on an I2S MEMS microphone (INMP441, ICS-43434, ...).
 * Wiring (change below): SCK -> GPIO 4, WS -> GPIO 5, SD -> GPIO 6, L/R -> GND, VDD -> 3.3 V. */
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "livekit_wakeword.h"

static const char *TAG = "example";

static void on_wake(float score, void *ctx)
{
    ESP_LOGI(TAG, "wake word! (score %.2f)", score);
}

void app_main(void)
{
    lkww_config_t cfg = LKWW_DEFAULT_CONFIG();
    cfg.on_detect = on_wake;
    lkww_handle_t ww;
    ESP_ERROR_CHECK(lkww_create(&cfg, &ww));

    lkww_i2s_config_t mic = LKWW_I2S_DEFAULT_CONFIG(GPIO_NUM_4, GPIO_NUM_5, GPIO_NUM_6);
    ESP_ERROR_CHECK(lkww_start_i2s(ww, &mic));

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        ESP_LOGI(TAG, "score %.2f, step %u us (budget 80000)", lkww_last_score(ww),
                 (unsigned)lkww_last_step_us(ww));
    }
}
