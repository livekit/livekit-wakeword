#include "esp_log.h"
#include "livekit.h"
#include "livekit_sandbox.h"
#include "media.h"
#include "board.h"
#include "example.h"

static const char *TAG = "livekit_example";

static livekit_room_handle_t room_handle;
static volatile bool session_ended = false;
static volatile bool was_connected = false;

static void on_state_changed(livekit_connection_state_t state, void* ctx)
{
    ESP_LOGI(TAG, "Room state changed: %s", livekit_connection_state_str(state));
    if (state == LIVEKIT_CONNECTION_STATE_CONNECTED) {
        was_connected = true;
    } else if (state == LIVEKIT_CONNECTION_STATE_FAILED ||
               (was_connected && state == LIVEKIT_CONNECTION_STATE_DISCONNECTED)) {
        session_ended = true;
    }

    livekit_failure_reason_t reason = livekit_room_get_failure_reason(room_handle);
    if (reason != LIVEKIT_FAILURE_REASON_NONE) {
        ESP_LOGE(TAG, "Failure reason: %s", livekit_failure_reason_str(reason));
    }
}

bool room_ended(void)
{
    return session_ended;
}

bool join_room(void)
{
    if (room_handle != NULL) {
        ESP_LOGE(TAG, "Room already created");
        return false;
    }
    session_ended = false;
    was_connected = false;

    livekit_room_options_t room_options = {
        .publish = {
            .kind = LIVEKIT_MEDIA_TYPE_AUDIO,
            .audio_encode = {
                .codec = LIVEKIT_AUDIO_CODEC_OPUS,
                .sample_rate = 16000,
                .channel_count = 1
            },
            .capturer = media_get_capturer()
        },
        .subscribe = {
            .kind = LIVEKIT_MEDIA_TYPE_AUDIO,
            .renderer = media_get_renderer()
        },
        .on_state_changed = on_state_changed
    };
    if (livekit_room_create(&room_handle, &room_options) != LIVEKIT_ERR_NONE) {
        ESP_LOGE(TAG, "Failed to create room");
        return false;
    }

    livekit_err_t connect_res;
#ifdef CONFIG_LK_EXAMPLE_USE_SANDBOX
    livekit_sandbox_res_t res = {};
    livekit_sandbox_options_t gen_options = {
        .sandbox_id = CONFIG_LK_EXAMPLE_SANDBOX_ID,
        .room_name = CONFIG_LK_EXAMPLE_ROOM_NAME,
        .participant_name = CONFIG_LK_EXAMPLE_PARTICIPANT_NAME
    };
    if (!livekit_sandbox_generate(&gen_options, &res)) {
        ESP_LOGE(TAG, "Failed to generate sandbox token");
        leave_room();
        return false;
    }
    connect_res = livekit_room_connect(room_handle, res.server_url, res.token);
    livekit_sandbox_res_free(&res);
#else
    connect_res = livekit_room_connect(
        room_handle,
        CONFIG_LK_EXAMPLE_SERVER_URL,
        CONFIG_LK_EXAMPLE_TOKEN);
#endif

    if (connect_res != LIVEKIT_ERR_NONE) {
        ESP_LOGE(TAG, "Failed to connect to room");
        leave_room();
        return false;
    }
    return true;
}

void leave_room(void)
{
    if (room_handle == NULL) {
        ESP_LOGE(TAG, "Room not created");
        return;
    }
    if (livekit_room_close(room_handle) != LIVEKIT_ERR_NONE) {
        ESP_LOGE(TAG, "Failed to leave room");
    }
    if (livekit_room_destroy(room_handle) != LIVEKIT_ERR_NONE) {
        ESP_LOGE(TAG, "Failed to destroy room");
        return;
    }
    room_handle = NULL;
}