#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Join the room (connects in the background); false if it could not start.
bool join_room(void);
void leave_room(void);

/// True once the room disconnected or failed after join_room().
bool room_ended(void);

#ifdef __cplusplus
}
#endif
