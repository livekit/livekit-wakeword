/* float32 audio on stdin -> float32 log-mels on stdout. Optional args: floor offset scale. */
#include <stdio.h>
#include <stdlib.h>

#include "../livekit_wakeword/src/lkww_micro.h"

int main(int argc, char **argv) {
    lkww_frontend_t fe = {0};
    float in[LKWW_CHUNK], out[LKWW_FRAMES * LKWW_N_MELS];
    if (argc == 4)
        lkww_frontend_init(&fe, strtof(argv[1], NULL), strtof(argv[2], NULL), strtof(argv[3], NULL));
    else
        lkww_frontend_reset(&fe);
    while (fread(in, sizeof(float), LKWW_CHUNK, stdin) == LKWW_CHUNK) {
        lkww_frontend_process_f32(&fe, in, out);
        fwrite(out, sizeof(float), LKWW_FRAMES * LKWW_N_MELS, stdout);
    }
    return 0;
}
