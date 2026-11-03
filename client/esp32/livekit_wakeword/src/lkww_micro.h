/* Copyright 2026 LiveKit, Inc. Licensed under the Apache License, Version 2.0.
 *
 * Log-mel front end of livekit-wakeword micro models: 1280 samples (80 ms, 16 kHz) in, the
 * model's `mel` input (8 x 40) out. C99, no allocation, no dependencies.
 */
#ifndef LKWW_MICRO_H
#define LKWW_MICRO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LKWW_SAMPLE_RATE 16000
#define LKWW_CHUNK 1280 /* samples per model step (80 ms) */
#define LKWW_N_FFT 512
#define LKWW_HOP 160
#define LKWW_N_MELS 40
#define LKWW_FRAMES 8 /* mel frames per step */
#define LKWW_TAIL (LKWW_N_FFT - LKWW_HOP)

/* log-mel = (log(mel + floor) + offset) / scale, from `mel_params` in `<name>_micro.json`. */
#ifndef LKWW_MEL_FLOOR
#define LKWW_MEL_FLOOR 1e-9f
#endif
#ifndef LKWW_MEL_OFFSET
#define LKWW_MEL_OFFSET 9.0f
#endif
#ifndef LKWW_MEL_SCALE
#define LKWW_MEL_SCALE 6.0f
#endif

typedef struct {
    float buf[LKWW_TAIL + LKWW_CHUNK];
    float mel_floor, mel_offset, mel_inv_scale; /* 0: the LKWW_MEL_* defaults */
    float re[LKWW_N_FFT], im[LKWW_N_FFT]; /* FFT scratch, kept off the caller's stack */
} lkww_frontend_t;

/* Set the log-mel parameters and zero the state. */
void lkww_frontend_init(lkww_frontend_t *fe, float floor, float offset, float scale);

/* Zero the state; keeps the LKWW_MEL_* defaults unless lkww_frontend_init was called. */
void lkww_frontend_reset(lkww_frontend_t *fe);

/* Consume LKWW_CHUNK samples; write LKWW_FRAMES x LKWW_N_MELS row-major log-mels. */
void lkww_frontend_process(lkww_frontend_t *fe, const int16_t *pcm, float *mel_out);

/* Same, for float samples in [-1, 1]. */
void lkww_frontend_process_f32(lkww_frontend_t *fe, const float *audio, float *mel_out);

#ifdef __cplusplus
}
#endif

#endif /* LKWW_MICRO_H */
