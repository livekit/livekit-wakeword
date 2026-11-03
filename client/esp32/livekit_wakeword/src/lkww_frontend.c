/* Copyright 2026 LiveKit, Inc. Licensed under the Apache License, Version 2.0. */
#include "lkww_micro.h"

#include <math.h>
#include <string.h>

#include "lkww_tables.h"

#define NB (LKWW_N_FFT / 2 + 1)

static float tw_cos[LKWW_N_FFT / 2], tw_sin[LKWW_N_FFT / 2];
static unsigned short bitrev[LKWW_N_FFT];
static int tables_ready = 0;

static void init_tables(void) {
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < LKWW_N_FFT / 2; i++) {
        tw_cos[i] = (float)cos(2.0 * pi * i / LKWW_N_FFT);
        tw_sin[i] = (float)-sin(2.0 * pi * i / LKWW_N_FFT);
    }
    int bits = 0;
    while ((1 << bits) < LKWW_N_FFT) bits++;
    for (int i = 0; i < LKWW_N_FFT; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
        bitrev[i] = (unsigned short)r;
    }
    tables_ready = 1;
}

static void fft(float *re, float *im) {
    for (int i = 0; i < LKWW_N_FFT; i++) {
        int j = bitrev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= LKWW_N_FFT; len <<= 1) {
        int half = len >> 1, step = LKWW_N_FFT / len;
        for (int s = 0; s < LKWW_N_FFT; s += len) {
            for (int k = 0; k < half; k++) {
                float wr = tw_cos[k * step], wi = tw_sin[k * step];
                int a = s + k, b = a + half;
                float xr = re[b] * wr - im[b] * wi;
                float xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr; im[a] += xi;
            }
        }
    }
}

void lkww_frontend_reset(lkww_frontend_t *fe) { memset(fe->buf, 0, sizeof(fe->buf)); }

void lkww_frontend_init(lkww_frontend_t *fe, float floor, float offset, float scale) {
    lkww_frontend_reset(fe);
    fe->mel_floor = floor;
    fe->mel_offset = offset;
    fe->mel_inv_scale = 1.0f / scale;
}

static void process_buf(lkww_frontend_t *fe, float *mel_out) {
    float *re = fe->re, *im = fe->im;
    const float *buf = fe->buf;
    if (!tables_ready) init_tables();
    const int custom = fe->mel_inv_scale != 0.0f;
    const float floor = custom ? fe->mel_floor : LKWW_MEL_FLOOR;
    const float offset = custom ? fe->mel_offset : LKWW_MEL_OFFSET;
    const float inv_scale = custom ? fe->mel_inv_scale : 1.0f / LKWW_MEL_SCALE;
    for (int f = 0; f < LKWW_FRAMES; f++) {
        const float *x = buf + f * LKWW_HOP;
        for (int t = 0; t < LKWW_WIN_OFFSET; t++) re[t] = 0.0f;
        for (int t = LKWW_WIN_OFFSET; t < LKWW_N_FFT; t++) re[t] = x[t] * lkww_window[t - LKWW_WIN_OFFSET];
        memset(im, 0, LKWW_N_FFT * sizeof(float));
        fft(re, im);
        for (int k = 0; k < NB; k++) re[k] = re[k] * re[k] + im[k] * im[k];
        const float *w = lkww_mel_weights;
        for (int m = 0; m < LKWW_N_MELS; m++) {
            float acc = 0.0f;
            const float *p = re + lkww_mel_start[m];
            for (int i = 0; i < lkww_mel_len[m]; i++) acc += p[i] * w[i];
            w += lkww_mel_len[m];
            mel_out[f * LKWW_N_MELS + m] = (logf(acc + floor) + offset) * inv_scale;
        }
    }
    memmove(fe->buf, fe->buf + LKWW_CHUNK, LKWW_TAIL * sizeof(float));
}

void lkww_frontend_process_f32(lkww_frontend_t *fe, const float *audio, float *mel_out) {
    memcpy(fe->buf + LKWW_TAIL, audio, LKWW_CHUNK * sizeof(float));
    process_buf(fe, mel_out);
}

void lkww_frontend_process(lkww_frontend_t *fe, const int16_t *pcm, float *mel_out) {
    float *dst = fe->buf + LKWW_TAIL;
    for (int i = 0; i < LKWW_CHUNK; i++) dst[i] = pcm[i] * (1.0f / 32768.0f);
    process_buf(fe, mel_out);
}
