"""Streaming resampling to 16 kHz in numpy: a polyphase windowed-sinc FIR (Kaiser window, about
90 dB stopband, like the Rust client's), with state carried between calls."""

from __future__ import annotations

from math import gcd

import numpy as np

MODEL_RATE = 16000
_ATTENUATION_DB = 90.0
_TRANSITION = 0.1  # transition band, as a fraction of the lower Nyquist rate
_BLOCK = 4096  # outputs per vectorised block (bounds memory for long clips)


class Resampler:
    """Resamples mono float32 audio from `rate` to 16 kHz, chunk by chunk.

    Output sample m is at input time m * down / up; it is computed as soon as the input reaching
    that time has arrived, from the last `taps` input samples (zeros before the stream starts).
    """

    def __init__(self, rate: int) -> None:
        if rate <= 0:
            raise ValueError(f"sample rate must be positive, got {rate}")
        g = gcd(int(rate), MODEL_RATE)
        self.rate = int(rate)
        self.up, self.down = MODEL_RATE // g, int(rate) // g
        # low-pass in the upsampled domain: cut below the lower of the two Nyquist rates
        nyq = 0.5 / max(self.up, self.down)  # cycles per upsampled sample
        width = _TRANSITION * nyq
        beta = 0.1102 * (_ATTENUATION_DB - 8.7)
        n = int(np.ceil((_ATTENUATION_DB - 7.95) / (2.285 * 2 * np.pi * width))) | 1
        taps = -(-n // self.up)  # input samples per output
        n = taps * self.up
        cutoff = nyq - width / 2
        t = np.arange(n) - (n - 1) / 2
        h = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(n, beta)
        h *= self.up / h.sum()  # unity gain after zero-stuffing
        # bank[p, k]: weight of input sample i - k for an output at upsampled phase p
        self._bank = h.reshape(taps, self.up).T.astype(np.float32).copy()
        self.taps = taps
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros(self.taps - 1, np.float32)  # last inputs, oldest first
        self._seen = 0  # input samples consumed
        self._made = 0  # output samples produced

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._history, np.asarray(audio, np.float32)])
        base = self._seen - (self.taps - 1)  # absolute index of x[0]
        self._seen += len(x) - len(self._history)
        # outputs whose newest input index i = floor(m * down / up) has arrived
        last = (self._seen * self.up - 1) // self.down  # largest m with i < seen
        m = np.arange(self._made, last + 1, dtype=np.int64)
        self._made = int(last) + 1
        out = np.empty(len(m), np.float32)
        k = np.arange(self.taps)
        for s in range(0, len(m), _BLOCK):
            mb = m[s : s + _BLOCK]
            i = (mb * self.down) // self.up
            p = (mb * self.down) % self.up
            window = x[(i - base)[:, None] - k[None, :]]
            out[s : s + _BLOCK] = np.einsum("nk,nk->n", window, self._bank[p])
        self._history = x[len(x) - (self.taps - 1) :] if self.taps > 1 else x[:0]
        return out
