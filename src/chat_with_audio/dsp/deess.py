"""Spectrale de-esser: dempt sissende s-klanken (5.5-9.5 kHz) alleen op de
frames waar de sibilance-energie uitschiet t.o.v. het spraakgebied."""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft


def deess(x: np.ndarray, sr: int, strength_db: float = 8.0,
          low_hz: float = 5500.0, high_hz: float = 9500.0,
          sensitivity: float = 2.2) -> np.ndarray:
    if sr < 2 * high_hz:
        return x.astype(np.float32)
    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    nfft, hop = 1024, 256
    floor = 10.0 ** (-abs(strength_db) / 20.0)

    out = np.empty_like(x2, dtype=np.float32)
    for ci in range(x2.shape[0]):
        f, _, X = stft(x2[ci], fs=sr, window="hann", nperseg=nfft,
                       noverlap=nfft - hop, padded=True)
        mag = np.abs(X)
        sib = (f >= low_hz) & (f <= high_hz)
        body = (f >= 300.0) & (f <= 3000.0)
        sib_e = mag[sib].mean(axis=0) + 1e-12
        body_e = mag[body].mean(axis=0) + 1e-12
        ratio = sib_e / body_e
        base = float(np.median(ratio)) + 1e-9
        # overschot boven sensitivity*mediane verhouding wordt weggeregeld
        excess = np.maximum(ratio / (base * sensitivity), 1.0)
        gain_frames = np.clip(1.0 / excess, floor, 1.0)
        # kort uitsmeren over de tijd zodat de demping niet klappert
        k = np.ones(3) / 3.0
        gain_frames = np.convolve(gain_frames, k, mode="same")
        G = np.ones_like(mag)
        G[sib, :] = gain_frames[None, :]
        _, y = istft(X * G, fs=sr, window="hann", nperseg=nfft, noverlap=nfft - hop)
        if y.shape[0] < n:
            y = np.pad(y, (0, n - y.shape[0]))
        out[ci] = y[:n].astype(np.float32)
    return out
