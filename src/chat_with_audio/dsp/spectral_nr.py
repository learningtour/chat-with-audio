"""Tier A ruisonderdrukking: STFT spectral gating met Wiener-achtige gains.

Ruisprofiel wordt per frequentiebin geschat als een laag percentiel van de
magnitude over de tijd; gains worden over frequentie en tijd gladgestreken om
musical noise te beperken. De maximale demping is `reduction_db`.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft


def spectral_denoise(x: np.ndarray, sr: int, reduction_db: float = 12.0,
                     quiet_fraction: float = 0.1, oversubtraction: float = 2.0,
                     time_smooth_ms: float = 40.0, freq_smooth_bins: int = 3) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("verwacht (channels, n)")
    n = x.shape[1]
    nfft = 2048 if sr >= 32000 else 1024
    hop = nfft // 4
    floor = 10.0 ** (-abs(reduction_db) / 20.0)

    out = np.empty_like(x, dtype=np.float32)
    for ci in range(x.shape[0]):
        _, _, X = stft(x[ci], fs=sr, window="hann", nperseg=nfft,
                       noverlap=nfft - hop, padded=True)
        mag = np.abs(X)
        # Ruisprofiel: gemiddelde magnitude van de stilste frames (op breedband-energie).
        energy = mag.mean(axis=0)
        n_quiet = max(3, int(energy.shape[0] * quiet_fraction))
        quiet = np.argsort(energy)[:n_quiet]
        noise = mag[:, quiet].mean(axis=1, keepdims=True)
        snr = np.maximum(mag / (oversubtraction * noise + 1e-12) - 1.0, 0.0)
        g = snr / (snr + 1.0)

        if freq_smooth_bins > 1:
            g = uniform_filter1d(g, freq_smooth_bins, axis=0, mode="nearest")

        if time_smooth_ms > 0:
            # Directe attack (spraak blijft intact), exponentiele afbouw van de
            # opening voorkomt flikkerende bins.
            a = float(np.exp(-hop / (sr * time_smooth_ms * 0.001)))
            prev = g[:, 0].copy()
            for i in range(g.shape[1]):
                prev = np.maximum(g[:, i], a * prev)
                g[:, i] = prev

        g = np.clip(g, floor, 1.0)
        _, y = istft(X * g, fs=sr, window="hann", nperseg=nfft, noverlap=nfft - hop)
        if y.shape[0] < n:
            y = np.pad(y, (0, n - y.shape[0]))
        out[ci] = y[:n].astype(np.float32)
    return out
