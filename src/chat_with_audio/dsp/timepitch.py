"""Tijd- en toonhoogtemotor: phase-vocoder time-stretch met piek-locking,
pitch-shift (stretch + resample) met optioneel formantbehoud, en varispeed
(tape-stijl: tempo en toonhoogte samen).

Puur numpy/scipy. De phase vocoder gebruikt identity phase locking: bins rond
een spectrale piek volgen de fasedraaiing van hun piek in plaats van elk hun
eigen gang te gaan — dat voorkomt het klassieke "onderwater"-fasegewapper.
Formantbehoud werkt met cepstrale omhullenden: na de shift wordt per frame de
omhullende van het origineel teruggelegd, zodat een stem hoger of lager wordt
zonder Mickey Mouse-klank.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly

N_FFT = 2048
OVERLAP = 4  # synthese-hop = N_FFT / OVERLAP


def _frac_ratio(rate: float, max_den: int = 1000) -> tuple[int, int]:
    """Benader rate als breuk up/down voor resample_poly."""
    from fractions import Fraction

    fr = Fraction(rate).limit_denominator(max_den)
    return fr.numerator, fr.denominator


def _stretch_mono(mono: np.ndarray, rate: float) -> np.ndarray:
    """Phase vocoder: 1/rate keer de duur (rate 1.25 = 25% sneller/korter)."""
    n = mono.shape[0]
    hs = N_FFT // OVERLAP
    win = np.hanning(N_FFT)
    n_syn = max(2, int(math.ceil(n / rate / hs)))

    # analyse-posities (float) en de echte integer-hop ertussen
    pos = np.minimum((np.arange(n_syn) * hs * rate), max(0, n - 1)).astype(np.int64)
    pad = np.pad(mono.astype(np.float64), (0, N_FFT + hs))
    frames = np.stack([pad[p:p + N_FFT] * win for p in pos])
    spec = np.fft.rfft(frames, axis=1)
    mag, phase = np.abs(spec), np.angle(spec)

    omega = 2 * np.pi * np.arange(N_FFT // 2 + 1) / N_FFT  # rad per sample
    phi_syn = np.empty_like(phase)
    phi_syn[0] = phase[0]
    for k in range(1, n_syn):
        ha = max(1, int(pos[k] - pos[k - 1]))
        expected = omega * ha
        dev = phase[k] - phase[k - 1] - expected
        dev = dev - 2 * np.pi * np.round(dev / (2 * np.pi))
        true_omega = omega + dev / ha
        phi_syn[k] = phi_syn[k - 1] + true_omega * hs

        # identity phase locking: niet-piek-bins volgen de draai van hun piek
        m = mag[k]
        peaks = np.where((m[1:-1] > m[:-2]) & (m[1:-1] > m[2:]))[0] + 1
        if peaks.size:
            owner_idx = np.searchsorted(peaks, np.arange(m.shape[0]))
            owner_idx = np.clip(owner_idx, 0, peaks.size - 1)
            left = peaks[np.maximum(owner_idx - 1, 0)]
            right = peaks[owner_idx]
            owner = np.where(np.abs(np.arange(m.shape[0]) - left) <
                             np.abs(right - np.arange(m.shape[0])), left, right)
            rot = phi_syn[k, owner] - phase[k, owner]
            locked = phase[k] + rot
            is_peak = np.zeros(m.shape[0], dtype=bool)
            is_peak[peaks] = True
            phi_syn[k] = np.where(is_peak, phi_syn[k], locked)

    out_spec = mag * np.exp(1j * phi_syn)
    frames_out = np.fft.irfft(out_spec, n=N_FFT, axis=1) * win
    y = np.zeros(n_syn * hs + N_FFT)
    norm = np.zeros_like(y)
    wsq = win**2
    for k in range(n_syn):
        y[k * hs:k * hs + N_FFT] += frames_out[k]
        norm[k * hs:k * hs + N_FFT] += wsq
    y = y / np.maximum(norm, 1e-8)
    n_out = int(round(n / rate))
    return y[:n_out]


def time_stretch(x: np.ndarray, sr: int, rate: float = 1.0) -> np.ndarray:
    """Duur veranderen zonder toonhoogte: rate 1.25 = 25% sneller (korter),
    0.8 = langzamer (langer). Bruikbaar bereik ~0.5-2.0."""
    if not 0.25 <= rate <= 4.0:
        raise ValueError(f"rate {rate} buiten bereik 0.25-4.0.")
    if rate == 1.0:
        return (x[None, :] if x.ndim == 1 else x).astype(np.float32)
    x2 = x[None, :] if x.ndim == 1 else x
    out = [_stretch_mono(ch, rate) for ch in x2]
    return np.stack(out).astype(np.float32)


def _cepstral_envelope(mag: np.ndarray, lifter: int = 40) -> np.ndarray:
    """Spectrale omhullende per frame via cepstrale liftering (log-magnitude)."""
    logm = np.log(mag + 1e-10)
    cep = np.fft.irfft(logm, axis=1)
    cep[:, lifter:-lifter if lifter < cep.shape[1] // 2 else None] = 0.0
    env = np.fft.rfft(cep, axis=1).real
    return np.exp(env[:, : mag.shape[1]])


def _match_formants(y: np.ndarray, ref: np.ndarray, sr: int,
                    max_db: float = 18.0) -> np.ndarray:
    """Leg per frame de spectrale omhullende van ref terug op y (beide mono,
    zelfde lengte). Correctie begrensd op ±max_db."""
    hs = N_FFT // OVERLAP
    win = np.hanning(N_FFT)
    n = min(y.shape[0], ref.shape[0])
    n_fr = max(1, (n - N_FFT) // hs + 1)
    pad_y = np.pad(y.astype(np.float64), (0, N_FFT + hs))
    pad_r = np.pad(ref.astype(np.float64), (0, N_FFT + hs))
    fy = np.stack([pad_y[k * hs:k * hs + N_FFT] * win for k in range(n_fr)])
    fr = np.stack([pad_r[k * hs:k * hs + N_FFT] * win for k in range(n_fr)])
    sy = np.fft.rfft(fy, axis=1)
    env_y = _cepstral_envelope(np.abs(sy))
    env_r = _cepstral_envelope(np.abs(np.fft.rfft(fr, axis=1)))
    lim = 10 ** (max_db / 20)
    corr = np.clip(env_r / np.maximum(env_y, 1e-10), 1 / lim, lim)
    frames_out = np.fft.irfft(sy * corr, n=N_FFT, axis=1) * win
    out = np.zeros(n_fr * hs + N_FFT)
    norm = np.zeros_like(out)
    wsq = win**2
    for k in range(n_fr):
        out[k * hs:k * hs + N_FFT] += frames_out[k]
        norm[k * hs:k * hs + N_FFT] += wsq
    return (out / np.maximum(norm, 1e-8))[:n]


def pitch_shift(x: np.ndarray, sr: int, semitones: float = 0.0,
                preserve_formants: bool = False) -> np.ndarray:
    """Toonhoogte verschuiven zonder duurverandering. preserve_formants houdt
    de spectrale omhullende (het stemkarakter) op zijn plek — omhoog zonder
    Mickey Mouse, omlaag zonder reus."""
    if abs(semitones) > 24:
        raise ValueError(f"semitones {semitones} buiten bereik ±24.")
    if semitones == 0:
        return (x[None, :] if x.ndim == 1 else x).astype(np.float32)
    f = 2.0 ** (semitones / 12.0)
    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    out = []
    for ch in x2:
        stretched = _stretch_mono(ch, 1.0 / f)     # f keer langer
        up, down = _frac_ratio(1.0 / f)
        shifted = resample_poly(stretched, up, down)  # sneller afspelen: ×f
        shifted = shifted[:n] if shifted.shape[0] >= n else \
            np.pad(shifted, (0, n - shifted.shape[0]))
        if preserve_formants:
            shifted = _match_formants(shifted, ch, sr)
            shifted = shifted[:n] if shifted.shape[0] >= n else \
                np.pad(shifted, (0, n - shifted.shape[0]))
        out.append(shifted)
    return np.stack(out).astype(np.float32)


def varispeed(x: np.ndarray, sr: int, rate: float = 1.0) -> np.ndarray:
    """Tape-varispeed: tempo én toonhoogte samen (rate 1.05 = 5% sneller en
    hoger). Eén resample — het schoonste wat er bestaat, als de koppeling
    tussen duur en pitch acceptabel is."""
    if not 0.25 <= rate <= 4.0:
        raise ValueError(f"rate {rate} buiten bereik 0.25-4.0.")
    if rate == 1.0:
        return (x[None, :] if x.ndim == 1 else x).astype(np.float32)
    x2 = x[None, :] if x.ndim == 1 else x
    up, down = _frac_ratio(1.0 / rate)
    return resample_poly(x2, up, down, axis=1).astype(np.float32)
