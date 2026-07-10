"""Pure-Python fallback voor de C++ DSP-kern (_dsp).

Zelfde functienamen en signaturen als de native module; werkt op float32-arrays
van vorm (channels, n). Dynamics draaien blok-gebaseerd (1 ms-blokken) zodat de
Python-lus kort blijft; filters gebruiken exact dezelfde RBJ-coefficienten als
de C++-kern.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def backend_info() -> str:
    return "fallback"


def _design(type_: str, sr: float, freq: float, gain_db: float, q: float):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * q)
    if type_ == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif type_ == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif type_ == "notch":
        b = [1.0, -2 * cw, 1.0]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif type_ == "peaking":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif type_ == "lowshelf":
        sqA = np.sqrt(A)
        b = [A * ((A + 1) - (A - 1) * cw + 2 * sqA * alpha),
             2 * A * ((A - 1) - (A + 1) * cw),
             A * ((A + 1) - (A - 1) * cw - 2 * sqA * alpha)]
        a = [(A + 1) + (A - 1) * cw + 2 * sqA * alpha,
             -2 * ((A - 1) + (A + 1) * cw),
             (A + 1) + (A - 1) * cw - 2 * sqA * alpha]
    elif type_ == "highshelf":
        sqA = np.sqrt(A)
        b = [A * ((A + 1) + (A - 1) * cw + 2 * sqA * alpha),
             -2 * A * ((A - 1) + (A + 1) * cw),
             A * ((A + 1) + (A - 1) * cw - 2 * sqA * alpha)]
        a = [(A + 1) - (A - 1) * cw + 2 * sqA * alpha,
             2 * ((A - 1) - (A + 1) * cw),
             (A + 1) - (A - 1) * cw - 2 * sqA * alpha]
    else:
        raise ValueError(f"unknown biquad type: {type_}")
    b = np.asarray(b, dtype=np.float64) / a[0]
    a = np.asarray(a, dtype=np.float64) / a[0]
    return b, a


def apply_gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    return (x * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def biquad_chain(x: np.ndarray, sr: float, bands) -> np.ndarray:
    y = x.astype(np.float64)
    for (t, f, g, q) in bands:
        b, a = _design(t, sr, f, g, q)
        y = lfilter(b, a, y, axis=1)
    return y.astype(np.float32)


def _block_max(det: np.ndarray, block: int) -> np.ndarray:
    n = det.shape[0]
    nb = (n + block - 1) // block
    padded = np.zeros(nb * block, dtype=det.dtype)
    padded[:n] = det
    return padded.reshape(nb, block).max(axis=1)


def _upsample(gains: np.ndarray, block: int, n: int) -> np.ndarray:
    centers = np.arange(gains.shape[0]) * block + block / 2.0
    return np.interp(np.arange(n), centers, gains).astype(np.float32)


def _coef(ms: float, sr: float, block: int) -> float:
    if ms <= 0:
        return 0.0
    return float(np.exp(-block / (0.001 * ms * sr)))


def noise_gate(x: np.ndarray, sr: float, threshold_db: float, attack_ms: float = 5.0,
               release_ms: float = 120.0, hold_ms: float = 50.0,
               range_db: float = 12.0) -> np.ndarray:
    n = x.shape[1]
    block = max(1, int(sr * 0.001))
    det = _block_max(np.abs(x).max(axis=0), block)
    thr = 10.0 ** (threshold_db / 20.0)
    floor_gain = 10.0 ** (-abs(range_db) / 20.0)
    env_a = _coef(1.0, sr, block)
    env_r = _coef(max(release_ms * 0.5, 10.0), sr, block)
    open_c = _coef(attack_ms, sr, block)
    close_c = _coef(release_ms, sr, block)
    hold_blocks = int(hold_ms * 0.001 * sr / block)

    gains = np.empty(det.shape[0], dtype=np.float64)
    env, g, hold = 0.0, 1.0, 0
    for i, d in enumerate(det):
        env = env_a * env + (1 - env_a) * d if d > env else env_r * env + (1 - env_r) * d
        if env >= thr:
            hold, target = hold_blocks, 1.0
        elif hold > 0:
            hold, target = hold - 1, 1.0
        else:
            target = floor_gain
        c = open_c if target > g else close_c
        g = c * g + (1 - c) * target
        gains[i] = g
    return (x * _upsample(gains, block, n)[None, :]).astype(np.float32)


def compressor(x: np.ndarray, sr: float, threshold_db: float, ratio: float = 3.0,
               attack_ms: float = 10.0, release_ms: float = 150.0,
               knee_db: float = 6.0, makeup_db: float = 0.0) -> np.ndarray:
    n = x.shape[1]
    block = max(1, int(sr * 0.001))
    det = _block_max(np.abs(x).max(axis=0), block)
    xg = 20.0 * np.log10(np.maximum(det, 1e-10))
    T, W, R = threshold_db, max(knee_db, 0.01), max(ratio, 1.0)

    yg = np.where(2 * (xg - T) < -W, xg,
                  np.where(2 * np.abs(xg - T) <= W,
                           xg + (1.0 / R - 1.0) * (xg - T + W / 2) ** 2 / (2 * W),
                           T + (xg - T) / R))
    static_gr = yg - xg  # <= 0

    aA = _coef(attack_ms, sr, block)
    aR = _coef(release_ms, sr, block)
    gr = np.empty_like(static_gr)
    g = 0.0
    for i, s in enumerate(static_gr):
        g = aA * g + (1 - aA) * s if s < g else aR * g + (1 - aR) * s
        gr[i] = g
    gains = 10.0 ** ((gr + makeup_db) / 20.0)
    return (x * _upsample(gains, block, n)[None, :]).astype(np.float32)


def limiter(x: np.ndarray, sr: float, ceiling_db: float = -1.5,
            release_ms: float = 60.0, lookahead_ms: float = 5.0) -> np.ndarray:
    n = x.shape[1]
    if n == 0:
        return x.astype(np.float32)
    c = 10.0 ** (ceiling_db / 20.0)
    L = max(1, int(lookahead_ms * 0.001 * sr))

    det = np.abs(x).max(axis=0)
    r = np.where(det > c, c / np.maximum(det, 1e-12), 1.0)

    # Forward sliding-window minimum over [i, i+L].
    rp = np.concatenate([r, np.full(L, r[-1])])
    m = np.lib.stride_tricks.sliding_window_view(rp, L + 1).min(axis=1)

    # Trailing moving average of length L (smooth attack, never above required gain).
    ps = np.concatenate([[0.0], np.cumsum(m, dtype=np.float64)])
    idx = np.arange(n)
    lo = np.maximum(idx + 1 - L, 0)
    ma = (ps[idx + 1] - ps[lo]) / (idx + 1 - lo)

    # Release at block rate: instant fall, slow rise.
    block = max(1, int(sr * 0.001))
    mab = -_block_max(-ma, block)  # block-wise minimum
    aR = _coef(release_ms, sr, block)
    rel = np.empty_like(mab)
    g = 1.0
    for i, v in enumerate(mab):
        g = v if v < g else aR * g + (1 - aR) * v
        rel[i] = min(g, 1.0)
    gains = np.minimum(_upsample(rel, block, n), ma)  # bewaakt de no-clip-garantie

    y = np.clip(x * gains[None, :], -c, c)
    return y.astype(np.float32)
