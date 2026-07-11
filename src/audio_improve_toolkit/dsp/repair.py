"""Spectrale reparatie: declip en declick (iZotope RX-achtig, maar dan van ons).

Declip: detecteert flat-top-regio's (opeenvolgende near-identieke samples nabij
lokale pieken) en reconstrueert de golfvorm met cubic-spline-interpolatie over
de omliggende gezonde samples.

Declick: detecteert impulsartefacten (uitschieters t.o.v. een lokale mediaan)
en vervangt ze via interpolatie.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation, median_filter

log = logging.getLogger(__name__)


def _clip_mask(mono: np.ndarray) -> np.ndarray:
    """Flat-top-detectie, robuust voor zowel 0 dBFS-clips als 32-bit float clips."""
    peak = float(np.abs(mono).max())
    if peak <= 0:
        return np.zeros(mono.shape[0], dtype=bool)
    if peak <= 1.001:
        near = np.abs(mono) >= 0.985
    else:
        near = np.abs(mono) >= 0.98 * peak
    flat = np.concatenate([[False], np.abs(np.diff(mono)) < 1e-6]) & near
    # runs van >= 2 flat samples, plus 1 sample marge aan weerszijden
    mask = binary_dilation(flat, iterations=1)
    return mask & near


def declip(x: np.ndarray, sr: int, max_gap_ms: float = 4.0) -> tuple[np.ndarray, int]:
    """Reconstrueer geclipte stukken; geeft (audio, aantal gerepareerde regio's)."""
    x2 = x[None, :] if x.ndim == 1 else x
    out = x2.astype(np.float32).copy()
    max_gap = max(2, int(max_gap_ms / 1000 * sr))
    total = 0
    for ci in range(out.shape[0]):
        ch = out[ci]
        mask = _clip_mask(ch)
        if not mask.any():
            continue
        edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        for a, b in zip(starts, ends):
            if b - a > max_gap:
                continue  # te groot om geloofwaardig te reconstrueren
            ctx = max(8, (b - a) * 3)
            lo, hi = max(0, a - ctx), min(ch.shape[0], b + ctx)
            good = np.concatenate([np.arange(lo, a), np.arange(b, hi)])
            if good.shape[0] < 8:
                continue
            spline = CubicSpline(good, ch[good].astype(np.float64))
            ch[a:b] = spline(np.arange(a, b)).astype(np.float32)
            total += 1
    return out, total


def _rolling_robust_env(mag: np.ndarray, sr: int, block_ms: float = 5.0,
                        span_blocks: int = 41, percentile: float = 25.0) -> np.ndarray:
    """Klik-ongevoelige lokale envelope: blok-medianen + rollend laag percentiel.

    Een klikcluster kan meerdere 5 ms-blokken beslaan; met een lang venster
    (~205 ms) en het 25e percentiel blijft de referentie op het echte
    omgevingsniveau liggen, ook vlak naast een groepje kliks.
    """
    from scipy.ndimage import percentile_filter

    n = mag.shape[0]
    block = max(1, int(block_ms / 1000 * sr))
    nb = (n + block - 1) // block
    padded = np.zeros(nb * block, dtype=mag.dtype)
    padded[:n] = mag
    block_med = np.median(padded.reshape(nb, block), axis=1)
    smooth = percentile_filter(block_med, percentile, size=span_blocks,
                               mode="nearest")
    centers = np.arange(nb) * block + block / 2.0
    return np.interp(np.arange(n), centers, smooth)


def _repair_runs(ch: np.ndarray, mask: np.ndarray, max_len: int) -> int:
    count = 0
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for a, b in zip(starts, ends):
        if b - a > max_len:
            continue  # te lang: waarschijnlijk echte content
        ctx = max(16, (b - a) // 2)
        lo, hi = max(0, a - ctx), min(ch.shape[0], b + ctx)
        good = np.concatenate([np.arange(lo, a), np.arange(b, hi)])
        if good.shape[0] < 6:
            continue
        ch[a:b] = np.interp(np.arange(a, b), good, ch[good].astype(np.float64))
        count += 1
    return count


def declick(x: np.ndarray, sr: int, threshold: float = 8.0, window_ms: float = 0.7,
            max_click_ms: float = 50.0,
            silence_floor_db: float = -45.0) -> tuple[np.ndarray, int]:
    """Verwijder klikken/impulsen; geeft (audio, aantal gerepareerde klikken).

    Twee lagen:
    1. naald-ticks (< ~1.5 ms) overal: uitschieters t.o.v. een fijne lopende
       mediaan, geschaald op de lokale omgeving (~25 ms);
    2. klik-bursts (tot max_click_ms) ALLEEN in stilte: envelope die ver boven
       een klik-ongevoelige 105 ms-referentie uitsteekt terwijl die referentie
       onder silence_floor_db ligt. In spraak/muziek wordt laag 2 nooit actief,
       zodat plosieven en woordaanzetten gegarandeerd blijven staan.
    """
    from scipy.ndimage import uniform_filter1d

    x2 = x[None, :] if x.ndim == 1 else x
    out = x2.astype(np.float32).copy()
    win = max(3, int(window_ms / 1000 * sr) | 1)  # oneven
    sil_floor = 10.0 ** (silence_floor_db / 20.0)
    total = 0
    for ci in range(out.shape[0]):
        ch = out[ci]

        # laag 1: naald-ticks via fijn mediaanresidu (bewezen spraakveilig)
        med = median_filter(ch, size=win, mode="nearest")
        resid = np.abs(ch - med)
        local_r = uniform_filter1d(resid, size=max(3, int(0.025 * sr)),
                                   mode="nearest") * 1.4826
        tick_mask = resid > threshold * np.maximum(local_r, 1e-4)
        total += _repair_runs(ch, tick_mask, max_len=max(2, int(0.0015 * sr)))

        # laag 2: bursts in stilte
        env = uniform_filter1d(np.abs(ch), size=max(3, int(0.001 * sr)),
                               mode="nearest")
        local_e = _rolling_robust_env(env, sr) * 1.4826
        burst_mask = (env > threshold * np.maximum(local_e, 2e-4)) \
            & (local_e < sil_floor)
        # beschermzone rond spraak: een woordfinale plosief ('-d', '-t') in
        # stilte is akoestisch bijna een klik — daar blijven we vanaf. Alleen
        # AANHOUDEND geluid (> 80 ms) telt als spraak, anders zouden de kliks
        # zichzelf beschermen.
        from scipy.ndimage import binary_opening

        speech_env = uniform_filter1d(np.abs(ch), size=max(3, int(0.025 * sr)),
                                      mode="nearest")
        block = max(1, int(0.005 * sr))
        nb = (ch.shape[0] + block - 1) // block
        pad_env = np.zeros(nb * block, dtype=speech_env.dtype)
        pad_env[:ch.shape[0]] = speech_env
        loud_b = np.asarray(pad_env.reshape(nb, block).max(axis=1) > 3.5 * sil_floor,
                            dtype=bool)
        sustained = binary_opening(loud_b, structure=np.ones(16, dtype=bool))
        protect_b = binary_dilation(sustained, iterations=30)  # +/- 150 ms
        protect = np.repeat(np.asarray(protect_b, dtype=bool), block)[: ch.shape[0]]
        burst_mask = np.asarray(burst_mask, dtype=bool) & ~protect
        # marge om de na-ring van de klik mee te nemen
        burst_mask = binary_dilation(burst_mask, iterations=max(1, int(0.003 * sr)))
        total += _repair_runs(ch, burst_mask,
                              max_len=max(2, int(max_click_ms / 1000 * sr))
                              + 2 * int(0.003 * sr))
    return out, total
