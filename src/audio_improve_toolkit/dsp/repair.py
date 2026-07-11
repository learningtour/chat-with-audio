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


def declick(x: np.ndarray, sr: int, threshold: float = 8.0,
            window_ms: float = 0.7) -> tuple[np.ndarray, int]:
    """Verwijder klikken/impulsen; geeft (audio, aantal gerepareerde klikken).

    De detectieschaal is lokaal (~25 ms): een klik moet boven zijn dírecte
    omgeving uitsteken. Een globale schaal zou bij materiaal met veel stilte
    (bv. spraakberichten) normale spraaktextuur als klik aanzien.
    """
    from scipy.ndimage import uniform_filter1d

    x2 = x[None, :] if x.ndim == 1 else x
    out = x2.astype(np.float32).copy()
    win = max(3, int(window_ms / 1000 * sr) | 1)  # oneven
    total = 0
    for ci in range(out.shape[0]):
        ch = out[ci]
        med = median_filter(ch, size=win, mode="nearest")
        resid = ch - med
        local = uniform_filter1d(np.abs(resid), size=max(3, int(0.025 * sr)),
                                 mode="nearest") * 1.4826
        mask = np.abs(resid) > threshold * np.maximum(local, 1e-4)
        # alleen korte impulsen (max ~1.5 ms), geen transienten van muziek/spraak
        max_len = max(2, int(0.0015 * sr))
        edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        for a, b in zip(starts, ends):
            if b - a > max_len:
                continue
            lo, hi = max(0, a - 16), min(ch.shape[0], b + 16)
            good = np.concatenate([np.arange(lo, a), np.arange(b, hi)])
            if good.shape[0] < 6:
                continue
            ch[a:b] = np.interp(np.arange(a, b), good, ch[good].astype(np.float64))
            total += 1
    return out, total
