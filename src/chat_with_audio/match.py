"""Reference matching: laat een opname klinken als een referentie.

Spectraal: 1/3-octaafbanden van bron en referentie vergelijken en het verschil
(begrensd en gladgestreken) corrigeren met een bank peaking-filters via de
C++ EQ. Daarna loudness-match naar de referentie (true-peak-bewaakt).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import welch

from chat_with_audio import chain, dsp
from chat_with_audio.analysis import measure_lufs

log = logging.getLogger(__name__)

# 1/3-octaaf centerfrequenties (ISO), begrensd tot het bruikbare spraak/muziekgebied
_CENTERS = [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
            1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000]


def _band_levels_db(x: np.ndarray, sr: int) -> np.ndarray:
    mono = x.mean(axis=0) if x.ndim == 2 else x
    nper = int(min(mono.shape[0], 8192))
    f, p = welch(mono, fs=sr, nperseg=nper)
    levels = np.full(len(_CENTERS), np.nan)
    for i, fc in enumerate(_CENTERS):
        if fc > sr / 2 / 1.2:
            continue
        lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
        sel = (f >= lo) & (f < hi)
        if sel.any():
            levels[i] = 10 * np.log10(float(p[sel].mean()) + 1e-20)
    return levels


def build_match_eq(x: np.ndarray, sr: int, ref: np.ndarray, ref_sr: int,
                   strength: float = 1.0, max_db: float = 6.0) -> tuple[list[dict], list[str]]:
    """Bepaal de match-EQ (bands voor de eq-stap) + leesbare beschrijving."""
    src = _band_levels_db(x, sr)
    dst = _band_levels_db(ref, ref_sr)
    valid = ~np.isnan(src) & ~np.isnan(dst)
    if valid.sum() < 6:
        raise ValueError("Te weinig spectrale overlap om te matchen "
                         "(is een van beide bestanden extreem kort of smalbandig?)")

    # normaliseer op het gemiddelde verschil (loudness komt later; hier alleen kleur)
    diff = np.where(valid, dst - src, 0.0)
    diff -= diff[valid].mean()
    diff = np.clip(diff * float(strength), -abs(max_db), abs(max_db))
    # glad over buurbanden zodat de filterbank geen kamstructuur wordt
    smooth = diff.copy()
    for i in range(len(diff)):
        lo, hi = max(0, i - 1), min(len(diff), i + 2)
        smooth[i] = diff[lo:hi].mean()

    bands = []
    for fc, g in zip(_CENTERS, smooth):
        if abs(g) >= 0.75 and fc < sr / 2 / 1.2:
            bands.append({"type": "peaking", "freq": float(fc),
                          "gain_db": round(float(g), 1), "q": 4.32})
    desc = [f"{b['freq']:.0f} Hz {b['gain_db']:+.1f} dB" for b in bands]
    return bands, desc


def match_reference(x: np.ndarray, sr: int, ref: np.ndarray, ref_sr: int,
                    strength: float = 1.0, max_db: float = 6.0,
                    match_loudness: bool = True) -> tuple[np.ndarray, dict]:
    bands, desc = build_match_eq(x, sr, ref, ref_sr, strength, max_db)
    y = dsp.eq(x, sr, bands) if bands else x
    info: dict = {"eq_bands": bands, "eq_description": desc}
    if match_loudness:
        ref_lufs = measure_lufs(ref if ref.ndim == 2 else ref[None, :], ref_sr)
        if ref_lufs is not None:
            y, norm = chain.normalize_loudness(y, sr, target_lufs=ref_lufs,
                                               true_peak_db=-1.0)
            info["loudness"] = norm
    return y, info
