"""Dynamiek-uitbreidingen (fase C): expander, multiband-compressor, transient
shaper.

Alle drie werken met een gelinkte detector over de kanalen (1 ms-blokken,
max-abs) zodat het stereobeeld nooit scheeftrekt, net als de native dynamics.
De multiband-compressor splitst met FFT-maskers waarvan de som exact 1 is:
onbewerkt tellen de banden bit-voor-bit op tot het origineel (geen
crossover-kleuring), en de compressie per band gebruikt de bestaande
dsp.compressor (native of scipy-fallback).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_BLOCK_S = 0.001


def _as2d(x: np.ndarray) -> np.ndarray:
    return x[None, :] if x.ndim == 1 else x


def _block_env_db(x2: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Gelinkte max-abs-envelop per 1 ms-blok, in dB."""
    block = max(1, int(sr * _BLOCK_S))
    n = x2.shape[1]
    nb = (n + block - 1) // block
    det = np.abs(x2).max(axis=0)
    padded = np.zeros(nb * block, dtype=det.dtype)
    padded[:n] = det
    env = 20.0 * np.log10(padded.reshape(nb, block).max(axis=1) + 1e-10)
    return env, block


def _smooth_updown(values: np.ndarray, up_coeff: float, down_coeff: float) -> np.ndarray:
    """Eén-pools smoothing met aparte coëfficiënten voor stijgen en dalen."""
    out = np.empty_like(values)
    v = values[0]
    for i, s in enumerate(values):
        c = up_coeff if s > v else down_coeff
        v = c * v + (1.0 - c) * s
        out[i] = v
    return out


def _per_sample(gain_db: np.ndarray, block: int, n: int) -> np.ndarray:
    centers = np.arange(gain_db.shape[0]) * block + block / 2.0
    return np.interp(np.arange(n), centers,
                     10.0 ** (gain_db / 20.0)).astype(np.float32)


def _coeff(ms: float, block_s: float) -> float:
    return float(np.exp(-block_s / (0.001 * ms))) if ms > 0 else 0.0


def expander(x: np.ndarray, sr: int, threshold_db: float = -45.0,
             ratio: float = 2.0, attack_ms: float = 5.0,
             release_ms: float = 120.0, range_db: float = 24.0) -> np.ndarray:
    """Downward expander: onder de drempel wordt zacht zachter (ratio), tot
    maximaal range_db demping. De mildere broer van de gate — pauzes zakken
    weg zonder het hakkerige open/dicht van een harde gate."""
    if ratio < 1.0:
        raise ValueError(f"ratio moet >= 1 zijn (niet {ratio}).")
    x2 = _as2d(x).astype(np.float32)
    env, block = _block_env_db(x2, sr)
    below = np.minimum(0.0, env - threshold_db)
    static = np.clip(below * (ratio - 1.0), -abs(range_db), 0.0)
    # openen (gain omhoog) met attack, sluiten met release
    aA, aR = _coeff(attack_ms, _BLOCK_S), _coeff(release_ms, _BLOCK_S)
    gain_db = _smooth_updown(static, aA, aR)
    g = _per_sample(gain_db, block, x2.shape[1])
    return x2 * g[None, :]


def transient_shaper(x: np.ndarray, sr: int, attack_db: float = 0.0,
                     sustain_db: float = 0.0, attack_window_ms: float = 30.0,
                     sustain_release_ms: float = 200.0) -> np.ndarray:
    """Transient shaper: attack_db regelt de aanzetten (+ = puntiger,
    - = ronder), sustain_db de staarten/nagalm (+ = langer aanvoelend,
    - = droger). Niveau-onafhankelijk, met twee differentiële detectors
    (SPL-stijl): aanzet = verschil van twee volgers met verschillende
    attack-tijd en gelijke release (dooft na ~attack_window_ms uit); staart =
    verschil van twee volgers met verschillende release-tijd en gelijke
    attack (alleen actief terwijl het signaal wegsterft)."""
    x2 = _as2d(x).astype(np.float32)
    if attack_db == 0.0 and sustain_db == 0.0:
        return x2
    env, block = _block_env_db(x2, sr)
    # stilte telt als -60 dB, niet als -oneindig: anders schaalt het
    # detectorverschil mee met een onbegrensde niveausprong uit een pauze
    env = np.maximum(env, -60.0)

    def follow(attack_ms: float, release_ms: float) -> np.ndarray:
        return _smooth_updown(env, _coeff(attack_ms, _BLOCK_S),
                              _coeff(release_ms, _BLOCK_S))

    fast = follow(1.0, 50.0)
    att = fast - follow(attack_window_ms, 50.0)        # >0 alleen op de aanzet
    sus = follow(1.0, sustain_release_ms) - fast       # >0 alleen in de staart
    gain_db = (attack_db * np.clip(att / 6.0, 0.0, 1.0)
               + sustain_db * np.clip(sus / 6.0, 0.0, 1.0))
    g = _per_sample(gain_db, block, x2.shape[1])
    return x2 * g[None, :]


def _band_masks(n: int, sr: int, crossovers: list[float]) -> np.ndarray:
    """Raised-cosine laagdoorlaatmaskers per crossover; de bandmaskers zijn de
    verschillen en tellen dus per definitie op tot exact 1."""
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    lps = []
    for fc in crossovers:
        w = fc * 0.25
        t = np.clip((freqs - (fc - w)) / (2 * w), 0.0, 1.0)
        lps.append(0.5 * (1.0 + np.cos(np.pi * t)))
    # opbouw: band0 = lp0, band_i = lp_i - lp_{i-1}, laatste = 1 - lp_laatste
    masks = [lps[0]]
    for i in range(1, len(lps)):
        masks.append(lps[i] - lps[i - 1])
    masks.append(1.0 - lps[-1])
    return np.stack(masks)


def multiband_compressor(x: np.ndarray, sr: int,
                         crossovers: list[float] | None = None,
                         threshold_db: float | list[float] = -24.0,
                         ratio: float | list[float] = 2.5,
                         attack_ms: float = 15.0, release_ms: float = 150.0,
                         knee_db: float = 6.0, makeup_db: float = 0.0,
                         band_gains_db: list[float] | None = None) -> np.ndarray:
    """Multiband-compressie: splits in banden (zero-fase FFT-maskers die exact
    optellen tot het origineel), comprimeer per band, som terug.

    crossovers: kantelfrequenties (default [200, 2000] = laag/mid/hoog);
    threshold_db en ratio mogen per band een lijst zijn; band_gains_db geeft
    elke band na compressie nog een eigen gain (klankbalans)."""
    from chat_with_audio import dsp

    crossovers = sorted(float(f) for f in (crossovers or [200.0, 2000.0]))
    if not crossovers or crossovers[0] < 40.0 or crossovers[-1] > sr / 2 - 500:
        raise ValueError("crossovers moeten tussen 40 Hz en Nyquist-500 liggen.")
    n_bands = len(crossovers) + 1

    def _per_band(v, name):
        vals = list(v) if isinstance(v, (list, tuple)) else [v] * n_bands
        if len(vals) != n_bands:
            raise ValueError(f"{name}: {len(vals)} waarden voor {n_bands} banden.")
        return [float(f) for f in vals]

    thresholds = _per_band(threshold_db, "threshold_db")
    ratios = _per_band(ratio, "ratio")
    gains = _per_band(band_gains_db if band_gains_db is not None else 0.0,
                      "band_gains_db")

    x2 = _as2d(x).astype(np.float32)
    n = x2.shape[1]
    spec = np.fft.rfft(x2.astype(np.float64), axis=1)
    masks = _band_masks(n, sr, crossovers)
    out = np.zeros_like(x2, dtype=np.float64)
    for i in range(n_bands):
        band = np.fft.irfft(spec * masks[i][None, :], n=n, axis=1).astype(np.float32)
        comp = dsp.compressor(band, sr, thresholds[i], ratios[i],
                              attack_ms, release_ms, knee_db, 0.0)
        out += comp.astype(np.float64) * (10.0 ** (gains[i] / 20.0))
    if makeup_db:
        out *= 10.0 ** (makeup_db / 20.0)
    return out.astype(np.float32)
