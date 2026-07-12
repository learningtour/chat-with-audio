"""Gereedschapsstappen: trim, kanaalbewerkingen, fase, dynamiek-extra's, M/S,
leader-generatoren. Puur numpy/scipy — geen native afhankelijkheid; alles is
(channels, n) float32 in en uit, net als de rest van de keten.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from chat_with_audio import dsp


def _2d(x: np.ndarray) -> np.ndarray:
    return x[None, :] if x.ndim == 1 else x


# ---------------------------------------------------------------- trim & tijd

def _frame_active(x2: np.ndarray, sr: int, threshold_db: float) -> np.ndarray:
    """Boolean per 10 ms-frame: zit er modulatie boven threshold_db?"""
    mono = x2.mean(axis=0).astype(np.float64)
    hop = max(1, int(sr * 0.01))
    n_fr = max(1, mono.shape[0] // hop)
    fr = mono[: n_fr * hop].reshape(n_fr, hop)
    rms_db = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-20)
    return rms_db > threshold_db


def trim(x: np.ndarray, sr: int, start_s: float | None = None,
         end_s: float | None = None, to_modulation: bool = False,
         threshold_db: float = -60.0, pad_s: float = 0.5) -> np.ndarray:
    """Knip kop/staart. Expliciet (start_s/end_s) of automatisch tot de eerste/
    laatste modulatie boven threshold_db, met pad_s marge eromheen."""
    x2 = _2d(x)
    n = x2.shape[1]
    if to_modulation:
        active = _frame_active(x2, sr, threshold_db)
        hop = max(1, int(sr * 0.01))
        idx = np.where(active)[0]
        if idx.size == 0:
            return x2.astype(np.float32)
        a = max(0, int(idx[0] * hop - pad_s * sr))
        b = min(n, int((idx[-1] + 1) * hop + pad_s * sr))
        return x2[:, a:b].astype(np.float32)
    a = int((start_s or 0.0) * sr)
    b = int(end_s * sr) if end_s is not None else n
    a, b = max(0, a), min(n, b)
    if b <= a:
        raise ValueError(f"Leeg trimvenster: start_s={start_s}, end_s={end_s}.")
    return x2[:, a:b].astype(np.float32)


def insert_silence(x: np.ndarray, sr: int, at_s: float = 0.0,
                   duration_s: float = 1.0) -> np.ndarray:
    """Voeg stilte in op at_s (0 = kop-offset); alles erna schuift op."""
    x2 = _2d(x)
    i = max(0, min(x2.shape[1], int(at_s * sr)))
    gap = np.zeros((x2.shape[0], int(duration_s * sr)), dtype=np.float32)
    return np.concatenate([x2[:, :i], gap, x2[:, i:]], axis=1).astype(np.float32)


# ---------------------------------------------------------------- fase & kanalen

def _channel_index(x2: np.ndarray, channel: str | int) -> list[int]:
    names = {"left": 0, "right": 1, "all": -1}
    c = names.get(channel, channel) if isinstance(channel, str) else channel
    if c == -1:
        return list(range(x2.shape[0]))
    if not isinstance(c, int) or not 0 <= c < x2.shape[0]:
        raise ValueError(f"Onbekend kanaal '{channel}' voor {x2.shape[0]} kanalen "
                         "(left/right/all of 0-gebaseerde index).")
    return [c]


def polarity_invert(x: np.ndarray, sr: int, channel: str | int = "all") -> np.ndarray:
    """Polariteit (fase) omkeren, per kanaal of allemaal."""
    x2 = _2d(x).astype(np.float32).copy()
    for c in _channel_index(x2, channel):
        x2[c] = -x2[c]
    return x2


def sample_delay(x: np.ndarray, sr: int, channel: str | int, samples: int = 0,
                 ms: float | None = None) -> np.ndarray:
    """Vertraag één kanaal (mic-paar-uitlijning, Haas): pad kop, staart eraf,
    lengte blijft gelijk. Negatief = vervroegen."""
    x2 = _2d(x).astype(np.float32).copy()
    d = int(round(ms * sr / 1000.0)) if ms is not None else int(samples)
    if d == 0:
        return x2
    for c in _channel_index(x2, channel):
        if d > 0:
            x2[c] = np.concatenate([np.zeros(d, dtype=np.float32), x2[c, :-d]])
        else:
            x2[c] = np.concatenate([x2[c, -d:], np.zeros(-d, dtype=np.float32)])
    return x2


def to_mono(x: np.ndarray, sr: int, mode: str = "sum") -> np.ndarray:
    """Downmix naar mono: sum (equal-power gemiddelde), left of right."""
    x2 = _2d(x)
    if mode == "sum":
        return x2.mean(axis=0, keepdims=True).astype(np.float32)
    if mode in ("left", "right"):
        return x2[_channel_index(x2, mode)].astype(np.float32)
    raise ValueError(f"Onbekende to_mono-modus '{mode}' (sum/left/right).")


def dual_mono(x: np.ndarray, sr: int, source: str = "sum") -> np.ndarray:
    """Mono (of één kanaal) naar identiek L=R stereo — dual-mono-levering."""
    m = to_mono(x, sr, mode=source)
    return np.vstack([m, m]).astype(np.float32)


def swap_channels(x: np.ndarray, sr: int) -> np.ndarray:
    """L en R wisselen (alleen stereo)."""
    x2 = _2d(x)
    if x2.shape[0] != 2:
        raise ValueError(f"swap_channels verwacht stereo, kreeg {x2.shape[0]} kanalen.")
    return x2[::-1].astype(np.float32).copy()


def mid_side(x: np.ndarray, sr: int, width: float = 1.0, mid_db: float = 0.0,
             side_db: float = 0.0) -> np.ndarray:
    """M/S-bewerking op stereo: width schaalt S (0 = mono, 1 = origineel,
    >1 = breder), mid_db/side_db zijn extra gains op M en S."""
    x2 = _2d(x).astype(np.float64)
    if x2.shape[0] != 2:
        raise ValueError(f"mid_side verwacht stereo, kreeg {x2.shape[0]} kanalen.")
    m = 0.5 * (x2[0] + x2[1]) * 10 ** (mid_db / 20)
    s = 0.5 * (x2[0] - x2[1]) * width * 10 ** (side_db / 20)
    return np.vstack([m + s, m - s]).astype(np.float32)


def _lr4_split(x2: np.ndarray, sr: int, freq: float) -> tuple[np.ndarray, np.ndarray]:
    """Linkwitz-Riley 4e-orde-splitsing (2x Butterworth 2e orde in cascade):
    laag + hoog sommeren vlak (allpass-fase)."""
    sos_lo = butter(2, freq, btype="low", fs=sr, output="sos")
    sos_hi = butter(2, freq, btype="high", fs=sr, output="sos")
    lo = sosfilt(sos_lo, sosfilt(sos_lo, x2, axis=1), axis=1)
    hi = sosfilt(sos_hi, sosfilt(sos_hi, x2, axis=1), axis=1)
    return lo, hi


def bass_mono(x: np.ndarray, sr: int, freq: float = 120.0) -> np.ndarray:
    """Laag onder freq naar mono (vinyl/club/translatie); daarboven blijft
    het stereobeeld intact. LR4-splitsing, dus de som blijft vlak."""
    x2 = _2d(x).astype(np.float64)
    if x2.shape[0] != 2:
        return x2.astype(np.float32)
    lo, hi = _lr4_split(x2, sr, freq)
    lo_m = lo.mean(axis=0, keepdims=True)
    return (np.vstack([lo_m, lo_m]) + hi).astype(np.float32)


# ---------------------------------------------------------------- dynamiek

def _envelope_db(x2: np.ndarray, sr: int, attack_ms: float,
                 release_ms: float) -> np.ndarray:
    """Gelinkte piek-envelope (max over kanalen) met attack/release, in dB.
    Blokgebaseerd (1 ms) zodat het zonder native code snel genoeg is."""
    block = max(1, int(sr * 0.001))
    det = np.abs(x2).max(axis=0)
    n_b = int(np.ceil(det.shape[0] / block))
    pad = np.pad(det, (0, n_b * block - det.shape[0]))
    peaks = pad.reshape(n_b, block).max(axis=1)
    a = np.exp(-block / (max(attack_ms, 0.1) / 1000 * sr))
    r = np.exp(-block / (max(release_ms, 1.0) / 1000 * sr))
    env = np.empty(n_b)
    e = 0.0
    for i, p in enumerate(peaks):
        e = (a * e + (1 - a) * p) if p > e else (r * e + (1 - r) * p)
        env[i] = e
    env_full = np.repeat(env, block)[: det.shape[0]]
    return 20 * np.log10(env_full + 1e-10)


def expander(x: np.ndarray, sr: int, threshold_db: float = -45.0,
             ratio: float = 2.0, attack_ms: float = 5.0,
             release_ms: float = 120.0, range_db: float = 24.0) -> np.ndarray:
    """Neerwaartse expander: onder de drempel wordt elke dB er (ratio-1) extra
    dB's onder geduwd, tot maximaal range_db. De zachte gate: ademruimte en
    ruis zakken weg zonder het hakkelen van een harde gate."""
    x2 = _2d(x).astype(np.float64)
    env_db = _envelope_db(x2, sr, attack_ms, release_ms)
    under = np.minimum(env_db - threshold_db, 0.0)
    gain_db = np.maximum(under * (ratio - 1.0), -abs(range_db))
    return (x2 * 10 ** (gain_db / 20)[None, :]).astype(np.float32)


def multiband_compressor(x: np.ndarray, sr: int,
                         crossovers: list | None = None,
                         threshold_db: float = -28.0, ratio: float = 2.0,
                         attack_ms: float = 15.0, release_ms: float = 150.0,
                         makeup_db: float = 0.0) -> np.ndarray:
    """Meerbands-compressie: LR4-splitsing op de crossovers (standaard 200 en
    2000 Hz), per band dezelfde compressor, som weer bij elkaar. Houdt de
    balans vast waar een breedbandcompressor zou gaan pompen."""
    x2 = _2d(x).astype(np.float64)
    freqs = sorted(float(f) for f in (crossovers or [200.0, 2000.0]))
    if any(f <= 20 or f >= sr / 2 for f in freqs):
        raise ValueError(f"Crossovers moeten tussen 20 Hz en Nyquist liggen: {freqs}")
    bands: list[np.ndarray] = []
    rest = x2
    for f in freqs:
        lo, rest = _lr4_split(rest, sr, f)
        bands.append(lo)
    bands.append(rest)
    out = np.zeros_like(x2)
    for band in bands:
        out += dsp.compressor(band.astype(np.float32), sr, threshold_db, ratio,
                              attack_ms, release_ms, 6.0, 0.0).astype(np.float64)
    return (out * 10 ** (makeup_db / 20)).astype(np.float32)


def transient_shaper(x: np.ndarray, sr: int, attack_db: float = 0.0,
                     sustain_db: float = 0.0) -> np.ndarray:
    """Transient shaper: attack_db kleurt de aanzetten (sneller/zachter),
    sustain_db de staarten/kamer. Werkt op het verschil tussen een snelle en
    een trage envelope — niveau-onafhankelijk, geen drempel nodig."""
    x2 = _2d(x).astype(np.float64)
    fast = _envelope_db(x2, sr, attack_ms=1.0, release_ms=50.0)
    slow = _envelope_db(x2, sr, attack_ms=30.0, release_ms=300.0)
    d = fast - slow  # >0 rond aanzetten, <0 in staarten
    gain_db = attack_db * np.clip(d / 6.0, 0.0, 1.0) + \
        sustain_db * np.clip(-d / 6.0, 0.0, 1.0)
    return (x2 * 10 ** (gain_db / 20)[None, :]).astype(np.float32)


def tilt_eq(x: np.ndarray, sr: int, tilt_db: float = 0.0,
            pivot_hz: float = 650.0) -> np.ndarray:
    """Tilt-EQ: kantel het spectrum rond pivot_hz (positief = helderder,
    negatief = warmer) met een gekoppeld shelvingpaar."""
    half = tilt_db / 2.0
    return dsp.eq(x, sr, [
        {"type": "lowshelf", "freq": pivot_hz, "gain_db": -half, "q": 0.5},
        {"type": "highshelf", "freq": pivot_hz, "gain_db": half, "q": 0.5},
    ])


# ---------------------------------------------------------------- leader

def tone_slate(x: np.ndarray, sr: int, tone_s: float = 10.0,
               level_db: float = -18.0, freq: float = 1000.0,
               gap_s: float = 1.0) -> np.ndarray:
    """Broadcast-leader: referentietoon (standaard 1 kHz op -18 dBFS) plus
    stilte vóór het programma. De toon krijgt 10 ms fades tegen klikken."""
    x2 = _2d(x)
    n_tone = int(tone_s * sr)
    t = np.arange(n_tone) / sr
    tone = (10 ** (level_db / 20) * np.sqrt(2.0)) * np.sin(2 * np.pi * freq * t)
    edge = min(int(0.01 * sr), max(1, n_tone // 2))
    env = np.ones(n_tone)
    env[:edge] = np.linspace(0, 1, edge)
    env[-edge:] = np.linspace(1, 0, edge)
    lead = np.tile(tone * env, (x2.shape[0], 1))
    gap = np.zeros((x2.shape[0], int(gap_s * sr)))
    return np.concatenate([lead, gap, x2], axis=1).astype(np.float32)


def two_pop(x: np.ndarray, sr: int, offset_s: float = 2.0, freq: float = 1000.0,
            pop_ms: float = 42.0, level_db: float = -18.0) -> np.ndarray:
    """2-pop: één framelang 1 kHz-piepje precies offset_s vóór programme start
    (sync-referentie voor beeld). Prepend [pop + stilte], programma schuift op."""
    x2 = _2d(x)
    n_pop = max(1, int(pop_ms / 1000 * sr))
    t = np.arange(n_pop) / sr
    pop = (10 ** (level_db / 20) * np.sqrt(2.0)) * np.sin(2 * np.pi * freq * t)
    edge = max(1, int(0.002 * sr))
    env = np.ones(n_pop)
    env[:edge] = np.linspace(0, 1, min(edge, n_pop))
    env[-edge:] = np.linspace(1, 0, min(edge, n_pop))
    silence = np.zeros((x2.shape[0], max(0, int(offset_s * sr) - n_pop)))
    lead = np.concatenate([np.tile(pop * env, (x2.shape[0], 1)), silence], axis=1)
    return np.concatenate([lead, x2], axis=1).astype(np.float32)
