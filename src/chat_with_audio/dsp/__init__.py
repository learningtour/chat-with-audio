"""DSP-laag: dispatch tussen de native C++-kern (_dsp) en de pure-Python fallback.

Alle functies accepteren mono (n,) of multichannel (channels, n) float-arrays en
geven een nieuwe array met dezelfde vorm terug.
"""

from __future__ import annotations

import numpy as np

try:
    from chat_with_audio import _dsp as _impl

    _BACKEND = "native"
except ImportError:  # native build ontbreekt: pure-Python fallback
    from chat_with_audio.dsp import fallback as _impl

    _BACKEND = "fallback"


def backend() -> str:
    """'native' (C++) of 'fallback' (scipy/numpy)."""
    return _BACKEND


def _as2d(x) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return np.ascontiguousarray(x[None, :]), True
    if x.ndim != 2:
        raise ValueError("audio moet 1D (mono) of 2D (channels, n) zijn")
    return np.ascontiguousarray(x), False


def _restore(y: np.ndarray, was_1d: bool) -> np.ndarray:
    return y[0] if was_1d else y


def _norm_bands(bands) -> list[tuple[str, float, float, float]]:
    out = []
    for b in bands:
        if isinstance(b, dict):
            out.append((str(b["type"]), float(b["freq"]), float(b.get("gain_db", 0.0)),
                        float(b.get("q", 0.707))))
        else:
            t, f, g, q = b
            out.append((str(t), float(f), float(g), float(q)))
    return out


def gain(x, db: float) -> np.ndarray:
    x2, was_1d = _as2d(x)
    return _restore(_impl.apply_gain(x2, float(db)), was_1d)


def eq(x, sr: int, bands) -> np.ndarray:
    """Serie biquads. bands: [{'type','freq','gain_db','q'}, ...] of tuples.

    Types: lowpass, highpass, peaking, lowshelf, highshelf, notch.
    """
    x2, was_1d = _as2d(x)
    return _restore(_impl.biquad_chain(x2, float(sr), _norm_bands(bands)), was_1d)


def highpass(x, sr: int, freq: float, q: float = 0.707) -> np.ndarray:
    return eq(x, sr, [("highpass", freq, 0.0, q)])


def lowpass(x, sr: int, freq: float, q: float = 0.707) -> np.ndarray:
    return eq(x, sr, [("lowpass", freq, 0.0, q)])


def notch(x, sr: int, freq: float, q: float = 30.0) -> np.ndarray:
    return eq(x, sr, [("notch", freq, 0.0, q)])


def noise_gate(x, sr: int, threshold_db: float, attack_ms: float = 5.0,
               release_ms: float = 120.0, hold_ms: float = 50.0,
               range_db: float = 12.0) -> np.ndarray:
    x2, was_1d = _as2d(x)
    y = _impl.noise_gate(x2, float(sr), float(threshold_db), float(attack_ms),
                         float(release_ms), float(hold_ms), float(range_db))
    return _restore(y, was_1d)


def compressor(x, sr: int, threshold_db: float, ratio: float = 3.0,
               attack_ms: float = 10.0, release_ms: float = 150.0,
               knee_db: float = 6.0, makeup_db: float = 0.0) -> np.ndarray:
    x2, was_1d = _as2d(x)
    y = _impl.compressor(x2, float(sr), float(threshold_db), float(ratio),
                         float(attack_ms), float(release_ms), float(knee_db),
                         float(makeup_db))
    return _restore(y, was_1d)


def limiter(x, sr: int, ceiling_db: float = -1.5, release_ms: float = 60.0,
            lookahead_ms: float = 5.0) -> np.ndarray:
    x2, was_1d = _as2d(x)
    y = _impl.limiter(x2, float(sr), float(ceiling_db), float(release_ms),
                      float(lookahead_ms))
    return _restore(y, was_1d)


def spectral_denoise(x, sr: int, reduction_db: float = 12.0, **kwargs) -> np.ndarray:
    """Tier A ruisonderdrukking: STFT spectral gating (altijd beschikbaar)."""
    from chat_with_audio.dsp.spectral_nr import spectral_denoise as _f

    x2, was_1d = _as2d(x)
    return _restore(_f(x2, sr, reduction_db=reduction_db, **kwargs), was_1d)


def ai_denoise_available() -> bool:
    from chat_with_audio.dsp import ai_nr

    return ai_nr.is_available()


def ai_denoise(x, sr: int, strength_db: float | None = None) -> np.ndarray:
    """Tier B ruisonderdrukking: DeepFilterNet (vereist het [ai]-extra)."""
    from chat_with_audio.dsp import ai_nr

    x2, was_1d = _as2d(x)
    return _restore(ai_nr.denoise(x2, sr, atten_lim_db=strength_db), was_1d)
