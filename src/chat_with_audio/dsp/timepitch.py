"""Time & pitch engine (fase B): Signalsmith Stretch + varispeed.

Drie bewerkingen, alle float32 (channels, n) in en uit:
  time_stretch — duur veranderen zonder de toonhoogte te raken
                 (factor = duurverhouding uit/in: 1.25 = 25% langer)
  pitch_shift  — toonhoogte veranderen zonder de duur te raken; met
                 preserve_formants wordt de spectrale envelop per frame
                 teruggecorrigeerd naar het origineel (cepstraal gladgemaakt),
                 zodat een stem niet gaat 'chipmunken'
  varispeed    — tape-stijl: duur én toonhoogte samen, via resampling
                 (factor = snelheid: 1.06 ≈ +1 halve toon)

De engine is Signalsmith Stretch (python-stretch, basisdependency): een
polyfase-STFT-stretcher die in- en uitgangslatentie zelf compenseert; de
uitvoerlengte is exact n/timeFactor. Varispeed heeft geen engine nodig
(resample_poly met een rationale benadering van de factor).
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction

import numpy as np

log = logging.getLogger(__name__)

INSTALL_HINT = ("python-stretch (Signalsmith Stretch) is niet geinstalleerd. "
                "Installeer met: uv sync (in de projectmap).")

STRETCH_RANGE = (0.25, 4.0)      # duur-/snelheidsfactoren buiten dit bereik zijn
SEMITONE_RANGE = (-24.0, 24.0)   # geen bewerking meer maar een effect

# formantcorrectie: STFT-raster + cepstrale envelopgrens
_NPERSEG = 2048
_ENV_MAX_GAIN_DB = 12.0


def is_available() -> bool:
    try:
        import python_stretch  # noqa: F401
        return True
    except Exception:
        return False


def _check(name: str, value: float, lo: float, hi: float) -> float:
    v = float(value)
    if not (lo <= v <= hi) or not math.isfinite(v):
        raise ValueError(f"{name} moet tussen {lo} en {hi} liggen (niet {value}).")
    return v


def _stretch(x2: np.ndarray, sr: int, time_factor: float = 1.0,
             semitones: float = 0.0, tonality_limit_hz: float = 0.0) -> np.ndarray:
    if not is_available():
        raise RuntimeError(INSTALL_HINT)
    import python_stretch as ps

    st = ps.Signalsmith.Stretch()
    st.preset(x2.shape[0], sr)
    st.timeFactor = float(time_factor)
    if semitones:
        st.setTransposeSemitones(float(semitones),
                                 float(tonality_limit_hz) / sr)
    return np.ascontiguousarray(st.process(
        np.ascontiguousarray(x2.astype(np.float32))))


def time_stretch(x: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Duur x factor (1.25 = 25% langer), toonhoogte blijft staan."""
    factor = _check("factor", factor, *STRETCH_RANGE)
    x2 = x[None, :] if x.ndim == 1 else x
    if factor == 1.0:
        return x2.astype(np.float32)
    # Signalsmith: timeFactor > 1 = sneller/korter, dus de inverse van 'langer'
    return _stretch(x2, sr, time_factor=1.0 / factor)


def _envelope(logmag: np.ndarray, cutoff: int) -> np.ndarray:
    """Cepstraal gladgemaakte spectrale envelop per frame.

    logmag: (frames, bins) log-magnitude van een rfft-STFT. De envelop is de
    lage-quefrency-helft van het cepstrum: pitch-harmonischen (hoge quefrency)
    vallen weg, de formantstructuur blijft.
    """
    n_fft = 2 * (logmag.shape[1] - 1)
    ceps = np.fft.irfft(logmag, n=n_fft, axis=1)
    lifter = np.zeros(n_fft)
    lifter[0] = 1.0
    lifter[1:cutoff] = 2.0  # symmetrisch deel dubbel tellen (reëel cepstrum)
    return np.fft.rfft(ceps * lifter[None, :], n=n_fft, axis=1).real


def _match_envelope(y2: np.ndarray, ref2: np.ndarray, sr: int) -> np.ndarray:
    """Corrigeer de spectrale envelop van y2 per frame terug naar die van ref2
    (even lang, zelfde STFT-raster). Dit is de formantcorrectie: de pitch is
    al verschoven, alleen de klankkleur-envelop gaat terug naar het origineel.
    """
    from scipy.signal import istft, stft

    nper = min(_NPERSEG, 1 << int(np.log2(max(256, y2.shape[1]))))
    nover = nper * 3 // 4
    # envelopdetail tot ~1.4 ms quefrency: ruim onder de laagste spreekpitch
    cutoff = max(8, min(int(sr / 700), nper // 4))
    max_gain = 10.0 ** (_ENV_MAX_GAIN_DB / 20.0)

    out = np.empty_like(y2)
    for c in range(y2.shape[0]):
        _, _, Y = stft(y2[c], fs=sr, nperseg=nper, noverlap=nover)
        _, _, R = stft(ref2[c], fs=sr, nperseg=nper, noverlap=nover)
        frames = min(Y.shape[1], R.shape[1])
        Y, R = Y[:, :frames], R[:, :frames]
        env_y = _envelope(np.log(np.abs(Y.T) + 1e-10), cutoff)
        env_r = _envelope(np.log(np.abs(R.T) + 1e-10), cutoff)
        gain = np.exp(np.clip(env_r - env_y, -np.log(max_gain), np.log(max_gain)))
        # stille frames niet 'corrigeren' naar de ruisvloer van het origineel
        quiet = (np.abs(Y.T) ** 2).sum(axis=1) < 1e-8
        gain[quiet] = 1.0
        _, rec = istft(Y * gain.T, fs=sr, nperseg=nper, noverlap=nover)
        n = min(rec.shape[0], y2.shape[1])
        out[c, :n] = rec[:n]
        if n < y2.shape[1]:
            out[c, n:] = y2[c, n:]
    return out.astype(np.float32)


def pitch_shift(x: np.ndarray, sr: int, semitones: float,
                preserve_formants: bool = True) -> np.ndarray:
    """Toonhoogte +/- semitones, duur blijft exact gelijk.

    preserve_formants: de spectrale envelop wordt per frame teruggecorrigeerd
    naar het origineel (en de engine krijgt een tonaliteitslimiet van 8 kHz),
    zodat spraak zijn klankkleur houdt. Zonder formantbehoud schuift de hele
    klankkleur mee — dat is juist het gereedschap voor stem-anonimisering.
    """
    semitones = _check("semitones", semitones, *SEMITONE_RANGE)
    x2 = x[None, :] if x.ndim == 1 else x
    if semitones == 0.0:
        return x2.astype(np.float32)
    limit = 8000.0 if preserve_formants else 0.0
    y = _stretch(x2, sr, semitones=semitones, tonality_limit_hz=limit)
    n = min(x2.shape[1], y.shape[1])
    y = y[:, :n]
    if preserve_formants:
        y = _match_envelope(y, x2[:, :n].astype(np.float32), sr)
    return y


def varispeed(x: np.ndarray, sr: int, factor: float | None = None,
              semitones: float | None = None) -> np.ndarray:
    """Tape-stijl snelheid: duur én toonhoogte samen (resampling).

    factor = snelheid (1.25 = 25% sneller én hoger); of geef semitones op
    (factor = 2**(semitones/12)). Geen engine nodig: rationale resampling.
    """
    from scipy.signal import resample_poly

    if factor is None and semitones is None:
        raise ValueError("Geef factor of semitones op voor varispeed.")
    if factor is not None and semitones is not None:
        raise ValueError("Geef factor óf semitones op, niet allebei.")
    if semitones is not None:
        semitones = _check("semitones", semitones, *SEMITONE_RANGE)
        factor = 2.0 ** (semitones / 12.0)
    factor = _check("factor", factor, *STRETCH_RANGE)
    x2 = x[None, :] if x.ndim == 1 else x
    if factor == 1.0:
        return x2.astype(np.float32)
    frac = Fraction(1.0 / factor).limit_denominator(1000)
    y = resample_poly(x2.astype(np.float64), frac.numerator, frac.denominator,
                      axis=1)
    return y.astype(np.float32)
