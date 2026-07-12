"""Spectrale reparatie ('painting'): een beschadigde tijd-frequentieplek
vervangen door wat de omgeving suggereert.

Het RX-achtige gereedschap van de restaurateur: een kuch over een stilte,
een stoelpiep in muziek, een tik in de galm — je wijst de plek (tijd en
optioneel frequentieband) aan en de patch wordt opnieuw 'geschilderd' door
de magnitudes per bin lineair te interpoleren tussen de context links en
rechts van de schade (mediaan over de contextframes, dus robuust tegen
toevallige uitschieters). De fase wordt coherent voortgezet vanuit de
context (vocoder-stijl: fasehoek + verwachte toename per frame), zodat
tonale inhoud gewoon dóórloopt in plaats van uit te doven in de
overlap-add. Buiten de patch blijft alles bit-voor-bit onaangetast
(harde splice met crossfade).

Eerlijk over de grens: over doorlopende spraak heen schilderen vervaagt de
spraak zelf — dit gereedschap is voor schade óver of náást programma, niet
voor het terugtoveren van verloren woorden.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import istft, stft

log = logging.getLogger(__name__)


def spectral_repair(x: np.ndarray, sr: int, start_s: float, end_s: float,
                    low_hz: float | None = None, high_hz: float | None = None,
                    context_s: float = 0.5, fade_ms: float = 15.0) -> np.ndarray:
    """Repareer [start_s, end_s] × [low_hz, high_hz] vanuit de context.

    low_hz/high_hz None = volledige band. Geeft nieuwe audio terug; alles
    buiten de patch (plus een korte crossfade-marge) is bit-voor-bit gelijk.
    """
    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    dur = n / sr
    start_s = max(0.0, float(start_s))
    end_s = min(dur, float(end_s))
    if end_s - start_s <= 0.005:
        raise ValueError("Patch te kort: geef een bereik van minstens 5 ms.")
    if end_s - start_s > 5.0:
        raise ValueError("Patch langer dan 5 s: spectrale reparatie is voor "
                         "korte schade; gebruik smart_edit/denoise voor lange "
                         "stukken.")

    nfft = 2048 if sr >= 32000 else 1024
    hop = nfft // 4
    lo = float(low_hz) if low_hz else 0.0
    hi = float(high_hz) if high_hz else sr / 2.0
    if hi <= lo:
        raise ValueError(f"high_hz ({hi}) moet boven low_hz ({lo}) liggen.")

    y = x2.astype(np.float32).copy()

    for ci in range(x2.shape[0]):
        freqs, times, spec = stft(x2[ci], fs=sr, window="hann", nperseg=nfft,
                                  noverlap=nfft - hop, padded=True)
        t_sel = np.where((times >= start_s) & (times <= end_s))[0]
        if t_sel.size == 0:
            continue
        f_sel = np.where((freqs >= lo) & (freqs <= hi))[0]
        ctx = max(3, int(context_s * sr / hop))
        left = np.arange(max(0, t_sel[0] - ctx), t_sel[0])
        right = np.arange(t_sel[-1] + 1, min(spec.shape[1], t_sel[-1] + 1 + ctx))
        if left.size == 0 and right.size == 0:
            raise ValueError("Geen context rond de patch om uit te schilderen.")

        mag = np.abs(spec)
        mag_l = (np.median(mag[np.ix_(f_sel, left)], axis=1)
                 if left.size else np.median(mag[np.ix_(f_sel, right)], axis=1))
        mag_r = (np.median(mag[np.ix_(f_sel, right)], axis=1)
                 if right.size else mag_l)

        w = (np.linspace(0.0, 1.0, t_sel.size + 2)[1:-1]
             if t_sel.size > 1 else np.array([0.5]))
        new_mag = mag_l[:, None] * (1.0 - w[None, :]) + mag_r[:, None] * w[None, :]
        # Coherente fase (phase vocoder): de wérkelijke fasetoename per bin
        # meten uit twee opeenvolgende contextframes en die voortzetten.
        # Bin-centerfrequenties gebruiken zou de mainlobe-bins van één toon
        # uit elkaar laten driften (hoorbaar als uitdoven/smeren).
        if left.size >= 2:
            base_phase = np.angle(spec[f_sel, left[-1]])
            dphi = np.angle(spec[f_sel, left[-1]]) - np.angle(spec[f_sel, left[-2]])
            steps = t_sel - left[-1]
        elif right.size >= 2:
            base_phase = np.angle(spec[f_sel, right[0]])
            dphi = np.angle(spec[f_sel, right[1]]) - np.angle(spec[f_sel, right[0]])
            steps = t_sel - right[0]
        else:  # minimale context: val terug op bin-centerfrequenties
            anchor = left[-1] if left.size else right[0]
            base_phase = np.angle(spec[f_sel, anchor])
            dphi = 2.0 * np.pi * freqs[f_sel] * hop / sr
            steps = t_sel - anchor
        phase = base_phase[:, None] + dphi[:, None] * steps[None, :]
        spec[np.ix_(f_sel, t_sel)] = new_mag * np.exp(1j * phase)

        _, rec = istft(spec, fs=sr, window="hann", nperseg=nfft,
                       noverlap=nfft - hop)
        rec = rec[:n].astype(np.float32)
        if rec.shape[0] < n:
            rec = np.pad(rec, (0, n - rec.shape[0]))

        # harde splice: alleen de patch (+ crossfade) komt uit de reconstructie
        a = max(0, int(start_s * sr) - nfft)
        b = min(n, int(end_s * sr) + nfft)
        fade = max(8, int(fade_ms / 1000 * sr))
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
        seg = rec[a:b].astype(np.float64)
        org = x2[ci, a:b].astype(np.float64)
        wmix = np.ones(b - a)
        if a > 0:
            wmix[:fade] = ramp
        if b < n:
            wmix[-fade:] = np.minimum(wmix[-fade:], ramp[::-1])
        y[ci, a:b] = (org * (1.0 - wmix) + seg * wmix).astype(np.float32)

    log.info("spectral_repair: %0.2f-%0.2f s, %0.0f-%0.0f Hz",
             start_s, end_s, lo, hi)
    return y
