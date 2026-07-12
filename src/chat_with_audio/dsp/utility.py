"""Utility-DSP (fase C): trim, polariteit, sample-delay, kanalen, M/S, bass-mono.

Kleine, exacte gereedschappen die in elke postproductielijst opduiken. Alles
float32 (channels, n) in en uit; geen enkel gereedschap verandert klank waar
het niet om gevraagd is (polarity flipt alleen het teken, channel_map mengt
alleen kanalen, bass-mono raakt alleen het laag onder de kantelfrequentie).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_FRAME_S = 0.010  # meetframe voor to_modulation


def _as2d(x: np.ndarray) -> np.ndarray:
    return x[None, :] if x.ndim == 1 else x


def trim(x: np.ndarray, sr: int, start_s: float | None = None,
         end_s: float | None = None, to_modulation: bool = False,
         threshold_db: float = -60.0, keep_s: float = 0.25,
         pad_head_s: float = 0.0, pad_tail_s: float = 0.0) -> np.ndarray:
    """Kop/staart wegsnijden en/of stilte aanzetten.

    start_s/end_s snijden expliciet; to_modulation snijdt tot de eerste/laatste
    modulatie boven threshold_db, met keep_s marge. pad_head_s/pad_tail_s
    zetten daarna digitale stilte vóór/achter (leader, frame-offsets).
    """
    x2 = _as2d(x).astype(np.float32)
    n = x2.shape[1]
    a = int(max(0.0, start_s or 0.0) * sr)
    b = int(end_s * sr) if end_s is not None else n
    b = max(a, min(b, n))
    if to_modulation:
        mono = x2.mean(axis=0).astype(np.float64)
        flen = max(1, int(sr * _FRAME_S))
        nf = max(1, mono.shape[0] // flen)
        lv = 10 * np.log10((mono[:nf * flen].reshape(nf, flen) ** 2).mean(axis=1)
                           + 1e-20)
        above = np.where(lv > threshold_db)[0]
        if above.size:
            a = max(a, int(above[0] * flen - keep_s * sr))
            b = min(b, int((above[-1] + 1) * flen + keep_s * sr))
        else:
            log.warning("trim: geen modulatie boven %.0f dB gevonden; "
                        "kop/staart onaangetast", threshold_db)
    if b - a < 1:
        raise ValueError("trim zou niets overlaten (start/eind controleren).")
    y = x2[:, a:b]
    if pad_head_s > 0 or pad_tail_s > 0:
        y = np.pad(y, ((0, 0), (int(pad_head_s * sr), int(pad_tail_s * sr))))
    return np.ascontiguousarray(y)


def polarity_invert(x: np.ndarray, channels: list[int] | None = None) -> np.ndarray:
    """Polariteit (fase) omklappen; channels = 0-gebaseerde lijst of alles."""
    x2 = _as2d(x).astype(np.float32).copy()
    idx = range(x2.shape[0]) if channels is None else channels
    for c in idx:
        if not 0 <= c < x2.shape[0]:
            raise ValueError(f"Kanaal {c} bestaat niet (bestand heeft "
                             f"{x2.shape[0]} kanalen).")
        x2[c] = -x2[c]
    return x2


def sample_delay(x: np.ndarray, sr: int, samples: int | None = None,
                 ms: float | None = None, channel: int | None = None) -> np.ndarray:
    """Verschuif een kanaal (of alles) in de tijd; lengte blijft gelijk.

    Positief = later (naar achteren), negatief = eerder. Voor microfoonpaar-
    uitlijning (channel=0/1) of een vaste AV-offset (channel=None).
    """
    if (samples is None) == (ms is None):
        raise ValueError("Geef samples óf ms op voor sample_delay.")
    d = int(samples) if samples is not None else int(round(ms / 1000.0 * sr))
    x2 = _as2d(x).astype(np.float32).copy()
    if channel is not None and not 0 <= channel < x2.shape[0]:
        raise ValueError(f"Kanaal {channel} bestaat niet (bestand heeft "
                         f"{x2.shape[0]} kanalen).")
    if d == 0:
        return x2
    if abs(d) >= x2.shape[1]:
        raise ValueError("Delay is langer dan het bestand.")
    rows = [channel] if channel is not None else list(range(x2.shape[0]))
    for c in rows:
        if d > 0:
            x2[c, d:] = x2[c, :-d].copy()
            x2[c, :d] = 0.0
        else:
            x2[c, :d] = x2[c, -d:].copy()
            x2[c, d:] = 0.0
    return x2


def channel_map(x: np.ndarray, mode: str | None = None,
                order: list[int] | None = None) -> np.ndarray:
    """Kanaalgereedschap: to_mono | dual_mono | swap, of een expliciete order.

    order is een lijst bron-kanaalindexen per uitgangskanaal, bv. [1, 0]
    (omwisselen) of [0, 0] (links als dual-mono).
    """
    x2 = _as2d(x).astype(np.float32)
    if order is not None:
        for c in order:
            if not 0 <= c < x2.shape[0]:
                raise ValueError(f"Kanaal {c} bestaat niet (bestand heeft "
                                 f"{x2.shape[0]} kanalen).")
        return np.ascontiguousarray(x2[order])
    if mode == "to_mono":
        return x2.mean(axis=0, keepdims=True).astype(np.float32)
    if mode == "dual_mono":
        m = x2.mean(axis=0, keepdims=True) if x2.shape[0] > 1 else x2
        return np.repeat(m, 2, axis=0).astype(np.float32)
    if mode == "swap":
        if x2.shape[0] < 2:
            raise ValueError("swap vraagt minstens 2 kanalen.")
        order2 = list(range(x2.shape[0]))
        order2[0], order2[1] = 1, 0
        return np.ascontiguousarray(x2[order2])
    raise ValueError(f"Onbekende mode '{mode}' "
                     "(to_mono|dual_mono|swap, of geef order).")


def mid_side(x: np.ndarray, width: float = 1.0, mid_db: float = 0.0,
             side_db: float = 0.0) -> np.ndarray:
    """M/S-bewerking op stereo: breedte (0 = mono, 1 = zoals het is, 2 = extra
    breed) plus aparte mid/side-gains in dB. Mono blijft onaangetast."""
    x2 = _as2d(x).astype(np.float64)
    if x2.shape[0] != 2:
        log.warning("mid_side: geen stereo (%d kanalen); onaangetast",
                    x2.shape[0])
        return x2.astype(np.float32)
    if not 0.0 <= width <= 4.0:
        raise ValueError(f"width moet tussen 0 en 4 liggen (niet {width}).")
    m = (x2[0] + x2[1]) * 0.5 * (10.0 ** (mid_db / 20.0))
    s = (x2[0] - x2[1]) * 0.5 * width * (10.0 ** (side_db / 20.0))
    return np.stack([m + s, m - s]).astype(np.float32)


def bass_mono(x: np.ndarray, sr: int, freq: float = 120.0) -> np.ndarray:
    """Maak het laag onder `freq` mono (vinyl/club/translatie); daarboven
    blijft het stereobeeld onaangetast. Zero-fase splitsing via FFT-masker,
    zodat laag + hoog exact optellen tot het origineel."""
    x2 = _as2d(x).astype(np.float64)
    if x2.shape[0] < 2:
        return x2.astype(np.float32)
    if not 20.0 <= freq <= 500.0:
        raise ValueError(f"freq moet tussen 20 en 500 Hz liggen (niet {freq}).")
    n = x2.shape[1]
    spec = np.fft.rfft(x2, axis=1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    ramp = max(10.0, freq * 0.25)
    mask = np.clip(((freq + ramp) - freqs) / (2 * ramp), 0.0, 1.0)
    low = np.fft.irfft(spec * mask[None, :], n=n, axis=1)
    high = x2 - low
    low_mono = low.mean(axis=0, keepdims=True)
    return (high + low_mono).astype(np.float32)
