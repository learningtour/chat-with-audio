"""Dialoog-automix (Dugan-stijl gain sharing) en mix-minus (N-1).

Automix lost het boom-vs-lav-probleem op: alle mics gewoon optellen geeft
ruis- en kamstapeling; hard schakelen knettert. Dugan-verdeling geeft elk
spoor per moment een aandeel evenredig met zijn (spraak)energie, en de som
van de aandelen is altijd 1 — de totale versterking blijft constant, mics
die niets bijdragen zakken vanzelf weg.

Mix-minus is de klassieke N-1: iedereen hoort de mix mínus zichzelf (geen
echo voor de inbeller, geen rondzingen in de studio).
"""

from __future__ import annotations

import numpy as np

from chat_with_audio import match as match_mod


def _mono(x: np.ndarray) -> np.ndarray:
    x2 = x[None, :] if x.ndim == 1 else x
    return x2.mean(axis=0).astype(np.float64)


def _pad_to(sig: np.ndarray, n: int) -> np.ndarray:
    return sig if sig.shape[0] >= n else np.pad(sig, (0, n - sig.shape[0]))


def automix(tracks: list[np.ndarray], sr: int, match_to: int | None = 0,
            window_s: float = 0.05, smooth_s: float = 0.25,
            floor_share: float = 0.05) -> tuple[np.ndarray, dict]:
    """Mix uitgelijnde dialoogsporen met Dugan-gain-sharing.

    match_to: index van het referentiespoor (meestal de boom) waarnaar de
    andere sporen spectraal worden gematcht vóór het mixen, zodat een lav niet
    ineens van klankkleur wisselt met de boom; None slaat het matchen over.
    floor_share houdt elk spoor minimaal hoorbaar aanwezig (geen gate-gevoel).
    Geeft (mono-mix (1, n), info met per spoor aandeel en match-EQ).
    """
    if len(tracks) < 2:
        raise ValueError("Automix heeft minstens 2 sporen nodig.")
    monos = [_mono(t) for t in tracks]
    n = max(m.shape[0] for m in monos)
    monos = [_pad_to(m, n) for m in monos]

    info: dict = {"tracks": []}
    if match_to is not None:
        ref = monos[match_to]
        matched = []
        for i, m in enumerate(monos):
            if i == match_to:
                matched.append(m)
                info["tracks"].append({"index": i, "match_eq": []})
                continue
            bands, desc = match_mod.build_match_eq(
                m[None, :].astype(np.float32), sr,
                ref[None, :].astype(np.float32), sr, strength=0.7)
            from chat_with_audio import dsp

            y = dsp.eq(m[None, :].astype(np.float32), sr, bands)[0] if bands else m
            matched.append(np.asarray(y, dtype=np.float64))
            info["tracks"].append({"index": i, "match_eq": desc})
        monos = matched
    else:
        info["tracks"] = [{"index": i, "match_eq": None}
                          for i in range(len(monos))]

    # energie per venster
    hop = max(1, int(window_s * sr))
    n_win = int(np.ceil(n / hop))
    energy = np.zeros((len(monos), n_win))
    for i, m in enumerate(monos):
        pad = _pad_to(m, n_win * hop)
        energy[i] = np.mean(pad.reshape(n_win, hop) ** 2, axis=1)

    # Dugan: aandeel_i = E_i / som(E); vloertje zodat niets hard dichtgaat
    total = energy.sum(axis=0) + 1e-20
    share = energy / total
    share = np.maximum(share, floor_share)
    share /= share.sum(axis=0)

    # glad in de tijd (geen zichtbare pompslagen)
    from scipy.ndimage import gaussian_filter1d

    sigma = max(smooth_s * sr / hop / 2.0, 1.0)
    share = gaussian_filter1d(share, sigma=sigma, axis=1)
    share /= share.sum(axis=0)

    t_win = np.arange(n_win) * hop + hop / 2
    mix = np.zeros(n)
    t_all = np.arange(n)
    for i, m in enumerate(monos):
        g = np.interp(t_all, t_win, share[i])
        mix += m * g
        info["tracks"][i]["avg_share"] = round(float(share[i].mean()), 3)
        info["tracks"][i]["active_pct"] = round(
            float((share[i] > 1.5 / len(monos)).mean() * 100), 1)

    peak = np.max(np.abs(mix)) + 1e-12
    if peak > 0.99:
        mix *= 0.99 / peak
        info["peak_guard_db"] = round(20 * np.log10(0.99 / peak), 2)
    return mix[None, :].astype(np.float32), info


def mix_minus(tracks: list[np.ndarray], exclude: int,
              headroom_db: float = 3.0) -> np.ndarray:
    """N-1: som van alle sporen behalve exclude, met peak-guard."""
    if not 0 <= exclude < len(tracks):
        raise ValueError(f"exclude {exclude} buiten bereik 0-{len(tracks) - 1}.")
    if len(tracks) < 2:
        raise ValueError("Mix-minus heeft minstens 2 sporen nodig.")
    monos = [_mono(t) for i, t in enumerate(tracks) if i != exclude]
    n = max(m.shape[0] for m in monos)
    mix = np.sum([_pad_to(m, n) for m in monos], axis=0)
    ceiling = 10 ** (-abs(headroom_db) / 20)
    peak = np.max(np.abs(mix)) + 1e-12
    if peak > ceiling:
        mix *= ceiling / peak
    return mix[None, :].astype(np.float32)
