"""Room-tone fill: digitale gaten vullen met de ambience van de opname zelf.

De klassieke dialoog-editorwens: een edit-gat, dropout of ADR-las valt op
doordat de 'lucht' van de opname wegvalt — niet door wat er wél klinkt.
We bemonsteren de rustigste échte ambience van het bestand (de room tone),
en vullen exacte-stilte-gaten met geshuffelde, overlappende stukjes daarvan
(shuffelen voorkomt hoorbare periodiciteit), met crossfades naar het
omliggende materiaal.

Alleen digitale stilte (exacte nullen) wordt gevuld: natuurlijke stilte in
de opname bevat al room tone en blijft onaangeroerd.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_FRAME_S = 0.025


def _frame_rms_db(mono: np.ndarray, sr: int) -> np.ndarray:
    flen = max(1, int(sr * _FRAME_S))
    nf = max(1, mono.shape[0] // flen)
    fr = mono[: nf * flen].reshape(nf, flen)
    return 10.0 * np.log10((fr**2).mean(axis=1) + 1e-20)


def find_donor(x2: np.ndarray, sr: int, min_s: float = 0.4,
               max_s: float = 2.0) -> tuple[int, int] | None:
    """Zoek het langste stuk échte ambience: frames dicht bij de ruisvloer,
    maar boven digitale stilte. Geeft (start, eind) in samples of None."""
    mono = x2.mean(axis=0).astype(np.float64)
    lv = _frame_rms_db(mono, sr)
    flen = max(1, int(sr * _FRAME_S))

    real = lv > -90.0  # geen digitale stilte
    if not real.any():
        return None
    floor = float(np.percentile(lv[real], 10))
    if floor > -35.0:
        return None  # geen rustige ambience te vinden (continu programma)
    quiet = real & (lv < floor + 6.0)

    best: tuple[int, int] | None = None
    run_start = None
    for i, q in enumerate(np.concatenate([quiet, [False]])):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            if best is None or i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    if best is None or (best[1] - best[0]) * _FRAME_S < min_s:
        return None
    a = best[0] * flen
    b = min(best[1] * flen, a + int(max_s * sr))
    return a, b


def _find_holes(mono: np.ndarray, sr: int, min_s: float = 0.02,
                max_s: float = 2.0) -> list[tuple[int, int]]:
    """Exacte-stilte-gaten met programmamateriaal aan weerszijden."""
    tiny = np.abs(mono) < 1e-7
    edges = np.diff(np.concatenate([[0], tiny.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    ctx = int(0.05 * sr)
    n = mono.shape[0]

    def _active(a: int, b: int) -> bool:
        seg = mono[max(0, a):min(n, b)]
        if seg.size == 0:
            return False
        return 10.0 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-20) > -70.0

    holes = []
    for a, b in zip(starts, ends, strict=True):
        if not (int(min_s * sr) <= b - a <= int(max_s * sr)):
            continue
        if _active(a - ctx, a) and _active(b, b + ctx):
            holes.append((int(a), int(b)))
    return holes


def _tone_patch(donor: np.ndarray, length: int, sr: int,
                rng: np.random.Generator) -> np.ndarray:
    """Bouw een patch van `length` samples uit geshuffelde donor-stukken met
    50%-overlap-add (Hann) — klinkt als doorlopende ambience, nooit als loop."""
    win = min(donor.shape[1], max(int(0.15 * sr), 32))
    hop = win // 2
    hann = np.hanning(win)
    out = np.zeros((donor.shape[0], length + win), dtype=np.float64)
    norm = np.zeros(length + win, dtype=np.float64)
    pos = 0
    max_start = donor.shape[1] - win
    while pos < length:
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        chunk = donor[:, start:start + win].astype(np.float64) * hann[None, :]
        out[:, pos:pos + win] += chunk
        norm[pos:pos + win] += hann
        pos += hop
    norm = np.maximum(norm, 1e-6)
    return (out[:, :length] / norm[None, :length]).astype(np.float32)


def fill_room_tone(x: np.ndarray, sr: int, fade_ms: float = 20.0,
                   seed: int = 1234) -> tuple[np.ndarray, dict]:
    """Vul digitale gaten met de eigen room tone. Geeft (audio, info) terug.

    info: {"filled": [{start_s, end_s}], "donor": {start_s, end_s}} of een
    reden waarom er niets is gedaan. Deterministisch (seed) zodat sessies
    reproduceerbaar zijn.
    """
    x2 = x[None, :] if x.ndim == 1 else x
    mono = x2.mean(axis=0).astype(np.float64)

    holes = _find_holes(mono, sr)
    if not holes:
        return x2.astype(np.float32), {"filled": [], "reason": "geen digitale gaten"}
    donor_span = find_donor(x2, sr)
    if donor_span is None:
        return x2.astype(np.float32), {
            "filled": [],
            "reason": "geen room tone om te bemonsteren (geen rustige echte "
                      "ambience in dit bestand)"}
    donor = x2[:, donor_span[0]:donor_span[1]]

    rng = np.random.default_rng(seed)
    y = x2.astype(np.float32).copy()
    fade = max(4, int(fade_ms / 1000 * sr))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
    filled = []
    for a, b in holes:
        patch = _tone_patch(donor, b - a, sr, rng)
        # korte fades bínnen het gat: geen abrupte tone-inzet; alles buiten
        # het gat blijft bit-voor-bit onaangetast
        if b - a >= fade * 2:
            patch[:, :fade] *= ramp[None, :].astype(np.float32)
            patch[:, -fade:] *= ramp[::-1][None, :].astype(np.float32)
        y[:, a:b] = patch
        filled.append({"start_s": round(a / sr, 3), "end_s": round(b / sr, 3)})
    log.info("fill_room_tone: %d gat(en) gevuld", len(filled))
    return y, {"filled": filled,
               "donor": {"start_s": round(donor_span[0] / sr, 2),
                         "end_s": round(donor_span[1] / sr, 2)}}
