"""32-sporen synchronisatie: opnames van verschillende recorders uitlijnen
op het geluid zelf ("link meerdere recorders").

Het multirecorder-probleem: een lav, een boom, een veldrecorder, camera-audio
en een telefoon draaien allemaal los van elkaar; elk bestand begint op een
ander moment en elke recorderklok loopt nét iets anders. Deze module vindt de
onderlinge offsets op basis van de audio-inhoud:

  fase 1 — envelope-GCC-PHAT: kruiscorrelatie van log-gecomprimeerde
           RMS-envelopes (500 Hz-raster). PHAT-whitening maakt de meting
           ongevoelig voor verschillen in mickleur en afstand.
  fase 2 — full-rate verfijning: GCC-PHAT op een venster in de overlap,
           op samplenauwkeurigheid.

Daarnaast: klokdrift-meting (offset aan het begin vs het einde van de
overlap) en optionele correctie, en een confidence-score per spoor zodat een
bestand zonder gedeelde audio nooit stilletjes op de verkeerde plek belandt.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

MAX_TRACKS = 32
_ENV_FS = 500.0          # envelope-raster (2 ms)
_CONF_SYNCED = 6.0       # confidence-drempel: daaronder niet verschuiven
_DRIFT_MIN_OVERLAP_S = 20.0


def _envelope(mono: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    block = max(1, int(round(sr / _ENV_FS)))
    nb = max(1, mono.shape[0] // block)
    env = np.sqrt((mono[: nb * block].reshape(nb, block) ** 2).mean(axis=1))
    env = np.log1p(env * 1e3)  # compressie: zachte passages tellen ook mee
    return env - env.mean(), sr / block


def _gcc_phat(a: np.ndarray, b: np.ndarray, fs: float,
              max_lag_s: float | None = None) -> tuple[float, float]:
    """(lag_s, confidence): lag > 0 betekent dat b's inhoud later op a's
    tijdlijn thuishoort (b's bestand start lag_s na a)."""
    from scipy.fft import irfft, next_fast_len, rfft

    nfft = next_fast_len(len(a) + len(b))
    spec = rfft(a, nfft) * np.conj(rfft(b, nfft))
    spec /= np.abs(spec) + 1e-12
    cc = irfft(spec, nfft)
    max_lag = (int(max_lag_s * fs) if max_lag_s
               else min(len(a), len(b)) - 1)
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])
    i = int(np.argmax(cc))
    peak = float(cc[i])
    guard = max(3, int(0.1 * fs))
    rest = np.concatenate([cc[:max(0, i - guard)], cc[i + guard:]])
    noise = float(np.sqrt(np.mean(rest ** 2))) + 1e-12
    return float((i - max_lag) / fs), peak / noise


def _refine(ref: np.ndarray, x: np.ndarray, sr: int, coarse_s: float,
            at_s: float | None = None, win_s: float = 10.0) -> float | None:
    """Sample-nauwkeurige verfijning van een grove offset, gemeten in een
    venster binnen de overlap (op tijdlijnpositie at_s)."""
    o0 = max(0.0, coarse_s)
    o1 = min(ref.shape[0] / sr, coarse_s + x.shape[0] / sr)
    if o1 - o0 < 1.0:
        return None
    win = min(win_s, (o1 - o0) * 0.8)
    center = at_s if at_s is not None else (o0 + o1) / 2.0
    w0 = min(max(o0, center - win / 2.0), o1 - win)
    a0 = int(w0 * sr)
    b0 = int((w0 - coarse_s) * sr)
    n = int(win * sr)
    if b0 < 0 or a0 < 0 or a0 + n > ref.shape[0] or b0 + n > x.shape[0]:
        return None
    d, _conf = _gcc_phat(ref[a0:a0 + n], x[b0:b0 + n], sr, max_lag_s=0.2)
    return coarse_s + d


def measure_offset(ref: np.ndarray, x: np.ndarray, sr: int) -> tuple[float, float]:
    """Offset van x t.o.v. ref (in s, positief = x start later) + confidence."""
    ea, efs = _envelope(ref, sr)
    eb, _ = _envelope(x, sr)
    coarse, conf = _gcc_phat(ea, eb, efs)
    fine = _refine(ref, x, sr, coarse)
    return (fine if fine is not None else coarse), conf


def measure_drift(ref: np.ndarray, x: np.ndarray, sr: int,
                  offset_s: float) -> float | None:
    """Klokdrift in ppm: verschil tussen de fijne offset vroeg en laat in de
    overlap. None als de overlap te kort is om zinnig te meten."""
    o0 = max(0.0, offset_s)
    o1 = min(ref.shape[0] / sr, offset_s + x.shape[0] / sr)
    span = o1 - o0
    if span < _DRIFT_MIN_OVERLAP_S:
        return None
    t1 = o0 + span * 0.15
    t2 = o0 + span * 0.85
    d1 = _refine(ref, x, sr, offset_s, at_s=t1, win_s=min(10.0, span * 0.25))
    d2 = _refine(ref, x, sr, offset_s, at_s=t2, win_s=min(10.0, span * 0.25))
    if d1 is None or d2 is None:
        return None
    return float((d2 - d1) / (t2 - t1) * 1e6)


def correct_drift(x2: np.ndarray, drift_ppm: float) -> np.ndarray:
    """Rek de tijdas van x met factor (1 + drift) zodat de offset constant
    wordt. Lineaire interpolatie: bij ppm-factoren is de fout verwaarloosbaar."""
    factor = 1.0 + drift_ppm / 1e6
    n = x2.shape[1]
    new_n = int(round(n * factor))
    src = np.arange(new_n) / factor
    idx = np.arange(n)
    return np.stack([np.interp(src, idx, x2[c]) for c in range(x2.shape[0])]
                    ).astype(np.float32)


def sync_all(monos: list[np.ndarray], sr: int, ref_i: int) -> list[dict]:
    """Synchroniseer alle sporen in twee passen.

    Pas 1 meet elk spoor tegen de referentie. Bij veel sporen met korte
    paarsgewijze overlap (32 recorders die elkaar maar net raken) is dat
    dubbelzinnig; pas 2 meet daarom elk spoor opnieuw tegen de SOM van de
    overige geplaatste sporen — die beslaat de hele tijdlijn, dus volledige
    overlap voor iedereen. Het eigen spoor wordt uit de som gehouden (anders
    bevestigt een fout geplaatst spoor zichzelf).
    """
    n = len(monos)
    results: list[dict] = []
    for i in range(n):
        if i == ref_i:
            results.append({"offset_s": 0.0, "confidence": None, "synced": True})
            continue
        off, conf = measure_offset(monos[ref_i], monos[i], sr)
        results.append({"offset_s": float(off), "confidence": float(conf),
                        "synced": conf >= _CONF_SYNCED})

    synced_idx = [i for i in range(n) if results[i]["synced"]]
    if len(synced_idx) < 2:
        return results
    t0 = min(results[i]["offset_s"] for i in synced_idx)
    total = max(int(round((results[i]["offset_s"] - t0) * sr)) + monos[i].shape[0]
                for i in synced_idx)
    mix = np.zeros(total, dtype=np.float64)
    own_start = {}
    for i in synced_idx:
        s = int(round((results[i]["offset_s"] - t0) * sr))
        own_start[i] = s
        mix[s:s + monos[i].shape[0]] += monos[i]

    for i in range(n):
        if i == ref_i:
            continue
        others = mix.copy()
        if i in own_start:
            s = own_start[i]
            others[s:s + monos[i].shape[0]] -= monos[i]
        off2, conf2 = measure_offset(others, monos[i], sr)
        off2 += t0  # mix-tijdlijn terug naar referentie-tijdlijn
        c1 = results[i]["confidence"] or 0.0
        if conf2 >= max(c1, _CONF_SYNCED):
            results[i] = {"offset_s": float(off2), "confidence": float(conf2),
                          "synced": True}
    return results


def align_tracks(tracks: list[dict], sr: int) -> tuple[list[dict], int]:
    """Zet per spoor de plaatsing op de gezamenlijke tijdlijn (t=0 = vroegste
    spoor) en geeft (tracks, totale lengte in samples) terug. Elke track-dict
    heeft 'audio' (channels, n), 'offset_s' en 'synced'."""
    placed = [t["offset_s"] if t["synced"] else 0.0 for t in tracks]
    t0 = min(placed)
    total = 0
    for t, p in zip(tracks, placed, strict=True):
        t["place_s"] = round(p - t0, 6)
        start = int(t["place_s"] * sr)
        total = max(total, start + t["audio"].shape[1])
    return tracks, total


def render_aligned(track: dict, sr: int, total: int) -> np.ndarray:
    """Track op de gezamenlijke tijdlijn: stilte ervoor/erna aangevuld."""
    x2 = track["audio"]
    start = int(track["place_s"] * sr)
    out = np.zeros((x2.shape[0], total), dtype=np.float32)
    out[:, start:start + x2.shape[1]] = x2
    return out


def mixdown(tracks_audio: list[np.ndarray], headroom_db: float = 3.0) -> np.ndarray:
    """Som van sporen (mono-gevouwen per spoor naar het maximum aantal
    kanalen), piekbewaakt naar -headroom_db."""
    ch = max(a.shape[0] for a in tracks_audio)
    n = max(a.shape[1] for a in tracks_audio)
    mix = np.zeros((ch, n), dtype=np.float64)
    for a in tracks_audio:
        rep = a if a.shape[0] == ch else np.repeat(a, ch, axis=0)[:ch]
        mix[:, :rep.shape[1]] += rep
    peak = float(np.abs(mix).max())
    ceiling = 10.0 ** (-abs(headroom_db) / 20.0)
    if peak > ceiling:
        mix *= ceiling / peak
    return mix.astype(np.float32)
