"""Dialoogbewerking: ademreductie, plosief-reparatie en muziekbed-ducking.

Het dagelijkse handwerk van dialoog-editors, geautomatiseerd maar behoudend:
adem wordt gedempt (niet weggeknipt — dat klinkt onnatuurlijk), plosieven
worden alleen op de pop zelf ontdaan van hun laagfrequente stoot, en
muziekbedden tussen de spraak worden naar een vast aantal dB onder het
spraakniveau gereden. Voor muziek ÓNDER spraak (overlappend) is stems-
scheiding nodig: rebalance_music.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_FRAME_S = 0.01  # 10 ms werk-resolutie


def _frame_rms_db(mono: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    flen = max(1, int(sr * _FRAME_S))
    nf = max(1, mono.shape[0] // flen)
    fr = mono[: nf * flen].reshape(nf, flen)
    return 10.0 * np.log10((fr**2).mean(axis=1) + 1e-20), flen


def _zcr(mono: np.ndarray, flen: int, nf: int) -> np.ndarray:
    """Zero-crossing rate per frame: goedkope stemhebbend/ruisachtig-detector."""
    fr = mono[: nf * flen].reshape(nf, flen)
    return (np.abs(np.diff(np.signbit(fr), axis=1)).sum(axis=1)) / flen


def _smooth_env(env: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    """Gain-envelope gladstrijken; randen ge-edge-pad zodat de convolutie het
    begin/einde van het bestand niet met impliciete nullen omlaag trekt."""
    fade = max(2, int(fade_ms / 1000 * sr))
    kernel = np.hanning(fade * 2 + 1)
    kernel /= kernel.sum()
    padded = np.pad(env, len(kernel) // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _apply_gain_runs(x2: np.ndarray, runs: list[tuple[int, int]], gain: float,
                     sr: int, fade_ms: float = 30.0) -> np.ndarray:
    """Demp de gegeven sample-bereiken met zachte randen (raised cosine)."""
    n = x2.shape[1]
    env = np.ones(n, dtype=np.float64)
    for a, b in runs:
        env[a:b] = gain
    env = _smooth_env(env, sr, fade_ms)
    return (x2.astype(np.float64) * env[None, :]).astype(np.float32)


def breath_control(x: np.ndarray, sr: int, reduction_db: float = 10.0,
                   max_breath_s: float = 0.9) -> tuple[np.ndarray, int]:
    """Detecteer ademhalingen en demp ze met reduction_db (niet wegknippen).

    Een adem is: kort (0.12-0.9 s), ruisachtig (hoge zero-crossing rate, geen
    toonhoogte), duidelijk zachter dan de spraak maar boven de vloer, en
    grenzend aan spraak. Sibilanten zitten op spraakniveau en blijven staan.
    Geeft (audio, aantal gedempte adems) terug.
    """
    x2 = x[None, :] if x.ndim == 1 else x
    mono = x2.mean(axis=0).astype(np.float64)
    lv, flen = _frame_rms_db(mono, sr)
    nf = len(lv)
    zcr = _zcr(mono, flen, nf)

    speech_level = float(np.percentile(lv[lv > -80], 90)) if (lv > -80).any() else -20.0
    # vloer over álle frames, geklemd: digitale stilte (-200) mag de adem-band
    # niet onder de werkelijkheid drukken
    floor = max(float(np.percentile(lv, 10)), -75.0)

    candidate = ((lv > max(floor + 4.0, -65.0)) & (lv < speech_level - 12.0)
                 & (zcr > 0.12))
    speechy = lv > speech_level - 8.0

    min_f, max_f = max(2, int(0.12 / _FRAME_S)), int(max_breath_s / _FRAME_S)
    near = max(1, int(0.3 / _FRAME_S))
    edges = np.diff(np.concatenate([[0], candidate.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    runs: list[tuple[int, int]] = []
    for a, b in zip(starts, ends, strict=True):
        if not (min_f <= b - a <= max_f):
            continue
        if speechy[max(0, a - near):a].any() or speechy[b:b + near].any():
            runs.append((a * flen, b * flen))
    if not runs:
        return x2.astype(np.float32), 0
    gain = 10.0 ** (-abs(reduction_db) / 20.0)
    log.info("breath_control: %d adem(s) gedempt met %.0f dB", len(runs), reduction_db)
    return _apply_gain_runs(x2, runs, gain, sr), len(runs)


def deplosive(x: np.ndarray, sr: int, cutoff_hz: float = 120.0,
              sensitivity_db: float = 6.0) -> tuple[np.ndarray, int]:
    """Repareer plosieven (p/b-pops): korte laagfrequente drukstoten.

    Detectie: frames waar de banda onder ~150 Hz het totaal domineert én ver
    boven het normale laag-niveau van de opname uitkomt, in stoten van
    20-250 ms. Fix: alleen dat stukje highpassen (cutoff_hz) en met zachte
    randen terugleggen — de stem eromheen blijft ongemoeid.
    """
    from scipy.signal import butter, sosfiltfilt

    x2 = x[None, :] if x.ndim == 1 else x
    mono = x2.mean(axis=0).astype(np.float64)
    n = x2.shape[1]
    if n < sr:
        return x2.astype(np.float32), 0

    sos_low = butter(4, 150.0, btype="lowpass", fs=sr, output="sos")
    low = sosfiltfilt(sos_low, mono)
    lv_low, flen = _frame_rms_db(low, sr)
    lv_all, _ = _frame_rms_db(mono, sr)

    active = lv_all > -60.0
    ref_low = (float(np.percentile(lv_low[active], 50)) if active.any() else -80.0)
    pop = (lv_low > lv_all - 3.0) & (lv_low > ref_low + abs(sensitivity_db)) \
        & (lv_low > -45.0)

    min_f, max_f = 2, int(0.25 / _FRAME_S)
    edges = np.diff(np.concatenate([[0], pop.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    y = x2.astype(np.float32).copy()
    sos_hp = butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    fixed = 0
    pad = int(0.03 * sr)
    for a, b in zip(starts, ends, strict=True):
        if not (min_f <= b - a <= max_f):
            continue
        s, e = max(0, a * flen - pad), min(n, b * flen + pad)
        chunk = y[:, s:e].astype(np.float64)
        proc = sosfiltfilt(sos_hp, chunk, axis=1)
        w = np.ones(e - s)
        ramp = min(pad, (e - s) // 2)
        if ramp > 1:
            r = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, ramp))
            w[:ramp] = r
            w[-ramp:] = np.minimum(w[-ramp:], r[::-1])
        y[:, s:e] = (chunk * (1 - w) + proc * w).astype(np.float32)
        fixed += 1
    if fixed:
        log.info("deplosive: %d plosief-pop(s) gerepareerd", fixed)
    return y, fixed


def sidechain_gain(vocals: np.ndarray, sr: int, duck_db: float = 6.0,
                   attack_ms: float = 15.0, release_ms: float = 250.0,
                   floor_db: float = -45.0) -> np.ndarray:
    """Sample-nauwkeurige duck-envelope gestuurd door de vocals: 10^(-duck/20)
    waar gezongen/gesproken wordt, terug naar 1.0 in de pauzes. Attack snel
    (wegduiken vóór het woord er is), release traag (geen gepomp)."""
    mono = vocals.mean(axis=0) if vocals.ndim == 2 else vocals
    n = mono.shape[0]
    flen = max(1, int(sr * 0.01))
    nf = max(1, n // flen)
    lv = 10.0 * np.log10((mono[: nf * flen].reshape(nf, flen) ** 2).mean(axis=1) + 1e-20)
    target = np.where(lv > floor_db, 10.0 ** (-abs(duck_db) / 20.0), 1.0)

    frame_ms = 10.0
    a_att = float(np.exp(-frame_ms / max(attack_ms, 1e-3)))
    a_rel = float(np.exp(-frame_ms / max(release_ms, 1e-3)))
    g = np.empty(nf)
    c = 1.0
    for i, tgt in enumerate(target):
        coef = a_att if tgt < c else a_rel
        c = coef * c + (1.0 - coef) * tgt
        g[i] = c
    centers = np.arange(nf) * flen + flen / 2.0
    return np.interp(np.arange(n), centers, g)


def _duck_music_stems(x2: np.ndarray, sr: int, duck_db: float,
                      attack_ms: float, release_ms: float) -> tuple[np.ndarray, dict]:
    """Echte sidechain-ducking voor muziek ónder spraak: Demucs scheidt
    vocals van de rest, de vocals-envelope duwt de begeleiding omlaag, remix."""
    from chat_with_audio.dsp import stems as stems_mod

    if not stems_mod.is_available():
        raise RuntimeError(stems_mod.INSTALL_HINT)
    parts = stems_mod.separate(x2, sr)
    vocals = parts.get("vocals")
    if vocals is None:
        raise RuntimeError("Demucs gaf geen vocals-stem terug.")
    rest = np.zeros_like(x2, dtype=np.float64)
    for name, part in parts.items():
        if name != "vocals":
            rest += part.astype(np.float64)
    g = sidechain_gain(vocals, sr, duck_db=duck_db,
                       attack_ms=attack_ms, release_ms=release_ms)
    y = (vocals.astype(np.float64) + rest * g[None, :]).astype(np.float32)
    active_pct = float((g < 0.9).mean() * 100)
    log.info("duck_music[stems]: begeleiding %0.0f%% van de tijd gedoken", active_pct)
    return y, {"mode": "stems", "duck_db": float(duck_db),
               "ducked_pct": round(active_pct, 1)}


def duck_music(x: np.ndarray, sr: int, gap_db: float = 6.0,
               fade_ms: float = 120.0, mode: str = "beds",
               attack_ms: float = 15.0,
               release_ms: float = 250.0) -> tuple[np.ndarray, dict]:
    """Muziek onder het spraakniveau brengen. Twee modi:

    mode="beds" (licht, standaard): rijdt muziekbedden tússen de spraak
    (intro's/outro's/bedden) naar gap_db onder het gemeten spraakniveau —
    segmentniveau, alleen omlaag.
    mode="stems" (zwaar, vereist het [stems]-extra): echte sidechain-ducking
    voor muziek die tegelijk mét de spraak klinkt — Demucs scheidt de vocals,
    hun envelope duwt de begeleiding gap_db omlaag met attack/release.
    """
    x2m = x[None, :] if x.ndim == 1 else x
    if mode == "stems":
        return _duck_music_stems(x2m, sr, gap_db, attack_ms, release_ms)
    if mode != "beds":
        raise ValueError(f"Onbekende mode '{mode}' (beds|stems).")
    from chat_with_audio.segments import classify_segments

    x2 = x[None, :] if x.ndim == 1 else x
    mono = x2.mean(axis=0).astype(np.float64)
    segs = classify_segments(x2, sr)
    lv, _flen = _frame_rms_db(mono, sr)

    def seg_level(seg, pct):
        a, b = int(seg["start_s"] / _FRAME_S), int(seg["end_s"] / _FRAME_S)
        sel = lv[a:max(a + 1, b)]
        sel = sel[sel > -80]
        return float(np.percentile(sel, pct)) if sel.size else None

    speech_lv = [seg_level(s, 90) for s in segs if s["kind"] == "speech"]
    speech_lv = [v for v in speech_lv if v is not None]
    if not speech_lv:
        return x2.astype(np.float32), {"ducked": [], "reason": "geen spraak gevonden"}
    ref = float(np.median(speech_lv))

    n = x2.shape[1]
    env = np.ones(n, dtype=np.float64)
    ducked = []
    for seg in segs:
        if seg["kind"] != "music":
            continue
        level = seg_level(seg, 70)
        if level is None:
            continue
        cut = level - (ref - abs(gap_db))
        if cut <= 0.5:
            continue  # bed zit al onder het doel
        a, b = int(seg["start_s"] * sr), int(seg["end_s"] * sr)
        env[a:b] = 10.0 ** (-cut / 20.0)
        ducked.append({"start_s": round(seg["start_s"], 2),
                       "end_s": round(seg["end_s"], 2),
                       "cut_db": round(cut, 1)})
    if not ducked:
        return x2.astype(np.float32), {"ducked": [], "reason": "geen bed boven het doel"}

    env = _smooth_env(env, sr, fade_ms)
    log.info("duck_music: %d muziekbed(den) gedempt", len(ducked))
    return (x2.astype(np.float64) * env[None, :]).astype(np.float32), {"ducked": ducked}
