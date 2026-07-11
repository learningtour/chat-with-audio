"""Audio-analyse: metrics, scores en issues waarover Claude kan praten.

Metrics-conventies: alle niveaus in dBFS, loudness in LUFS, true peak in dBTP.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly, welch


def _db(v: float | np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(v, 1e-10))


def _frame_rms_db(mono: np.ndarray, sr: int, frame_ms: float = 25.0) -> np.ndarray:
    flen = max(1, int(sr * frame_ms / 1000))
    nf = mono.shape[0] // flen
    if nf < 1:
        return np.asarray([float(_db(np.sqrt(np.mean(mono**2) + 1e-20)))])
    fr = mono[: nf * flen].reshape(nf, flen)
    return _db(np.sqrt((fr**2).mean(axis=1) + 1e-20))


def measure_lufs(x: np.ndarray, sr: int) -> float | None:
    """Geintegreerde loudness (BS.1770) via pyloudnorm; None bij te korte/stille audio."""
    import pyloudnorm as pyln

    if x.shape[1] / sr < 0.5:
        return None
    meter = pyln.Meter(sr)
    data = x.T if x.shape[0] > 1 else x[0]
    val = float(meter.integrated_loudness(np.asarray(data, dtype=np.float64)))
    return val if math.isfinite(val) else None


def _short_term_lufs(x: np.ndarray, sr: int, win_s: float = 3.0, hop_s: float = 1.0) -> list[float]:
    import pyloudnorm as pyln

    n = x.shape[1]
    win, hop = int(win_s * sr), int(hop_s * sr)
    if n < win:
        v = measure_lufs(x, sr)
        return [v] if v is not None else []
    meter = pyln.Meter(sr)
    vals = []
    for start in range(0, n - win + 1, hop):
        seg = x[:, start:start + win]
        data = seg.T if seg.shape[0] > 1 else seg[0]
        v = float(meter.integrated_loudness(np.asarray(data, dtype=np.float64)))
        if math.isfinite(v) and v > -70.0:
            vals.append(v)
    return vals


def _true_peak_db(x: np.ndarray, sr: int) -> float:
    up = resample_poly(x.astype(np.float64), 4, 1, axis=1)
    return float(_db(np.abs(up).max()))


def _momentary_lufs_max(x: np.ndarray, sr: int) -> float | None:
    """Hoogste momentary loudness (400 ms-vensters, EBU-terminologie)."""
    import pyloudnorm as pyln

    n = x.shape[1]
    win = int(0.4 * sr)
    if n < win:
        return None
    hop = int(0.2 * sr) if n / sr <= 1800 else sr  # lange bestanden: grover raster
    meter = pyln.Meter(sr, block_size=0.4)
    best = None
    for start in range(0, n - win + 1, hop):
        seg = x[:, start:start + win]
        data = seg.T if seg.shape[0] > 1 else seg[0]
        v = float(meter.integrated_loudness(np.asarray(data, dtype=np.float64)))
        if math.isfinite(v) and v > -70.0 and (best is None or v > best):
            best = v
    return best


def _stereo_qc(x: np.ndarray) -> dict | None:
    """Technische stereo-checks (eerste twee kanalen): fasecorrelatie, balans,
    dood kanaal, dual-mono en polariteitsinversie — de standaard QC-lijst."""
    if x.shape[0] < 2:
        return None
    left = x[0].astype(np.float64)
    right = x[1].astype(np.float64)
    e_l, e_r = float(np.mean(left**2)), float(np.mean(right**2))
    silent = 1e-12
    dead = None
    if e_l > silent and e_r <= max(e_l * 1e-6, silent):
        dead = "right"
    elif e_r > silent and e_l <= max(e_r * 1e-6, silent):
        dead = "left"

    if e_l <= silent or e_r <= silent:
        corr = None
    else:
        corr = float(np.mean(left * right) / (np.sqrt(e_l * e_r) + 1e-20))
    balance = (round(10.0 * np.log10((e_l + 1e-20) / (e_r + 1e-20)), 1)
               if dead is None else None)

    mid = 0.5 * float(np.mean((left + right) ** 2))
    side = 0.5 * float(np.mean((left - right) ** 2))
    dual_mono = bool(corr is not None and corr > 0.999
                     and side < mid * 1e-6)

    return {
        "correlation": round(corr, 3) if corr is not None else None,
        "balance_db": balance,
        "dead_channel": dead,
        "dual_mono": dual_mono,
        "polarity_inverted": bool(corr is not None and corr < -0.9),
    }


def _detect_dropouts(mono: np.ndarray, sr: int, max_report: int = 5) -> dict:
    """Digitale dropouts: exacte-stilte-gaten midden in signaal (3 ms - 0.5 s),
    met hoorbaar materiaal direct ervoor en erna."""
    tiny = np.abs(mono) < 1e-7
    edges = np.diff(np.concatenate([[0], tiny.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    min_len, max_len = int(0.003 * sr), int(0.5 * sr)
    ctx = int(0.05 * sr)

    def _active(a: int, b: int) -> bool:
        seg = mono[max(0, a):b]
        if seg.size == 0:
            return False
        return 10.0 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-20) > -50.0

    positions = []
    for a, b in zip(starts, ends, strict=True):
        if not (min_len <= b - a <= max_len):
            continue
        if _active(a - ctx, a) and _active(b, b + ctx):
            positions.append(round(a / sr, 3))
    return {"count": len(positions), "positions_s": positions[:max_report]}


def _edge_silence(frames_db: np.ndarray, frame_s: float, noise_floor: float) -> tuple[float, float]:
    """Stilteduur aan kop en staart (frames onder vloer+6 dB, minimaal -60)."""
    thr = max(noise_floor + 6.0, -60.0)
    active = frames_db > thr
    if not active.any():
        dur = len(frames_db) * frame_s
        return round(dur, 2), round(dur, 2)
    first = int(np.argmax(active))
    last = len(active) - 1 - int(np.argmax(active[::-1]))
    return round(first * frame_s, 2), round((len(active) - 1 - last) * frame_s, 2)


def _detect_hum(mono: np.ndarray, sr: int) -> dict:
    if mono.shape[0] < sr:
        return {"detected": False}
    nper = int(min(mono.shape[0], sr * 4))
    f, p = welch(mono, fs=sr, nperseg=nper)
    pdb = 10.0 * np.log10(p + 1e-20)
    best = {"detected": False}
    for f0 in (50.0, 60.0):
        proms = []
        for h in (1, 2, 3):
            fc = f0 * h
            if fc > sr / 2 - 30:
                break
            band = (np.abs(f - fc) <= 2.0)
            neigh = (np.abs(f - fc) >= 5.0) & (np.abs(f - fc) <= 25.0)
            if not band.any() or not neigh.any():
                continue
            proms.append(float(pdb[band].max() - np.median(pdb[neigh])))
        if proms and proms[0] > 6.0 and float(np.mean(proms)) > 8.0:
            score = float(np.mean(proms))
            if not best["detected"] or score > best["prominence_db"]:
                best = {"detected": True, "freq": f0, "prominence_db": round(score, 1)}
    return best


def _detect_resonances(mono: np.ndarray, sr: int, lo: float = 150.0,
                       hi: float = 4500.0, min_excess_db: float = 8.0,
                       max_count: int = 3) -> list[dict]:
    """Smalle pieken die boven het gladgestreken spectrum uitsteken (dozige
    roomresonanties, pieptonen)."""
    from scipy.ndimage import median_filter

    nper = int(min(mono.shape[0], 8192))
    if nper < 2048:
        return []
    f, p = welch(mono, fs=sr, nperseg=nper)
    pdb = 10.0 * np.log10(p + 1e-20)
    smooth = median_filter(pdb, size=max(9, len(f) // 40) | 1, mode="nearest")
    excess = pdb - smooth
    sel = (f >= lo) & (f <= min(hi, sr / 2 - 500))
    found = []
    idx = np.where(sel & (excess > min_excess_db))[0]
    for i in idx:
        if 0 < i < len(f) - 1 and pdb[i] >= pdb[i - 1] and pdb[i] >= pdb[i + 1]:
            found.append({"freq": round(float(f[i]), 0),
                          "excess_db": round(float(excess[i]), 1)})
    found.sort(key=lambda r: -r["excess_db"])
    # dichte buren (zelfde resonantie over meerdere bins) ontdubbelen
    dedup: list[dict] = []
    for r in found:
        if all(abs(r["freq"] - d["freq"]) > d["freq"] * 0.15 for d in dedup):
            dedup.append(r)
        if len(dedup) >= max_count:
            break
    return dedup


def _spectral(mono: np.ndarray, sr: int) -> dict:
    nper = int(min(mono.shape[0], 4096))
    f, p = welch(mono, fs=sr, nperseg=nper)
    total = float(p.sum()) + 1e-20
    low = float(p[f < 250].sum()) / total * 100
    mid = float(p[(f >= 250) & (f < 4000)].sum()) / total * 100
    high = float(p[f >= 4000].sum()) / total * 100
    centroid = float((f * p).sum() / total)

    sel = (f >= 100) & (f <= 10000) & (p > 0)
    tilt = 0.0
    if sel.sum() > 8:
        slope, _ = np.polyfit(np.log2(f[sel]), 10.0 * np.log10(p[sel] + 1e-20), 1)
        tilt = float(slope)
    return {
        "band_energy_pct": {"low": round(low, 1), "mid": round(mid, 1), "high": round(high, 1)},
        "spectral_centroid_hz": round(centroid, 0),
        "tilt_db_per_octave": round(tilt, 2),
    }


def analyze(x: np.ndarray, sr: int) -> dict:
    """Volledige kwaliteitsanalyse van (channels, n) float32-audio."""
    x = x[None, :] if x.ndim == 1 else x
    mono = x.mean(axis=0)
    n = x.shape[1]

    frames = _frame_rms_db(mono, sr)
    noise_floor = float(np.mean(np.sort(frames)[: max(1, len(frames) // 10)]))
    signal_level = float(np.percentile(frames, 90))
    rms_db = float(_db(np.sqrt(np.mean(mono**2) + 1e-20)))
    peak = float(np.abs(x).max())
    peak_db = float(_db(peak))

    # Stilte = frames dicht bij de ruisvloer, mits er echt onderscheid is tussen
    # signaal en vloer (anders is 'stilte' betekenisloos, bv. bij een constante toon).
    if signal_level - noise_floor < 10.0:
        silence_pct = 0.0
    else:
        silence_thr = max(noise_floor + 6.0, signal_level - 30.0)
        silence_pct = float((frames < silence_thr).mean() * 100)

    if peak <= 1.001:
        clip_mask = np.abs(mono) >= 0.999
    else:
        # 32-bit float opname met headroom boven 0 dBFS: dat is geen clipping.
        # Alleen echte flat-tops (opeenvolgende identieke samples nabij de piek) tellen.
        near_peak = np.abs(mono) >= 0.98 * peak
        flat = np.concatenate([[False], np.diff(mono) == 0.0]) & near_peak
        clip_mask = flat
    clipped = int(clip_mask.sum())
    runs = np.diff(np.concatenate([[0], clip_mask.astype(np.int8), [0]]))
    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
    clip_events = int(((ends - starts) >= 3).sum())

    st = _short_term_lufs(x, sr)
    lra = float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) >= 4 else None

    lufs = measure_lufs(x, sr)
    true_peak = _true_peak_db(x, sr)
    momentary = _momentary_lufs_max(x, sr)
    lead_s, tail_s = _edge_silence(frames, 0.025, noise_floor)
    metrics = {
        "duration_s": round(n / sr, 2),
        "sample_rate": sr,
        "channels": int(x.shape[0]),
        "lufs_integrated": round(lufs, 2) if lufs is not None else None,
        "lufs_short_term_max": round(max(st), 2) if st else None,
        "lufs_momentary_max": round(momentary, 2) if momentary is not None else None,
        "lra_db": round(lra, 1) if lra is not None else None,
        "sample_peak_db": round(peak_db, 2),
        "true_peak_dbtp": round(true_peak, 2),
        "plr_db": round(true_peak - lufs, 1) if lufs is not None else None,
        "rms_db": round(rms_db, 2),
        "crest_factor_db": round(peak_db - rms_db, 1),
        "noise_floor_db": round(noise_floor, 1),
        "snr_db": round(signal_level - noise_floor, 1),
        "silence_pct": round(silence_pct, 1),
        "clipped_samples": clipped,
        "clip_events": clip_events,
        "dc_offset": round(float(np.abs(x.mean(axis=1)).max()), 5),
        "lead_silence_s": lead_s,
        "tail_silence_s": tail_s,
        "dropouts": _detect_dropouts(mono, sr),
        "stereo": _stereo_qc(x),
        "hum": _detect_hum(mono, sr),
        "resonances": _detect_resonances(mono, sr),
    }
    metrics.update(_spectral(mono, sr))
    return metrics


def _clamp_score(v: float) -> int:
    return int(np.clip(round(v), 0, 100))


def score_and_issues(m: dict, target_lufs: float = -15.0) -> tuple[dict, list[dict]]:
    """0-100 scores per categorie + concrete issues met suggesties."""
    issues: list[dict] = []

    lufs = m.get("lufs_integrated")
    if lufs is None:
        loudness = 50
    else:
        dist = abs(lufs - target_lufs)
        loudness = _clamp_score(100 - max(dist - 1.5, 0) * 8)
        if lufs < target_lufs - 4:
            issues.append({"severity": "medium", "code": "low_loudness",
                           "message": f"Loudness is laag ({lufs} LUFS, streefwaarde "
                                      f"rond {target_lufs}).",
                           "suggestion": "normalize_loudness of improve_audio"})
        elif lufs > target_lufs + 3:
            issues.append({"severity": "low", "code": "high_loudness",
                           "message": f"Loudness is hoog ({lufs} LUFS).",
                           "suggestion": "normalize_loudness met lager target"})

    snr = m.get("snr_db", 40)
    noise = _clamp_score((snr - 10) / 30 * 100)
    if snr < 25:
        sev = "high" if snr < 15 else "medium"
        issues.append({"severity": sev, "code": "noisy",
                       "message": f"Veel achtergrondruis (SNR {snr} dB, ruisvloer "
                                  f"{m.get('noise_floor_db')} dB).",
                       "suggestion": "reduce_noise"})
    if m.get("hum", {}).get("detected"):
        issues.append({"severity": "medium", "code": "hum",
                       "message": f"Netbrom gedetecteerd rond {m['hum']['freq']:.0f} Hz "
                                  f"(+{m['hum']['prominence_db']} dB).",
                       "suggestion": "improve_audio (plaatst automatisch notch-filters)"})
        noise = min(noise, 60)

    crest = m.get("crest_factor_db", 14)
    lra = m.get("lra_db")
    dynamics = 100
    if crest < 6:
        dynamics = _clamp_score(crest / 6 * 60)
        issues.append({"severity": "medium", "code": "over_compressed",
                       "message": f"Erg platte dynamiek (crest {crest} dB) — mogelijk "
                                  "overgecomprimeerd.",
                       "suggestion": "bron met meer dynamiek gebruiken; verdere "
                                     "compressie vermijden"})
    elif crest > 22 or (lra is not None and lra > 15):
        dynamics = 65
        issues.append({"severity": "low", "code": "very_dynamic",
                       "message": f"Grote dynamiek (crest {crest} dB, LRA {lra} dB) — "
                                  "stille delen kunnen wegvallen.",
                       "suggestion": "improve_audio (past lichte compressie toe)"})

    clarity = 100
    tilt = m.get("tilt_db_per_octave", -4)
    if tilt < -9:
        clarity -= 30
        issues.append({"severity": "medium", "code": "dull",
                       "message": f"Dof klankbeeld (spectrale tilt {tilt} dB/octaaf).",
                       "suggestion": "improve_audio (highshelf) of apply_chain met eq"})
    if m.get("band_energy_pct", {}).get("low", 0) > 50:
        clarity -= 25
        issues.append({"severity": "medium", "code": "muddy",
                       "message": "Veel energie onder 250 Hz (dreun/modder).",
                       "suggestion": "improve_audio (highpass + peaking-cut rond 300 Hz)"})
    if m.get("sample_peak_db", 0) > 0.5 and m.get("clip_events", 0) == 0:
        issues.append({"severity": "medium", "code": "hot_float",
                       "message": f"32-bit float opname met pieken tot "
                                  f"{m['sample_peak_db']} dBFS boven nul — geen vervorming, "
                                  "maar normaliseren is vereist voor gebruik/export.",
                       "suggestion": "improve_audio of normalize_loudness"})
    if m.get("clip_events", 0) > 0:
        clarity -= 35
        issues.append({"severity": "high", "code": "clipping",
                       "message": f"{m['clip_events']} clip-momenten "
                                  f"({m['clipped_samples']} samples) — "
                                  "vervorming in de opname.",
                       "suggestion": "kan niet volledig hersteld worden; voortaan lager opnemen"})
    if m.get("dc_offset", 0) > 0.005:
        issues.append({"severity": "low", "code": "dc_offset",
                       "message": f"DC-offset aanwezig ({m['dc_offset']}).",
                       "suggestion": "improve_audio (highpass verwijdert dit)"})

    stereo = m.get("stereo") or {}
    if stereo.get("dead_channel"):
        clarity -= 20
        issues.append({"severity": "high", "code": "dead_channel",
                       "message": f"Stereobestand met een dood kanaal "
                                  f"({stereo['dead_channel']}) — half beeld.",
                       "suggestion": "als mono aanleveren of het goede kanaal dupliceren"})
    if stereo.get("polarity_inverted"):
        clarity -= 20
        issues.append({"severity": "high", "code": "polarity_inverted",
                       "message": f"Kanalen in tegenfase (correlatie "
                                  f"{stereo.get('correlation')}) — valt weg bij "
                                  "mono-weergave (tv, telefoon, smartspeaker).",
                       "suggestion": "polariteit van één kanaal omdraaien in de DAW"})
    if stereo.get("dual_mono"):
        issues.append({"severity": "low", "code": "dual_mono",
                       "message": "Stereobestand met identieke kanalen (dual-mono).",
                       "suggestion": "als mono aanleveren scheelt de helft in "
                                     "bestandsgrootte; klinkt identiek"})
    bal = stereo.get("balance_db")
    if bal is not None and abs(bal) > 3.0:
        issues.append({"severity": "medium", "code": "unbalanced",
                       "message": f"Stereo-balans scheef ({bal:+.1f} dB links t.o.v. rechts).",
                       "suggestion": "apply_chain met gain per kanaal, of pan corrigeren"})

    drops = m.get("dropouts") or {}
    if drops.get("count"):
        clarity -= 25
        pos = ", ".join(f"{p:.1f}s" for p in drops.get("positions_s", [])[:3])
        issues.append({"severity": "high", "code": "dropouts",
                       "message": f"{drops['count']} digitale dropout(s) gedetecteerd "
                                  f"(o.a. rond {pos}).",
                       "suggestion": "bron opnieuw exporteren/overzetten; repair_audio "
                                     "kan korte gaten niet betrouwbaar vullen"})

    for edge, key in (("kop", "lead_silence_s"), ("staart", "tail_silence_s")):
        if (m.get(key) or 0) > 3.0:
            issues.append({"severity": "low", "code": f"{key}",
                           "message": f"{m[key]:.1f} s stilte aan de {edge} van het bestand.",
                           "suggestion": "wegknippen voor aflevering"})

    scores = {
        "loudness": loudness,
        "noise": noise,
        "dynamics": dynamics,
        "clarity": _clamp_score(clarity),
    }
    scores["overall"] = _clamp_score(np.mean(list(scores.values())))
    return scores, issues
