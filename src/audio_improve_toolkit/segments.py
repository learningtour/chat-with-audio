"""Spraak/muziek/stilte-segmentatie, zodat elk deel zijn eigen behandeling krijgt.

Kern-features per venster: energie (stilte), 3-9 Hz envelope-modulatie (het
lettergreepritme van spraak) en spectrale spreiding. Grof maar robuust; de
segmenten worden gladgestreken en mini-eilandjes opgeslokt door hun buren.
"""

from __future__ import annotations

import numpy as np


def _window_features(mono: np.ndarray, sr: int, win_s: float = 1.0,
                     hop_s: float = 0.5) -> dict:
    n = mono.shape[0]
    win, hop = int(win_s * sr), int(hop_s * sr)
    if n < win:
        win, hop = n, n
    starts = np.arange(0, max(n - win + 1, 1), hop)

    rms_db = np.empty(len(starts))
    mod_ratio = np.empty(len(starts))
    env_hop = max(1, int(sr * 0.02))  # 20 ms envelope-resolutie

    for i, s in enumerate(starts):
        seg = mono[s:s + win].astype(np.float64)
        rms_db[i] = 10 * np.log10(np.mean(seg**2) + 1e-20)
        nf = seg.shape[0] // env_hop
        env = np.sqrt((seg[: nf * env_hop].reshape(nf, env_hop) ** 2).mean(axis=1))
        env = env - env.mean()
        spec = np.abs(np.fft.rfft(env)) ** 2
        freqs = np.fft.rfftfreq(nf, d=env_hop / sr)
        total = spec[freqs > 0.5].sum() + 1e-20
        mod_ratio[i] = float(spec[(freqs >= 3.0) & (freqs <= 9.0)].sum() / total)

    return {"starts_s": starts / sr, "hop_s": hop / sr, "win_s": win / sr,
            "rms_db": rms_db, "mod_ratio": mod_ratio}


def _otsu_split(values: np.ndarray) -> tuple[float, float]:
    """Drempel die de tussen-klasse-variantie maximaliseert + klassescheiding (dB)."""
    if len(values) < 4:
        return float(values.mean()) if len(values) else 0.0, 0.0
    best_t, best_var, best_sep = float(np.median(values)), -1.0, 0.0
    for t in np.percentile(values, np.arange(10, 91, 2.5)):
        lo, hi = values[values < t], values[values >= t]
        if len(lo) < 2 or len(hi) < 2:
            continue
        w = len(lo) * len(hi) / len(values) ** 2
        var = w * (hi.mean() - lo.mean()) ** 2
        if var > best_var:
            best_t, best_var, best_sep = float(t), float(var), float(hi.mean() - lo.mean())
    return best_t, best_sep


def classify_segments(x: np.ndarray, sr: int, speech_mod: float = 0.40,
                      min_seg_s: float = 1.0) -> list[dict]:
    """Verdeel de tijdlijn volledig in segmenten: speech | music | silence.

    Primair onderscheid tussen (stille) dialoog en (luide) muziek is het niveau,
    met een automatische drempel. Alleen als de niveauverdeling geen twee duidelijke
    klassen heeft (>= 10 dB uit elkaar), valt de classificatie terug op het
    3-9 Hz-modulatieritme van spraak.
    """
    x2 = x[None, :] if x.ndim == 1 else x
    mono = x2.mean(axis=0)
    f = _window_features(mono, sr)

    quiet = np.sort(f["rms_db"])[: max(1, len(f["rms_db"]) // 8)]
    floor = float(quiet.mean())
    silence = f["rms_db"] < floor + 6.0

    active = f["rms_db"][~silence]
    split, sep = _otsu_split(active)
    if sep >= 10.0:
        speech = (~silence) & (f["rms_db"] < split)
    else:
        speech = (~silence) & (f["mod_ratio"] >= speech_mod)

    kinds = np.where(silence, 0, np.where(speech, 1, 2))  # 0=silence 1=speech 2=music

    # mediaan-achtige smoothing (meerderheidsstem over 3 vensters)
    if len(kinds) >= 3:
        sm = kinds.copy()
        for i in range(1, len(kinds) - 1):
            trio = kinds[i - 1:i + 2]
            vals, counts = np.unique(trio, return_counts=True)
            sm[i] = vals[counts.argmax()]
        kinds = sm

    names = {0: "silence", 1: "speech", 2: "music"}
    dur = x2.shape[1] / sr
    segs: list[dict] = []
    for i, k in enumerate(kinds):
        t0 = f["starts_s"][i]
        t1 = min(t0 + f["hop_s"], dur) if i < len(kinds) - 1 else dur
        if segs and segs[-1]["kind"] == names[k]:
            segs[-1]["end_s"] = t1
        else:
            segs.append({"start_s": float(t0), "end_s": float(t1), "kind": names[k]})

    # eilandjes korter dan min_seg_s opslokken (voorkeur: niet-stilte buur)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, seg in enumerate(segs):
            if seg["end_s"] - seg["start_s"] >= min_seg_s:
                continue
            nb = segs[i - 1] if i > 0 else segs[i + 1]
            if i > 0 and i < len(segs) - 1 and segs[i - 1]["kind"] == "silence":
                nb = segs[i + 1]
            nb_is_prev = nb is segs[i - 1] if i > 0 else False
            if nb_is_prev:
                nb["end_s"] = seg["end_s"]
            else:
                nb["start_s"] = seg["start_s"]
            segs.pop(i)
            # aangrenzende gelijke soorten samenvoegen
            j = 0
            while j < len(segs) - 1:
                if segs[j]["kind"] == segs[j + 1]["kind"]:
                    segs[j]["end_s"] = segs[j + 1]["end_s"]
                    segs.pop(j + 1)
                else:
                    j += 1
            changed = True
            break

    return segs


def segment_slices(segs: list[dict], sr: int, kind: str) -> list[slice]:
    return [slice(int(s["start_s"] * sr), int(s["end_s"] * sr))
            for s in segs if s["kind"] == kind]
