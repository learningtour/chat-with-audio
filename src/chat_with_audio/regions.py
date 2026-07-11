"""Slimme probleemregio's: vind wáár op de tijdlijn iets mis is en behandel
alleen dat stuk, in plaats van effecten over het hele bestand.

Detectoren (venstergebaseerd, samengevoegd tot regio's):
  hum   — netbrom (50/60 Hz + harmonischen) die maar een deel van de tijd
          aanwezig is (koelkast, dimmer, apparaat dat aan/uit gaat)
  noise — ruisvloer die tijdelijk boven de schoonste ambience van het bestand
          uitkomt (airco, verkeer, ventilator die aanslaat)
  clip  — clusters van afgekapte golftoppen
  boom  — laagfrequente dreun die het beeld tijdelijk domineert (passerende
          vrachtwagen, pop, tafelbons)

`apply_regions` voert per regio een eigen mini-keten uit en smeedt de bewerkte
stukken met raised-cosine-crossfades terug in het origineel; alles buiten de
regio's (plus pad/fade-marge) blijft bit-voor-bit onaangetast.
"""

from __future__ import annotations

import logging

import numpy as np

from chat_with_audio.analysis import _detect_hum, _frame_rms_db

log = logging.getLogger(__name__)

FRAME_MS = 25.0  # zelfde frameresolutie als analysis._frame_rms_db

KIND_LABELS = {"hum": "netbrom", "noise": "ruis", "clip": "clipping", "boom": "dreun"}


def fmt_ts(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _merge_flags(flags: np.ndarray, max_gap_wins: int = 1) -> list[list[int]]:
    """Boolean venstervlaggen -> inclusieve [i0, i1]-indexbereiken; gaten tot
    max_gap_wins vensters worden overbrugd (een detector mag even knipperen)."""
    spans: list[list[int]] = []
    for i, on in enumerate(flags):
        if not on:
            continue
        if spans and i - spans[-1][1] <= max_gap_wins:
            spans[-1][1] = i
        else:
            spans.append([i, i])
    return spans


def _mono(x: np.ndarray) -> np.ndarray:
    x2 = x[None, :] if x.ndim == 1 else x
    return x2.mean(axis=0).astype(np.float64)


# ---------------------------------------------------------------- detectoren

def _hum_regions(mono: np.ndarray, sr: int, dur: float) -> list[dict]:
    win_s, hop_s = 4.0, 2.0
    win, hop = int(win_s * sr), int(hop_s * sr)
    if mono.shape[0] < win * 2:
        h = _detect_hum(mono, sr)
        if not h.get("detected"):
            return []
        return [{"kind": "hum", "start_s": 0.0, "end_s": dur,
                 "freq": h["freq"], "severity_db": h["prominence_db"]}]
    starts = np.arange(0, mono.shape[0] - win + 1, hop)
    hits = [_detect_hum(mono[s:s + win], sr) for s in starts]
    regions = []
    for f0 in (50.0, 60.0):
        flags = np.array([h.get("detected", False) and h["freq"] == f0 for h in hits])
        for i0, i1 in _merge_flags(flags):
            proms = [hits[i]["prominence_db"] for i in range(i0, i1 + 1) if flags[i]]
            regions.append({"kind": "hum", "freq": f0,
                            "start_s": float(starts[i0] / sr),
                            "end_s": float(min(starts[i1] / sr + win_s, dur)),
                            "severity_db": round(float(np.mean(proms)), 1)})
    return regions


def _noise_regions(mono: np.ndarray, sr: int, dur: float) -> list[dict]:
    """Vensters waar de lokale ruisvloer ver boven de schoonste ambience van het
    bestand ligt. De vloer per venster is het 10e percentiel van de 25 ms-frames
    (= de pauzes); vensters zonder meetbare pauzes tellen alleen mee als ze in
    hun geheel stil-maar-ruizig zijn."""
    fr = _frame_rms_db(mono, sr)
    fps = 1000.0 / FRAME_MS
    win_f, hop_f = int(2.0 * fps), int(1.0 * fps)
    if len(fr) < win_f * 2:
        return []
    starts_f = np.arange(0, len(fr) - win_f + 1, hop_f)
    floors = np.array([np.percentile(fr[s:s + win_f], 10) for s in starts_f])
    p90 = np.array([np.percentile(fr[s:s + win_f], 90) for s in starts_f])
    global_floor = float(np.percentile(floors, 10))

    excess = floors - global_floor
    # Continue content (muziek zonder pauzes) heeft geen meetbare venstervloer:
    # daar is floor ~ signaalniveau en houdt 'measurable' de detector dicht.
    measurable = (p90 - floors) > 12.0        # venster bevat echte pauzes
    quiet = p90 < global_floor + 20.0         # venster is (ruizige) stilte
    flags = (excess > 8.0) & (floors > -65.0) & (measurable | quiet)

    regions = []
    for i0, i1 in _merge_flags(flags, max_gap_wins=2):
        sev = float(np.mean(excess[i0:i1 + 1]))
        regions.append({"kind": "noise",
                        "start_s": float(starts_f[i0] / fps),
                        "end_s": float(min((starts_f[i1] + win_f) / fps, dur)),
                        "severity_db": round(sev, 1)})
    return [r for r in regions if r["end_s"] - r["start_s"] >= 1.0]


def _clip_regions(mono: np.ndarray, sr: int, dur: float) -> list[dict]:
    peak = float(np.abs(mono).max())
    if peak < 1e-6:
        return []
    if peak <= 1.001:
        mask = np.abs(mono) >= 0.999
    else:
        # 32-bit float met headroom: alleen echte flat-tops tellen
        near = np.abs(mono) >= 0.98 * peak
        mask = np.concatenate([[False], np.diff(mono) == 0.0]) & near
    runs = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.where(runs == 1)[0], np.where(runs == -1)[0]
    events = [(a, b) for a, b in zip(starts, ends, strict=True) if b - a >= 3]
    if not events:
        return []
    # events clusteren tot regio's (gat <= 0.5 s), met 50 ms marge
    regions: list[list[float]] = []
    count: list[int] = []
    for a, b in events:
        t0, t1 = a / sr - 0.05, b / sr + 0.05
        if regions and t0 - regions[-1][1] <= 0.5:
            regions[-1][1] = t1
            count[-1] += 1
        else:
            regions.append([t0, t1])
            count.append(1)
    return [{"kind": "clip", "start_s": max(0.0, a), "end_s": min(b, dur),
             "events": c} for (a, b), c in zip(regions, count, strict=True)]


def _boom_regions(mono: np.ndarray, sr: int, dur: float,
                  segments: list[dict]) -> list[dict]:
    """Laagfrequente dreun (30-160 Hz) die het venster tijdelijk domineert."""
    from scipy.signal import butter, sosfiltfilt

    if mono.shape[0] < sr * 3:
        return []
    sos = butter(4, 160.0, btype="lowpass", fs=sr, output="sos")
    low = sosfiltfilt(sos, mono)

    win, hop = sr, sr // 2
    starts = np.arange(0, mono.shape[0] - win + 1, hop)

    def _lvl(sig: np.ndarray, s: int) -> float:
        return 10.0 * np.log10(np.mean(sig[s:s + win] ** 2) + 1e-20)

    low_lvl = np.array([_lvl(low, s) for s in starts])
    full_lvl = np.array([_lvl(mono, s) for s in starts])
    ref = float(np.median(low_lvl))  # normale laag-inhoud van dit bestand

    dominates = low_lvl > full_lvl - 3.0        # laag draagt het venster
    sticks_out = low_lvl > ref + 10.0           # en is atypisch voor dit bestand
    audible = low_lvl > -50.0
    flags = dominates & sticks_out & audible

    # muziek heeft legitiem veel laag: alleen extreem uitstekende dreun telt daar
    for seg in segments or []:
        if seg["kind"] != "music":
            continue
        a, b = seg["start_s"] * sr, seg["end_s"] * sr
        inside = (starts >= a - hop) & (starts + win <= b + hop)
        flags[inside] &= low_lvl[inside] > ref + 20.0

    regions = []
    for i0, i1 in _merge_flags(flags):
        sev = float(np.mean((low_lvl - ref)[i0:i1 + 1]))
        regions.append({"kind": "boom",
                        "start_s": float(starts[i0] / sr),
                        "end_s": float(min((starts[i1] + win) / sr, dur)),
                        "severity_db": round(sev, 1)})
    return [r for r in regions if r["end_s"] - r["start_s"] >= 0.75]


def detect_regions(x: np.ndarray, sr: int, segments: list[dict] | None = None) -> list[dict]:
    """Vind alle probleemregio's; segments (classify_segments) geeft context zodat
    muziek niet als 'ruis' of 'dreun' wordt aangezien."""
    x2 = x[None, :] if x.ndim == 1 else x
    mono = _mono(x2)
    dur = x2.shape[1] / sr
    if segments is None:
        from chat_with_audio.segments import classify_segments

        segments = classify_segments(x2, sr)
    regions = (_hum_regions(mono, sr, dur) + _noise_regions(mono, sr, dur)
               + _clip_regions(mono, sr, dur) + _boom_regions(mono, sr, dur, segments))
    regions.sort(key=lambda r: (r["start_s"], r["kind"]))
    return regions


# ------------------------------------------------------------------- plannen

def _overlaps_speech(r: dict, segments: list[dict] | None) -> bool:
    return any(s["kind"] == "speech" and s["start_s"] < r["end_s"]
               and s["end_s"] > r["start_s"] for s in (segments or []))


def plan_region_fixes(regions: list[dict], sr: int, ai_available: bool = False,
                      segments: list[dict] | None = None) -> tuple[list[dict], list[str]]:
    """Vul per regio de mini-keten (steps) en een menselijke uitleg (label) in."""
    rationale: list[str] = []
    planned: list[dict] = []
    for r in regions:
        r = dict(r)
        span = f"{fmt_ts(r['start_s'])}–{fmt_ts(r['end_s'])}"
        if r["kind"] == "hum":
            f0 = r["freq"]
            r["steps"] = [{"type": "notch", "freq": f0 * h, "q": 30.0}
                          for h in (1, 2, 3) if f0 * h < sr / 2 - 100]
            r["label"] = f"netbrom {f0:.0f} Hz (+{r['severity_db']} dB)"
            rationale.append(f"{span}: netbrom rond {f0:.0f} Hz "
                             f"(+{r['severity_db']} dB) — notch-filters alleen hier.")
        elif r["kind"] == "noise":
            strength = float(np.clip(r["severity_db"], 6.0, 18.0))
            use_ai = ai_available and _overlaps_speech(r, segments)
            r["steps"] = [{"type": "denoise", "strength_db": round(strength, 1),
                           "method": "ai" if use_ai else "spectral"}]
            r["label"] = f"ruis (+{r['severity_db']} dB boven ambience)"
            rationale.append(f"{span}: ruisvloer +{r['severity_db']} dB boven de "
                             f"schoonste ambience — "
                             f"{'DeepFilterNet (AI)' if use_ai else 'spectral gating'} "
                             f"met {strength:.0f} dB reductie, alleen hier.")
        elif r["kind"] == "clip":
            r["steps"] = [{"type": "declip"}]
            r["label"] = f"clipping ({r['events']} events)"
            rationale.append(f"{span}: {r['events']} clip-moment(en) — golfvorm-"
                             "reconstructie alleen rond de afgekapte toppen.")
        elif r["kind"] == "boom":
            cut = float(np.clip(r["severity_db"] * 0.6, 3.0, 12.0))
            r["steps"] = [{"type": "highpass", "freq": 60.0},
                          {"type": "eq", "bands": [{"type": "lowshelf", "freq": 160.0,
                                                    "gain_db": round(-cut, 1),
                                                    "q": 0.707}]}]
            r["label"] = f"dreun (+{r['severity_db']} dB laag)"
            rationale.append(f"{span}: laagfrequente dreun (+{r['severity_db']} dB "
                             f"boven normaal) — highpass + lowshelf {-cut:.1f} dB, "
                             "alleen hier.")
        else:  # onbekende soort: niets doen, wel doorgeven
            r["steps"] = []
            r["label"] = r["kind"]
        planned.append(r)
    return planned, rationale


# ----------------------------------------------------------------- toepassen

def apply_regions(x: np.ndarray, sr: int, regions: list[dict],
                  fade_ms: float = 80.0) -> tuple[np.ndarray, list[dict]]:
    """Voer per regio zijn mini-keten uit en crossfade het resultaat terug.

    Regio's worden sequentieel op de tussenstand toegepast, zodat overlappende
    regio's (bv. ruis + dreun) componeren. Geeft (audio, toegepaste regio's met
    resolved steps) terug.
    """
    from chat_with_audio.chain import run_chain

    x2 = x[None, :] if x.ndim == 1 else x
    y = x2.astype(np.float32).copy()
    n = y.shape[1]
    fade = max(8, int(fade_ms / 1000 * sr))
    pad = max(fade, int(0.25 * sr))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))

    applied: list[dict] = []
    for r in regions:
        steps = r.get("steps") or []
        if not steps:
            continue
        a = max(0, int(r["start_s"] * sr) - pad)
        b = min(n, int(r["end_s"] * sr) + pad)
        if b - a < fade * 2 + 8:
            continue
        chunk = y[:, a:b]
        try:
            proc, resolved = run_chain(chunk, sr, steps)
        except Exception as exc:  # één mislukte regio mag de rest niet blokkeren
            log.warning("regio %s (%s-%s) overgeslagen: %s", r["kind"],
                        fmt_ts(r["start_s"]), fmt_ts(r["end_s"]), exc)
            continue
        proc2 = proc[None, :] if proc.ndim == 1 else proc
        w = np.ones(b - a, dtype=np.float64)
        if a > 0:
            w[:fade] = ramp
        if b < n:
            w[-fade:] = np.minimum(w[-fade:], ramp[::-1])
        y[:, a:b] = (chunk.astype(np.float64) * (1.0 - w)
                     + proc2.astype(np.float64) * w).astype(np.float32)
        applied.append({**r, "steps": resolved})
    return y, applied
