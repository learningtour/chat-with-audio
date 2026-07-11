"""Smaakmodel: leer van door de gebruiker gelabelde voorbeelden wat 'goed' en
'slecht' geluid is, en scoor nieuwe audio daartegen.

Bewust eenvoudig en uitlegbaar (geen black box): z-genormaliseerde perceptuele
features, afstand tot het goed- vs slecht-centroid, plus de dimensies die de
score het meest bepalen — zodat Claude kan uitleggen WAAROM iets afwijkt en er
gericht op kan bijsturen.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

_FEATURES = [
    ("lufs_integrated", None), ("lra_db", None), ("crest_factor_db", None),
    ("tilt_db_per_octave", None), ("snr_db", None), ("noise_floor_db", None),
    ("silence_pct", None), ("rms_db", None), ("true_peak_dbtp", None),
    ("spectral_centroid_hz", "log"),
    ("band_energy_pct.low", None), ("band_energy_pct.mid", None),
    ("band_energy_pct.high", None),
]

_LABELS = {"good", "bad"}
MIN_PER_CLASS = 2


def taste_dir() -> Path:
    d = Path(os.environ.get("AIT_TASTE_DIR",
                            str(Path.home() / "AudioImprove" / "taste"))).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(m: dict, dotted: str):
    cur = m
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def _vector(m: dict) -> list[float]:
    out = []
    for key, tf in _FEATURES:
        v = _get(m, key)
        if v is None:
            out.append(float("nan"))
        else:
            out.append(math.log10(max(float(v), 1.0)) if tf == "log" else float(v))
    return out


def add_example(metrics: dict, label: str, source: str, note: str = "") -> dict:
    if label not in _LABELS:
        raise ValueError(f"label moet 'good' of 'bad' zijn, niet '{label}'")
    entry = {"label": label, "source": source, "note": note,
             "created": time.strftime("%Y-%m-%d %H:%M:%S"),
             "features": _vector(metrics)}
    with open(taste_dir() / "examples.jsonl", "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return counts()


def load_examples() -> list[dict]:
    p = taste_dir() / "examples.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def counts() -> dict:
    ex = load_examples()
    return {"good": sum(1 for e in ex if e["label"] == "good"),
            "bad": sum(1 for e in ex if e["label"] == "bad"),
            "needed_per_class": MIN_PER_CLASS}


def score(metrics: dict) -> dict | None:
    """0-100 (100 = klinkt als je 'goed'-voorbeelden) + de grootste afwijkingen."""
    ex = load_examples()
    good = np.array([e["features"] for e in ex if e["label"] == "good"], dtype=float)
    bad = np.array([e["features"] for e in ex if e["label"] == "bad"], dtype=float)
    if len(good) < MIN_PER_CLASS or len(bad) < MIN_PER_CLASS:
        return None
    import warnings

    all_x = np.vstack([good, bad])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # kolommen met alleen NaN
        col_mean = np.nanmean(all_x, axis=0)
        col_std = np.nanstd(all_x, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    col_std = np.where(np.isnan(col_std) | (col_std < 1e-6), 1.0, col_std)

    def norm(v):
        v = np.where(np.isnan(v), col_mean, v)
        return (v - col_mean) / col_std

    gc = norm(np.nanmean(good, axis=0))
    bc = norm(np.nanmean(bad, axis=0))
    z = norm(np.array(_vector(metrics), dtype=float))
    dg = float(np.linalg.norm(z - gc))
    db = float(np.linalg.norm(z - bc))
    taste = 100.0 * db / (dg + db + 1e-9)

    dev = np.abs(z - gc)
    order = np.argsort(-dev)[:3]
    hints = []
    for i in order:
        if dev[i] < 0.5:
            continue
        key = _FEATURES[i][0]
        richting = "hoger" if z[i] > gc[i] else "lager"
        hints.append(f"{key} is duidelijk {richting} dan je 'goed'-voorbeelden")
    return {"taste_score": round(taste, 1), "counts": counts(),
            "largest_deviations": hints}
