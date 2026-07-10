"""Visualisaties voor de viewer: waveform-peaks (JSON) en spectrogram (PNG)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import stft

# Compacte viridis-benadering (ankers, lineair geinterpoleerd).
_VIRIDIS = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype=np.float64)


def _colormap(v: np.ndarray) -> np.ndarray:
    """v in [0,1] -> uint8 RGB via de viridis-ankers."""
    pos = np.clip(v, 0.0, 1.0) * (len(_VIRIDIS) - 1)
    i = np.clip(pos.astype(int), 0, len(_VIRIDIS) - 2)
    frac = (pos - i)[..., None]
    rgb = _VIRIDIS[i] * (1 - frac) + _VIRIDIS[i + 1] * frac
    return rgb.astype(np.uint8)


def waveform_json(x: np.ndarray, sr: int, out_path: str | Path,
                  buckets: int = 2000) -> Path:
    """Min/max-peaks per bucket over de gemixte kanalen, voor canvas-rendering."""
    x = x[None, :] if x.ndim == 1 else x
    mono = x.mean(axis=0)
    n = mono.shape[0]
    buckets = min(buckets, n)
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    mins = np.empty(buckets, dtype=np.float32)
    maxs = np.empty(buckets, dtype=np.float32)
    for i in range(buckets):
        seg = mono[edges[i]:edges[i + 1]]
        mins[i], maxs[i] = (seg.min(), seg.max()) if seg.size else (0.0, 0.0)
    data = {
        "duration_s": round(n / sr, 3),
        "min": [round(float(v), 4) for v in mins],
        "max": [round(float(v), 4) for v in maxs],
    }
    out_path = Path(out_path)
    out_path.write_text(json.dumps(data))
    return out_path


def spectrogram_png(x: np.ndarray, sr: int, out_path: str | Path, width: int = 1200,
                    height: int = 400, db_range: float = 80.0) -> Path:
    x = x[None, :] if x.ndim == 1 else x
    mono = x.mean(axis=0)
    nfft = 2048 if sr >= 32000 else 1024
    hop = max(1, mono.shape[0] // (width * 2))
    hop = min(max(hop, nfft // 8), nfft)  # genoeg frames voor de gevraagde breedte
    _, _, X = stft(mono, fs=sr, window="hann", nperseg=nfft, noverlap=nfft - hop)
    mag_db = 20.0 * np.log10(np.abs(X) + 1e-10)
    top = float(mag_db.max())
    v = (mag_db - (top - db_range)) / db_range  # [0,1], hoog = fel
    img = _colormap(v[::-1, :])  # lage frequenties onderaan
    pil = Image.fromarray(img, "RGB").resize((width, height), Image.BILINEAR)
    out_path = Path(out_path)
    pil.save(out_path)
    return out_path
