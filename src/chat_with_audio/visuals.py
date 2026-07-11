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


def _log_spec_db(mono: np.ndarray, sr: int, width: int = 860, height: int = 190,
                 fmin: float = 60.0, fmax: float = 16000.0) -> np.ndarray:
    """Log-frequentie (gehoorschaal) spectrogram in dB, vorm (height, width)."""
    nfft = 2048 if sr >= 32000 else 1024
    hop = min(max(mono.shape[0] // (width * 2), nfft // 8), nfft)
    f, _, X = stft(mono, fs=sr, window="hann", nperseg=nfft, noverlap=nfft - hop)
    mag_db = 20.0 * np.log10(np.abs(X) + 1e-10)
    edges = np.geomspace(fmin, min(fmax, sr / 2 * 0.95), height + 1)
    rows = np.empty((height, mag_db.shape[1]))
    for i in range(height):
        sel = (f >= edges[i]) & (f < edges[i + 1])
        rows[i] = mag_db[sel].mean(axis=0) if sel.any() else -120.0
    img = Image.fromarray(rows[::-1, :].astype(np.float32), "F").resize(
        (width, height), Image.BILINEAR)
    return np.asarray(img)


def _diverging(v: np.ndarray) -> np.ndarray:
    """v in [-1,1] -> blauw (weggehaald) / donker (gelijk) / rood (toegevoegd)."""
    v = np.clip(v, -1.0, 1.0)
    r = np.where(v > 0, 40 + 215 * v, 40 * (1 + v))
    b = np.where(v < 0, 40 - 215 * v, 40 * (1 - v))
    g = 40 * (1 - np.abs(v))
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def perceptual_panel(x_original: np.ndarray, sr: int,
                     x_processed: np.ndarray | None = None,
                     labels: tuple[str, str] = ("A - origineel", "B - bewerkt")) -> bytes:
    """Vergelijkingspaneel voor AI-ogen: gehoorschaal-spectrogrammen, een
    verschilkaart (rood = toegevoegd, blauw = weggehaald) en levelcurves."""
    import io as _io

    from PIL import ImageDraw

    xo = x_original[None, :] if x_original.ndim == 1 else x_original
    mono_o = xo.mean(axis=0)
    W, SH, CH, LBL = 860, 190, 120, 16
    spec_o = _log_spec_db(mono_o, sr, W, SH)

    panels: list[tuple[str, np.ndarray]] = []
    top = float(spec_o.max())
    if x_processed is not None:
        xp = x_processed[None, :] if x_processed.ndim == 1 else x_processed
        mono_p = xp.mean(axis=0)
        spec_p = _log_spec_db(mono_p, sr, W, SH)
        top = max(top, float(spec_p.max()))
        rng = 80.0
        panels.append((f"{labels[0]} - spectrogram (log-frequentie 60 Hz - 16 kHz)",
                       _colormap((spec_o - (top - rng)) / rng)))
        panels.append((f"{labels[1]} - spectrogram",
                       _colormap((spec_p - (top - rng)) / rng)))
        panels.append(("Verschil B - A (rood = toegevoegd, blauw = weggehaald, +/-18 dB)",
                       _diverging((spec_p - spec_o) / 18.0)))
    else:
        rng = 80.0
        panels.append((f"{labels[0]} - spectrogram (log-frequentie 60 Hz - 16 kHz)",
                       _colormap((spec_o - (top - rng)) / rng)))

    H = len(panels) * (SH + LBL) + CH + LBL + 8
    canvas = Image.new("RGB", (W, H), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for title, arr in panels:
        draw.text((4, y + 2), title, fill=(200, 205, 215))
        canvas.paste(Image.fromarray(arr, "RGB"), (0, y + LBL))
        y += SH + LBL

    # levelcurves (25 ms RMS, dB): A grijs, B blauw
    draw.text((4, y + 2), "Niveau (dB, 25 ms RMS): A = grijs, B = blauw; "
                          "stippellijnen -12/-24/-36/-48", fill=(200, 205, 215))
    y0 = y + LBL
    for gdb in (-12, -24, -36, -48):
        gy = y0 + int((-gdb) / 60.0 * CH)
        for gx in range(0, W, 7):
            draw.point((gx, gy), fill=(60, 64, 72))

    def _curve(mono, color):
        flen = max(1, int(sr * 0.025))
        nf = mono.shape[0] // flen
        r = 10 * np.log10((mono[: nf * flen].reshape(nf, flen) ** 2).mean(axis=1) + 1e-20)
        pts = [(int(i / nf * W), y0 + int(np.clip(-r[i], 0, 60) / 60.0 * CH))
               for i in range(nf)]
        draw.line(pts, fill=color, width=1)

    _curve(mono_o, (150, 150, 155))
    if x_processed is not None:
        _curve(mono_p, (86, 156, 255))

    buf = _io.BytesIO()
    canvas.save(buf, "PNG")
    return buf.getvalue()


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
