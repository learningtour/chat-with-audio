"""Stem-separatie via Demucs (htdemucs): vocals / drums / bass / other.

Optioneel [stems]-extra. Het model (~80 MB) wordt bij eerste gebruik gedownload.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

INSTALL_HINT = ("Stem-separatie (Demucs) is niet geinstalleerd. "
                "Installeer met: uv sync --all-extras (in de projectmap).")

_MODEL = None


def is_available() -> bool:
    try:
        from demucs import pretrained  # noqa: F401
        return True
    except Exception:
        return False


def _model():
    global _MODEL
    if _MODEL is None:
        if not is_available():
            raise RuntimeError(INSTALL_HINT)
        from demucs.pretrained import get_model

        log.info("Demucs htdemucs laden (eerste keer: download)...")
        _MODEL = get_model("htdemucs")
        _MODEL.eval()
    return _MODEL


def _resample(sig: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return sig.astype(np.float32)
    g = math.gcd(src, dst)
    return resample_poly(sig.astype(np.float64), dst // g, src // g,
                         axis=-1).astype(np.float32)


def separate(x: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Splits (channels, n) audio in stems; elke stem heeft dezelfde vorm en sr."""
    import torch
    from demucs.apply import apply_model

    model = _model()
    model_sr = int(model.samplerate)
    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    mono = x2.shape[0] == 1

    xr = _resample(np.ascontiguousarray(x2, dtype=np.float32), sr, model_sr)
    wav = torch.from_numpy(xr)
    if mono:
        wav = wav.repeat(2, 1)

    # normalisatie zoals demucs.separate die toepast
    ref = wav.mean(0)
    mean, std = float(ref.mean()), float(ref.std()) + 1e-8
    wav = (wav - mean) / std
    with torch.no_grad():
        sources = apply_model(model, wav[None], device="cpu", split=True,
                              overlap=0.25, progress=False)[0]
    sources = sources * std + mean

    out: dict[str, np.ndarray] = {}
    for name, t in zip(model.sources, sources):
        y = t.cpu().numpy().astype(np.float32)
        if mono:
            y = y.mean(axis=0, keepdims=True)
        y = _resample(y, model_sr, sr)
        if y.shape[1] < n:
            y = np.pad(y, ((0, 0), (0, n - y.shape[1])))
        out[name] = y[:, :n]
    return out
