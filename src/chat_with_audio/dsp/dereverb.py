"""Dereverberatie/spraakverbetering via ClearVoice (MossFormer2, 48 kHz).

Optioneel [enhance]-extra. Het model wordt bij eerste gebruik gedownload.
Output wordt in niveau teruggeschaald naar de input (RMS-match), zodat de
balans in de keten intact blijft.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

INSTALL_HINT = ("Dereverberatie (ClearVoice) is niet geinstalleerd. "
                "Installeer met: uv sync --all-extras (in de projectmap).")

MODEL_SR = 48000
_CV = None


def is_available() -> bool:
    try:
        import clearvoice  # noqa: F401
        return True
    except Exception:
        return False


def _model():
    global _CV
    if _CV is None:
        if not is_available():
            raise RuntimeError(INSTALL_HINT)
        from clearvoice import ClearVoice

        log.info("ClearVoice MossFormer2_SE_48K laden (eerste keer: download)...")
        _CV = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    return _CV


def _resample(sig: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return sig.astype(np.float32)
    g = math.gcd(src, dst)
    return resample_poly(sig.astype(np.float64), dst // g, src // g).astype(np.float32)


def dereverb(x: np.ndarray, sr: int) -> np.ndarray:
    """Verwerk (channels, n) audio per kanaal; geeft dezelfde vorm terug."""
    import soundfile as sf

    cv = _model()
    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    out = np.empty_like(x2, dtype=np.float32)
    for ci in range(x2.shape[0]):
        xr = _resample(x2[ci], sr, MODEL_SR)
        peak = float(np.abs(xr).max()) or 1.0
        scale = 0.7 / peak if peak > 0.7 else 1.0  # modelinput netjes binnen [-1, 1]
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "in.wav"
            sf.write(str(tmp), (xr * scale), MODEL_SR, subtype="FLOAT")
            result = cv(input_path=str(tmp), online_write=False)
        y = np.asarray(result, dtype=np.float32).squeeze()
        if y.ndim > 1:
            y = y[0] if y.shape[0] < y.shape[-1] else y[:, 0]
        y = _resample(y, MODEL_SR, sr)
        if y.shape[0] < n:
            y = np.pad(y, (0, n - y.shape[0]))
        y = y[:n]
        # RMS-match naar de input zodat niveaus in de keten kloppen
        in_rms = float(np.sqrt(np.mean(x2[ci] ** 2) + 1e-20))
        out_rms = float(np.sqrt(np.mean(y**2) + 1e-20))
        out[ci] = y * (in_rms / out_rms if out_rms > 1e-9 else 1.0)
    return out
