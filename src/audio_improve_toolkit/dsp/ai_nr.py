"""Tier B ruisonderdrukking: DeepFilterNet (spraak, state-of-the-art).

Vereist het optionele [ai]-extra: `uv sync --all-extras`. Het model (DeepFilterNet3)
wordt bij eerste gebruik automatisch gedownload. Audio wordt naar de modelrate
(48 kHz) geresampled, per kanaal in chunks met crossfade verwerkt, en terug
geresampled.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

INSTALL_HINT = ("AI-ruisonderdrukking (DeepFilterNet) is niet geinstalleerd. "
                "Installeer met: uv sync --all-extras (in de projectmap). "
                "Vereist Python 3.11 (DeepFilterNet levert geen wheels voor 3.12+).")

_STATE = None
_IMPORT_ERROR: str | None = None


def is_available() -> bool:
    if _STATE is not None:
        return True
    try:
        import df.enhance  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # ook torch/torchaudio-importfouten
        global _IMPORT_ERROR
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False
    return True


def unavailable_reason() -> str:
    is_available()
    return f"{INSTALL_HINT} (import: {_IMPORT_ERROR})" if _IMPORT_ERROR else INSTALL_HINT


def _load():
    global _STATE
    if _STATE is None:
        if not is_available():
            raise RuntimeError(unavailable_reason())
        from df.enhance import enhance, init_df

        log.info("DeepFilterNet-model laden (eerste keer: download)...")
        model, df_state, _ = init_df(log_level="ERROR")
        _STATE = (enhance, model, df_state)
    return _STATE


def denoise(x: np.ndarray, sr: int, atten_lim_db: float | None = None,
            chunk_s: float = 60.0, overlap_s: float = 0.5) -> np.ndarray:
    """Ontruis (channels, n) float32-audio met DeepFilterNet."""
    import torch

    enhance, model, df_state = _load()
    model_sr = int(df_state.sr())
    n = x.shape[1]

    def _resample(sig: np.ndarray, src: int, dst: int) -> np.ndarray:
        if src == dst:
            return sig
        g = math.gcd(src, dst)
        return resample_poly(sig.astype(np.float64), dst // g, src // g).astype(np.float32)

    kwargs = {}
    if atten_lim_db is not None and atten_lim_db < 100:
        kwargs["atten_lim_db"] = float(abs(atten_lim_db))

    chunk = max(int(chunk_s * model_sr), model_sr)
    overlap = min(int(overlap_s * model_sr), chunk // 4)

    out = np.empty_like(x, dtype=np.float32)
    for ci in range(x.shape[0]):
        xr = _resample(x[ci], sr, model_sr)
        m = xr.shape[0]
        if m <= chunk:
            with torch.no_grad():
                y = enhance(model, df_state, torch.from_numpy(xr[None, :]).float(), **kwargs)
            yr = y.squeeze(0).cpu().numpy()
        else:
            yr = np.zeros(m, dtype=np.float32)
            weight = np.zeros(m, dtype=np.float32)
            step = chunk - overlap
            for start in range(0, m, step):
                end = min(start + chunk, m)
                seg = xr[start:end]
                with torch.no_grad():
                    y = enhance(model, df_state, torch.from_numpy(seg[None, :]).float(), **kwargs)
                seg_y = y.squeeze(0).cpu().numpy()[: end - start]
                w = np.ones(end - start, dtype=np.float32)
                ramp = min(overlap, end - start)
                if start > 0:
                    w[:ramp] = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                if end < m:
                    w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1.0, 0.0, ramp, dtype=np.float32))
                yr[start:end] += seg_y * w
                weight[start:end] += w
                if end >= m:
                    break
            yr /= np.maximum(weight, 1e-6)
        yb = _resample(yr, model_sr, sr)
        if yb.shape[0] < n:
            yb = np.pad(yb, (0, n - yb.shape[0]))
        out[ci] = yb[:n]
    return out
