"""Ketenuitvoering: een lijst stappen (dicts) toepassen op audio.

Elke stap is {"type": <naam>, ...params}. run_chain valideert, voert uit en
geeft de daadwerkelijk gebruikte parameters terug (voor chain.json en de chat).
"""

from __future__ import annotations

import inspect
import logging

import numpy as np

from audio_improve_toolkit import dsp
from audio_improve_toolkit.analysis import measure_lufs

log = logging.getLogger(__name__)


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -16.0,
                       true_peak_db: float = -1.5, max_iter: int = 2) -> tuple[np.ndarray, dict]:
    """Gain naar target-LUFS met een true-peak-veilige limiter erachter.

    De limiter-ceiling ligt 0.3 dB onder het true-peak-target omdat inter-sample
    pieken boven de sample-piek kunnen uitkomen.
    """
    before = measure_lufs(x, sr)
    if before is None:
        return x, {"skipped": "audio te kort of stil voor loudness-meting"}
    ceiling = true_peak_db - 0.3
    applied = 0.0
    y = x
    for _ in range(max_iter):
        cur = measure_lufs(y, sr)
        if cur is None or abs(cur - target_lufs) < 0.5:
            break
        step = target_lufs - cur
        applied += step
        y = dsp.limiter(dsp.gain(y, step), sr, ceiling_db=ceiling)
    after = measure_lufs(y, sr)
    return y, {
        "lufs_before": round(before, 2),
        "lufs_after": round(after, 2) if after is not None else None,
        "gain_db": round(applied, 2),
        "limiter_ceiling_db": round(ceiling, 2),
    }


def _step_highpass(x, sr, freq: float = 80.0, q: float = 0.707):
    return dsp.highpass(x, sr, freq, q)


def _step_lowpass(x, sr, freq: float = 16000.0, q: float = 0.707):
    return dsp.lowpass(x, sr, freq, q)


def _step_notch(x, sr, freq: float, q: float = 30.0):
    return dsp.notch(x, sr, freq, q)


def _step_eq(x, sr, bands: list):
    return dsp.eq(x, sr, bands)


def _step_gain(x, sr, gain_db: float):
    return dsp.gain(x, gain_db)


def _step_denoise(x, sr, strength_db: float = 12.0, method: str = "spectral"):
    if method == "ai":
        return dsp.ai_denoise(x, sr, strength_db=strength_db)
    return dsp.spectral_denoise(x, sr, reduction_db=strength_db)


def _step_smart_denoise(x, sr, speech_strength_db: float = 100.0,
                        music_strength_db: float = 6.0,
                        silence_strength_db: float = 18.0, fade_ms: float = 120.0):
    """Segment-gestuurde ontruising: AI (DeepFilterNet) op spraak, milde spectral
    gating op muziek, stevige reductie op stiltes. Segmenten worden met
    crossfades weer aaneengesmeed."""
    from audio_improve_toolkit.segments import classify_segments

    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    segs = classify_segments(x2, sr)
    fade = max(1, int(fade_ms / 1000 * sr))
    pad = max(fade, int(0.3 * sr))
    ai_ok = dsp.ai_denoise_available()

    out = np.zeros_like(x2, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for seg in segs:
        a, b = int(seg["start_s"] * sr), int(seg["end_s"] * sr)
        if b <= a:
            continue
        aa, bb = max(0, a - pad), min(n, b + pad)
        chunk = x2[:, aa:bb]
        kind = seg["kind"]
        if kind == "speech" and speech_strength_db > 0:
            if ai_ok:
                proc = dsp.ai_denoise(chunk, sr, strength_db=speech_strength_db)
            else:
                proc = dsp.spectral_denoise(chunk, sr,
                                            reduction_db=min(speech_strength_db, 18))
        elif kind == "music" and music_strength_db > 0:
            proc = dsp.spectral_denoise(chunk, sr, reduction_db=music_strength_db)
        elif kind == "silence" and silence_strength_db > 0:
            proc = dsp.spectral_denoise(chunk, sr, reduction_db=silence_strength_db)
        else:
            proc = np.asarray(chunk, dtype=np.float32)
        w = np.ones(bb - aa)
        ramp = max(1, min(fade, (bb - aa) // 2))
        if aa > 0:
            w[:ramp] = np.linspace(0.0, 1.0, ramp)
        if bb < n:
            w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1.0, 0.0, ramp))
        out[:, aa:bb] += np.asarray(proc, dtype=np.float64) * w
        wsum[aa:bb] += w

    holes = wsum <= 1e-9
    out[:, holes] = x2[:, holes]
    wsum[holes] = 1.0
    return (out / wsum[None, :]).astype(np.float32)


def _step_gate(x, sr, threshold_db: float, attack_ms: float = 5.0,
               release_ms: float = 120.0, hold_ms: float = 50.0, range_db: float = 12.0):
    return dsp.noise_gate(x, sr, threshold_db, attack_ms, release_ms, hold_ms, range_db)


def _step_compressor(x, sr, threshold_db: float, ratio: float = 3.0, attack_ms: float = 10.0,
                     release_ms: float = 150.0, knee_db: float = 6.0, makeup_db: float = 0.0):
    return dsp.compressor(x, sr, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_db)


def _step_leveler(x, sr, target_db: float = -18.0, max_boost_db: float = 20.0,
                  max_cut_db: float = 12.0, floor_db: float | None = None,
                  smooth_s: float = 0.8):
    """Automatische gain-riding: stille passages (spraak) omhoog, luide (muziek)
    omlaag naar een gezamenlijk kortetermijnniveau. Stilte/ruis onder floor_db
    wordt niet opgetild."""
    from scipy.ndimage import gaussian_filter1d

    x2 = x[None, :] if x.ndim == 1 else x
    mono = x2.mean(axis=0).astype(np.float64)
    n = mono.shape[0]
    hop = max(1, int(sr * 0.05))
    half = max(1, int(sr * 0.2))  # 400 ms meetvenster

    cs = np.concatenate([[0.0], np.cumsum(mono**2)])
    centers = np.arange(0, n, hop)
    lo = np.maximum(centers - half, 0)
    hi = np.minimum(centers + half, n)
    level = 10.0 * np.log10((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1) + 1e-20)

    if floor_db is None:
        quiet = np.sort(level)[: max(1, len(level) // 10)]
        floor_db = float(quiet.mean()) + 8.0

    active = level > floor_db
    if not active.any():
        return x2.astype(np.float32)
    gain_db = np.clip(target_db - level, -abs(max_cut_db), abs(max_boost_db))
    idx = np.where(active)[0]
    # inactieve frames (pauzes) volgen hun actieve buren, zodat ruis niet wordt opgepompt
    gain_db = np.interp(np.arange(len(level)), idx, gain_db[idx])
    sigma_frames = max(smooth_s * sr / hop / 2.0, 1.0)
    gain_db = gaussian_filter1d(gain_db, sigma=sigma_frames)

    gains = (10.0 ** (gain_db / 20.0)).astype(np.float64)
    per_sample = np.interp(np.arange(n), centers, gains)
    return (x2 * per_sample[None, :]).astype(np.float32)


def _step_limiter(x, sr, ceiling_db: float = -1.5, release_ms: float = 60.0,
                  lookahead_ms: float = 5.0):
    return dsp.limiter(x, sr, ceiling_db, release_ms, lookahead_ms)


def _step_loudness_normalize(x, sr, target_lufs: float = -16.0, true_peak_db: float = -1.5):
    y, _info = normalize_loudness(x, sr, target_lufs, true_peak_db)
    return y


STEP_REGISTRY = {
    "highpass": _step_highpass,
    "lowpass": _step_lowpass,
    "notch": _step_notch,
    "eq": _step_eq,
    "gain": _step_gain,
    "denoise": _step_denoise,
    "smart_denoise": _step_smart_denoise,
    "gate": _step_gate,
    "compressor": _step_compressor,
    "leveler": _step_leveler,
    "limiter": _step_limiter,
    "loudness_normalize": _step_loudness_normalize,
}


def run_chain(x: np.ndarray, sr: int, steps: list[dict],
              progress=None) -> tuple[np.ndarray, list[dict]]:
    """Voer de stappen uit; geeft (audio, resolved_steps incl. defaults) terug."""
    y = x
    resolved: list[dict] = []
    for i, step in enumerate(steps):
        step = dict(step)
        stype = step.pop("type", None)
        fn = STEP_REGISTRY.get(stype)
        if fn is None:
            raise ValueError(f"Onbekende stap '{stype}'. Geldig: {sorted(STEP_REGISTRY)}")
        sig = inspect.signature(fn)
        valid = {k for k in sig.parameters if k not in ("x", "sr")}
        unknown = set(step) - valid
        if unknown:
            raise ValueError(f"Onbekende parameter(s) {sorted(unknown)} voor stap "
                             f"'{stype}'. Geldig: {sorted(valid)}")
        bound = sig.bind(None, sr, **step)
        bound.apply_defaults()
        params = {k: v for k, v in bound.arguments.items() if k not in ("x", "sr")}
        log.info("stap %d/%d: %s %s", i + 1, len(steps), stype, params)
        if progress:
            progress(i, len(steps), stype)
        y = fn(y, sr, **step)
        resolved.append({"type": stype, **params})
    return y, resolved
