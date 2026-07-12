"""Ruimte & karakter: convolutiegalm (IR-bestand of gesynthetiseerde kamer),
saturatie, slapback-delay en RT60-schatting voor ADR-room-matching.

De gesynthetiseerde IR is octaafband-ruis met per band een eigen vervaltijd
(hoog dooft sneller — `damping` bepaalt hoeveel): geen kathedraalpreset maar
een geloofwaardige kámer, wat precies is wat worldizing en ADR nodig hebben.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from chat_with_audio import io as audio_io


def _2d(x: np.ndarray) -> np.ndarray:
    return x[None, :] if x.ndim == 1 else x


def synth_ir(sr: int, rt60: float = 0.4, damping: float = 0.35,
             predelay_ms: float = 8.0, seed: int = 7) -> np.ndarray:
    """Synthetiseer een kamer-IR (mono): octaafband-ruis, per band exponentieel
    verval. rt60 = brede-band-nagalmtijd (s); damping = hoeveel sneller elke
    octaaf boven 500 Hz dooft (0 = overal even lang, 0.35 = natuurlijke kamer)."""
    rng = np.random.default_rng(seed)
    n = max(int(rt60 * 1.5 * sr), int(0.05 * sr))
    t = np.arange(n) / sr
    centers = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    ir = np.zeros(n)
    for fc in centers:
        octaves_up = max(0.0, np.log2(fc / 500.0))
        rt = rt60 / (1.0 + damping * octaves_up)
        lo, hi = fc / np.sqrt(2), min(fc * np.sqrt(2), sr / 2 * 0.95)
        if lo >= hi:
            continue
        sos = butter(2, [lo, hi], btype="band", fs=sr, output="sos")
        band = sosfilt(sos, rng.standard_normal(n))
        ir += band * np.exp(-6.91 * t / rt)  # -60 dB bij t = rt
    pre = int(predelay_ms / 1000 * sr)
    if pre > 0:
        ir = np.concatenate([np.zeros(pre), ir])
    ir /= np.sqrt(np.sum(ir**2)) + 1e-12  # eenheids-energie
    return ir.astype(np.float64)


def convolve_ir(x: np.ndarray, sr: int, ir_path: str | None = None,
                mix: float = 0.3, rt60: float = 0.4, damping: float = 0.35,
                predelay_ms: float = 8.0, keep_tail: bool = False) -> np.ndarray:
    """Convolutiegalm. Met ir_path een echte IR-wav; zonder wordt een kamer
    gesynthetiseerd (rt60/damping/predelay_ms). mix = wet-aandeel (0-1); de
    wet-tak wordt op dry-RMS genormaliseerd zodat mix zich gedraagt als een
    fader. keep_tail laat de galmstaart voorbij het bestandseinde doorlopen."""
    if not 0.0 <= mix <= 1.0:
        raise ValueError(f"mix {mix} buiten bereik 0-1.")
    x2 = _2d(x).astype(np.float64)
    if ir_path:
        ir_x, ir_sr = audio_io.load_audio(ir_path, mono=True)
        ir = ir_x[0].astype(np.float64)
        if ir_sr != sr:
            ir_x2, _ = audio_io.resample(ir_x, ir_sr, sr)
            ir = ir_x2[0].astype(np.float64)
        ir /= np.sqrt(np.sum(ir**2)) + 1e-12
    else:
        ir = synth_ir(sr, rt60=rt60, damping=damping, predelay_ms=predelay_ms)
    wet = fftconvolve(x2, ir[None, :], mode="full", axes=1)
    dry_rms = np.sqrt(np.mean(x2**2)) + 1e-12
    wet_rms = np.sqrt(np.mean(wet[:, : x2.shape[1]] ** 2)) + 1e-12
    wet *= dry_rms / wet_rms
    n_out = wet.shape[1] if keep_tail else x2.shape[1]
    dry = np.zeros((x2.shape[0], n_out))
    dry[:, : x2.shape[1]] = x2
    y = dry * (1.0 - mix) + wet[:, :n_out] * mix
    return y.astype(np.float32)


def saturate(x: np.ndarray, sr: int, drive_db: float = 6.0,
             mode: str = "tape", mix: float = 1.0) -> np.ndarray:
    """Saturatie: tape (tanh, oneven harmonischen), soft (zachter) of hard
    (agressieve clip — megafoon). Uitgang wordt op ingangs-RMS teruggelegd."""
    x2 = _2d(x).astype(np.float64)
    g = 10 ** (drive_db / 20)
    driven = x2 * g
    if mode == "tape":
        shaped = np.tanh(driven)
    elif mode == "soft":
        shaped = driven / (1.0 + np.abs(driven))
    elif mode == "hard":
        shaped = np.clip(driven, -0.7, 0.7)
    else:
        raise ValueError(f"Onbekende saturatiemodus '{mode}' (tape/soft/hard).")
    in_rms = np.sqrt(np.mean(x2**2)) + 1e-12
    out_rms = np.sqrt(np.mean(shaped**2)) + 1e-12
    shaped *= in_rms / out_rms
    return (x2 * (1 - mix) + shaped * mix).astype(np.float32)


def delay(x: np.ndarray, sr: int, time_ms: float = 120.0, feedback: float = 0.3,
          mix: float = 0.25) -> np.ndarray:
    """Slapback/echo: feedback-comb. time_ms = tapafstand, feedback = herhaal-
    sterkte (<0.95), mix = wet-aandeel. Lengte blijft gelijk."""
    if not 0.0 <= feedback < 0.95:
        raise ValueError(f"feedback {feedback} buiten bereik 0-0.95.")
    x2 = _2d(x).astype(np.float64)
    d = max(1, int(time_ms / 1000 * sr))
    n = x2.shape[1]
    wet = np.zeros_like(x2)
    for start in range(d, n, d):
        end = min(start + d, n)
        span = end - start
        wet[:, start:end] = x2[:, start - d:start - d + span] + \
            feedback * wet[:, start - d:start - d + span]
    return (x2 * (1 - mix) + wet * mix).astype(np.float32)


def estimate_rt60(x: np.ndarray, sr: int, max_events: int = 12) -> float | None:
    """Schat de nagalmtijd uit spraak/programma: zoek offsets (niveau valt
    ≥20 dB) en fit de vervalhelling van -5 naar -20 dB (Schroeder-achtig,
    T20 → RT60). Mediaan over events; None als er geen bruikbaar verval is.
    Benadering: goed genoeg om een kamer te matchen, geen akoestiekrapport."""
    mono = _2d(x).mean(axis=0).astype(np.float64)
    hop = max(1, int(sr * 0.01))
    n_fr = mono.shape[0] // hop
    if n_fr < 40:
        return None
    fr = mono[: n_fr * hop].reshape(n_fr, hop)
    env_db = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-20)
    peak_db = np.percentile(env_db, 95)
    floor_db = np.percentile(env_db, 10)
    if peak_db - floor_db < 25:
        return None

    # gaten tussen bursts zoeken: daar leeft het verval
    quiet = env_db < peak_db - 15
    slopes: list[float] = []
    i = 1
    while i < n_fr and len(slopes) < max_events:
        if not (quiet[i] and not quiet[i - 1]):
            i += 1
            continue
        j = i
        while j < n_fr and quiet[j]:
            j += 1
        if (j - i) * hop / sr >= 0.25:  # gat van minstens 250 ms
            # laatste 80 ms afknippen: daar kan de fade-in van de volgende
            # burst al onder de drempel meegroeien en de integraal vervuilen
            j_cut = j - max(1, int(0.08 * sr / hop))
            seg = mono[i * hop: j_cut * hop]
            # Schroeder-achterwaartse integratie: monotone vervalcurve
            energy = np.cumsum(seg[::-1] ** 2)[::-1]
            sch = 10 * np.log10(energy / (energy[0] + 1e-30) + 1e-30)
            sel = (sch <= -5) & (sch >= -20)
            if sel.sum() >= max(8, int(0.005 * sr)):
                t_sel = np.where(sel)[0] / sr
                coef = np.polyfit(t_sel, sch[sel], 1)
                if coef[0] < -20:  # echt verval, geen vlakke vloer
                    rt = -60.0 / coef[0]
                    if 0.05 <= rt <= 3.0:
                        slopes.append(rt)
        i = j
    if not slopes:
        return None
    return round(float(np.median(slopes)), 2)
