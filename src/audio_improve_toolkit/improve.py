"""Auto-improve: analyse -> beslisregels -> verbeterketen (de 'maak het beter'-knop).

Elke regel motiveert zichzelf met een rationale-zin, zodat Claude in de chat kan
uitleggen wat er is gebeurd en waarom.
"""

from __future__ import annotations

import numpy as np

from audio_improve_toolkit import dsp

TARGETS = {
    "speech": {"lufs": -16.0, "true_peak_db": -1.5, "hpf_hz": 80.0},
    "music": {"lufs": -14.0, "true_peak_db": -1.0, "hpf_hz": 30.0},
}


def detect_profile(m: dict) -> str:
    """Heuristische spraak/muziek-detectie op de analysemetrics."""
    mid = m.get("band_energy_pct", {}).get("mid", 50)
    centroid = m.get("spectral_centroid_hz", 2000)
    silence = m.get("silence_pct", 0)
    score = 0
    if silence > 12:
        score += 1  # spraak bevat pauzes
    if mid > 60:
        score += 1  # stem-energie zit vrijwel geheel in het middengebied
    if centroid < 3200:
        score += 1
    if m.get("lra_db") is not None and m["lra_db"] > 10:
        score += 1  # spraak heeft doorgaans grotere loudness range dan een mix
    return "speech" if score >= 3 else "music"


def build_improve_chain(m: dict, profile: str = "auto", target_lufs: float | None = None,
                        denoise_method: str = "auto") -> tuple[str, list[dict], list[str]]:
    """Bepaal (profiel, stappen, rationale) op basis van de analyse."""
    if profile in (None, "auto"):
        profile = detect_profile(m)
    if profile not in TARGETS:
        raise ValueError(f"Onbekend profiel '{profile}'. Geldig: auto, speech, music")
    t = TARGETS[profile]
    lufs_target = target_lufs if target_lufs is not None else t["lufs"]

    steps: list[dict] = []
    rationale: list[str] = [
        f"Profiel: {'spraak' if profile == 'speech' else 'muziek'} "
        f"(target {lufs_target} LUFS, true peak {t['true_peak_db']} dBTP)."
    ]

    steps.append({"type": "highpass", "freq": t["hpf_hz"]})
    why = f"Highpass op {t['hpf_hz']:.0f} Hz tegen rumble en DC-offset"
    if m.get("dc_offset", 0) > 0.001:
        why += f" (DC-offset {m['dc_offset']} gemeten)"
    rationale.append(why + ".")

    hum = m.get("hum", {})
    if hum.get("detected"):
        f0 = hum["freq"]
        for h in (1, 2, 3):
            if f0 * h < m["sample_rate"] / 2 - 100:
                steps.append({"type": "notch", "freq": f0 * h, "q": 30.0})
        rationale.append(f"Netbrom rond {f0:.0f} Hz (+{hum['prominence_db']} dB): "
                         "notch-filters op de grondtoon en harmonischen.")

    snr = m.get("snr_db", 40)
    if snr < 25:
        strength = float(np.clip(30 - snr, 6, 18))
        method = denoise_method
        if method == "auto":
            method = "ai" if profile == "speech" and dsp.ai_denoise_available() else "spectral"
        if method == "ai" and not dsp.ai_denoise_available():
            from audio_improve_toolkit.dsp import ai_nr

            method = "spectral"
            rationale.append("AI-ruisonderdrukking is niet geinstalleerd; teruggevallen op "
                             f"spectral gating. ({ai_nr.INSTALL_HINT})")
        label = "DeepFilterNet (AI)" if method == "ai" else "spectral gating"
        steps.append({"type": "denoise", "strength_db": round(strength, 1), "method": method})
        rationale.append(f"SNR {snr} dB is laag: {label} met {strength:.0f} dB reductie.")

    if profile == "speech" and m.get("noise_floor_db", -80) > -60:
        thr = min(m["noise_floor_db"] + 6, -30)
        steps.append({"type": "gate", "threshold_db": round(thr, 1), "range_db": 10.0})
        rationale.append(f"Ruisvloer {m['noise_floor_db']} dB: zachte noise gate op "
                         f"{thr:.0f} dB maakt pauzes stil.")

    eq_bands: list[dict] = []
    tilt = m.get("tilt_db_per_octave", -4)
    if tilt < -6:
        eq_bands.append({"type": "highshelf", "freq": 8000, "gain_db": 3.0, "q": 0.707})
        rationale.append(f"Dof klankbeeld (tilt {tilt} dB/octaaf): +3 dB highshelf op 8 kHz.")
    if m.get("band_energy_pct", {}).get("low", 0) > 45:
        eq_bands.append({"type": "peaking", "freq": 300, "gain_db": -2.5, "q": 1.2})
        rationale.append("Veel laag-energie: -2.5 dB rond 300 Hz tegen modder.")
    if eq_bands:
        steps.append({"type": "eq", "bands": eq_bands})

    crest = m.get("crest_factor_db", 12)
    lra = m.get("lra_db") or 0
    if profile == "speech" and (crest > 18 or lra > 13):
        thr = round(m.get("rms_db", -24) + 4, 1)
        steps.append({"type": "compressor", "threshold_db": thr, "ratio": 2.5, "knee_db": 6.0})
        rationale.append(f"Grote dynamiek (crest {crest} dB, LRA {lra} dB): lichte "
                         f"compressie (2.5:1 vanaf {thr} dB) voor een egaler niveau.")

    steps.append({"type": "loudness_normalize", "target_lufs": lufs_target,
                  "true_peak_db": t["true_peak_db"]})
    cur = m.get("lufs_integrated")
    rationale.append(f"Loudness{f' van {cur} LUFS' if cur is not None else ''} naar "
                     f"{lufs_target} LUFS gebracht, met true-peak-limiter op "
                     f"{t['true_peak_db']} dBTP tegen clipping.")

    return profile, steps, rationale
