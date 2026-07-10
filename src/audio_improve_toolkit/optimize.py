"""Optimalisatie-harnas: kandidaat-pijplijnen draaien en objectief scoren.

De 'nachtrun'-modus: elke variant doorloopt de volledige verfijnlus, wordt
gemeten (doelen) en door Whisper beoordeeld (verstaanbaarheid); de beste wint.
De score beloont woordretentie en Whisper-zekerheid, en straft afwijking van de
spraakpiek-/balansdoelen en te luide pauzes.
"""

from __future__ import annotations

import logging

import numpy as np

from audio_improve_toolkit import asr, refine as refine_mod
from audio_improve_toolkit.segments import classify_segments, segment_slices

log = logging.getLogger(__name__)


def default_variants() -> list[dict]:
    """Kandidaten; AI-ontruising is een echte wedstrijddimensie ("denoise"), en
    dereverb-varianten doen alleen mee als het [enhance]-extra er is."""
    from audio_improve_toolkit.dsp import dereverb

    rustig = {"leveler": {"smooth_s": 1.6}, "compressor": {"attack_ms": 15.0}}
    presence_mild = {"eq_bands": [
        {"type": "peaking", "freq": 300, "gain_db": -3.0, "q": 1.2},
        {"type": "peaking", "freq": 3150, "gain_db": 2.0, "q": 0.9},
        {"type": "highshelf", "freq": 8000, "gain_db": 2.0, "q": 0.707}]}

    v = [
        {"name": "basis", "denoise": "auto", "tuning": {}},
        {"name": "basis-geen-ai", "denoise": "off", "tuning": {}},
        {"name": "rustig-geen-ai", "denoise": "off", "tuning": dict(rustig)},
    ]
    if dereverb.is_available():
        drv = {"pre_extra": [{"type": "dereverb"}]}
        v += [
            {"name": "dereverb-puur", "denoise": "off", "tuning": dict(drv)},
            {"name": "dereverb-deess", "denoise": "off",
             "tuning": {"pre_extra": [{"type": "dereverb"}, {"type": "deess"}]}},
            {"name": "dereverb-deess-rustig", "denoise": "off",
             "tuning": {"pre_extra": [{"type": "dereverb"}, {"type": "deess"}],
                         **rustig}},
            {"name": "dereverb-rustig", "denoise": "off", "tuning": {**drv, **rustig}},
            {"name": "dereverb-presence-mild", "denoise": "off",
             "tuning": {**drv, **presence_mild}},
            {"name": "dereverb-met-ai", "denoise": "auto", "tuning": dict(drv)},
        ]
    return v


def _score(meas: dict, targets: dict, asr_result: dict | None) -> float:
    s = 0.0
    if asr_result:
        s += 100.0 * asr_result["word_retention"]
        s += 8.0 * float(np.clip(asr_result["logprob_processed"]
                                 - asr_result["logprob_original"], -3.0, 1.0))
    if "speech_peak_db" in meas:
        s -= 4.0 * abs(meas["speech_peak_db"] - targets["speech_peak_db"])
    if "music_vs_speech_gap_db" in meas:
        s -= 3.0 * abs(meas["music_vs_speech_gap_db"] - targets["music_gap_db"])
    if "pause_floor_db" in meas:
        s -= 0.15 * max(meas["pause_floor_db"] + 44.0, 0.0)
    return round(s, 2)


def optimize(x: np.ndarray, sr: int, variants: list[dict] | None = None,
             speech_peak_db: float = -6.0, music_gap_db: float = 2.0,
             max_iterations: int = 4, denoise: str = "auto",
             judge_model: str = "small", progress=None) -> tuple[np.ndarray, dict]:
    """Draai alle varianten, scoor ze en geef (beste_audio, rapport) terug."""
    x2 = x[None, :] if x.ndim == 1 else x
    variants = variants or default_variants()
    targets = {"speech_peak_db": speech_peak_db, "music_gap_db": music_gap_db}

    segs = classify_segments(x2, sr)
    speech_sl = segment_slices(segs, sr, "speech")
    use_asr = asr.is_available() and bool(speech_sl)
    ref = None
    if use_asr:
        if progress:
            progress("referentietranscript (Whisper) maken")
        ref = asr.transcribe(refine_mod._cat(x2, speech_sl), sr, model_size=judge_model)
        ref_logprob = round(float(np.mean([s["avg_logprob"]
                                           for s in ref["segments"]] or [0])), 2)

    results = []
    best = None
    for i, variant in enumerate(variants, 1):
        if progress:
            progress(f"variant {i}/{len(variants)}: {variant['name']}")
        try:
            y, info = refine_mod.refine(
                x2, sr, speech_peak_db=speech_peak_db, music_gap_db=music_gap_db,
                max_iterations=max_iterations,
                denoise=variant.get("denoise", denoise),
                asr_check=False, tuning=variant.get("tuning", {}))
        except Exception as exc:
            log.warning("variant %s faalde: %s", variant["name"], exc)
            results.append({"name": variant["name"], "error": str(exc)})
            continue
        meas = info["report"]["final_measurements"]
        asr_result = None
        if use_asr:
            t = asr.transcribe(refine_mod._cat(y, speech_sl), sr, model_size=judge_model)
            asr_result = {
                "word_retention": asr.word_retention(ref["text"], t["text"]),
                "logprob_original": ref_logprob,
                "logprob_processed": round(float(np.mean(
                    [s["avg_logprob"] for s in t["segments"]] or [0])), 2),
                "transcript_processed": t["text"],
            }
        entry = {
            "name": variant["name"],
            "tuning": variant.get("tuning", {}),
            "score": _score(meas, targets, asr_result),
            "measurements": meas,
            "asr": asr_result,
            "decisions": info["report"].get("decisions", []),
        }
        results.append(entry)
        if best is None or entry["score"] > best["entry"]["score"]:
            best = {"entry": entry, "audio": y, "info": info}

    if best is None:
        raise RuntimeError("Alle varianten faalden; zie de log.")
    ranking = sorted([r for r in results if "score" in r],
                     key=lambda r: r["score"], reverse=True)
    report = {
        "winner": best["entry"]["name"],
        "targets": targets,
        "ranking": ranking,
        "failed": [r for r in results if "error" in r],
        "reference_transcript": ref["text"] if ref else None,
        "refine_report": best["info"]["report"],
    }
    return best["audio"], {"report": report, "steps": best["info"]["steps"]}
