"""Iteratieve verfijning: meten -> bijsturen -> opnieuw, tot de doelen kloppen.

De 'bulk' (AI-ontruising, leveling) gebeurt door de DSP/ML-laag; deze lus stuurt
per iteratie de parameters bij op basis van segmentbewuste metingen en geeft de
volledige meetgeschiedenis terug, zodat Claude in de chat kan meekijken en
bijsturen op de details.
"""

from __future__ import annotations

import logging

import numpy as np

from audio_improve_toolkit import chain
from audio_improve_toolkit.segments import classify_segments, segment_slices

log = logging.getLogger(__name__)


def _cat(y: np.ndarray, slices: list[slice]) -> np.ndarray | None:
    parts = [y[:, sl] for sl in slices if sl.stop > sl.start]
    return np.concatenate(parts, axis=1) if parts else None


def measure(y: np.ndarray, sr: int, segs: list[dict]) -> dict:
    """Segmentbewuste metingen waar de verfijnlus (en Claude) op stuurt."""
    def rms_db(seg):
        return round(float(10 * np.log10(np.mean(np.asarray(seg, np.float64) ** 2)
                                         + 1e-20)), 2)

    def peak_db(seg):
        return round(float(20 * np.log10(np.percentile(np.abs(seg), 99.9) + 1e-12)), 2)

    out: dict = {"true_peak_est_db": round(float(20 * np.log10(np.abs(y).max() + 1e-12)), 2)}
    speech = _cat(y, segment_slices(segs, sr, "speech"))
    music = _cat(y, segment_slices(segs, sr, "music"))
    silence = _cat(y, segment_slices(segs, sr, "silence"))
    if speech is not None:
        out["speech_peak_db"] = peak_db(speech)
        out["speech_rms_db"] = rms_db(speech)
    if music is not None:
        out["music_peak_db"] = peak_db(music)
        out["music_rms_db"] = rms_db(music)
    if speech is not None and music is not None:
        out["music_vs_speech_gap_db"] = round(out["music_rms_db"] - out["speech_rms_db"], 2)
    if silence is not None:
        out["pause_floor_db"] = rms_db(silence)
    return out


def _duck_silence(y: np.ndarray, sr: int, segs: list[dict], duck_db: float = 18.0,
                  fade_ms: float = 100.0) -> np.ndarray:
    """Druk stiltesegmenten terug die door de leveler zijn meegetild; zachte
    fades houden een natuurlijke roomtone over."""
    y = y.copy()
    g = 10.0 ** (-abs(duck_db) / 20.0)
    fade = max(1, int(fade_ms / 1000 * sr))
    for sl in segment_slices(segs, sr, "silence"):
        length = sl.stop - sl.start
        if length <= 0:
            continue
        w = np.full(length, g, dtype=np.float32)
        r = min(fade, length // 2)
        if r > 0:
            w[:r] = np.linspace(1.0, g, r)
            w[-r:] = np.linspace(g, 1.0, r)
        y[:, sl] *= w[None, :]
    return y


def _speech_snr_db(x: np.ndarray, sr: int, speech_slices: list[slice]) -> float | None:
    sp = _cat(x, speech_slices)
    if sp is None:
        return None
    mono = sp.mean(axis=0)
    flen = max(1, int(sr * 0.025))
    nf = mono.shape[0] // flen
    if nf < 8:
        return None
    fr = 10 * np.log10((mono[: nf * flen].reshape(nf, flen) ** 2).mean(axis=1) + 1e-20)
    floor = float(np.sort(fr)[: max(1, nf // 10)].mean())
    return round(float(np.percentile(fr, 90)) - floor, 1)


def refine(x: np.ndarray, sr: int, speech_peak_db: float = -6.0,
           music_gap_db: float = 2.0, max_iterations: int = 5,
           denoise: str = "auto", tone: bool = True, silence_duck_db: float = 18.0,
           asr_check: bool = True,
           progress=None) -> tuple[np.ndarray, dict]:
    """Verfijn tot spraakpieken en spraak/muziek-balans op de millimeter kloppen.

    denoise: "auto" (aan als de spraak-SNR laag is, en alleen als Whisper
    bevestigt dat het de transcribeerbaarheid niet schaadt), "on" of "off".
    Met asr_check en het [asr]-extra wordt het eindresultaat altijd op
    transcribeerbaarheid vergeleken met het origineel (report["asr"]).
    """
    from audio_improve_toolkit import asr

    x2 = x[None, :] if x.ndim == 1 else x
    segs = classify_segments(x2, sr)
    has_speech = any(s["kind"] == "speech" for s in segs)
    has_music = any(s["kind"] == "music" for s in segs)
    speech_sl = segment_slices(segs, sr, "speech")
    decision_log: list[str] = []

    # Besluit over AI-ontruising: alleen bij lage spraak-SNR. Bij hoge SNR is het
    # 'vuil' doorgaans zaalgalm — daar maakt een denoiser spraak juist slechter.
    use_denoise = denoise == "on"
    if denoise == "auto" and has_speech:
        snr = _speech_snr_db(x2, sr, speech_sl)
        if snr is not None and snr < 32.0:
            use_denoise = True
            decision_log.append(f"Spraak-SNR {snr} dB is laag: AI-ontruising ingezet.")
        else:
            decision_log.append(f"Spraak-SNR {snr} dB is al hoog; AI-ontruising "
                                "overgeslagen (het restgeluid is vooral zaalgalm, "
                                "daar helpt een denoiser niet tegen).")

    pre_steps: list[dict] = [{"type": "highpass", "freq": 80}]
    if use_denoise:
        pre_steps.append({"type": "smart_denoise", "speech_strength_db": 100})
    if tone:
        pre_steps.append({"type": "eq", "bands": [
            {"type": "peaking", "freq": 300, "gain_db": -3.0, "q": 1.2},
            {"type": "peaking", "freq": 3150, "gain_db": 3.5, "q": 0.9},  # verstaanbaarheid
            {"type": "highshelf", "freq": 8000, "gain_db": 2.0, "q": 0.707},
        ]})
    if progress:
        progress("voorbewerking")
    x_clean, pre_resolved = chain.run_chain(x2, sr, pre_steps)

    # Whisper als scheidsrechter: schaadt de ontruising de transcribeerbaarheid?
    ref_transcript = None
    if asr_check and has_speech and asr.is_available():
        ref_transcript = asr.transcribe(_cat(x2, speech_sl), sr)
        if use_denoise:
            t_clean = asr.transcribe(_cat(x_clean, speech_sl), sr)
            retention = asr.word_retention(ref_transcript["text"], t_clean["text"])
            if retention < 0.75:
                decision_log.append(f"Whisper-check: woordretentie zakte naar "
                                    f"{retention:.0%} door de ontruising — "
                                    "teruggedraaid, doorgegaan zonder AI-ontruising.")
                use_denoise = False
                pre_steps = [s for s in pre_steps if s["type"] != "smart_denoise"]
                x_clean, pre_resolved = chain.run_chain(x2, sr, pre_steps)
            else:
                decision_log.append(f"Whisper-check: woordretentie {retention:.0%} "
                                    "na ontruising — akkoord.")

    lufs_t, cut = -18.0, 14.0
    history: list[dict] = []
    y, loop_resolved = x_clean, []
    for it in range(1, max_iterations + 1):
        loop_steps: list[dict] = []
        if has_speech and has_music:
            loop_steps.append({"type": "leveler", "target_db": -18.0,
                               "max_boost_db": 20.0, "max_cut_db": round(cut, 1)})
        # dichter op de spraak: egaler = beter verstaanbaar
        loop_steps.append({"type": "compressor", "threshold_db": -14.0, "ratio": 2.5,
                           "attack_ms": 5.0, "release_ms": 180.0})
        loop_steps.append({"type": "loudness_normalize",
                           "target_lufs": round(lufs_t, 2), "true_peak_db": -1.5})
        y, loop_resolved = chain.run_chain(x_clean, sr, loop_steps)
        if silence_duck_db > 0:
            y = _duck_silence(y, sr, segs, duck_db=silence_duck_db)
        meas = measure(y, sr, segs)

        err_pk = (speech_peak_db - meas["speech_peak_db"]) if has_speech else 0.0
        err_gap = ((meas.get("music_vs_speech_gap_db", music_gap_db) - music_gap_db)
                   if (has_speech and has_music) else 0.0)
        entry = {"iteration": it, "params": {"target_lufs": round(lufs_t, 2),
                                             "leveler_max_cut_db": round(cut, 1)},
                 "measurements": meas,
                 "errors": {"speech_peak": round(err_pk, 2), "balance_gap": round(err_gap, 2)}}
        history.append(entry)
        log.info("verfijning %d: %s", it, entry)
        if progress:
            progress(f"iteratie {it}: spraakpiek-afwijking {err_pk:+.1f} dB, "
                     f"balans-afwijking {err_gap:+.1f} dB")

        if abs(err_pk) <= 0.5 and abs(err_gap) <= 0.75:
            entry["converged"] = True
            break
        lufs_t += float(np.clip(err_pk, -5.0, 5.0))
        cut = float(np.clip(cut + np.clip(err_gap, -6.0, 6.0), 6.0, 26.0))

    # Eindrapport transcribeerbaarheid: origineel vs resultaat.
    asr_report = None
    if ref_transcript is not None:
        t_final = asr.transcribe(_cat(y, speech_sl), sr)
        asr_report = {
            "word_retention": asr.word_retention(ref_transcript["text"], t_final["text"]),
            "logprob_original": round(float(np.mean(
                [s["avg_logprob"] for s in ref_transcript["segments"]] or [0])), 2),
            "logprob_processed": round(float(np.mean(
                [s["avg_logprob"] for s in t_final["segments"]] or [0])), 2),
            "transcript_original": ref_transcript["text"],
            "transcript_processed": t_final["text"],
        }
        decision_log.append(f"Eindcheck Whisper: woordretentie "
                            f"{asr_report['word_retention']:.0%}.")

    report = {
        "segments": segs,
        "iterations": history,
        "converged": bool(history and history[-1].get("converged", False)),
        "final_measurements": history[-1]["measurements"] if history else {},
        "targets": {"speech_peak_db": speech_peak_db, "music_gap_db": music_gap_db},
        "decisions": decision_log,
        "asr": asr_report,
    }
    return y, {"report": report, "steps": pre_resolved + loop_resolved}
