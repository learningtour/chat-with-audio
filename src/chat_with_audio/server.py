"""MCP-server (stdio) voor Claude Desktop en Claude Code.

Servernaam: chat-with-audio. Alle logging gaat naar stderr; stdout is
gereserveerd voor het MCP-protocol.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

from chat_with_audio import analysis, chain, dsp, improve, io, sessions

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

mcp = FastMCP("chat-with-audio")

VIEWER_PORT = int(os.environ.get("AIT_VIEWER_PORT", "8471"))


def _viewer_url(session_id: str | None = None) -> str:
    base = f"http://127.0.0.1:{VIEWER_PORT}/"
    return f"{base}#/session/{session_id}" if session_id else base


def _viewer_running() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{VIEWER_PORT}/health", timeout=1):
            return True
    except Exception:
        return False


def _ensure_viewer() -> bool:
    if _viewer_running():
        return True
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED | NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, "-m", "chat_with_audio.viewer"], **kwargs)
    for _ in range(25):
        time.sleep(0.2)
        if _viewer_running():
            return True
    return False


def _process(file_path: str, steps: list[dict], rationale: list[str],
             profile: str | None = None, out_path: str | None = None,
             user_request: str | None = None) -> dict:
    """Gedeelde route: laden -> keten -> analyse voor/na -> sessie -> resultaat."""
    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    y, resolved = chain.run_chain(x, sr, steps)
    m1 = analysis.analyze(y, sr)
    session = sessions.create_session(file_path, x, sr, m0, y, m1, resolved,
                                      rationale, profile, user_request=user_request)
    scores_after, issues_after = analysis.score_and_issues(m1)

    result = {
        "session_id": session["session_id"],
        "output_path": str(sessions.session_path(session["session_id"]) / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "profile": profile,
        "chain": resolved,
        "rationale": rationale,
        "metrics_before": m0,
        "metrics_after": m1,
        "scores_after": scores_after,
        "remaining_issues": issues_after,
        "deltas": session["deltas"],
        "hint": "Gebruik open_viewer om origineel en resultaat naast elkaar te beluisteren.",
    }
    if out_path:
        wav = sessions.session_path(session["session_id"]) / "processed.wav"
        result["export_path"] = str(io.encode_wav_to(wav, out_path))
    return result


@mcp.tool()
def analyze_audio(file_path: str, create_session: bool = True) -> dict:
    """Analyseer de audiokwaliteit van een bestand (wav/mp3/m4a/flac/ogg).

    Geeft metrics (LUFS, true peak, ruisvloer, SNR, dynamiek, spectrum, brom,
    clipping), 0-100 scores per categorie en concrete issues met suggesties.
    Met create_session=True is het bestand daarna ook in de viewer te bekijken.
    """
    x, sr = io.load_audio(file_path)
    m = analysis.analyze(x, sr)
    scores, issues = analysis.score_and_issues(m)
    result = {
        "file": str(Path(file_path).expanduser()),
        "container": io.probe(file_path),
        "detected_profile": improve.detect_profile(m),
        "metrics": m,
        "scores": scores,
        "issues": issues,
        "ai_denoise_available": dsp.ai_denoise_available(),
        "dsp_backend": dsp.backend(),
    }
    from chat_with_audio import training

    taste = training.score(m)
    result["taste"] = taste if taste else {
        "hint": "Nog geen smaakmodel: label voorbeelden met rate_audio "
                "(minimaal 2x 'good' en 2x 'bad')."}
    if create_session:
        session = sessions.create_session(file_path, x, sr, m)
        result["session_id"] = session["session_id"]
        result["viewer_url"] = _viewer_url(session["session_id"])
    return result


@mcp.tool()
def improve_audio(file_path: str, profile: str = "auto", target_lufs: float | None = None,
                  denoise_method: str = "auto", out_path: str | None = None,
                  user_request: str = "") -> dict:
    """Verbeter audio automatisch ("maak dit geluid beter").

    Analyseert het bestand en kiest zelf een keten (highpass, brom-notches,
    ruisonderdrukking, gate, EQ, compressie, loudness-normalisatie) met
    onderbouwing per stap. profile: auto|speech|music (auto detecteert).
    denoise_method: auto|spectral|ai (ai = DeepFilterNet, best voor spraak).
    out_path: optioneel exportpad; het formaat volgt de extensie (.mp3, .m4a,
    .flac, .ogg, .wav). Het resultaat staat altijd ook als wav in de sessie.
    Geef in user_request de letterlijke vraag van de gebruiker door: die wordt
    opgenomen in het sessielogboek (log.md) voor volledige herleidbaarheid.
    """
    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    used_profile, steps, rationale = improve.build_improve_chain(
        m0, profile=profile, target_lufs=target_lufs, denoise_method=denoise_method)
    return _process(file_path, steps, rationale, profile=used_profile, out_path=out_path,
                    user_request=user_request or None)


@mcp.tool()
def reduce_noise(file_path: str, strength_db: float = 12.0, method: str = "auto",
                 use_gate: bool = True, out_path: str | None = None,
                 user_request: str = "") -> dict:
    """Verminder alleen de ruis; loudness blijft verder ongemoeid.

    method: auto|spectral|ai. 'ai' gebruikt DeepFilterNet (state-of-the-art voor
    spraak); 'spectral' is STFT spectral gating (muziekveilig); 'auto' kiest ai
    bij spraak als het beschikbaar is. strength_db is de maximale demping.
    use_gate voegt een zachte noise gate toe die pauzes stil maakt (alleen zinvol
    bij spraak).
    """
    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    detected = improve.detect_profile(m0)
    if method == "auto":
        method = "ai" if detected == "speech" and dsp.ai_denoise_available() else "spectral"
    rationale = []
    if method == "ai" and not dsp.ai_denoise_available():
        from chat_with_audio.dsp import ai_nr

        method = "spectral"
        rationale.append(f"AI-methode niet beschikbaar; spectral gating gebruikt. "
                         f"({ai_nr.INSTALL_HINT})")
    label = "DeepFilterNet (AI)" if method == "ai" else "spectral gating"
    steps: list[dict] = [{"type": "denoise", "strength_db": strength_db, "method": method}]
    rationale.append(f"Ruisonderdrukking via {label}, maximaal {strength_db:.0f} dB demping.")
    if use_gate and detected == "speech" and m0.get("noise_floor_db", -80) > -65:
        thr = min(m0["noise_floor_db"] + 6, -30)
        steps.append({"type": "gate", "threshold_db": round(thr, 1), "range_db": 10.0})
        rationale.append(f"Zachte noise gate op {thr:.0f} dB maakt de pauzes stil.")
    return _process(file_path, steps, rationale, profile=detected, out_path=out_path,
                    user_request=user_request or None)


@mcp.tool()
def normalize_loudness(file_path: str, target_lufs: float = -16.0,
                       true_peak_db: float = -1.5, out_path: str | None = None,
                       user_request: str = "") -> dict:
    """Trek het niveau over de hele breedte op (of omlaag) zonder te clippen.

    Meet de loudness (BS.1770) en brengt die naar target_lufs; een look-ahead
    true-peak-limiter bewaakt true_peak_db. Richtwaarden: spraak/podcast -16,
    muziek/streaming -14 LUFS.
    """
    steps = [{"type": "loudness_normalize", "target_lufs": target_lufs,
              "true_peak_db": true_peak_db}]
    rationale = [f"Loudness genormaliseerd naar {target_lufs} LUFS met "
                 f"true-peak-limiter op {true_peak_db} dBTP."]
    return _process(file_path, steps, rationale, out_path=out_path,
                    user_request=user_request or None)


@mcp.tool()
def apply_chain(file_path: str, steps: list[dict], out_path: str | None = None,
                user_request: str = "") -> dict:
    """Pas een expliciete bewerkingsketen toe (voor fijnsturing in de chat).

    steps is een lijst van stap-objecten, uitgevoerd in volgorde. Beschikbaar:
      {"type": "highpass", "freq": 80, "q": 0.707}
      {"type": "lowpass", "freq": 16000, "q": 0.707}
      {"type": "notch", "freq": 50, "q": 30}
      {"type": "eq", "bands": [{"type": "peaking", "freq": 300, "gain_db": -2.5, "q": 1.2},
                                {"type": "highshelf", "freq": 8000, "gain_db": 3, "q": 0.707}]}
        (band-types: peaking, lowshelf, highshelf, lowpass, highpass, notch)
      {"type": "gain", "gain_db": 3.0}
      {"type": "denoise", "strength_db": 12, "method": "spectral"|"ai"}
      {"type": "gate", "threshold_db": -50, "attack_ms": 5, "release_ms": 120,
       "hold_ms": 50, "range_db": 12}
      {"type": "compressor", "threshold_db": -20, "ratio": 3, "attack_ms": 10,
       "release_ms": 150, "knee_db": 6, "makeup_db": 0}
      {"type": "leveler", "target_db": -18, "max_boost_db": 20, "max_cut_db": 18,
       "floor_db": -40, "smooth_s": 0.8}
        (gain-riding: stille passages omhoog, luide omlaag naar een gezamenlijk
         niveau — voor spraak/muziek-balans; floor_db beschermt stilte/ruis)
      {"type": "limiter", "ceiling_db": -1.5, "release_ms": 60, "lookahead_ms": 5}
      {"type": "loudness_normalize", "target_lufs": -16, "true_peak_db": -1.5}
      {"type": "breath_control", "reduction_db": 10}
        (ademhalingen dempen, niet wegknippen — dialoogbewerking)
      {"type": "deplosive", "cutoff_hz": 120, "sensitivity_db": 6}
        (p/b-pops: alleen de laagfrequente stoot zelf wordt gehighpasst)
      {"type": "duck_music", "gap_db": 6, "mode": "beds"|"stems"}
        (beds = muziekbedden tussen de spraak omlaag; stems = echte
         sidechain-ducking voor muziek ónder spraak via Demucs, [stems]-extra)

    Tip: sluit af met een limiter of loudness_normalize als eerdere stappen het
    niveau verhogen.
    """
    rationale = [f"Handmatige keten met {len(steps)} stap(pen), samengesteld in de chat."]
    return _process(file_path, steps, rationale, out_path=out_path,
                    user_request=user_request or None)


@mcp.tool()
def refine_audio(file_path: str, speech_peak_db: float = -6.0, music_gap_db: float = 2.0,
                 max_iterations: int = 5, denoise: str = "auto", tone: bool = True,
                 asr_check: bool = True, out_path: str | None = None,
                 user_request: str = "") -> dict:
    """Iteratieve verfijning tot spraakniveau en spraak/muziek-balans exact kloppen.

    Segmenteert het bestand (spraak/muziek/stilte) en draait een meet-en-bijstuur-
    lus (leveler + compressor + loudness) tot de spraakpieken op speech_peak_db
    (dBFS) zitten en de muziek music_gap_db daarboven. denoise: auto|on|off —
    'auto' zet AI-ontruising alleen in bij lage spraak-SNR en laat Whisper
    (indien geinstalleerd) verifieren dat de transcribeerbaarheid niet daalt;
    het rapport bevat de meetgeschiedenis, de genomen beslissingen en een
    woordretentie-eindcheck ('report.asr') om verstaanbaarheidsverlies te zien.
    """
    from chat_with_audio import refine as refine_mod

    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    y, info = refine_mod.refine(x, sr, speech_peak_db=speech_peak_db,
                                music_gap_db=music_gap_db,
                                max_iterations=max_iterations,
                                denoise=denoise, tone=tone, asr_check=asr_check)
    m1 = analysis.analyze(y, sr)
    rep = info["report"]
    rationale = [f"Iteratieve verfijning ({len(rep['iterations'])} iteraties, "
                 f"{'geconvergeerd' if rep['converged'] else 'maximum bereikt'}): "
                 f"doel spraakpiek {speech_peak_db} dBFS, muziek {music_gap_db:+.1f} dB daarbij."]
    rationale.extend(rep.get("decisions", []))
    for it in rep["iterations"]:
        e = it["errors"]
        rationale.append(f"Iteratie {it['iteration']}: spraakpiek-afwijking "
                         f"{e['speech_peak']:+.1f} dB, balans-afwijking "
                         f"{e['balance_gap']:+.1f} dB.")
    session = sessions.create_session(file_path, x, sr, m0, y, m1, info["steps"],
                                      rationale, "speech",
                                      label=f"{Path(file_path).name} — verfijnd",
                                      user_request=user_request or None,
                                      asr_report=rep.get("asr"))
    result = {
        "session_id": session["session_id"],
        "output_path": str(sessions.session_path(session["session_id"]) / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "report": rep,
        "chain": info["steps"],
        "rationale": rationale,
        "metrics_before": m0,
        "metrics_after": m1,
        "deltas": session["deltas"],
    }
    if out_path:
        wav = sessions.session_path(session["session_id"]) / "processed.wav"
        result["export_path"] = str(io.encode_wav_to(wav, out_path))
    return result


_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".aiff", ".aif"}


@mcp.tool()
def improve_folder(dir_path: str, mode: str = "improve", profile: str = "auto",
                   out_dir: str | None = None) -> dict:
    """Batchverwerking: verbeter alle audiobestanden in een map in een keer.

    mode: improve (snel, regelgestuurd) | refine (meet-en-bijstuur-lus) |
    optimize (varianten-wedstrijd per bestand; traag maar maximaal). Elke
    verwerking wordt een eigen sessie; out_dir exporteert de resultaten daar
    in het bronformaat. Handig voor afleveringen, opnamedagen of archieven.
    """
    d = Path(dir_path).expanduser()
    if not d.is_dir():
        raise ValueError(f"Geen map: {d}")
    files = sorted(p for p in d.iterdir()
                   if p.suffix.lower() in _AUDIO_EXTS and not p.name.startswith("."))
    if not files:
        raise ValueError(f"Geen audiobestanden gevonden in {d} "
                         f"(gezocht naar: {', '.join(sorted(_AUDIO_EXTS))})")
    results, failures = [], []
    for p in files:
        out_path = str(Path(out_dir).expanduser() / p.name) if out_dir else None
        try:
            if mode == "improve":
                r = improve_audio(str(p), profile=profile, out_path=out_path)
            elif mode == "refine":
                r = refine_audio(str(p), out_path=out_path)
            elif mode == "optimize":
                r = optimize_audio(str(p), out_path=out_path)
            else:
                raise ValueError(f"Onbekende mode '{mode}' (improve|refine|optimize)")
            results.append({"file": p.name, "session_id": r["session_id"],
                            "deltas": r.get("deltas")})
        except Exception as exc:
            log.warning("batch: %s faalde: %s", p.name, exc)
            failures.append({"file": p.name, "error": str(exc)})
    return {"processed": len(results), "failed": len(failures),
            "results": results, "failures": failures,
            "viewer_url": _viewer_url()}


@mcp.tool()
def list_sessions(session_id: str | None = None) -> dict:
    """Toon eerdere sessies, of met session_id de volledige voor/na-vergelijking.

    Met session_id krijg je beide analyses (metrics, scores, issues), de
    uitgevoerde keten met rationale en de deltas — dezelfde data als de viewer
    toont, dus hierover kun je doorpraten.
    """
    if session_id:
        data = sessions.load_session(session_id)
        data["viewer_url"] = _viewer_url(session_id)
        return data
    items = sessions.list_sessions()
    return {"count": len(items), "sessions": items,
            "sessions_dir": str(sessions.sessions_dir())}


@mcp.tool()
def match_reference(file_path: str, reference_path: str, strength: float = 1.0,
                    max_db: float = 6.0, match_loudness: bool = True,
                    out_path: str | None = None) -> dict:
    """Laat een opname klinken als een referentiebestand ("klink zoals dit").

    Vergelijkt het spectrum in 1/3-octaafbanden en corrigeert het verschil met
    een begrensde match-EQ (max_db per band), plus optioneel loudness-match naar
    de referentie. strength 0-1 regelt hoe ver richting de referentie (0.5 =
    halverwege). Ideaal om afleveringen/opnamedagen consistent te maken.
    """
    from chat_with_audio import match as match_mod

    x, sr = io.load_audio(file_path)
    ref, ref_sr = io.load_audio(reference_path)
    m0 = analysis.analyze(x, sr)
    y, info = match_mod.match_reference(x, sr, ref, ref_sr, strength=strength,
                                        max_db=max_db, match_loudness=match_loudness)
    m1 = analysis.analyze(y, sr)
    rationale = [f"Spectraal gematcht aan {Path(reference_path).name} "
                 f"(sterkte {strength:.0%}, begrensd op ±{max_db:.0f} dB per band)."]
    if info["eq_bands"]:
        rationale.append("Match-EQ: " + ", ".join(info["eq_description"]) + ".")
    else:
        rationale.append("Spectra kwamen al vrijwel overeen; geen EQ nodig.")
    if info.get("loudness"):
        rationale.append(f"Loudness gematcht naar {info['loudness'].get('lufs_after')} LUFS "
                         "(true-peak-bewaakt).")
    steps = ([{"type": "eq", "bands": info["eq_bands"]}] if info["eq_bands"] else [])
    session = sessions.create_session(file_path, x, sr, m0, y, m1, steps, rationale,
                                      None, label=f"{Path(file_path).name} — match")
    result = {
        "session_id": session["session_id"],
        "output_path": str(sessions.session_path(session["session_id"]) / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "match": info["eq_description"],
        "rationale": rationale,
        "deltas": session["deltas"],
    }
    if out_path:
        wav = sessions.session_path(session["session_id"]) / "processed.wav"
        result["export_path"] = str(io.encode_wav_to(wav, out_path))
    return result


@mcp.tool()
def separate_stems(file_path: str, out_dir: str | None = None) -> dict:
    """Splits muziek in stems: vocals, drums, bass en other (Demucs AI-model).

    De stems worden als losse wav's in de sessiemap (of out_dir) gezet — direct
    bruikbaar in een DAW. Voor herbalanceren in een keer: rebalance_music.
    Vereist het [stems]-extra (uv sync --all-extras).
    """
    from chat_with_audio.dsp import stems as stems_mod

    if not stems_mod.is_available():
        raise RuntimeError(stems_mod.INSTALL_HINT)
    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    parts = stems_mod.separate(x, sr)
    session = sessions.create_session(file_path, x, sr, m0,
                                      label=f"{Path(file_path).name} — stems")
    stems_dir = Path(out_dir).expanduser() if out_dir else (
        sessions.session_path(session["session_id"]) / "stems")
    stems_dir.mkdir(parents=True, exist_ok=True)
    result_stems = {}
    for name, y in parts.items():
        p = io.save_wav(stems_dir / f"{name}.wav", y, sr)
        rms = float(20 * np.log10(np.sqrt((y**2).mean()) + 1e-12))
        result_stems[name] = {"path": str(p), "rms_db": round(rms, 1)}
    return {"session_id": session["session_id"], "stems": result_stems,
            "hint": "Gebruik rebalance_music om de mix te herbalanceren "
                    "(bv. vocals_db=+3, of vocals_db=-60 voor karaoke)."}


@mcp.tool()
def rebalance_music(file_path: str, vocals_db: float = 0.0, drums_db: float = 0.0,
                    bass_db: float = 0.0, other_db: float = 0.0,
                    target_lufs: float | None = None,
                    out_path: str | None = None) -> dict:
    """Herbalanceer een mix per stem: "zang 3 dB erbij", "drums zachter", of een
    karaoke-versie (vocals_db=-60). Splitst met Demucs, past per stem gain toe,
    mixt terug en bewaakt de pieken met een limiter. target_lufs normaliseert
    de eindmix optioneel. Resultaat staat als A/B-sessie in de viewer.
    """
    from chat_with_audio.dsp import stems as stems_mod

    if not stems_mod.is_available():
        raise RuntimeError(stems_mod.INSTALL_HINT)
    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    parts = stems_mod.separate(x, sr)
    gains = {"vocals": vocals_db, "drums": drums_db, "bass": bass_db, "other": other_db}
    y = np.zeros_like(x, dtype=np.float32)
    for name, part in parts.items():
        y = y + part * (10.0 ** (gains.get(name, 0.0) / 20.0))
    steps: list[dict] = []
    peak = float(np.abs(y).max())
    if target_lufs is not None:
        steps.append({"type": "loudness_normalize", "target_lufs": target_lufs,
                      "true_peak_db": -1.0})
    elif peak > 10.0 ** (-1.0 / 20.0):
        # statische gain naar -2 dBFS piek (dynamiek intact), limiter alleen als vangnet
        steps.append({"type": "gain",
                      "gain_db": round(20.0 * np.log10(10.0 ** (-2.0 / 20.0) / peak), 2)})
        steps.append({"type": "limiter", "ceiling_db": -1.0})
    y, resolved = chain.run_chain(y, sr, steps)
    m1 = analysis.analyze(y, sr)
    changed = ", ".join(f"{k} {v:+.1f} dB" for k, v in gains.items() if v)
    rationale = [f"Stems gescheiden (Demucs) en geherbalanceerd: {changed or 'geen wijziging'}.",
                 "Eindmix piekbewaakt" + (f" en genormaliseerd naar {target_lufs} LUFS."
                                          if target_lufs is not None else ".")]
    session = sessions.create_session(file_path, x, sr, m0, y, m1,
                                      [{"type": "rebalance", **gains}] + resolved,
                                      rationale, None,
                                      label=f"{Path(file_path).name} — rebalance")
    result = {
        "session_id": session["session_id"],
        "output_path": str(sessions.session_path(session["session_id"]) / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "rationale": rationale,
        "deltas": session["deltas"],
    }
    if out_path:
        wav = sessions.session_path(session["session_id"]) / "processed.wav"
        result["export_path"] = str(io.encode_wav_to(wav, out_path))
    return result


@mcp.tool()
def repair_audio(file_path: str, declip: bool = True, declick: bool = True,
                 out_path: str | None = None, user_request: str = "") -> dict:
    """Repareer beschadigde audio: declip (reconstrueert afgekapte golftoppen via
    spline-interpolatie) en declick (verwijdert korte impulsen/klikken).

    Verandert verder niets aan klank of niveau; combineer met improve_audio of
    refine_audio voor het volledige traject. Werkt ook op 32-bit float opnames
    met capsule-oversturing (flat-tops boven 0 dBFS).
    """
    steps: list[dict] = []
    rationale: list[str] = []
    if declip:
        steps.append({"type": "declip"})
        rationale.append("Declip: afgekapte golftoppen gereconstrueerd.")
    if declick:
        steps.append({"type": "declick"})
        rationale.append("Declick: impulsartefacten gerepareerd.")
    if not steps:
        raise ValueError("Niets te doen: declip en declick staan beide uit.")
    return _process(file_path, steps, rationale, out_path=out_path,
                    user_request=user_request or None)


@mcp.tool()
def smart_edit(file_path: str, problems: str = "auto", denoise_method: str = "auto",
               out_path: str | None = None, user_request: str = "") -> dict:
    """Chirurgische bewerking: AI vindt probleemregio's en behandelt alléén die delen.

    Waar improve_audio het hele bestand mastert, zoekt smart_edit op de tijdlijn
    naar plekken waar iets mis is en repareert uitsluitend daar, met crossfades:
      hum   — netbrom die aan/uit gaat (koelkast, dimmer): notch alleen daar
      noise — ruis die tijdelijk opkomt (airco, verkeer): ontruising alleen daar
      clip  — clusters afgekapte toppen: declip alleen rond de schade
      boom  — laagfrequente dreun (passerende vrachtwagen): laag-cut alleen daar
    Alles buiten de regio's blijft bit-voor-bit onaangetast; niveau en klank van
    het geheel veranderen niet. problems: "auto" (alles) of kommalijst uit
    hum,noise,clip,boom. denoise_method: auto|spectral|ai (voor ruisregio's die
    spraak raken). De regiokaart staat als tijdlijn in de viewer en in het
    resultaat, met per regio de tijden, de diagnose en de toegepaste fix.
    """
    from chat_with_audio import regions as regions_mod
    from chat_with_audio.segments import classify_segments

    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    segs = classify_segments(x, sr)
    found = regions_mod.detect_regions(x, sr, segments=segs)
    if problems not in ("auto", "", "all"):
        wanted = {p.strip() for p in problems.split(",")}
        unknown = wanted - set(regions_mod.KIND_LABELS)
        if unknown:
            raise ValueError(f"Onbekende probleemsoort(en) {sorted(unknown)}. "
                             f"Geldig: {sorted(regions_mod.KIND_LABELS)} of 'auto'.")
        found = [r for r in found if r["kind"] in wanted]
    if not found:
        return {"regions": [],
                "message": "Geen probleemregio's gevonden: geen plaatselijke brom, "
                           "ruis, clipping of dreun. Voor algehele verbetering "
                           "(loudness, EQ, dynamiek) is improve_audio de weg."}

    ai_ok = denoise_method != "spectral" and dsp.ai_denoise_available()
    planned, rationale = regions_mod.plan_region_fixes(found, sr, ai_available=ai_ok,
                                                       segments=segs)
    if denoise_method == "ai" and not dsp.ai_denoise_available():
        from chat_with_audio.dsp import ai_nr

        rationale.append(f"AI-ontruising niet beschikbaar; spectral gating "
                         f"gebruikt. ({ai_nr.INSTALL_HINT})")
    y, applied = regions_mod.apply_regions(x, sr, planned)
    m1 = analysis.analyze(y, sr)

    region_summary = [{"kind": r["kind"], "label": r.get("label", r["kind"]),
                       "start_s": round(r["start_s"], 2), "end_s": round(r["end_s"], 2),
                       "severity_db": r.get("severity_db")} for r in applied]
    chain_steps = [{"type": "region", "kind": r["kind"],
                    "start_s": round(r["start_s"], 2), "end_s": round(r["end_s"], 2),
                    "steps": r["steps"]} for r in applied]
    rationale = [f"Chirurgische bewerking: {len(applied)} regio('s) behandeld, "
                 "alles daarbuiten bit-voor-bit onaangetast."] + rationale
    session = sessions.create_session(
        file_path, x, sr, m0, y, m1, chain_steps, rationale, None,
        label=f"{Path(file_path).name} — chirurgisch",
        user_request=user_request or None,
        timeline={"segments": segs, "regions": region_summary})
    result = {
        "session_id": session["session_id"],
        "output_path": str(sessions.session_path(session["session_id"]) / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "regions": region_summary,
        "rationale": rationale,
        "metrics_before": m0,
        "metrics_after": m1,
        "deltas": session["deltas"],
        "hint": "De tijdlijn in de viewer toont waar is ingegrepen; toets r laat "
                "je per regio horen wat er is weggehaald.",
    }
    if out_path:
        wav = sessions.session_path(session["session_id"]) / "processed.wav"
        result["export_path"] = str(io.encode_wav_to(wav, out_path))
    return result


@mcp.tool()
def check_compliance(file_path: str, spec: str = "ebu-r128") -> dict:
    """Controleer een bestand tegen een aflever-spec ("is dit broadcast-proof?").

    Specs: ebu-r128 (Europese omroep, -23 LUFS), atsc-a85 (VS-tv, -24),
    netflix-2.0 (dialogue-gated -27), apple-podcast (-16), spotify (-14),
    youtube (-14), acx-audiobook (RMS/piek/ruisvloer). Het rapport geeft per
    criterium gemeten vs vereist met pass/fail, plus de universele technische
    QC-poorten (clipping, dropouts, dood kanaal, tegenfase, kop/staart-stilte).
    Dialogue-gated loudness wordt benaderd via BS.1770 over de gedetecteerde
    spraaksegmenten. Repareren: master_for.
    """
    from chat_with_audio import compliance as comp

    x, sr = io.load_audio(file_path)
    m = analysis.analyze(x, sr)
    dlg = None
    if comp.SPECS.get(spec, {}).get("gating") == "dialogue":
        from chat_with_audio.segments import classify_segments

        dlg = comp.dialogue_loudness(x, sr, classify_segments(x, sr))
    report = comp.check(m, spec, dialogue_lufs=dlg, file_info=io.probe(file_path))
    report["file"] = str(Path(file_path).expanduser())
    report["available_specs"] = comp.list_specs()
    if not report["passed"]:
        report["hint"] = (f"master_for(file_path, spec='{spec}') mastert naar deze "
                          "spec en controleert opnieuw.")
    return report


@mcp.tool()
def master_for(file_path: str, spec: str = "ebu-r128", out_path: str | None = None,
               sample_rate: int | None = None, bit_depth: int | None = None,
               user_request: str = "") -> dict:
    """Master een bestand naar een aflever-spec en controleer het resultaat.

    Brengt de loudness naar het spec-doel (dialogue-gated specs sturen op de
    spraaksegmenten) met een true-peak-limiter onder het spec-plafond, draait
    daarna check_compliance opnieuw en levert het pass/fail-rapport mee.
    out_path exporteert het resultaat; sample_rate (bv. 48000) en bit_depth
    (16|24|32, 32 = float) maken er een leveringsbestand van — broadcast wil
    doorgaans 48 kHz / 24-bit wav. Het rapport staat ook als compliance.json
    in de sessie en als paneel in de viewer.
    """
    from chat_with_audio import compliance as comp
    from chat_with_audio.segments import classify_segments

    spec_def = comp.SPECS.get(spec)
    if spec_def is None:
        raise ValueError(f"Onbekende spec '{spec}'. "
                         f"Beschikbaar: {', '.join(sorted(comp.SPECS))}")
    if bit_depth is not None and bit_depth not in io.BIT_DEPTH_SUBTYPES:
        raise ValueError(f"bit_depth moet 16, 24 of 32 zijn (niet {bit_depth}).")

    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    segs = classify_segments(x, sr)
    tp_max = spec_def.get("true_peak_max", -1.0)
    loud = spec_def.get("loudness")
    rationale = [f"Master naar {spec_def['name']}."]
    steps: list[dict] = []

    if loud and spec_def.get("gating") == "dialogue":
        dlg0 = comp.dialogue_loudness(x, sr, segs)
        if dlg0 is None:
            raise ValueError("Geen spraak gevonden om dialogue-gated op te sturen; "
                             "gebruik een integrated spec (bv. ebu-r128).")
        gain = loud["target"] - dlg0
        steps.append({"type": "gain", "gain_db": round(gain, 2)})
        steps.append({"type": "limiter", "ceiling_db": round(tp_max - 0.3, 2)})
        rationale.append(f"Dialogue-gated loudness {dlg0:.1f} -> {loud['target']} LUFS "
                         f"({gain:+.1f} dB), true-peak-limiter op {tp_max - 0.3:.1f} dBTP.")
    elif loud:
        steps.append({"type": "loudness_normalize", "target_lufs": loud["target"],
                      "true_peak_db": tp_max})
        rationale.append(f"Loudness naar {loud['target']} LUFS, true peak <= {tp_max} dBTP.")
    elif "rms_range" in spec_def:  # ACX: stuur op RMS-midden met piekbewaking
        lo, hi = spec_def["rms_range"]
        gain = (lo + hi) / 2.0 - (m0.get("rms_db") or -20.0)
        steps.append({"type": "gain", "gain_db": round(gain, 2)})
        steps.append({"type": "limiter",
                      "ceiling_db": spec_def.get("sample_peak_max", -3.0) - 0.2})
        rationale.append(f"RMS naar {(lo + hi) / 2:.0f} dB ({gain:+.1f} dB), "
                         "piekbewaking voor de ACX-piekeis.")

    y, resolved = chain.run_chain(x, sr, steps)
    m1 = analysis.analyze(y, sr)
    dlg1 = (comp.dialogue_loudness(y, sr, classify_segments(y, sr))
            if spec_def.get("gating") == "dialogue" else None)

    # levering: specs met een formaateis (Netflix: 48 kHz/24-bit) vullen de
    # exportdefaults zelf in; de eindcheck keurt het échte leveringsbestand
    fmt = spec_def.get("format") or {}
    export = None
    export_info = None
    if out_path:
        if sample_rate is None:
            sample_rate = fmt.get("sample_rate")
        if bit_depth is None:
            bit_depth = fmt.get("min_bit_depth")
        y_out = y
        if fmt.get("channels") == 2 and y_out.shape[0] == 1:
            # mono-bron voor een 2.0-levering: dual-mono is de standaardpraktijk
            y_out = np.repeat(y_out, 2, axis=0)
            rationale.append("Mono-bron als dual-mono stereo geëxporteerd "
                             "(2.0-leveringseis).")
        y_out, sr_out = (io.resample(y_out, sr, sample_rate)
                         if sample_rate else (y_out, sr))
        out = Path(out_path).expanduser()
        subtype = io.BIT_DEPTH_SUBTYPES.get(bit_depth or 24, "PCM_24")
        if out.suffix.lower() in ("", ".wav"):
            export = io.save_wav(out if out.suffix else out.with_suffix(".wav"),
                                 y_out, sr_out, subtype=subtype)
        else:
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "master.wav"
                io.save_wav(tmp, y_out, sr_out, subtype=subtype)
                export = io.encode_wav_to(tmp, out)
        export_info = io.probe(export)
    elif fmt:
        # geen leveringsbestand: keur de sessie-wav (PCM_24 op bron-sample-rate)
        export_info = {"sample_rate": sr, "subtype": "PCM_24", "bit_depth": 24,
                       "codec": "pcm_s24le", "channels": int(x.shape[0])}

    report_after = comp.check(m1, spec, dialogue_lufs=dlg1, file_info=export_info)
    rationale.append("Eindcontrole: " + ("GESLAAGD voor " if report_after["passed"]
                     else "nog NIET geslaagd voor ") + spec_def["name"] +
                     ("" if report_after["passed"] else
                      f" (open: {', '.join(report_after['failed_checks'])})"))
    fmt_checks = {"Sample rate", "Kanalen (formaat)", "Leveringsformaat"}
    if fmt and not out_path and not report_after["passed"] \
            and fmt_checks & set(report_after["failed_checks"]):
        rationale.append("Tip: geef out_path op — de export krijgt dan automatisch "
                         f"het spec-formaat ({fmt.get('sample_rate')} Hz / "
                         f"{fmt.get('min_bit_depth')}-bit"
                         + (", mono wordt dual-mono" if fmt.get("channels") == 2
                            else "") + ").")

    session = sessions.create_session(
        file_path, x, sr, m0, y, m1, resolved, rationale, None,
        label=f"{Path(file_path).name} — master {spec}",
        user_request=user_request or None,
        timeline={"segments": segs})
    d = sessions.session_path(session["session_id"])
    import json as _json

    (d / "compliance.json").write_text(
        _json.dumps(report_after, indent=2, ensure_ascii=False))

    result = {
        "session_id": session["session_id"],
        "output_path": str(d / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "compliance": report_after,
        "rationale": rationale,
        "deltas": session["deltas"],
    }
    if export is not None:
        result["export"] = {"path": str(export),
                            "sample_rate": (export_info or {}).get("sample_rate"),
                            "bit_depth": (export_info or {}).get("bit_depth")
                            or bit_depth or 24}
    return result


@mcp.tool()
def list_recipes() -> dict:
    """Toon alle beschikbare recepten: bewaarde bewerkingsketens om te hergebruiken.

    Recepten zijn losse JSON-bestanden: ingebouwde presets (gedestilleerd uit
    echte sessies) plus eigen recepten in ~/AudioImprove/recipes/. Delen kan
    door het bestand door te geven; apply_recipe accepteert ook een pad naar
    een recept-JSON van iemand anders.
    """
    from chat_with_audio import recipes as recipes_mod

    items = recipes_mod.list_recipes()
    return {"count": len(items), "recipes": items,
            "recipes_dir": str(recipes_mod.recipes_dir()),
            "hint": "apply_recipe past een recept toe op een bestand; save_recipe "
                    "bewaart de keten van een geslaagde sessie als nieuw recept."}


@mcp.tool()
def save_recipe(name: str, session_id: str | None = None,
                steps: list[dict] | None = None, description: str = "") -> dict:
    """Bewaar een bewerkingsketen als herbruikbaar recept ("bewaar dit als preset").

    Bron: session_id (neemt de uitgevoerde keten van die sessie over — de
    natuurlijke route na "dit klinkt goed") of een expliciete steps-lijst
    (zelfde formaat als apply_chain). Het recept wordt een JSON-bestand in
    ~/AudioImprove/recipes/ dat je kunt delen; toepassen gaat met apply_recipe.
    """
    from chat_with_audio import recipes as recipes_mod

    if session_id:
        data = sessions.load_session(session_id)
        chain_steps = (data.get("chain") or {}).get("steps") or []
        if not chain_steps:
            raise ValueError(f"Sessie {session_id} bevat geen bewerkingsketen "
                             "(alleen analyse).")
        if any(s.get("type") == "region" for s in chain_steps):
            raise ValueError("Deze sessie is een chirurgische regio-bewerking; die "
                             "is aan dít bestand gebonden en niet als recept "
                             "herbruikbaar. smart_edit vindt de regio's per bestand "
                             "opnieuw.")
        steps = chain_steps
        if not description:
            rat = (data.get("chain") or {}).get("rationale") or []
            description = rat[0] if rat else ""
    if not steps:
        raise ValueError("Geef session_id of steps op.")
    rec = recipes_mod.save_recipe(name, steps, description=description,
                                  source_session=session_id)
    return {"saved": rec,
            "hint": f"Toepassen: apply_recipe(file_path, recipe='{rec['name']}'). "
                    f"Delen: geef {rec['path']} door."}


@mcp.tool()
def apply_recipe(file_path: str, recipe: str, out_path: str | None = None,
                 user_request: str = "") -> dict:
    """Pas een bewaard recept toe ("doe dit bestand zoals mijn podcast-preset").

    recipe: een naam uit list_recipes of een pad naar een recept-JSON (bv.
    gedeeld door iemand anders). De stappen worden gevalideerd voordat er iets
    wordt uitgevoerd; het resultaat is een gewone A/B-sessie in de viewer.
    """
    from chat_with_audio import recipes as recipes_mod

    rec = recipes_mod.load_recipe(recipe)
    rationale = [f"Recept '{rec['name']}' toegepast"
                 + (f": {rec['description']}" if rec.get("description") else ".")]
    return _process(file_path, rec["steps"], rationale, out_path=out_path,
                    user_request=user_request or None)


@mcp.tool()
def optimize_audio(file_path: str, speech_peak_db: float = -6.0, music_gap_db: float = 2.0,
                   max_iterations: int = 4, denoise: str = "auto",
                   judge_model: str = "small", out_path: str | None = None) -> dict:
    """Nachtrun-optimalisatie: draai meerdere pijplijnvarianten en laat de beste winnen.

    Elke variant (EQ-, leveler-, compressor- en dereverb-combinaties) doorloopt
    de volledige verfijnlus en wordt objectief gescoord: Whisper-woordretentie en
    -zekerheid (verstaanbaarheid) plus afwijking van de spraakpiek-/balansdoelen.
    Traag maar grondig; het rapport bevat de volledige ranglijst zodat je in de
    chat kunt zien waarom de winnaar won en gericht kunt bijsturen.
    """
    from chat_with_audio import optimize as optimize_mod

    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    y, info = optimize_mod.optimize(x, sr, speech_peak_db=speech_peak_db,
                                    music_gap_db=music_gap_db,
                                    max_iterations=max_iterations, denoise=denoise,
                                    judge_model=judge_model)
    m1 = analysis.analyze(y, sr)
    rep = info["report"]
    rationale = [f"Optimalisatie over {len(rep['ranking'])} varianten; winnaar: "
                 f"'{rep['winner']}'."]
    for r in rep["ranking"][:5]:
        asr_txt = (f", retentie {r['asr']['word_retention']:.0%}" if r.get("asr") else "")
        rationale.append(f"  {r['name']}: score {r['score']}{asr_txt}")
    rationale.extend(rep["refine_report"].get("decisions", []))
    session = sessions.create_session(file_path, x, sr, m0, y, m1, info["steps"],
                                      rationale, "speech",
                                      label=f"{Path(file_path).name} — geoptimaliseerd "
                                            f"({rep['winner']})")
    result = {
        "session_id": session["session_id"],
        "output_path": str(sessions.session_path(session["session_id"]) / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "report": rep,
        "rationale": rationale,
        "deltas": session["deltas"],
    }
    if out_path:
        wav = sessions.session_path(session["session_id"]) / "processed.wav"
        result["export_path"] = str(io.encode_wav_to(wav, out_path))
    return result


@mcp.tool()
def transcribe_audio(file_path: str, model_size: str = "small", language: str = "nl",
                     start_s: float | None = None, end_s: float | None = None) -> dict:
    """Transcribeer (een deel van) een audiobestand met Whisper.

    Handig als verstaanbaarheidscheck: transcribeer origineel en bewerking en
    vergelijk. Vereist het [asr]-extra (uv sync --all-extras). model_size:
    tiny|base|small|medium (groter = beter maar trager).
    """
    from chat_with_audio import asr

    if not asr.is_available():
        raise RuntimeError(asr.INSTALL_HINT)
    x, sr = io.load_audio(file_path)
    a = int((start_s or 0) * sr)
    b = int(end_s * sr) if end_s else x.shape[1]
    return asr.transcribe(x[:, a:b], sr, model_size=model_size, language=language)


@mcp.tool()
def view_audio(session_id: str | None = None, file_path: str | None = None):
    """Render het perceptuele vergelijkingspaneel als afbeelding, zodat de AI kan
    ZIEN wat mensen zullen horen.

    Paneel-indeling: (1) spectrogram A op gehoorschaal (log-frequentie 60 Hz -
    16 kHz, laag onderaan), (2) spectrogram B, (3) verschilkaart B-A waarin
    ROOD = door de bewerking toegevoegd en BLAUW = weggehaald (bereik +/-18 dB),
    (4) levelcurves (A grijs, B blauw). Kijk naar: verdwenen medeklinker-
    transienten (verticale strepen), aangetaste harmonischen (horizontale
    lijnen), galmstaarten na woorden, en of de blauwe curve rustiger loopt.
    Geef session_id (vergelijkt origineel vs bewerking) of file_path (alleen
    analyse van een bestand).
    """
    from mcp.server.fastmcp import Image as MCPImage

    from chat_with_audio import visuals

    if session_id:
        d = sessions.session_path(session_id)
        xo, sr = io.load_audio(d / "original.wav")
        xp = None
        if (d / "processed.wav").exists():
            xp, _ = io.load_audio(d / "processed.wav")
        png = visuals.perceptual_panel(xo, sr, xp)
        (d / "perceptual_panel.png").write_bytes(png)
    elif file_path:
        x, sr = io.load_audio(file_path)
        png = visuals.perceptual_panel(x, sr, None)
    else:
        raise ValueError("Geef session_id of file_path op.")
    return MCPImage(data=png, format="png")


@mcp.tool()
def rate_audio(label: str, session_id: str | None = None, file_path: str | None = None,
               note: str = "") -> dict:
    """Train het smaakmodel: label audio als 'good' of 'bad'.

    Bij een session_id wordt de bewerkte versie gelabeld (dat is wat je hoorde);
    bij een file_path het bestand zelf. Vanaf 2 voorbeelden per klasse scoort
    analyze_audio nieuwe audio automatisch tegen jouw smaak (taste_score 0-100,
    met de grootste afwijkingen als aanknopingspunt voor verbeteringen).
    """
    from chat_with_audio import training

    if session_id:
        d = sessions.session_path(session_id)
        target = d / "processed.wav" if (d / "processed.wav").exists() else d / "original.wav"
        source = f"sessie {session_id} ({target.name})"
    elif file_path:
        target, source = Path(file_path).expanduser(), file_path
    else:
        raise ValueError("Geef session_id of file_path op.")
    x, sr = io.load_audio(target)
    m = analysis.analyze(x, sr)
    c = training.add_example(m, label, str(source), note)
    ready = c["good"] >= training.MIN_PER_CLASS and c["bad"] >= training.MIN_PER_CLASS
    return {"counts": c, "model_active": ready,
            "hint": "Smaakmodel actief: analyze_audio geeft nu een taste_score."
                    if ready else
                    f"Nog {max(0, 2 - c['good'])}x good en {max(0, 2 - c['bad'])}x bad "
                    "nodig om het model te activeren."}


@mcp.tool()
def export_to_audition(session_id: str | None = None, file_path: str | None = None,
                       source: str = "original", include_mix: bool = True,
                       open_app: bool = True) -> dict:
    """Splits audio in stems en zet ze klaar als Adobe Audition-multitracksessie.

    Maakt vocals/drums/bass/other-wav's plus een .sesx-sessiebestand (elke stem
    op een eigen spoor) en opent die in Audition. source: original|processed
    (bij een session_id). De losse wav's staan er altijd naast voor handmatig
    importeren. Vereist het [stems]-extra.
    """
    from chat_with_audio import audition
    from chat_with_audio.dsp import stems as stems_mod

    if not stems_mod.is_available():
        raise RuntimeError(stems_mod.INSTALL_HINT)
    if session_id:
        d = sessions.session_path(session_id)
        wav = d / ("processed.wav" if source == "processed"
                   and (d / "processed.wav").exists() else "original.wav")
        name, out_dir = session_id, d / "audition"
    elif file_path:
        wav = Path(file_path).expanduser()
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", wav.stem)
        out_dir = sessions.sessions_dir() / f"audition-{name}"
    else:
        raise ValueError("Geef session_id of file_path op.")

    x, sr = io.load_audio(wav)
    parts = stems_mod.separate(x, sr)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = []
    for tname, y in parts.items():
        p = io.save_wav(out_dir / f"{tname}.wav", y, sr)
        tracks.append((tname, p, int(y.shape[1])))
    if include_mix:
        p = io.save_wav(out_dir / "mix.wav", x, sr)
        tracks.append(("mix", p, int(x.shape[1])))
    sesx = audition.write_sesx(out_dir, name[:48], tracks, sr)

    opened = False
    if open_app:
        opened = audition.open_in_audition([sesx])
    return {"sesx": str(sesx), "tracks": {t: str(p) for t, p, _ in tracks},
            "opened_in_audition": opened,
            "hint": None if opened else
                    ("Audition niet gevonden of open_app=False; open het "
                     ".sesx-bestand handmatig of sleep de wav's in een sessie.")}


@mcp.tool()
def spectral_repair(file_path: str, start_s: float, end_s: float,
                    low_hz: float | None = None, high_hz: float | None = None,
                    out_path: str | None = None, user_request: str = "") -> dict:
    """Spectrale reparatie ('painting'): poets een kuch, stoelpiep, tik of
    bons weg door die tijd-frequentieplek opnieuw te schilderen vanuit de
    omgeving.

    Geef het tijdvak (start_s-end_s, max 5 s) en optioneel de frequentieband
    (low_hz-high_hz; weglaten = volledige band). De magnitudes in de patch
    worden per bin geinterpoleerd tussen de context links en rechts; buiten
    de patch blijft alles bit-voor-bit onaangetast. Vind de plek met
    view_audio (verticale strepen/vlekken in het spectrogram) of op het
    gehoor via de viewer. Voor schade óver of naast programma — niet om
    verloren woorden terug te toveren (dat zegt het rapport er eerlijk bij).
    """
    from chat_with_audio.dsp.spectral_repair import spectral_repair as _repair
    from chat_with_audio.regions import fmt_ts

    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    y = _repair(x, sr, start_s, end_s, low_hz=low_hz, high_hz=high_hz)
    m1 = analysis.analyze(y, sr)
    band = (f"{low_hz or 0:.0f}-{high_hz:.0f} Hz" if high_hz
            else ("volledige band" if not low_hz else f"vanaf {low_hz:.0f} Hz"))
    rationale = [f"Spectrale reparatie {fmt_ts(start_s)}-{fmt_ts(end_s)} ({band}): "
                 "magnitudes per bin geinterpoleerd uit de context links en rechts; "
                 "alles buiten de patch bit-voor-bit onaangetast.",
                 "Kanttekening: painting over doorlopende spraak vervaagt de spraak "
                 "zelf; dit gereedschap is voor schade over/naast het programma."]
    session = sessions.create_session(
        file_path, x, sr, m0, y, m1,
        [{"type": "spectral_repair", "start_s": start_s, "end_s": end_s,
          "low_hz": low_hz, "high_hz": high_hz}],
        rationale, None, label=f"{Path(file_path).name} — spectral repair",
        user_request=user_request or None,
        timeline={"segments": [],
                  "regions": [{"kind": "repair", "label": f"painting {band}",
                               "start_s": round(float(start_s), 2),
                               "end_s": round(float(end_s), 2)}]})
    result = {
        "session_id": session["session_id"],
        "output_path": str(sessions.session_path(session["session_id"]) / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "rationale": rationale,
        "deltas": session["deltas"],
        "hint": "Beluister de R-knop (residu) in de viewer: dat is exact wat er "
                "is weggepoetst.",
    }
    if out_path:
        wav = sessions.session_path(session["session_id"]) / "processed.wav"
        result["export_path"] = str(io.encode_wav_to(wav, out_path))
    return result


@mcp.tool()
def fill_room_tone(file_path: str, out_path: str | None = None,
                   user_request: str = "") -> dict:
    """Vul digitale gaten met de room tone van de opname zelf (dialoogbewerking).

    Een dropout, edit-gat of ADR-las valt op doordat de 'lucht' van de opname
    wegvalt. Deze tool bemonstert de rustigste echte ambience van het bestand
    en vult exacte-stilte-gaten met geshuffelde, overlappende stukjes daarvan —
    het klinkt als doorlopende ruimte, nooit als loop. Natuurlijke stilte
    (die al room tone bevat) en alles buiten de gaten blijven bit-voor-bit
    onaangetast. Zonder gaten of zonder bruikbare ambience legt het resultaat
    uit waarom er niets is gedaan.
    """
    from chat_with_audio.dsp.roomtone import fill_room_tone as _fill
    from chat_with_audio.regions import fmt_ts

    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    y, info = _fill(x, sr)
    if not info["filled"]:
        return {"filled": [], "message": f"Niets gevuld: {info['reason']}."}
    m1 = analysis.analyze(y, sr)
    spans = ", ".join(fmt_ts(f["start_s"]) for f in info["filled"][:6])
    rationale = [f"{len(info['filled'])} digitale gat(en) gevuld met de eigen room "
                 f"tone (donor: {info['donor']['start_s']}-{info['donor']['end_s']} s); "
                 f"posities: {spans}.",
                 "Alles buiten de gaten is bit-voor-bit onaangetast."]
    session = sessions.create_session(
        file_path, x, sr, m0, y, m1,
        [{"type": "room_tone_fill", "holes": info["filled"],
          "donor": info["donor"]}],
        rationale, None, label=f"{Path(file_path).name} — room tone",
        user_request=user_request or None)
    result = {
        "session_id": session["session_id"],
        "output_path": str(sessions.session_path(session["session_id"]) / "processed.wav"),
        "viewer_url": _viewer_url(session["session_id"]),
        "filled": info["filled"],
        "donor": info["donor"],
        "rationale": rationale,
        "deltas": session["deltas"],
    }
    if out_path:
        wav = sessions.session_path(session["session_id"]) / "processed.wav"
        result["export_path"] = str(io.encode_wav_to(wav, out_path))
    return result


@mcp.tool()
def export_markers(session_id: str, out_dir: str | None = None,
                   include_segments: bool = False) -> dict:
    """Exporteer de AI-regiokaart van een sessie als DAW-markers.

    Schrijft drie formaten: Adobe Audition marker-CSV (importeren via het
    Markers-paneel), een Audacity label track (start/end/label, importeert in
    veel tools) en markers.json. Zo landt wat de AI vond — netbrom hier, ruis
    daar, clipping daar — als navigeerbare markers in je editor.
    include_segments neemt ook de spraak/muziek/stilte-segmenten mee.
    Regiokaarten ontstaan bij smart_edit; andere sessies hebben alleen
    segmenten.
    """
    from chat_with_audio import markers as markers_mod

    data = sessions.load_session(session_id)
    timeline = data.get("timeline")
    if not timeline:
        raise ValueError(f"Sessie {session_id} heeft geen tijdlijndata "
                         "(oudere sessie?). Draai de bewerking opnieuw.")
    d = Path(out_dir).expanduser() if out_dir else (
        sessions.session_path(session_id) / "markers")
    result = markers_mod.write_markers(timeline, d, include_segments=include_segments)
    result["hint"] = ("Audition: Markers-paneel > import; Audacity: "
                      "File > Import > Labels.")
    return result


@mcp.tool()
def qc_report(file_path: str, spec: str | None = None,
              out_path: str | None = None) -> dict:
    """Genereer één leesbaar QC-rapport (markdown) voor een bestand.

    De sheet die een facility wil zien vóór acceptatie: bestandsgegevens,
    loudness-metingen (integrated/short-term/momentary, true peak, PLR),
    technische QC (stereo, dropouts, clipping, DC, kop/staart-stilte),
    bevindingen met ernst en suggestie, en — met spec — de aflever-check
    (zie check_compliance voor de spec-lijst). Het rapport wordt in de
    sessiemap gezet (qc_report.md) en optioneel naar out_path geschreven;
    de inhoud komt ook mee in het resultaat zodat je er direct over kunt
    doorpraten.
    """
    from chat_with_audio import compliance as comp
    from chat_with_audio import qcsheet

    x, sr = io.load_audio(file_path)
    m = analysis.analyze(x, sr)
    scores, issues = analysis.score_and_issues(m)
    container = io.probe(file_path)
    report = None
    if spec:
        dlg = None
        if comp.SPECS.get(spec, {}).get("gating") == "dialogue":
            from chat_with_audio.segments import classify_segments

            dlg = comp.dialogue_loudness(x, sr, classify_segments(x, sr))
        report = comp.check(m, spec, dialogue_lufs=dlg, file_info=container)
    sheet = qcsheet.build_qc_sheet(str(Path(file_path).expanduser()),
                                   container, m, scores, issues,
                                   compliance_report=report)
    session = sessions.create_session(file_path, x, sr, m,
                                      label=f"{Path(file_path).name} — QC")
    d = sessions.session_path(session["session_id"])
    (d / "qc_report.md").write_text(sheet)
    result = {
        "session_id": session["session_id"],
        "report_path": str(d / "qc_report.md"),
        "viewer_url": _viewer_url(session["session_id"]),
        "passed_compliance": report["passed"] if report else None,
        "issues": issues,
        "report_markdown": sheet,
    }
    if out_path:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(sheet)
        result["export_path"] = str(out)
    return result


@mcp.tool()
def qc_folder(dir_path: str, spec: str | None = None,
              out_path: str | None = None) -> dict:
    """Keur een hele map audio in één keer (inkomende leveringen, archieven).

    Per bestand: volledige analyse + bevindingen en optioneel de spec-check
    (zie check_compliance). Het resultaat is een samenvattingstabel per
    bestand — duur, loudness, true peak, score, bevindingen, verdict — plus
    de markdown-index (out_path schrijft die weg). Bestanden die niet laden
    komen als fout in de tabel in plaats van de batch te breken. Voor een
    diepe sheet per bestand: qc_report.
    """
    from chat_with_audio import compliance as comp

    d = Path(dir_path).expanduser()
    if not d.is_dir():
        raise ValueError(f"Geen map: {d}")
    files = sorted(p for p in d.iterdir()
                   if p.suffix.lower() in _AUDIO_EXTS and not p.name.startswith("."))
    if not files:
        raise ValueError(f"Geen audiobestanden gevonden in {d}")
    dialogue_gated = bool(spec) and comp.SPECS.get(spec, {}).get("gating") == "dialogue"

    rows: list[dict] = []
    for p in files:
        try:
            x, sr = io.load_audio(p)
            m = analysis.analyze(x, sr)
            scores, issues = analysis.score_and_issues(m)
            rep = None
            if spec:
                dlg = None
                if dialogue_gated:
                    from chat_with_audio.segments import classify_segments

                    dlg = comp.dialogue_loudness(x, sr, classify_segments(x, sr))
                rep = comp.check(m, spec, dialogue_lufs=dlg,
                                 file_info=io.probe(p))
            high = [i for i in issues if i["severity"] == "high"]
            rows.append({
                "file": p.name,
                "duration_s": m["duration_s"],
                "lufs": m.get("lufs_integrated"),
                "true_peak_dbtp": m.get("true_peak_dbtp"),
                "score": scores["overall"],
                "issues": len(issues),
                "high_issues": [i["code"] for i in high],
                "compliance_passed": rep["passed"] if rep else None,
                "failed_checks": rep["failed_checks"] if rep else None,
            })
        except Exception as exc:
            log.warning("qc_folder: %s faalde: %s", p.name, exc)
            rows.append({"file": p.name, "error": str(exc)})

    spec_name = comp.SPECS[spec]["name"] if spec else None
    lines = [f"# QC-index — {d.name}", "",
             f"_{len(rows)} bestand(en)"
             + (f", gekeurd tegen {spec_name}" if spec else "") + "._", "",
             "| Bestand | Duur | LUFS | TP (dBTP) | Score | Bevindingen | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['file']} | — | — | — | — | — | ⚠️ fout: {r['error'][:60]} |")
            continue
        if r["compliance_passed"] is True:
            verdict = "✅ geslaagd"
        elif r["compliance_passed"] is False:
            verdict = "❌ " + ", ".join(r["failed_checks"][:3])
        elif r["high_issues"]:
            verdict = "🔴 " + ", ".join(r["high_issues"][:3])
        else:
            verdict = "—"
        lines.append(f"| {r['file']} | {r['duration_s']} s | {r['lufs']} | "
                     f"{r['true_peak_dbtp']} | {r['score']} | {r['issues']} | {verdict} |")
    index_md = "\n".join(lines) + "\n"

    result = {"count": len(rows), "rows": rows, "spec": spec,
              "summary_markdown": index_md}
    if out_path:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(index_md)
        result["export_path"] = str(out)
    return result


@mcp.tool()
def open_viewer(session_id: str | None = None) -> dict:
    """Open de lokale A/B-viewer in de browser (start hem zo nodig).

    In de viewer kun je origineel en bewerking gesynchroniseerd vergelijken
    (A/B-knop of toets 'b'), met golfvormen, spectrogrammen en alle metrics.
    """
    if not _ensure_viewer():
        raise RuntimeError(f"Viewer wilde niet starten op poort {VIEWER_PORT}. "
                           f"Controleer of de poort vrij is (AIT_VIEWER_PORT om te wijzigen).")
    url = _viewer_url(session_id)
    webbrowser.open(url)
    return {"url": url, "status": "geopend in de browser"}


def main() -> None:
    log.info("chat-with-audio MCP-server gestart (dsp backend: %s)", dsp.backend())
    mcp.run()


if __name__ == "__main__":
    main()
