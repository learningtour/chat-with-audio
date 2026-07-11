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
