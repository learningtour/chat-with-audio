"""MCP-server (stdio) voor Claude Desktop en Claude Code.

Servernaam: audio-improve. Alle logging gaat naar stderr; stdout is
gereserveerd voor het MCP-protocol.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from audio_improve_toolkit import analysis, chain, improve, io, sessions
from audio_improve_toolkit import dsp

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

mcp = FastMCP("audio-improve")

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
    subprocess.Popen([sys.executable, "-m", "audio_improve_toolkit.viewer"], **kwargs)
    for _ in range(25):
        time.sleep(0.2)
        if _viewer_running():
            return True
    return False


def _process(file_path: str, steps: list[dict], rationale: list[str],
             profile: str | None = None, out_path: str | None = None) -> dict:
    """Gedeelde route: laden -> keten -> analyse voor/na -> sessie -> resultaat."""
    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    y, resolved = chain.run_chain(x, sr, steps)
    m1 = analysis.analyze(y, sr)
    session = sessions.create_session(file_path, x, sr, m0, y, m1, resolved,
                                      rationale, profile)
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
    if create_session:
        session = sessions.create_session(file_path, x, sr, m)
        result["session_id"] = session["session_id"]
        result["viewer_url"] = _viewer_url(session["session_id"])
    return result


@mcp.tool()
def improve_audio(file_path: str, profile: str = "auto", target_lufs: float | None = None,
                  denoise_method: str = "auto", out_path: str | None = None) -> dict:
    """Verbeter audio automatisch ("maak dit geluid beter").

    Analyseert het bestand en kiest zelf een keten (highpass, brom-notches,
    ruisonderdrukking, gate, EQ, compressie, loudness-normalisatie) met
    onderbouwing per stap. profile: auto|speech|music (auto detecteert).
    denoise_method: auto|spectral|ai (ai = DeepFilterNet, best voor spraak).
    out_path: optioneel exportpad; het formaat volgt de extensie (.mp3, .m4a,
    .flac, .ogg, .wav). Het resultaat staat altijd ook als wav in de sessie.
    """
    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    used_profile, steps, rationale = improve.build_improve_chain(
        m0, profile=profile, target_lufs=target_lufs, denoise_method=denoise_method)
    return _process(file_path, steps, rationale, profile=used_profile, out_path=out_path)


@mcp.tool()
def reduce_noise(file_path: str, strength_db: float = 12.0, method: str = "auto",
                 use_gate: bool = True, out_path: str | None = None) -> dict:
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
        from audio_improve_toolkit.dsp import ai_nr

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
    return _process(file_path, steps, rationale, profile=detected, out_path=out_path)


@mcp.tool()
def normalize_loudness(file_path: str, target_lufs: float = -16.0,
                       true_peak_db: float = -1.5, out_path: str | None = None) -> dict:
    """Trek het niveau over de hele breedte op (of omlaag) zonder te clippen.

    Meet de loudness (BS.1770) en brengt die naar target_lufs; een look-ahead
    true-peak-limiter bewaakt true_peak_db. Richtwaarden: spraak/podcast -16,
    muziek/streaming -14 LUFS.
    """
    steps = [{"type": "loudness_normalize", "target_lufs": target_lufs,
              "true_peak_db": true_peak_db}]
    rationale = [f"Loudness genormaliseerd naar {target_lufs} LUFS met "
                 f"true-peak-limiter op {true_peak_db} dBTP."]
    return _process(file_path, steps, rationale, out_path=out_path)


@mcp.tool()
def apply_chain(file_path: str, steps: list[dict], out_path: str | None = None) -> dict:
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
    return _process(file_path, steps, rationale, out_path=out_path)


@mcp.tool()
def refine_audio(file_path: str, speech_peak_db: float = -6.0, music_gap_db: float = 2.0,
                 max_iterations: int = 5, denoise: bool = True, tone: bool = True,
                 out_path: str | None = None) -> dict:
    """Iteratieve verfijning tot spraakniveau en spraak/muziek-balans exact kloppen.

    Segmenteert het bestand (spraak/muziek/stilte), ontruist per segment (AI op
    spraak, mild op muziek), en draait dan een meet-en-bijstuur-lus: leveler +
    compressor + loudness, net zo lang tot de spraakpieken op speech_peak_db
    (dBFS) zitten en de muziek music_gap_db daarboven. Het resultaat bevat de
    volledige meetgeschiedenis per iteratie ('report'), zodat je in de chat kunt
    beoordelen wat er gebeurde en gericht kunt bijsturen.
    """
    from audio_improve_toolkit import refine as refine_mod

    x, sr = io.load_audio(file_path)
    m0 = analysis.analyze(x, sr)
    y, info = refine_mod.refine(x, sr, speech_peak_db=speech_peak_db,
                                music_gap_db=music_gap_db,
                                max_iterations=max_iterations,
                                denoise=denoise, tone=tone)
    m1 = analysis.analyze(y, sr)
    rep = info["report"]
    rationale = [f"Iteratieve verfijning ({len(rep['iterations'])} iteraties, "
                 f"{'geconvergeerd' if rep['converged'] else 'maximum bereikt'}): "
                 f"doel spraakpiek {speech_peak_db} dBFS, muziek {music_gap_db:+.1f} dB daarbij."]
    for it in rep["iterations"]:
        e = it["errors"]
        rationale.append(f"Iteratie {it['iteration']}: spraakpiek-afwijking "
                         f"{e['speech_peak']:+.1f} dB, balans-afwijking {e['balance_gap']:+.1f} dB.")
    session = sessions.create_session(file_path, x, sr, m0, y, m1, info["steps"],
                                      rationale, "speech",
                                      label=f"{Path(file_path).name} — verfijnd")
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
    log.info("audio-improve MCP-server gestart (dsp backend: %s)", dsp.backend())
    mcp.run()


if __name__ == "__main__":
    main()
