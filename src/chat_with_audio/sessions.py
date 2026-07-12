"""Sessiemodel: elke bewerking is een map onder ~/AudioImprove/sessions/.

De viewer en Claude lezen exact dezelfde bestanden, zodat je in de chat kunt
doorpraten over wat je in de viewer ziet en hoort.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from chat_with_audio import analysis, io, visuals


def sessions_dir() -> Path:
    root = os.environ.get("AIT_SESSIONS_DIR")
    d = Path(root).expanduser() if root else Path.home() / "AudioImprove" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", Path(name).stem.lower()).strip("-")
    return s[:32] or "audio"


def create_session(source_path: str | Path, x_original: np.ndarray, sr: int,
                   metrics_original: dict, x_processed: np.ndarray | None = None,
                   metrics_processed: dict | None = None, chain: list | None = None,
                   rationale: list[str] | None = None, profile: str | None = None,
                   label: str | None = None, user_request: str | None = None,
                   asr_report: dict | None = None,
                   timeline: dict | None = None) -> dict:
    """Schrijf een complete sessiemap; geeft session.json-inhoud terug.

    timeline: {"segments": [...], "regions": [...]} voor de tijdlijnbalk in de
    viewer; zonder opgave worden de spraak/muziek/stilte-segmenten zelf bepaald.
    """
    session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + _slug(str(source_path))
    d = sessions_dir() / session_id
    # twee bewerkingen van hetzelfde bestand binnen één seconde mogen elkaar
    # nooit overschrijven
    suffix = 1
    while d.exists():
        suffix += 1
        d = sessions_dir() / f"{session_id}-{suffix}"
    session_id = d.name
    d.mkdir(parents=True)

    if timeline is None:
        try:
            from chat_with_audio.segments import classify_segments

            timeline = {"segments": classify_segments(x_original, sr)}
        except Exception:  # tijdlijn is nice-to-have, nooit blokkerend
            timeline = None
    if timeline:
        tidy = {k: [{**item, "start_s": round(float(item["start_s"]), 2),
                     "end_s": round(float(item["end_s"]), 2)} for item in v]
                for k, v in timeline.items()}
        (d / "timeline.json").write_text(json.dumps(tidy, indent=2, ensure_ascii=False))

    io.save_wav(d / "original.wav", x_original, sr)
    visuals.waveform_json(x_original, sr, d / "waveform_original.json")
    visuals.spectrogram_png(x_original, sr, d / "spectrogram_original.png")
    scores_o, issues_o = analysis.score_and_issues(metrics_original)
    (d / "analysis_original.json").write_text(json.dumps(
        {"metrics": metrics_original, "scores": scores_o, "issues": issues_o},
        indent=2, ensure_ascii=False))

    deltas = None
    if x_processed is not None and metrics_processed is not None:
        io.save_wav(d / "processed.wav", x_processed, sr)
        visuals.waveform_json(x_processed, sr, d / "waveform_processed.json")
        visuals.spectrogram_png(x_processed, sr, d / "spectrogram_processed.png")
        scores_p, issues_p = analysis.score_and_issues(metrics_processed)
        (d / "analysis_processed.json").write_text(json.dumps(
            {"metrics": metrics_processed, "scores": scores_p, "issues": issues_p},
            indent=2, ensure_ascii=False))
        deltas = compute_deltas(metrics_original, metrics_processed)
        _write_residual(d, x_original, x_processed, sr,
                        metrics_original, metrics_processed)

    if chain is not None:
        (d / "chain.json").write_text(json.dumps(
            {"steps": chain, "rationale": rationale or []}, indent=2, ensure_ascii=False))

    session = {
        "session_id": session_id,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_path": str(Path(source_path).expanduser().resolve()),
        "label": label or Path(source_path).name,
        "profile": profile,
        "sample_rate": sr,
        "duration_s": round(x_original.shape[-1] / sr, 2),
        "has_processed": x_processed is not None,
        "deltas": deltas,
    }
    (d / "session.json").write_text(json.dumps(session, indent=2, ensure_ascii=False))
    try:
        from chat_with_audio.session_log import write_log

        write_log(d, session, x_original, sr, metrics_original,
                  x_processed=x_processed, metrics_processed=metrics_processed,
                  chain=chain, rationale=rationale, user_request=user_request,
                  asr_report=asr_report)
    except Exception:  # het logboek mag een sessie nooit laten falen
        import logging

        logging.getLogger(__name__).exception("logboek schrijven mislukt")
    return session


def _write_residual(d: Path, x_original: np.ndarray, x_processed: np.ndarray,
                    sr: int, m_orig: dict, m_proc: dict) -> None:
    """Residu = bewerking minus loudness-gematcht origineel: precies wat de
    keten heeft veranderd. Onmisbaar voor artefact-controle ('wat is er
    weggehaald of bijgemaakt?')."""
    lo, lp = m_orig.get("lufs_integrated"), m_proc.get("lufs_integrated")
    gain = 10.0 ** ((lp - lo) / 20.0) if (lo is not None and lp is not None) else 1.0
    xo = x_original[None, :] if x_original.ndim == 1 else x_original
    xp = x_processed[None, :] if x_processed.ndim == 1 else x_processed
    n = min(xo.shape[1], xp.shape[1])
    ch = min(xo.shape[0], xp.shape[0])
    residual = xp[:ch, :n] - xo[:ch, :n] * gain
    io.save_wav(d / "residual.wav", residual, sr)


_DELTA_KEYS = ("lufs_integrated", "true_peak_dbtp", "rms_db", "noise_floor_db",
               "snr_db", "crest_factor_db", "lra_db", "silence_pct")


def compute_deltas(before: dict, after: dict) -> dict:
    out = {}
    for k in _DELTA_KEYS:
        a, b = before.get(k), after.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[k] = round(b - a, 2)
    return out


def list_sessions() -> list[dict]:
    out = []
    for d in sorted(sessions_dir().iterdir(), reverse=True):
        f = d / "session.json"
        if f.exists():
            try:
                out.append(json.loads(f.read_text()))
            except Exception:
                continue
    return out


def search_sessions(query: str | None = None) -> list[dict]:
    """Sessies gefilterd op een zoekterm (in id, label, bronpad of verzoek)."""
    items = list_sessions()
    if not query:
        return items
    q = query.lower()
    hits = []
    for s in items:
        haystack = " ".join(str(s.get(k) or "") for k in
                            ("session_id", "label", "source_path",
                             "user_request", "profile")).lower()
        if q in haystack:
            hits.append(s)
    return hits


def dir_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def delete_session(session_id: str) -> int:
    """Verwijder een sessiemap; geeft het aantal vrijgekomen bytes terug."""
    import shutil

    d = sessions_dir() / session_id
    if "/" in session_id or ".." in session_id or not d.is_dir():
        raise FileNotFoundError(f"Sessie '{session_id}' niet gevonden.")
    size = dir_size(d)
    shutil.rmtree(d)
    return size


def load_session(session_id: str) -> dict:
    """Volledige sessiedata: session.json + analyses + chain."""
    d = sessions_dir() / session_id
    if not d.is_dir() or not (d / "session.json").exists():
        raise FileNotFoundError(f"Sessie '{session_id}' niet gevonden in {sessions_dir()}")
    data = json.loads((d / "session.json").read_text())
    for key, fname in (("original", "analysis_original.json"),
                       ("processed", "analysis_processed.json"),
                       ("chain", "chain.json"),
                       ("timeline", "timeline.json"),
                       ("compliance", "compliance.json")):
        f = d / fname
        if f.exists():
            data[key] = json.loads(f.read_text())
    return data


def session_path(session_id: str) -> Path:
    return sessions_dir() / session_id
