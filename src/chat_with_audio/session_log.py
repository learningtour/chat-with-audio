"""Sessielogboek: volledige provenance van invoer tot geverifieerd resultaat.

Elke sessie krijgt een leesbaar `log.md` en een gestructureerd `log.json`:
wat er binnenkwam (inclusief de samples als getallen), hoe er geanalyseerd is,
wat de gebruiker vroeg, welke ingrepen er waren en met welk bewijs het
eindresultaat is gecontroleerd.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_AANPAK = """\
- **Loudness**: BS.1770 (pyloudnorm) — K-gewogen, met gating; integrated over het
  hele bestand, short-term over 3 s-vensters. Eenheid LUFS.
- **True peak**: 4x oversampling (resample_poly) en dan de piek — vangt
  inter-sample-pieken die een gewone sample-piek mist. Eenheid dBTP.
- **Ruisvloer**: gemiddelde van het stilste deciel van 25 ms-frames (RMS, dB).
- **SNR**: 90e-percentiel frameniveau minus de ruisvloer.
- **Stilte-%**: frames dicht bij de ruisvloer, alleen geteld als signaal en vloer
  minstens 10 dB uit elkaar liggen.
- **Clipping**: samples >= 0.999 full scale, of flat-tops (identieke opeenvolgende
  samples nabij de piek) bij 32-bit float opnames met headroom boven 0 dBFS.
- **Netbrom**: Welch-spectrum, prominentie van 50/60 Hz en harmonischen t.o.v.
  de spectrale omgeving.
- **Spectrum**: energieverdeling laag/mid/hoog, zwaartepunt (centroid) en
  spectrale tilt (dB/octaaf, fit 100 Hz - 10 kHz).
- **Verstaanbaarheid** (waar Whisper is ingezet): transcript van origineel en
  resultaat vergeleken; woordretentie = aandeel woorden van het origineel dat
  terugkomt in het resultaat.\
"""


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _metrics_table(m: dict) -> str:
    rows = []
    for k, v in m.items():
        if isinstance(v, dict):
            v = ", ".join(f"{kk}: {_fmt(vv)}" for kk, vv in v.items())
        elif isinstance(v, list):
            v = "; ".join(str(i) for i in v) or "-"
        rows.append(f"| {k} | {_fmt(v)} |")
    return "\n".join(["| meting | waarde |", "|---|---|", *rows])


def _sample_excerpt(x: np.ndarray, sr: int) -> tuple[str, str, float]:
    mono = x.mean(axis=0)
    head = ", ".join(f"{v:+.5f}" for v in mono[:8])
    p = int(np.argmax(np.abs(mono)))
    a = max(0, p - 3)
    peak = ", ".join(f"{v:+.5f}" for v in mono[a:a + 8])
    return head, peak, p / sr


def write_log(d: Path, session: dict, x_original: np.ndarray, sr: int,
              metrics_original: dict, x_processed: np.ndarray | None = None,
              metrics_processed: dict | None = None, chain: list | None = None,
              rationale: list[str] | None = None, user_request: str | None = None,
              asr_report: dict | None = None) -> Path:
    """Schrijf log.md + log.json in sessiemap d; geeft het pad van log.md terug."""
    from chat_with_audio import analysis

    x2 = x_original[None, :] if x_original.ndim == 1 else x_original
    head, peak, peak_t = _sample_excerpt(x2, sr)
    scores_o, issues_o = analysis.score_and_issues(metrics_original)

    length_ok = None
    if x_processed is not None:
        xp = x_processed[None, :] if x_processed.ndim == 1 else x_processed
        length_ok = bool(xp.shape[1] == x2.shape[1])

    logj = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": {
            "source_path": session.get("source_path"),
            "sample_rate": sr,
            "channels": int(x2.shape[0]),
            "n_samples": int(x2.shape[1]),
            "duration_s": session.get("duration_s"),
        },
        "user_request": user_request,
        "analysis_before": {"metrics": metrics_original, "scores": scores_o,
                            "issues": issues_o},
        "interventions": {"steps": chain or [], "rationale": rationale or []},
        "analysis_after": ({"metrics": metrics_processed}
                           if metrics_processed else None),
        "deltas": session.get("deltas"),
        "verification": {"length_preserved": length_ok, "asr": asr_report},
    }
    (d / "log.json").write_text(json.dumps(logj, indent=2, ensure_ascii=False))

    md: list[str] = [f"# Logboek — {session.get('label', d.name)}",
                     f"_Sessie {session.get('session_id', d.name)}, "
                     f"aangemaakt {logj['created']}_", ""]

    md += ["## 1. Wat kwam er binnen",
           f"- Bronbestand: `{session.get('source_path')}`",
           f"- {x2.shape[0]} kana(a)l(en), {sr} Hz, {x2.shape[1]:,} samples "
           f"= {session.get('duration_s')} s", "",
           "### Audio als data",
           f"Digitale audio is een rij getallen: {sr}x per seconde de uitwijking "
           "van de luidsprekerconus, tussen -1.0 en +1.0 (32-bit float mag daar "
           "tijdelijk boven). Zo zien de eerste acht samples van dit bestand er "
           "letterlijk uit:", "", f"    [{head}]", "",
           f"En de acht samples rond het luidste punt (t={peak_t:.3f}s):", "",
           f"    [{peak}]", "",
           "Alle bewerkingen hieronder zijn wiskundige operaties op deze rij.", ""]

    if user_request:
        md += ["## 2. De vraag", f"> {user_request}", ""]
    else:
        md += ["## 2. De vraag", "_Niet meegegeven bij deze sessie._", ""]

    md += ["## 3. Hoe er geanalyseerd is", _AANPAK, ""]

    md += ["## 4. Bevindingen vooraf", _metrics_table(metrics_original), "",
           "Scores (0-100): " + ", ".join(f"{k} {v}" for k, v in scores_o.items()), ""]
    if issues_o:
        md += ["Gevonden issues:"]
        md += [f"- **{i['severity']}** [{i['code']}] {i['message']} "
               f"→ _{i['suggestion']}_" for i in issues_o]
        md += [""]

    md += ["## 5. Interventies"]
    if chain:
        for n, step in enumerate(chain, 1):
            s = dict(step)
            t = s.pop("type", "?")
            params = ", ".join(f"{k}={_fmt(v)}" for k, v in s.items()) or "standaard"
            md += [f"{n}. **{t}** — {params}"]
    else:
        md += ["_Alleen analyse, geen bewerking._"]
    if rationale:
        md += ["", "Onderbouwing:"]
        md += [f"- {r}" for r in rationale]
    md += [""]

    if metrics_processed:
        md += ["## 6. Eindanalyse", _metrics_table(metrics_processed), ""]
        if session.get("deltas"):
            md += ["Verschillen (na - voor):",
                   "", "| meting | delta |", "|---|---|"]
            md += [f"| {k} | {v:+g} |" for k, v in session["deltas"].items()]
            md += [""]

    md += ["## 7. Verificatie"]
    if length_ok is not None:
        md += [f"- Lengte behouden: {'ja' if length_ok else 'NEE'} "
               f"({x2.shape[1]:,} samples in = uit)"]
    if asr_report:
        md += [f"- Whisper-woordretentie: {asr_report.get('word_retention', '?'):.0%}"
               if isinstance(asr_report.get("word_retention"), float) else
               f"- Whisper: {asr_report}",
               f"- Transcript origineel: \"{asr_report.get('transcript_original', '')}\"",
               f"- Transcript resultaat: \"{asr_report.get('transcript_processed', '')}\""]
    if metrics_processed:
        md += [f"- True peak na bewerking: {metrics_processed.get('true_peak_dbtp')} dBTP",
               f"- Loudness na bewerking: {metrics_processed.get('lufs_integrated')} LUFS"]
    md += ["", "## 8. Bestanden in deze sessie"]
    md += [f"- `{f.name}`" for f in sorted(d.iterdir()) if f.is_file()]
    md += [""]

    out = d / "log.md"
    out.write_text("\n".join(md))
    return out


def write_log_for_existing(session_id: str, user_request: str | None = None,
                           asr_report: dict | None = None) -> Path:
    """Bouw het logboek achteraf voor een bestaande sessie (uit de opgeslagen data)."""
    from chat_with_audio import io, sessions

    d = sessions.session_path(session_id)
    data = sessions.load_session(session_id)
    x, sr = io.load_audio(d / "original.wav")
    xp = None
    if (d / "processed.wav").exists():
        xp, _ = io.load_audio(d / "processed.wav")
    return write_log(
        d, data, x, sr, data.get("original", {}).get("metrics", {}),
        x_processed=xp,
        metrics_processed=data.get("processed", {}).get("metrics"),
        chain=data.get("chain", {}).get("steps"),
        rationale=data.get("chain", {}).get("rationale"),
        user_request=user_request, asr_report=asr_report)
