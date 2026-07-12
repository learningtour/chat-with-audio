"""QC-sheet: één leesbaar kwaliteitsrapport (markdown) per bestand.

Wat een facility op papier wil zien voordat een levering wordt geaccepteerd:
bestandsgegevens, loudness-metingen, technische QC (stereo, dropouts,
clipping), issues met ernst, en optioneel de compliance-check tegen een
aflever-spec. De sheet is bewust markdown: leesbaar in de terminal, in de
viewer, op GitHub en printbaar via elke converter.
"""

from __future__ import annotations

import time

_SEV = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def _row(label: str, value) -> str:
    return f"| {label} | {value if value is not None else '—'} |"


def build_qc_sheet(file_path: str, container: dict, metrics: dict, scores: dict,
                   issues: list[dict], compliance_report: dict | None = None) -> str:
    m = metrics
    lines = [
        f"# QC-rapport — {file_path.rsplit('/', 1)[-1]}",
        "",
        f"_Gegenereerd {time.strftime('%Y-%m-%d %H:%M:%S')} door Chat with Audio._",
        "",
        "## Bestand",
        "",
        "| | |",
        "|---|---|",
        _row("Pad", file_path),
        _row("Formaat", f"{container.get('format')} / {container.get('codec')}"),
        _row("Duur", f"{m.get('duration_s')} s"),
        _row("Sample rate", f"{m.get('sample_rate')} Hz"),
        _row("Kanalen", m.get("channels")),
        "",
        "## Loudness & niveaus",
        "",
        "| Meting | Waarde |",
        "|---|---|",
        _row("Integrated loudness", f"{m.get('lufs_integrated')} LUFS"),
        _row("Short-term max", f"{m.get('lufs_short_term_max')} LUFS"),
        _row("Momentary max", f"{m.get('lufs_momentary_max')} LUFS"),
        _row("Loudness range", f"{m.get('lra_db')} LU"),
        _row("True peak", f"{m.get('true_peak_dbtp')} dBTP"),
        _row("Sample peak", f"{m.get('sample_peak_db')} dBFS"),
        _row("PLR", f"{m.get('plr_db')} dB"),
        _row("RMS", f"{m.get('rms_db')} dB"),
        _row("Crest factor", f"{m.get('crest_factor_db')} dB"),
        "",
        "## Technische QC",
        "",
        "| Check | Waarde |",
        "|---|---|",
        _row("Ruisvloer", f"{m.get('noise_floor_db')} dB"),
        _row("SNR", f"{m.get('snr_db')} dB"),
        _row("Clipping", f"{m.get('clip_events')} events "
                          f"({m.get('clipped_samples')} samples)"),
        _row("Dropouts", (m.get("dropouts") or {}).get("count", 0)),
        _row("DC-offset", m.get("dc_offset")),
        _row("Stilte kop/staart", f"{m.get('lead_silence_s')} s / "
                                   f"{m.get('tail_silence_s')} s"),
    ]
    surround = m.get("surround")
    if surround:
        lv = surround.get("channel_levels_db") or {}
        lines += [
            _row("Layout", surround.get("layout")),
            _row("Kanaalniveaus", ", ".join(f"{k} {v} dB" for k, v in lv.items())),
            _row("Dode kanalen", ", ".join(surround.get("dead_channels") or []) or "geen"),
            _row("LFE", "stil" if surround.get("lfe_silent") else "actief"),
            _row("Downmix true peak (ITU)",
                 f"{surround.get('downmix_true_peak_dbtp')} dBTP"),
        ]
    stereo = m.get("stereo")
    if stereo:
        lines += [
            _row("Fasecorrelatie", stereo.get("correlation")),
            _row("Stereobalans (L/R)", f"{stereo.get('balance_db')} dB"),
            _row("Dood kanaal", stereo.get("dead_channel") or "nee"),
            _row("Dual-mono", "ja" if stereo.get("dual_mono") else "nee"),
            _row("Tegenfase", "JA" if stereo.get("polarity_inverted") else "nee"),
        ]
    hum = m.get("hum") or {}
    lines += [_row("Netbrom", f"{hum.get('freq'):.0f} Hz "
                              f"(+{hum.get('prominence_db')} dB)"
                   if hum.get("detected") else "niet gedetecteerd")]

    lines += ["", "## Scores (0-100)", "",
              "| Categorie | Score |", "|---|---|"]
    for k in ("loudness", "noise", "dynamics", "clarity", "overall"):
        lines.append(_row(k.capitalize(), scores.get(k)))

    lines += ["", "## Bevindingen", ""]
    if issues:
        for i in issues:
            lines.append(f"- {_SEV.get(i['severity'], '•')} **{i['code']}** — "
                         f"{i['message']} _Suggestie: {i['suggestion']}._")
    else:
        lines.append("- ✅ Geen bevindingen.")

    if compliance_report:
        rep = compliance_report
        verdict = "✅ GESLAAGD" if rep["passed"] else "❌ NIET GESLAAGD"
        lines += ["", f"## Aflever-check — {rep['spec_name']}", "",
                  f"**{verdict}**", "",
                  "| Criterium | Gemeten | Vereist | |",
                  "|---|---|---|---|"]
        for c in rep["checks"]:
            mark = "✓" if c["passed"] else ("△" if c["advisory"] else "✗")
            adv = " _(richtlijn)_" if c["advisory"] else ""
            measured = c["measured"] if c["measured"] is not None else "—"
            lines.append(f"| {c['name']}{adv} | {measured} "
                         f"| {c['requirement']} | {mark} |")
        if not rep["passed"]:
            lines += ["", f"Open punten: {', '.join(rep['failed_checks'])}."]

    return "\n".join(lines) + "\n"
