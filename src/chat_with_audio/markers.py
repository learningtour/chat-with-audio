"""Markers-export: de AI-regiokaart van een sessie als DAW-markers.

Formaten:
  - Adobe Audition marker-CSV (tab-gescheiden, 'decimal' tijdformaat)
  - Audacity label track (start<TAB>end<TAB>label — importeert ook in veel
    andere tools)
  - markers.json (machine-leesbaar, zelfde inhoud)
De regio's (ingrepen) zijn de markers; met include_segments gaan ook de
spraak/muziek/stilte-segmenten mee.
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt_audition(t: float) -> str:
    m = int(t // 60)
    return f"{m}:{t - 60 * m:06.3f}"


def _marker_rows(timeline: dict, include_segments: bool) -> list[dict]:
    rows = []
    for r in timeline.get("regions") or []:
        rows.append({"start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                     "name": r.get("label") or r.get("kind", "regio"),
                     "kind": r.get("kind", "region"), "group": "ingreep"})
    if include_segments:
        for s in timeline.get("segments") or []:
            rows.append({"start_s": float(s["start_s"]), "end_s": float(s["end_s"]),
                         "name": s.get("kind", "segment"),
                         "kind": s.get("kind", "segment"), "group": "inhoud"})
    rows.sort(key=lambda r: r["start_s"])
    return rows


def write_markers(timeline: dict, out_dir: str | Path,
                  include_segments: bool = False) -> dict:
    """Schrijf alle marker-formaten naar out_dir; geeft paden + telling terug."""
    rows = _marker_rows(timeline, include_segments)
    if not rows:
        raise ValueError("Geen regio's in deze sessie om te exporteren "
                         "(alleen smart_edit-sessies hebben een regiokaart; "
                         "include_segments=True exporteert de inhoudssegmenten).")
    d = Path(out_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)

    audition = d / "audition_markers.csv"
    lines = ["Name\tStart\tDuration\tTime Format\tType\tDescription"]
    for r in rows:
        dur = max(0.0, r["end_s"] - r["start_s"])
        lines.append(f"{r['name']}\t{_fmt_audition(r['start_s'])}\t"
                     f"{_fmt_audition(dur)}\tdecimal\tCue\t{r['group']}: {r['kind']}")
    audition.write_text("\n".join(lines) + "\n")

    audacity = d / "audacity_labels.txt"
    audacity.write_text("".join(
        f"{r['start_s']:.6f}\t{r['end_s']:.6f}\t{r['name']}\n" for r in rows))

    as_json = d / "markers.json"
    as_json.write_text(json.dumps({"markers": rows}, indent=2, ensure_ascii=False))

    return {"count": len(rows),
            "audition_csv": str(audition),
            "audacity_labels": str(audacity),
            "json": str(as_json)}
