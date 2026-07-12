"""Tekstgestuurd dialoogmonteren: van woordtijdstempels naar een montagelijst.

De planner werkt op een platte woordenlijst ({word, start, end, prob} — van
Whisper met word timestamps, of elke andere bron met dezelfde vorm) en levert
bewerkingen op: stopwoorden ("eh", "uhm") en verdubbelingen eruit, pauzes
inkorten tot een doel, tekstfragmenten verwijderen of bliepen/redigeren.

De renderer voert de lijst uit met raised-cosine-crossfades op elke lasnaad
zodat de knips onhoorbaar zijn; bliepen zijn lengte-neutraal (de tijdlijn
verschuift niet), knips maken het bestand korter. De motor is puur DSP en
volledig testbaar zonder model — tests voeden de planner met synthetische
woordenlijsten.
"""

from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

# Stopwoorden per taal: alléén klanken die nooit betekenis dragen. Echte
# woorden die ook als vulsel dienen ("nou", "like", "you know") zijn bewust
# weggelaten — die knipt de chat gericht via remove_text.
FILLERS = {
    "nl": {"eh", "uh", "ehm", "uhm", "euh", "euhm", "hm", "hmm", "mm", "mmm"},
    "en": {"uh", "um", "erm", "uhm", "em", "hm", "hmm", "mm", "mmm"},
    "de": {"äh", "ähm", "eh", "ehm", "hm", "hmm", "mm"},
    "fr": {"euh", "heu", "hein", "hm", "hmm", "mm"},
}

GUARD_S = 0.05         # marge rond een woordknip, begrensd door de buurwoorden
PAUSE_SLACK_S = 0.15   # pauzes net boven het doel blijven met rust
DOUBLE_MAX_GAP_S = 0.4  # verdubbeling telt alleen bij direct herstel


def _norm(word: str) -> str:
    """Normaliseer een Whisper-woord: spaties/interpunctie weg, kleine letters."""
    return re.sub(r"[^\w'\-]+", "", word.strip().lower())


def _context(words: list[dict], i0: int, i1: int, span: int = 3) -> str:
    """Transcriptcontext rond woorden i0..i1 (inclusief), met [..] om de knip."""
    pre = " ".join(w["word"].strip() for w in words[max(0, i0 - span):i0])
    mid = " ".join(w["word"].strip() for w in words[i0:i1 + 1])
    post = " ".join(w["word"].strip() for w in words[i1 + 1:i1 + 1 + span])
    return f"{pre} [{mid}] {post}".strip()


def _word_span(words: list[dict], i0: int, i1: int, dur: float,
               guard_s: float = GUARD_S) -> tuple[float, float]:
    """Knipvenster voor woorden i0..i1: guard eromheen, nooit in buurwoorden."""
    start = words[i0]["start"] - guard_s
    end = words[i1]["end"] + guard_s
    if i0 > 0:
        start = max(start, words[i0 - 1]["end"])
    else:
        start = max(start, 0.0)
    if i1 + 1 < len(words):
        end = min(end, words[i1 + 1]["start"])
    else:
        end = min(end, dur)
    return max(0.0, start), max(start, end)


# ---------------------------------------------------------------- planners

def plan_fillers(words: list[dict], dur: float, language: str = "nl") -> list[dict]:
    """Stopwoorden ('eh', 'uhm', ...) als delete-bewerkingen."""
    lexicon = FILLERS.get(language, FILLERS["nl"])
    edits = []
    for i, w in enumerate(words):
        if _norm(w["word"]) in lexicon:
            s, e = _word_span(words, i, i, dur)
            edits.append({"action": "delete", "start_s": s, "end_s": e,
                          "reason": "filler", "label": f"stopwoord '{w['word'].strip()}'",
                          "context": _context(words, i, i)})
    return edits


def plan_doubles(words: list[dict], dur: float) -> list[dict]:
    """Directe woordverdubbelingen ('ik ik ga'): de eerste instantie eruit,
    de uiteindelijke (meestal beste) uitvoering blijft staan."""
    edits = []
    for i in range(len(words) - 1):
        a, b = _norm(words[i]["word"]), _norm(words[i + 1]["word"])
        gap = words[i + 1]["start"] - words[i]["end"]
        if a and a == b and gap <= DOUBLE_MAX_GAP_S:
            s, e = _word_span(words, i, i, dur)
            edits.append({"action": "delete", "start_s": s, "end_s": e,
                          "reason": "double", "label": f"verdubbeling '{words[i]['word'].strip()}'",
                          "context": _context(words, i, i + 1)})
    return edits


def plan_pauses(words: list[dict], dur: float, max_pause_s: float = 0.6) -> list[dict]:
    """Pauzes tussen woorden inkorten tot max_pause_s: het midden van de pauze
    verdwijnt, kop en staart blijven — de ademruimte rond de woorden blijft echt."""
    edits = []
    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        if gap <= max_pause_s + PAUSE_SLACK_S:
            continue
        keep = max_pause_s / 2.0
        s = words[i]["end"] + keep
        e = words[i + 1]["start"] - keep
        edits.append({"action": "delete", "start_s": s, "end_s": e,
                      "reason": "pause",
                      "label": f"pauze {gap:.1f}s → {max_pause_s:.1f}s",
                      "context": _context(words, i, i + 1)})
    return edits


def find_phrase(words: list[dict], phrase: str) -> list[tuple[int, int]]:
    """Alle voorkomens van een frase als (i0, i1)-woordindexbereiken.
    Matching is interpunctie- en hoofdletterongevoelig."""
    tokens = [_norm(t) for t in phrase.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return []
    normed = [_norm(w["word"]) for w in words]
    hits = []
    for i in range(len(normed) - len(tokens) + 1):
        if normed[i:i + len(tokens)] == tokens:
            hits.append((i, i + len(tokens) - 1))
    return hits


def plan_text_edits(words: list[dict], dur: float, phrases: list[str],
                    action: str = "delete") -> tuple[list[dict], list[str]]:
    """Frases verwijderen of bliepen; geeft (bewerkingen, niet-gevonden frases)."""
    edits, missing = [], []
    for phrase in phrases:
        hits = find_phrase(words, phrase)
        if not hits:
            missing.append(phrase)
            continue
        for i0, i1 in hits:
            guard = GUARD_S if action == "delete" else 0.02
            s, e = _word_span(words, i0, i1, dur, guard_s=guard)
            edits.append({"action": action, "start_s": s, "end_s": e,
                          "reason": "text",
                          "label": f"{'knip' if action == 'delete' else 'bliep'} "
                                   f"'{phrase}'",
                          "context": _context(words, i0, i1)})
    return edits, missing


def merge_edits(edits: list[dict]) -> list[dict]:
    """Sorteer en ontdubbel: overlappende deletes versmelten; een bliep die
    binnen een delete valt vervalt (weg is al geredigeerd)."""
    deletes = sorted((e for e in edits if e["action"] == "delete"),
                     key=lambda e: e["start_s"])
    merged: list[dict] = []
    for e in deletes:
        if merged and e["start_s"] <= merged[-1]["end_s"]:
            if e["end_s"] > merged[-1]["end_s"]:
                merged[-1] = {**merged[-1], "end_s": e["end_s"],
                              "label": f"{merged[-1]['label']} + {e['label']}"}
            continue
        merged.append(dict(e))
    bleeps = sorted((e for e in edits if e["action"] == "bleep"),
                    key=lambda e: e["start_s"])
    kept_bleeps: list[dict] = []
    for b in bleeps:
        inside_delete = any(d["start_s"] <= b["start_s"] and b["end_s"] <= d["end_s"]
                            for d in merged)
        overlaps_prev = kept_bleeps and b["start_s"] < kept_bleeps[-1]["end_s"]
        if inside_delete:
            continue
        if overlaps_prev:
            kept_bleeps[-1]["end_s"] = max(kept_bleeps[-1]["end_s"], b["end_s"])
            continue
        kept_bleeps.append(dict(b))
    return sorted(merged + kept_bleeps, key=lambda e: e["start_s"])


def plan_edits(words: list[dict], dur: float, language: str = "nl",
               remove_fillers: bool = True, remove_doubles: bool = True,
               tighten_pauses_to_s: float | None = None,
               remove_text: list[str] | None = None,
               bleep_text: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Volledige montagelijst; geeft (bewerkingen, niet-gevonden frases)."""
    edits: list[dict] = []
    missing: list[str] = []
    if remove_fillers:
        edits += plan_fillers(words, dur, language)
    if remove_doubles:
        edits += plan_doubles(words, dur)
    if tighten_pauses_to_s is not None:
        edits += plan_pauses(words, dur, tighten_pauses_to_s)
    if remove_text:
        found, miss = plan_text_edits(words, dur, remove_text, action="delete")
        edits += found
        missing += miss
    if bleep_text:
        found, miss = plan_text_edits(words, dur, bleep_text, action="bleep")
        edits += found
        missing += miss
    return merge_edits(edits), missing


# ---------------------------------------------------------------- renderer

def _raised_cosine(n: int) -> np.ndarray:
    """Equal-power fade-in (0→1); 1 - curve is de bijbehorende fade-out."""
    return np.sin(0.5 * np.pi * np.linspace(0.0, 1.0, n, dtype=np.float64)) ** 2


def _render_bleep(x2: np.ndarray, sr: int, i0: int, i1: int,
                  style: str = "tone") -> None:
    """Vervang [i0:i1) in-place door een bliep (toon) of stilte, met korte
    raised-cosine-randen zodat de overgang niet klikt. Lengte-neutraal."""
    n = i1 - i0
    if n <= 0:
        return
    if style == "mute":
        repl = np.zeros((x2.shape[0], n), dtype=np.float64)
    else:
        seg = x2[:, i0:i1]
        rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
        rms = min(max(rms, 10 ** (-40 / 20)), 10 ** (-14 / 20))
        t = np.arange(n) / sr
        tone = (rms * np.sqrt(2.0)) * np.sin(2 * np.pi * 1000.0 * t)
        repl = np.tile(tone, (x2.shape[0], 1))
    edge = min(int(0.005 * sr), n // 2)
    if edge > 0:
        fade = _raised_cosine(edge)
        repl[:, :edge] = repl[:, :edge] * fade + x2[:, i0:i0 + edge] * (1 - fade)
        repl[:, -edge:] = repl[:, -edge:] * (1 - fade) + x2[:, i1 - edge:i1] * fade
    x2[:, i0:i1] = repl.astype(x2.dtype)


def apply_edits(x: np.ndarray, sr: int, edits: list[dict],
                crossfade_ms: float = 12.0,
                bleep_style: str = "tone") -> tuple[np.ndarray, dict]:
    """Voer de montagelijst uit. Bliepen eerst (lengte-neutraal, op de
    oorspronkelijke tijdlijn), dan de knips met een equal-power crossfade op
    elke lasnaad. Geeft (audio, verslag met per knip de verwijderde tijd)."""
    x2 = (x[None, :] if x.ndim == 1 else x).astype(np.float32).copy()
    n = x2.shape[1]
    xf = max(2, int(sr * crossfade_ms / 1000.0))

    applied: list[dict] = []
    for e in edits:
        if e["action"] != "bleep":
            continue
        i0 = max(0, min(n, int(round(e["start_s"] * sr))))
        i1 = max(i0, min(n, int(round(e["end_s"] * sr))))
        _render_bleep(x2, sr, i0, i1, style=bleep_style)
        applied.append({**e, "start_s": round(i0 / sr, 3), "end_s": round(i1 / sr, 3)})

    cuts = []
    for e in edits:
        if e["action"] != "delete":
            continue
        i0 = max(0, min(n, int(round(e["start_s"] * sr))))
        i1 = max(i0, min(n, int(round(e["end_s"] * sr))))
        if i1 > i0:
            cuts.append((i0, i1, e))

    if not cuts:
        report = {"edits": applied, "removed_s": 0.0,
                  "duration_before_s": round(n / sr, 3),
                  "duration_after_s": round(n / sr, 3),
                  "crossfade_ms": crossfade_ms, "cuts": 0,
                  "bleeps": len(applied)}
        return x2, report

    # te behouden spans tussen de knips
    keep: list[tuple[int, int]] = []
    pos = 0
    for i0, i1, _e in cuts:
        if i0 > pos:
            keep.append((pos, i0))
        pos = max(pos, i1)
    if pos < n:
        keep.append((pos, n))

    pieces: list[np.ndarray] = []
    for k, (a, b) in enumerate(keep):
        piece = x2[:, a:b].astype(np.float64)
        if k > 0 and pieces:
            prev = pieces[-1]
            f = min(xf, prev.shape[1], piece.shape[1])
            if f > 1:
                fi = _raised_cosine(f)
                prev[:, -f:] = prev[:, -f:] * (1 - fi) + piece[:, :f] * fi
                piece = piece[:, f:]
        pieces.append(piece)
    y = np.concatenate(pieces, axis=1).astype(np.float32) if pieces else \
        np.zeros((x2.shape[0], 0), dtype=np.float32)

    # verslag: per knip de echte (sample-gekwantiseerde) tijden
    shift = 0
    for i0, i1, e in cuts:
        applied.append({**e, "start_s": round(i0 / sr, 3), "end_s": round(i1 / sr, 3),
                        "removed_s": round((i1 - i0) / sr, 3),
                        "new_start_s": round(max(0, i0 - shift) / sr, 3)})
        shift += i1 - i0
    applied.sort(key=lambda e: e["start_s"])
    report = {
        "edits": applied,
        "removed_s": round(sum(e.get("removed_s", 0.0) for e in applied), 3),
        "duration_before_s": round(n / sr, 3),
        "duration_after_s": round(y.shape[1] / sr, 3),
        "crossfade_ms": crossfade_ms,
        "cuts": len(cuts),
        "bleeps": sum(1 for e in applied if e["action"] == "bleep"),
    }
    return y, report
