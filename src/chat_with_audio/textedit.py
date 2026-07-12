"""Tekstgestuurde spraakbewerking (fase A): woordtimestamps -> knipplan -> render.

plan_edits() is puur (woordenlijst + duur in, bewerkingsplan uit) en
render_edits() voert het plan uit op de audio; beide zijn zonder Whisper te
testen. De MCP-tool edit_speech koppelt ze aan asr.transcribe_words().

Een edit is {"kind", "action", "start_s", "end_s", "text"}:
  filler/repeat/text/unselected/pause -> action "cut": materiaal eruit, met
  een korte raised-cosine-crossfade op elke las;
  bleep -> action "bleep": het woord wordt vervangen door een toon of room
  tone, de tijdlijn blijft daar intact.

Whisper-woordgrenzen zijn niet perfect. Knipvensters houden daarom
WORD_GUARD_S afstand van de buurwoorden en nemen de kortste aangrenzende
pauze mee, zodat er één natuurlijke pauze overblijft in plaats van twee
aaneengeplakte.
"""

from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FILLERS = {
    "nl": {"eh", "ehm", "uh", "uhm", "euh", "euhm", "hm", "hmm", "mmm"},
    "en": {"um", "uh", "uhm", "erm", "er", "hm", "hmm", "mm"},
}

WORD_GUARD_S = 0.03       # minimale afstand tot een buurwoord bij het knippen
PAD_S = 0.02              # marge rond het te knippen/bliepen woord zelf
REPEAT_MAX_GAP_S = 0.6    # max stilte binnen een woordherhaling (valse start)
BIGRAM_MAX_GAP_S = 0.8
KEEP_PAD_S = 0.15         # ademruimte rond bewaarde passages (keep_text)

KIND_LABELS = {"filler": "vulwoord", "repeat": "herhaling",
               "text": "tekst verwijderd", "unselected": "buiten selectie",
               "pause": "pauze ingekort", "bleep": "bleep"}


def _norm(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.lower()).strip("'_")


def _words_text(words: list[dict], i0: int, i1: int) -> str:
    return " ".join(w["word"].strip() for w in words[i0:i1 + 1]).strip()


def _find_phrase(norms: list[str], phrase: str) -> list[tuple[int, int]]:
    """Alle voorkomens van de (genormaliseerde) frase als woordindexbereiken."""
    toks = [t for t in (_norm(p) for p in re.split(r"\s+", phrase)) if t]
    if not toks:
        return []
    return [(i, i + len(toks) - 1) for i in range(len(norms) - len(toks) + 1)
            if norms[i:i + len(toks)] == toks]


def _merge_spans(spans: list[tuple[float, float]],
                 join_gap: float = 0.02) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= join_gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b > a]


def _word_covered(w: dict, spans: list[tuple[float, float]]) -> bool:
    return any(a <= w["start"] and w["end"] <= b for a, b in spans)


def _cut_span(words: list[dict], i0: int, i1: int,
              dur: float) -> tuple[float, float] | None:
    """Knipvenster voor woorden i0..i1: neem midden in het materiaal de
    kortste aangrenzende pauze mee (er blijft dan één natuurlijke pauze over),
    maar blijf WORD_GUARD_S van de buurwoorden vandaan."""
    w0, w1 = words[i0], words[i1]
    prev_end = words[i0 - 1]["end"] if i0 > 0 else 0.0
    next_start = words[i1 + 1]["start"] if i1 + 1 < len(words) else dur
    lo, hi = w0["start"] - PAD_S, w1["end"] + PAD_S
    if 0 < i0 and i1 + 1 < len(words):
        if w0["start"] - prev_end <= next_start - w1["end"]:
            lo = prev_end + WORD_GUARD_S
        else:
            hi = next_start - WORD_GUARD_S
    lo = max(0.0, lo, prev_end + (WORD_GUARD_S if i0 > 0 else 0.0))
    hi = min(dur, hi, next_start - (WORD_GUARD_S if i1 + 1 < len(words) else 0.0))
    if hi - lo < 0.01:  # buurwoorden staan er strak tegenaan: alleen het woord zelf
        lo, hi = max(0.0, w0["start"]), min(dur, w1["end"])
    return (lo, hi) if hi - lo >= 0.01 else None


def _subtract(a: float, b: float,
              spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Deel van [a, b] dat níet door de (gesorteerde) spans wordt gedekt."""
    out: list[tuple[float, float]] = []
    pos = a
    for s, e in spans:
        if e <= pos or s >= b:
            continue
        if s > pos:
            out.append((pos, min(s, b)))
        pos = max(pos, e)
        if pos >= b:
            break
    if pos < b:
        out.append((pos, b))
    return [(x, y) for x, y in out if y - x > 1e-6]


def _shorten_intervals(subs: list[tuple[float, float]],
                       target: float) -> list[tuple[float, float]]:
    """Knipspans die de totale duur van subs terugbrengen tot target, met de
    kop (target/2) vooraan en de staart automatisch achteraan bewaard."""
    total = sum(b - a for a, b in subs)
    if total <= target:
        return []
    remaining_head = target / 2.0
    remaining_cut = total - target
    cuts: list[tuple[float, float]] = []
    for a, b in subs:
        skip = min(remaining_head, b - a)
        remaining_head -= skip
        start = a + skip
        take = min(remaining_cut, b - start)
        if take > 0:
            cuts.append((start, start + take))
            remaining_cut -= take
        if remaining_cut <= 0:
            break
    return cuts


def plan_edits(words: list[dict], duration_s: float, *, language: str = "nl",
               remove_fillers: bool = True, remove_repeats: bool = True,
               extra_fillers: list[str] | None = None,
               max_pause_s: float | None = 1.5, target_pause_s: float = 0.6,
               remove_text: list[str] | None = None,
               keep_text: list[str] | None = None,
               bleep_text: list[str] | None = None) -> dict:
    """Maak het bewerkingsplan uit een woordenlijst (asr.transcribe_words).

    Geeft {"edits": [...], "not_found": [frases], "transcript_after": str}.
    """
    dur = float(duration_s)
    words = [w for w in words if _norm(w["word"])]
    norms = [_norm(w["word"]) for w in words]
    edits: list[dict] = []
    not_found: list[str] = []
    cut_spans: list[tuple[float, float]] = []

    def add_cut(kind: str, span: tuple[float, float] | None, text: str) -> None:
        if span is None:
            return
        a, b = round(span[0], 3), round(span[1], 3)
        if b - a < 0.01:
            return
        edits.append({"kind": kind, "action": "cut",
                      "start_s": a, "end_s": b, "text": text})
        cut_spans.append((a, b))

    # 1) keep_text: alléén de genoemde passages blijven over
    if keep_text:
        kept: list[tuple[float, float]] = []
        for phrase in keep_text:
            hits = _find_phrase(norms, phrase)
            if not hits:
                not_found.append(phrase)
                continue
            for i0, i1 in hits:
                lo = max(0.0, words[i0]["start"] - KEEP_PAD_S)
                if i0 > 0:
                    lo = max(lo, words[i0 - 1]["end"])
                hi = min(dur, words[i1]["end"] + KEEP_PAD_S)
                if i1 + 1 < len(words):
                    hi = min(hi, words[i1 + 1]["start"])
                kept.append((lo, hi))
        if not kept:
            raise ValueError("Geen van de keep_text-passages komt voor in de "
                             "transcriptie; er zou niets overblijven.")
        pos = 0.0
        for a, b in _merge_spans(kept, join_gap=0.2):
            if a - pos > 0.05:
                add_cut("unselected", (pos, a), "")
            pos = max(pos, b)
        if dur - pos > 0.05:
            add_cut("unselected", (pos, dur), "")

    # 2) remove_text: genoemde passages eruit
    for phrase in remove_text or []:
        hits = _find_phrase(norms, phrase)
        if not hits:
            not_found.append(phrase)
            continue
        for i0, i1 in hits:
            if all(_word_covered(words[i], cut_spans) for i in range(i0, i1 + 1)):
                continue
            add_cut("text", _cut_span(words, i0, i1, dur),
                    _words_text(words, i0, i1))

    # 3) vulwoorden
    if remove_fillers:
        fillers = set(DEFAULT_FILLERS.get(language, set()))
        fillers |= {_norm(f) for f in extra_fillers or [] if _norm(f)}
        for i, nw in enumerate(norms):
            if nw in fillers and not _word_covered(words[i], cut_spans):
                add_cut("filler", _cut_span(words, i, i, dur), words[i]["word"])

    # 4) herhalingen/valse starts: eerst woordparen, dan losse woorden
    if remove_repeats:
        claimed: set[int] = set()
        for i in range(len(words) - 3):
            if norms[i:i + 2] != norms[i + 2:i + 4]:
                continue
            if words[i + 2]["start"] - words[i + 1]["end"] > BIGRAM_MAX_GAP_S:
                continue
            if _word_covered(words[i], cut_spans) \
                    or _word_covered(words[i + 1], cut_spans):
                continue
            lo = max(0.0, words[i]["start"] - PAD_S)
            if i > 0:
                lo = max(lo, words[i - 1]["end"] + WORD_GUARD_S)
            add_cut("repeat", (lo, words[i + 2]["start"]),
                    _words_text(words, i, i + 1))
            claimed.update(range(i, i + 4))
        for i in range(len(words) - 1):
            if i in claimed or i + 1 in claimed:
                continue
            if norms[i] != norms[i + 1] or len(norms[i]) < 2:
                continue
            if words[i + 1]["start"] - words[i]["end"] > REPEAT_MAX_GAP_S:
                continue
            if _word_covered(words[i], cut_spans):
                continue
            lo = max(0.0, words[i]["start"] - PAD_S)
            if i > 0:
                lo = max(lo, words[i - 1]["end"] + WORD_GUARD_S)
            add_cut("repeat", (lo, words[i + 1]["start"]), words[i]["word"])

    # 5) pauzes inkorten (op de tijdlijn die na de cuts overblijft)
    if max_pause_s and max_pause_s > 0:
        target = max(0.0, min(target_pause_s, max_pause_s))
        merged = _merge_spans(cut_spans)
        kept_idx = [i for i, w in enumerate(words) if not _word_covered(w, merged)]
        for i, j in zip(kept_idx, kept_idx[1:], strict=False):
            p0, p1 = words[i]["end"], words[j]["start"]
            if p1 - p0 <= max_pause_s:
                continue
            subs = _subtract(p0, p1, merged)
            total = sum(b - a for a, b in subs)
            if total <= max_pause_s:
                continue
            label = f"pauze {total:.1f} s → {target:.1f} s"
            for a, b in _shorten_intervals(subs, target):
                add_cut("pause", (a, b), label)

    # 6) bleeps: tijdlijn intact, alleen het woord onhoorbaar
    merged = _merge_spans(cut_spans)
    for phrase in bleep_text or []:
        hits = _find_phrase(norms, phrase)
        if not hits:
            not_found.append(phrase)
            continue
        for i0, i1 in hits:
            if all(_word_covered(words[i], merged) for i in range(i0, i1 + 1)):
                continue  # wordt toch al weggeknipt
            a = max(0.0, words[i0]["start"] - PAD_S)
            if i0 > 0:
                a = max(a, words[i0 - 1]["end"])
            b = min(dur, words[i1]["end"] + PAD_S)
            if i1 + 1 < len(words):
                b = min(b, words[i1 + 1]["start"])
            if b - a < 0.02:
                continue
            edits.append({"kind": "bleep", "action": "bleep",
                          "start_s": round(a, 3), "end_s": round(b, 3),
                          "text": _words_text(words, i0, i1)})

    edits.sort(key=lambda e: (e["start_s"], e["end_s"]))
    merged = _merge_spans(cut_spans)
    transcript_after = " ".join(w["word"].strip() for w in words
                                if not _word_covered(w, merged))
    return {"edits": edits, "not_found": not_found,
            "transcript_after": transcript_after}


# ----------------------------------------------------------------- renderen

def _render_bleep(y: np.ndarray, sr: int, a: int, b: int, mode: str,
                  donor: np.ndarray | None, rng: np.random.Generator,
                  freq: float = 1000.0) -> None:
    span = b - a
    seg = y[:, a:b].astype(np.float64)
    if mode == "tone":
        rms = float(np.sqrt((seg ** 2).mean()))
        level_db = float(np.clip(20.0 * np.log10(rms + 1e-12), -30.0, -12.0))
        amp = 10.0 ** (level_db / 20.0) * np.sqrt(2.0)
        t = np.arange(span) / sr
        patch = np.repeat((amp * np.sin(2 * np.pi * freq * t))[None, :],
                          y.shape[0], axis=0)
    elif donor is not None:
        from chat_with_audio.dsp import roomtone

        patch = roomtone._tone_patch(donor, span, sr, rng).astype(np.float64)
    else:
        patch = np.zeros((y.shape[0], span))
    fade = max(4, min(int(0.01 * sr), span // 3))
    w = np.ones(span)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
    w[:fade] = ramp
    w[-fade:] = np.minimum(w[-fade:], ramp[::-1])
    y[:, a:b] = (seg * (1.0 - w) + patch * w).astype(np.float32)


def _render_cuts(y: np.ndarray, sr: int, cuts: list[tuple[int, int]],
                 fade_ms: float) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Verwijder de (gesorteerde, samengevoegde) cuts met een crossfade op elke
    las. Geeft (audio, stukkenkaart) terug; elk kaartitem is
    (origineel_start, origineel_eind, uitvoer_start) in samples."""
    n = y.shape[1]
    if not cuts:
        return y, [(0, n, 0)]
    fade = max(4, int(fade_ms / 1000.0 * sr))
    spans: list[tuple[int, int]] = []
    pos = 0
    for a, b in cuts:
        if a > pos:
            spans.append((pos, a))
        pos = max(pos, b)
    if pos < n:
        spans.append((pos, n))
    if not spans:
        return y[:, :0], []
    out = np.zeros((y.shape[0], sum(b - a for a, b in spans)), dtype=np.float32)
    a0, b0 = spans[0]
    w = b0 - a0
    out[:, :w] = y[:, a0:b0]
    pieces = [(a0, b0, 0)]
    for a, b in spans[1:]:
        chunk = y[:, a:b].astype(np.float64)
        f = min(fade, w, b - a)
        if f >= 4:
            up = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f))
            out[:, w - f:w] = (out[:, w - f:w] * (1.0 - up)
                               + chunk[:, :f] * up).astype(np.float32)
            out[:, w:w + (b - a) - f] = chunk[:, f:].astype(np.float32)
            pieces.append((a, b, w - f))
            w += (b - a) - f
        else:
            out[:, w:w + (b - a)] = chunk.astype(np.float32)
            pieces.append((a, b, w))
            w += b - a
    return out[:, :w], pieces


def render_edits(x: np.ndarray, sr: int, edits: list[dict], *,
                 fade_ms: float = 12.0, bleep_mode: str = "tone",
                 seed: int = 1234) -> tuple[np.ndarray, dict]:
    """Voer het plan uit: eerst bleeps (tijdlijn intact), dan de cuts.

    Geeft (audio, info) met per edit ook de positie in de bewerkte tijdlijn
    (edited_start_s), plus duur voor/na en de samengevoegde kniplijst.
    """
    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    y = x2.astype(np.float32).copy()

    bleeps = [e for e in edits if e["action"] == "bleep"]
    donor = None
    if bleeps and bleep_mode != "tone":
        from chat_with_audio.dsp import roomtone

        span = roomtone.find_donor(y, sr)
        if span is not None:
            donor = y[:, span[0]:span[1]].copy()
    rng = np.random.default_rng(seed)
    for e in bleeps:
        a, b = max(0, int(e["start_s"] * sr)), min(n, int(e["end_s"] * sr))
        if b - a >= 16:
            _render_bleep(y, sr, a, b, bleep_mode, donor, rng)

    cut_spans = _merge_spans([(e["start_s"], e["end_s"])
                              for e in edits if e["action"] == "cut"])
    cuts = [(max(0, int(a * sr)), min(n, int(b * sr))) for a, b in cut_spans]
    cuts = [(a, b) for a, b in cuts if b > a]
    out, pieces = _render_cuts(y, sr, cuts, fade_ms)
    if out.shape[1] < sr // 10:
        raise ValueError("Het knipplan laat vrijwel niets van het bestand over; "
                         "bewerking geweigerd.")

    def to_edited(t_s: float) -> float:
        s = t_s * sr
        for a, b, o in pieces:
            if s < a:
                return o / sr  # in een cut: de las zelf
            if s < b:
                return (o + (s - a)) / sr
        return out.shape[1] / sr

    resolved = []
    for e in edits:
        e = dict(e)
        e["edited_start_s"] = round(to_edited(e["start_s"]), 3)
        if e["action"] == "bleep":
            e["edited_end_s"] = round(to_edited(e["end_s"]), 3)
        else:
            e["removed_s"] = round(e["end_s"] - e["start_s"], 3)
        resolved.append(e)
    log.info("render_edits: %d cut(s), %d bleep(s), %.2f s verwijderd",
             len(cuts), len(bleeps), (n - out.shape[1]) / sr)
    return out, {
        "edits": resolved,
        "duration_before_s": round(n / sr, 3),
        "duration_after_s": round(out.shape[1] / sr, 3),
        "removed_s": round((n - out.shape[1]) / sr, 3),
        "cuts": [[round(a / sr, 3), round(b / sr, 3)] for a, b in cuts],
        "crossfade_ms": float(fade_ms),
    }
