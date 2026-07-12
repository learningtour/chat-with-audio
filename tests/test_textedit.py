"""Tekstgestuurde bewerking: knipplan (puur) + render (audio), zonder Whisper."""

import numpy as np
import pytest

from chat_with_audio import textedit


def W(word, start, end, p=0.9):
    return {"word": word, "start": start, "end": end, "probability": p}


# ------------------------------------------------------------------ plannen

def test_filler_plan_respects_neighbours():
    words = [W("dit", 0.5, 0.8), W("eh", 1.1, 1.4), W("is", 1.7, 1.9),
             W("een", 2.1, 2.3), W("uhm,", 2.6, 2.9), W("test", 3.2, 3.5)]
    plan = textedit.plan_edits(words, 4.0, max_pause_s=0)
    fillers = [e for e in plan["edits"] if e["kind"] == "filler"]
    assert len(fillers) == 2
    e = fillers[0]
    assert e["action"] == "cut"
    assert "eh" in e["text"]
    # blijft van de buurwoorden af, maar neemt de kortste pauze mee
    assert 0.8 < e["start_s"] < 1.11
    assert e["end_s"] < 1.7
    assert "uhm" in fillers[1]["text"]
    assert plan["transcript_after"].split() == ["dit", "is", "een", "test"]


def test_repeat_plan_single_and_bigram():
    words = [W("dat", 0.5, 0.7), W("dat", 0.9, 1.1), W("klopt", 1.3, 1.7)]
    plan = textedit.plan_edits(words, 2.5, max_pause_s=0)
    reps = [e for e in plan["edits"] if e["kind"] == "repeat"]
    assert len(reps) == 1
    # eerste voorkomen + het gat tot de tweede inzet wordt geknipt
    assert reps[0]["end_s"] == pytest.approx(0.9, abs=1e-6)
    assert reps[0]["start_s"] < 0.5

    words = [W("ik", 0.5, 0.6), W("heb", 0.7, 0.9), W("ik", 1.0, 1.1),
             W("heb", 1.2, 1.4), W("honger", 1.6, 2.0)]
    plan = textedit.plan_edits(words, 2.5, max_pause_s=0)
    reps = [e for e in plan["edits"] if e["kind"] == "repeat"]
    assert len(reps) == 1
    assert reps[0]["text"] == "ik heb"
    assert reps[0]["end_s"] == pytest.approx(1.0, abs=1e-6)
    assert plan["transcript_after"].split() == ["ik", "heb", "honger"]


def test_pause_plan_keeps_head_and_tail():
    words = [W("een", 0.5, 0.9), W("twee", 4.0, 4.4)]
    plan = textedit.plan_edits(words, 5.0, max_pause_s=1.0, target_pause_s=0.4)
    pauses = [e for e in plan["edits"] if e["kind"] == "pause"]
    assert len(pauses) == 1
    e = pauses[0]
    # pauze 3.1 s -> 0.4 s: kop en staart van 0.2 s blijven staan
    assert e["start_s"] == pytest.approx(1.1, abs=0.01)
    assert e["end_s"] == pytest.approx(3.8, abs=0.01)
    # korte pauzes blijven met rust
    plan2 = textedit.plan_edits(words, 5.0, max_pause_s=4.0)
    assert [e for e in plan2["edits"] if e["kind"] == "pause"] == []


def test_text_remove_bleep_and_not_found():
    words = [W("Dit", 0.5, 0.7), W("is", 0.9, 1.1), W("een", 1.3, 1.5),
             W("geheim", 1.7, 2.2), W("verhaal.", 2.4, 2.9)]
    plan = textedit.plan_edits(words, 3.5, max_pause_s=0,
                               remove_text=["een geheim"],
                               bleep_text=["verhaal", "bestaatniet"])
    kinds = {e["kind"]: e for e in plan["edits"]}
    assert kinds["text"]["text"] == "een geheim"
    assert kinds["text"]["start_s"] < 1.3 and kinds["text"]["end_s"] > 2.2
    b = kinds["bleep"]
    assert b["action"] == "bleep"
    assert b["start_s"] <= 2.4 and b["end_s"] >= 2.9
    assert plan["not_found"] == ["bestaatniet"]
    # gebliepte woorden blijven in de tijdlijn (en dus in het transcript)
    assert plan["transcript_after"].split() == ["Dit", "is", "verhaal."]


def test_keep_text_cuts_the_rest():
    words = [W("intro", 0.5, 1.0), W("de", 2.0, 2.1), W("kern", 2.2, 2.6),
             W("quote", 2.7, 3.1), W("einde", 4.0, 4.5)]
    plan = textedit.plan_edits(words, 5.0, max_pause_s=0,
                               keep_text=["kern quote"])
    uns = [e for e in plan["edits"] if e["kind"] == "unselected"]
    assert len(uns) == 2
    assert uns[0]["start_s"] == 0.0 and uns[0]["end_s"] < 2.2
    assert uns[1]["start_s"] > 3.1 and uns[1]["end_s"] == 5.0
    assert plan["transcript_after"].split() == ["kern", "quote"]

    with pytest.raises(ValueError, match="keep_text"):
        textedit.plan_edits(words, 5.0, keep_text=["nietsgevonden"])


def test_extra_fillers_and_language():
    words = [W("right", 0.5, 0.8), W("um", 1.0, 1.2), W("so", 1.4, 1.6)]
    plan = textedit.plan_edits(words, 2.0, language="en", max_pause_s=0,
                               extra_fillers=["so"])
    cut_texts = {e["text"] for e in plan["edits"]}
    assert cut_texts == {"um", "so"}


# ------------------------------------------------------------------ renderen

@pytest.fixture
def three_tone(sr):
    """10 s: 300 Hz, met 500 Hz tussen 3 en 5 s (het te knippen stuk)."""
    t = np.arange(sr * 10) / sr
    x = 0.1 * np.sin(2 * np.pi * 300 * t)
    mid = (t >= 3.0) & (t < 5.0)
    x[mid] = 0.1 * np.sin(2 * np.pi * 500 * t[mid])
    return x.astype(np.float32)


def _band_energy(x, sr, freq, width=50.0):
    spec = np.abs(np.fft.rfft(x))
    f = np.fft.rfftfreq(x.shape[0], 1.0 / sr)
    return float(spec[(f > freq - width) & (f < freq + width)].sum())


def test_render_cut_and_bleep_mapping(three_tone, sr):
    edits = [
        {"kind": "text", "action": "cut", "start_s": 3.0, "end_s": 5.0, "text": "x"},
        {"kind": "bleep", "action": "bleep", "start_s": 7.0, "end_s": 7.5, "text": "p"},
    ]
    y, info = textedit.render_edits(three_tone, sr, edits)
    assert y.dtype == np.float32 and y.ndim == 2
    # 2 s geknipt + één las van 12 ms
    assert info["duration_after_s"] == pytest.approx(8.0 - 0.012, abs=0.01)
    assert info["removed_s"] == pytest.approx(2.012, abs=0.01)
    # het 500 Hz-stuk is weg
    mono = y[0]
    assert _band_energy(mono, sr, 500) < 0.01 * _band_energy(mono, sr, 300)
    # vóór de eerste ingreep is het bestand bit-voor-bit onaangetast
    n0 = int(2.9 * sr)
    assert np.array_equal(y[0, :n0], three_tone[:n0])
    # de bleep ligt in de bewerkte tijdlijn ~2 s (+ las) eerder en is 1 kHz
    b = [e for e in info["edits"] if e["kind"] == "bleep"][0]
    assert b["edited_start_s"] == pytest.approx(7.0 - 2.012, abs=0.02)
    a, bb = int(b["edited_start_s"] * sr), int(b["edited_end_s"] * sr)
    seg = y[0, a + int(0.05 * sr):bb - int(0.05 * sr)]  # midden: buiten de fades
    assert _band_energy(seg, sr, 1000) > 5 * _band_energy(seg, sr, 300)
    # geen klippende lassen
    assert float(np.abs(y).max()) < 0.25


def test_render_bleep_silence_mode(three_tone, sr):
    edits = [{"kind": "bleep", "action": "bleep", "start_s": 7.0, "end_s": 7.5,
              "text": "p"}]
    y, info = textedit.render_edits(three_tone, sr, edits, bleep_mode="silence")
    # continu programma heeft geen room-tone-donor: het woord wordt stilte
    assert info["duration_after_s"] == pytest.approx(10.0)
    seg = y[0, int(7.1 * sr):int(7.4 * sr)]
    orig = three_tone[int(7.1 * sr):int(7.4 * sr)]
    assert float(np.sqrt((seg ** 2).mean())) < 0.01 * float(np.sqrt((orig ** 2).mean()))


def test_render_refuses_cutting_everything(three_tone, sr):
    edits = [{"kind": "text", "action": "cut", "start_s": 0.0, "end_s": 10.0,
              "text": "alles"}]
    with pytest.raises(ValueError, match="niets"):
        textedit.render_edits(three_tone, sr, edits)


def test_render_overlapping_cuts_merge(three_tone, sr):
    edits = [
        {"kind": "text", "action": "cut", "start_s": 3.0, "end_s": 4.5, "text": "a"},
        {"kind": "pause", "action": "cut", "start_s": 4.4, "end_s": 5.0, "text": "b"},
    ]
    y, info = textedit.render_edits(three_tone, sr, edits)
    assert len(info["cuts"]) == 1
    assert info["duration_after_s"] == pytest.approx(8.0 - 0.012, abs=0.01)
