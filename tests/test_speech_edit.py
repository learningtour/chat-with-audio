"""Tekstmontage: planner (synthetische woordenlijsten) + renderer + MCP-tool.

Er is bewust geen Whisper-model nodig: de motor is puur DSP en de planner
werkt op elke woordenlijst; de E2E-test monkeypatcht de transcriptie.
"""

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import speech_edit

SR = 44100


def w(word, start, end):
    return {"word": word, "start": start, "end": end, "prob": 0.95}


WORDS = [w("Dit", 0.5, 0.8), w("is", 0.9, 1.1), w("eh,", 1.3, 1.6),
         w("een", 1.8, 2.0), w("een", 2.1, 2.3), w("test.", 2.4, 2.8),
         w("Echt", 4.5, 4.8), w("waar", 4.9, 5.2)]
DUR = 6.0


def burst_audio(words, dur=DUR, sr=SR, freq=440.0, amp=0.2):
    """Toonstootje op elke woordpositie; stilte ertussen."""
    x = np.zeros(int(dur * sr), dtype=np.float32)
    t = np.arange(x.shape[0]) / sr
    for word in words:
        i0, i1 = int(word["start"] * sr), int(word["end"] * sr)
        seg = np.sin(2 * np.pi * freq * t[i0:i1]) * amp
        edge = int(0.01 * sr)
        env = np.ones(i1 - i0)
        env[:edge] = np.linspace(0, 1, edge)
        env[-edge:] = np.linspace(1, 0, edge)
        x[i0:i1] = (seg * env).astype(np.float32)
    return x


# ---------------------------------------------------------------- planner

def test_fillers_found_with_guard_bounded_by_neighbours():
    edits = speech_edit.plan_fillers(WORDS, DUR, language="nl")
    assert len(edits) == 1
    e = edits[0]
    assert e["reason"] == "filler" and "eh" in e["label"]
    # guard van 50 ms rond het woord, maar nooit in de buurwoorden
    assert 1.1 <= e["start_s"] <= 1.3 and 1.6 <= e["end_s"] <= 1.8
    assert "eh," in e["context"]


def test_fillers_respect_language_lexicon():
    words = [w("um", 0.5, 0.7), w("right", 0.9, 1.2)]
    assert speech_edit.plan_fillers(words, 2.0, language="en")
    assert not speech_edit.plan_fillers([w("right", 0.9, 1.2)], 2.0, language="en")


def test_doubles_removes_first_instance_only():
    edits = speech_edit.plan_doubles(WORDS, DUR)
    assert len(edits) == 1
    e = edits[0]
    assert e["reason"] == "double"
    assert 1.6 <= e["start_s"] <= 1.8 and e["end_s"] <= 2.1  # eerste 'een'


def test_doubles_ignores_slow_repetition():
    words = [w("ja", 0.5, 0.7), w("ja", 2.0, 2.2)]  # gap > 0.4 s: retorisch
    assert speech_edit.plan_doubles(words, 3.0) == []


def test_pauses_only_above_target_plus_slack():
    edits = speech_edit.plan_pauses(WORDS, DUR, max_pause_s=0.6)
    assert len(edits) == 1  # alleen de 1.7 s-pauze tussen 'test.' en 'Echt'
    e = edits[0]
    # kop en staart van de pauze blijven staan (elk max_pause/2)
    assert e["start_s"] == pytest.approx(2.8 + 0.3, abs=0.01)
    assert e["end_s"] == pytest.approx(4.5 - 0.3, abs=0.01)


def test_find_phrase_is_punctuation_and_case_insensitive():
    hits = speech_edit.find_phrase(WORDS, "EEN TEST")
    assert hits == [(4, 5)]
    assert speech_edit.find_phrase(WORDS, "niet aanwezig") == []


def test_text_edits_report_missing_phrases():
    edits, missing = speech_edit.plan_text_edits(WORDS, DUR, ["een test", "foo bar"])
    assert len(edits) == 1 and missing == ["foo bar"]


def test_merge_overlapping_deletes_and_drop_bleep_inside_delete():
    edits = [
        {"action": "delete", "start_s": 1.0, "end_s": 2.0, "label": "a", "reason": "x"},
        {"action": "delete", "start_s": 1.5, "end_s": 2.5, "label": "b", "reason": "x"},
        {"action": "bleep", "start_s": 1.6, "end_s": 1.9, "label": "c", "reason": "x"},
        {"action": "bleep", "start_s": 3.0, "end_s": 3.4, "label": "d", "reason": "x"},
    ]
    merged = speech_edit.merge_edits(edits)
    deletes = [e for e in merged if e["action"] == "delete"]
    bleeps = [e for e in merged if e["action"] == "bleep"]
    assert len(deletes) == 1 and deletes[0]["end_s"] == 2.5
    assert len(bleeps) == 1 and bleeps[0]["start_s"] == 3.0


def test_plan_edits_full_pipeline():
    edits, missing = speech_edit.plan_edits(
        WORDS, DUR, language="nl", tighten_pauses_to_s=0.6,
        remove_text=["een test"], bleep_text=["waar"])
    actions = [e["action"] for e in edits]
    assert missing == []
    assert actions.count("bleep") == 1
    # filler + double + pauze + frase, waarbij overlappen versmolten mogen zijn
    assert actions.count("delete") >= 3


# ---------------------------------------------------------------- renderer

def test_apply_edits_removes_time_without_clicks():
    x = burst_audio(WORDS)
    edits = speech_edit.plan_fillers(WORDS, DUR)
    y, report = speech_edit.apply_edits(x, SR, edits, crossfade_ms=12.0)
    assert report["cuts"] == 1
    removed = report["duration_before_s"] - report["duration_after_s"]
    assert removed == pytest.approx(report["removed_s"], abs=0.02)
    assert 0.3 <= removed <= 0.8
    # crossfade: geen sprongen groter dan de natuurlijke helling van de toon
    natural = 0.2 * 2 * np.pi * 440 / SR
    assert np.max(np.abs(np.diff(y[0]))) < 4 * natural


def test_apply_edits_keeps_untouched_words_intact():
    x = burst_audio(WORDS)
    edits, _ = speech_edit.plan_edits(WORDS, DUR, remove_fillers=True,
                                      remove_doubles=False)
    y, report = speech_edit.apply_edits(x, SR, edits)
    # eerste woord ligt vóór alle knips: bit-voor-bit gelijk
    i0, i1 = int(0.5 * SR), int(0.8 * SR)
    np.testing.assert_array_equal(y[0, i0:i1], x[i0:i1])


def test_bleep_is_length_neutral_and_masks_content():
    x = burst_audio(WORDS)
    edits, missing = speech_edit.plan_edits(
        WORDS, DUR, remove_fillers=False, remove_doubles=False,
        bleep_text=["test"])
    assert missing == []
    y, report = speech_edit.apply_edits(x, SR, edits)
    assert report["duration_after_s"] == report["duration_before_s"]
    assert report["bleeps"] == 1
    seg = y[0, int(2.5 * SR):int(2.7 * SR)]
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(seg.shape[0], 1 / SR)
    dominant = freqs[int(np.argmax(spec))]
    assert dominant == pytest.approx(1000.0, abs=15.0)  # bliep, geen 440 meer


def test_bleep_mute_style_silences_span():
    x = burst_audio(WORDS)
    edits, _ = speech_edit.plan_edits(WORDS, DUR, remove_fillers=False,
                                      remove_doubles=False, bleep_text=["test"])
    y, _ = speech_edit.apply_edits(x, SR, edits, bleep_style="mute")
    seg = y[0, int(2.5 * SR):int(2.7 * SR)]
    assert np.max(np.abs(seg)) < 1e-4


def test_no_edits_is_identity():
    x = burst_audio(WORDS)
    y, report = speech_edit.apply_edits(x, SR, [])
    np.testing.assert_array_equal(y[0], x)
    assert report["removed_s"] == 0.0


# ---------------------------------------------------------------- MCP-tool

@pytest.fixture
def speech_wav(tmp_path):
    p = tmp_path / "speech.wav"
    sf.write(str(p), burst_audio(WORDS), SR)
    return p


@pytest.fixture
def fake_asr(monkeypatch):
    from chat_with_audio import asr

    monkeypatch.setattr(asr, "is_available", lambda: True)
    monkeypatch.setattr(asr, "transcribe_words", lambda x, sr, **kw: {
        "text": "Dit is eh, een een test. Echt waar",
        "language": "nl", "words": [dict(word) for word in WORDS]})


def test_edit_speech_tool_end_to_end(speech_wav, fake_asr):
    from chat_with_audio import server

    res = server.edit_speech(str(speech_wav), tighten_pauses_to_s=0.6,
                             bleep_text=["waar"])
    assert res["applied"] is True
    assert res["report"]["cuts"] >= 3 and res["report"]["bleeps"] == 1
    assert res["report"]["duration_after_s"] < res["report"]["duration_before_s"]
    data, sr = sf.read(res["output_path"])
    assert data.shape[0] / sr == pytest.approx(res["report"]["duration_after_s"],
                                               abs=0.01)
    # kniplijst staat als tijdlijnregio's in de sessie -> export_markers werkt
    detail = server.list_sessions(session_id=res["session_id"])
    kinds = {r["kind"] for r in detail["timeline"]["regions"]}
    assert {"cut", "pause", "bleep"} <= kinds


def test_edit_speech_plan_mode_makes_no_audio(speech_wav, fake_asr):
    from chat_with_audio import server

    res = server.edit_speech(str(speech_wav), apply=False,
                             tighten_pauses_to_s=0.6)
    assert res["applied"] is False and len(res["plan"]) >= 3
    assert "output_path" not in res
    assert all("context" in e for e in res["plan"])


def test_edit_speech_reports_missing_phrases(speech_wav, fake_asr):
    from chat_with_audio import server

    res = server.edit_speech(str(speech_wav), remove_fillers=False,
                             remove_doubles=False, remove_text=["bestaat niet"])
    assert res["phrases_not_found"] == ["bestaat niet"]
    assert "message" in res  # niets te monteren
