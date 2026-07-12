"""MCP-servertests: toolregistratie + directe aanroep van de toolfuncties."""

import asyncio

import pytest

from chat_with_audio import server

EXPECTED = {"analyze_audio", "improve_audio", "reduce_noise", "normalize_loudness",
            "apply_chain", "repair_audio", "match_reference", "refine_audio",
            "optimize_audio", "transcribe_audio", "separate_stems", "rebalance_music",
            "improve_folder", "view_audio", "rate_audio", "export_to_audition",
            "list_sessions", "open_viewer", "smart_edit",
            "list_recipes", "save_recipe", "apply_recipe",
            "check_compliance", "master_for", "export_markers",
            "fill_room_tone", "qc_report", "spectral_repair", "qc_folder",
            "sync_tracks", "edit_speech"}


def test_tool_registry():
    tools = asyncio.run(server.mcp.list_tools())
    assert {t.name for t in tools} == EXPECTED


def test_analyze_and_sessions(noisy_wav):
    res = server.analyze_audio(str(noisy_wav), create_session=True)
    assert res["metrics"]["duration_s"] == 10.0
    assert res["scores"]["overall"] >= 0
    assert res["detected_profile"] in ("speech", "music")
    sid = res["session_id"]

    listing = server.list_sessions()
    assert listing["count"] == 1

    detail = server.list_sessions(session_id=sid)
    assert detail["session_id"] == sid
    assert "original" in detail


def test_apply_chain_tool(noisy_wav):
    res = server.apply_chain(str(noisy_wav), steps=[
        {"type": "highpass", "freq": 100},
        {"type": "gain", "gain_db": 3},
        {"type": "limiter", "ceiling_db": -3},
    ])
    assert res["metrics_after"]["true_peak_dbtp"] <= -2.5
    assert len(res["chain"]) == 3
    assert res["deltas"]["rms_db"] > 1


def test_smart_edit_tool(tmp_path, sr):
    import numpy as np
    import soundfile as sf

    t = np.arange(sr * 12) / sr
    syllables = (np.sin(2 * np.pi * 5.0 * t) > 0).astype(np.float64)
    sentences = (np.sin(2 * np.pi * 0.25 * t) > -0.6).astype(np.float64)
    base = 0.1 * np.sin(2 * np.pi * 300 * t) * syllables * sentences
    hum = 0.02 * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t))
    x = (base + hum * ((t >= 4) & (t < 8))).astype(np.float32)
    p = tmp_path / "hum_middle.wav"
    sf.write(str(p), x, sr)

    res = server.smart_edit(str(p))
    assert res["regions"], res.get("message")
    assert any(r["kind"] == "hum" for r in res["regions"])
    detail = server.list_sessions(session_id=res["session_id"])
    assert detail["timeline"]["regions"], "regiokaart hoort in de sessietijdlijn"
    assert detail["chain"]["steps"][0]["type"] == "region"


def test_recipe_tools_roundtrip(noisy_wav):
    res = server.apply_chain(str(noisy_wav), steps=[{"type": "highpass", "freq": 120}])
    saved = server.save_recipe("hoogdoorlaat", session_id=res["session_id"])
    assert saved["saved"]["name"] == "hoogdoorlaat"

    out = server.apply_recipe(str(noisy_wav), "hoogdoorlaat")
    assert out["chain"][0]["type"] == "highpass"
    assert out["chain"][0]["freq"] == 120
    assert out["session_id"] != res["session_id"]

    listing = server.list_recipes()
    assert any(r["name"] == "hoogdoorlaat" for r in listing["recipes"])
    assert any(r["builtin"] for r in listing["recipes"])


def test_save_recipe_refuses_region_sessions(tmp_path, sr):
    import numpy as np
    import soundfile as sf

    t = np.arange(sr * 12) / sr
    base = (0.1 * np.sin(2 * np.pi * 300 * t)
            * (np.sin(2 * np.pi * 5.0 * t) > 0))
    hum = 0.02 * np.sin(2 * np.pi * 50 * t) * ((t >= 4) & (t < 8))
    p = tmp_path / "hum.wav"
    sf.write(str(p), (base + hum).astype(np.float32), sr)
    res = server.smart_edit(str(p))
    assert res["regions"]
    import pytest

    with pytest.raises(ValueError, match="chirurgische"):
        server.save_recipe("mag-niet", session_id=res["session_id"])


def test_export_markers_from_smart_edit_session(tmp_path, sr):
    import numpy as np
    import soundfile as sf

    t = np.arange(sr * 12) / sr
    base = (0.1 * np.sin(2 * np.pi * 300 * t)
            * (np.sin(2 * np.pi * 5.0 * t) > 0)
            * (np.sin(2 * np.pi * 0.25 * t) > -0.6))
    hum = 0.02 * np.sin(2 * np.pi * 50 * t) * ((t >= 4) & (t < 8))
    p = tmp_path / "hum.wav"
    sf.write(str(p), (base + hum).astype(np.float32), sr)
    res = server.smart_edit(str(p))
    assert res["regions"]

    out = server.export_markers(res["session_id"])
    assert out["count"] >= 1
    from pathlib import Path

    csv_lines = Path(out["audition_csv"]).read_text().strip().splitlines()
    assert csv_lines[0].startswith("Name\tStart\tDuration")
    assert len(csv_lines) == out["count"] + 1
    labels = Path(out["audacity_labels"]).read_text().strip().splitlines()
    start, end, name = labels[0].split("\t")
    assert float(end) > float(start)
    assert name

    both = server.export_markers(res["session_id"], include_segments=True)
    assert both["count"] > out["count"]


def test_export_markers_needs_regions(noisy_wav):
    import pytest

    res = server.analyze_audio(str(noisy_wav), create_session=True)
    with pytest.raises(ValueError, match="regio's|tijdlijndata"):
        server.export_markers(res["session_id"])


def test_fill_room_tone_tool(tmp_path, sr):
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(4)
    t = np.arange(sr * 10) / sr
    speech = (0.1 * np.sin(2 * np.pi * 300 * t)
              * (np.sin(2 * np.pi * 5.0 * t) > 0)
              * (np.sin(2 * np.pi * 0.25 * t) > -0.3))
    x = (speech + rng.normal(0, 10 ** (-52 / 20), t.size)).astype(np.float32)
    x[int(4 * sr):int(4.3 * sr)] = 0.0
    p = tmp_path / "gat.wav"
    sf.write(str(p), x, sr)
    res = server.fill_room_tone(str(p))
    assert len(res["filled"]) == 1
    assert res["donor"]["end_s"] > res["donor"]["start_s"]

    # schoon bestand: duidelijke no-op-melding, geen sessie
    p2 = tmp_path / "schoon.wav"
    sf.write(str(p2), (speech + rng.normal(0, 10 ** (-52 / 20), t.size)
                       ).astype(np.float32), sr)
    res2 = server.fill_room_tone(str(p2))
    assert res2["filled"] == [] and "message" in res2


def test_qc_report_tool(tmp_path, noisy_wav):
    res = server.qc_report(str(noisy_wav), spec="ebu-r128",
                           out_path=str(tmp_path / "qc.md"))
    sheet = res["report_markdown"]
    assert "# QC-rapport" in sheet
    assert "Integrated loudness" in sheet
    assert "Aflever-check" in sheet and "EBU R128" in sheet
    assert res["passed_compliance"] is False  # ruwe testfile haalt -23 niet
    from pathlib import Path

    assert Path(res["report_path"]).is_file()
    assert Path(res["export_path"]).read_text() == sheet

    res2 = server.qc_report(str(noisy_wav))
    assert res2["passed_compliance"] is None
    assert "Aflever-check" not in res2["report_markdown"]


def test_qc_folder_tool(tmp_path, sr, noisy_bursts):
    import numpy as np
    import soundfile as sf

    d = tmp_path / "leveringen"
    d.mkdir()
    sf.write(str(d / "ruw.wav"), noisy_bursts, sr)
    t = np.arange(sr * 4) / sr
    sf.write(str(d / "netjes.wav"),
             (0.12 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)
    (d / "kapot.wav").write_text("dit is geen audio")

    res = server.qc_folder(str(d), spec="apple-podcast",
                           out_path=str(tmp_path / "index.md"))
    assert res["count"] == 3
    by_name = {r["file"]: r for r in res["rows"]}
    assert "error" in by_name["kapot.wav"]
    assert by_name["ruw.wav"]["compliance_passed"] is False
    md = res["summary_markdown"]
    assert "ruw.wav" in md and "netjes.wav" in md and "fout" in md
    from pathlib import Path

    assert Path(res["export_path"]).read_text() == md


def _speech_wav(tmp_path, sr, words, dur_s=8.0):
    """Toonstoten op de woordposities + zachte ruisvloer: 'spraak' voor de test."""
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(7)
    t = np.arange(int(sr * dur_s)) / sr
    x = rng.normal(0, 10 ** (-60 / 20), t.size)
    for w in words:
        a, b = int(w["start"] * sr), int(w["end"] * sr)
        x[a:b] += 0.1 * np.sin(2 * np.pi * 300 * t[a:b])
    p = tmp_path / "spraak.wav"
    sf.write(str(p), x.astype(np.float32), sr)
    return p


def test_edit_speech_tool(tmp_path, sr, monkeypatch):
    from chat_with_audio import asr

    words = [
        {"word": "Dit", "start": 0.5, "end": 0.8, "probability": 0.95},
        {"word": "eh", "start": 1.1, "end": 1.4, "probability": 0.5},
        {"word": "is", "start": 1.7, "end": 1.9, "probability": 0.95},
        {"word": "geheim.", "start": 2.1, "end": 2.6, "probability": 0.9},
        {"word": "Klaar.", "start": 6.5, "end": 6.9, "probability": 0.9},
    ]
    p = _speech_wav(tmp_path, sr, words)
    monkeypatch.setattr(asr, "is_available", lambda: True)
    monkeypatch.setattr(asr, "transcribe_words", lambda *a, **k: {
        "text": "Dit eh is geheim. Klaar.", "language": "nl", "words": words})

    prev = server.edit_speech(str(p), max_pause_s=1.0, target_pause_s=0.4,
                              bleep_text=["geheim"], preview=True)
    assert prev["preview"] is True and "session_id" not in prev
    assert {"filler", "pause", "bleep"} <= {e["kind"] for e in prev["edits"]}

    res = server.edit_speech(str(p), max_pause_s=1.0, target_pause_s=0.4,
                             bleep_text=["geheim"], user_request="strakker graag")
    assert {"filler", "pause", "bleep"} <= set(res["counts"])
    # vulwoord ~0.6 s + pauze 3.9 s -> 0.4 s: ruim 3.5 s korter
    assert 3.0 < res["duration_after_s"] < 4.7
    assert res["removed_s"] == pytest.approx(
        res["duration_before_s"] - res["duration_after_s"], abs=0.01)
    assert "eh" not in res["transcript_after"].split()
    for e in res["edits"]:
        assert "edited_start_s" in e

    detail = server.list_sessions(session_id=res["session_id"])
    assert detail["chain"]["steps"][0]["type"] == "edit_speech"
    assert len(detail["timeline"]["regions"]) == len(res["edits"])
    markers = server.export_markers(res["session_id"])
    assert markers["count"] == len(res["edits"])

    with pytest.raises(ValueError, match="tekstgestuurde"):
        server.save_recipe("mag-niet", session_id=res["session_id"])


def test_edit_speech_needs_whisper_and_valid_mode(noisy_wav, monkeypatch):
    from chat_with_audio import asr

    monkeypatch.setattr(asr, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Whisper"):
        server.edit_speech(str(noisy_wav))
    monkeypatch.setattr(asr, "is_available", lambda: True)
    with pytest.raises(ValueError, match="bleep_mode"):
        server.edit_speech(str(noisy_wav), bleep_mode="fluit")


def test_edit_speech_nothing_to_do(tmp_path, sr, monkeypatch):
    from chat_with_audio import asr

    words = [{"word": "alles", "start": 0.5, "end": 0.9, "probability": 0.9},
             {"word": "prima", "start": 1.1, "end": 1.5, "probability": 0.9}]
    p = _speech_wav(tmp_path, sr, words, dur_s=2.5)
    monkeypatch.setattr(asr, "is_available", lambda: True)
    monkeypatch.setattr(asr, "transcribe_words", lambda *a, **k: {
        "text": "Alles prima.", "language": "nl", "words": words})
    res = server.edit_speech(str(p))
    assert res["edits"] == [] and "message" in res


def test_smart_edit_clean_file_does_nothing(tmp_path, sr):
    import numpy as np
    import soundfile as sf

    t = np.arange(sr * 8) / sr
    x = (0.1 * np.sin(2 * np.pi * 300 * t)
         * (np.sin(2 * np.pi * 5.0 * t) > 0)).astype(np.float32)
    p = tmp_path / "clean.wav"
    sf.write(str(p), x, sr)
    res = server.smart_edit(str(p))
    assert res["regions"] == []
    assert "improve_audio" in res["message"]
