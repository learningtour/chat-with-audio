"""MCP-servertests: toolregistratie + directe aanroep van de toolfuncties."""

import asyncio

from chat_with_audio import server

EXPECTED = {"analyze_audio", "improve_audio", "reduce_noise", "normalize_loudness",
            "apply_chain", "repair_audio", "match_reference", "refine_audio",
            "optimize_audio", "transcribe_audio", "separate_stems", "rebalance_music",
            "improve_folder", "view_audio", "rate_audio", "export_to_audition",
            "list_sessions", "open_viewer", "smart_edit",
            "list_recipes", "save_recipe", "apply_recipe",
            "check_compliance", "master_for"}


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
