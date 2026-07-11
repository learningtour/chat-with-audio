"""MCP-servertests: toolregistratie + directe aanroep van de toolfuncties."""

import asyncio

from chat_with_audio import server

EXPECTED = {"analyze_audio", "improve_audio", "reduce_noise", "normalize_loudness",
            "apply_chain", "repair_audio", "match_reference", "refine_audio", "optimize_audio",
            "transcribe_audio", "separate_stems", "rebalance_music",
            "improve_folder", "view_audio", "rate_audio", "export_to_audition", "list_sessions", "open_viewer"}


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
