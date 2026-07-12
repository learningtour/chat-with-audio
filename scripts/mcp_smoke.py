"""MCP-rooktest: start de server over stdio, lijst de tools en analyseert een wav.

Draaien met: uv run python scripts/mcp_smoke.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {"analyze_audio", "improve_audio", "reduce_noise", "normalize_loudness",
                  "apply_chain", "repair_audio", "match_reference", "refine_audio",
                  "optimize_audio", "transcribe_audio", "separate_stems", "rebalance_music",
                  "improve_folder", "view_audio", "rate_audio", "export_to_audition",
                  "list_sessions", "open_viewer", "smart_edit",
                  "list_recipes", "save_recipe", "apply_recipe",
                  "check_compliance", "master_for", "export_markers",
                  "fill_room_tone", "qc_report", "spectral_repair", "qc_folder",
                  "sync_tracks", "edit_speech", "retime_audio"}


async def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "smoke.wav"
        sr = 44100
        t = np.arange(sr * 2) / sr
        sf.write(str(wav), (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

        params = StdioServerParameters(
            command=sys.executable, args=["-m", "chat_with_audio.server"],
            env={"AIT_SESSIONS_DIR": td})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                print("tools:", sorted(names))
                assert names == EXPECTED_TOOLS, f"onverwachte toolset: {names}"

                res = await session.call_tool(
                    "analyze_audio", {"file_path": str(wav), "create_session": False})
                assert not res.isError, res.content
                data = json.loads(res.content[0].text)
                print("lufs:", data["metrics"]["lufs_integrated"],
                      "| backend:", data["dsp_backend"])
                assert data["metrics"]["duration_s"] == 2.0
    print("MCP-rooktest OK")


if __name__ == "__main__":
    asyncio.run(main())
