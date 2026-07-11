# Night roadmap — July 10/11, 2026

Assignment from Serge: "Keep building overnight. Super advanced audio
applications, groundbreaking, every production house will want to download
this." Method: finish each feature (implementation + tests + commit) before
starting the next; the repo stays green.

## Status

- [x] 1. Spectral repair: declip + declick (dsp/repair.py, chain steps,
      MCP tool repair_audio, auto-declip in improve when clip_events > 0)
- [x] 2. Stem separation (Demucs htdemucs): separate_stems + rebalance_music
      (incl. karaoke = vocals -60 dB); A/B session for rebalance
- [x] 3. Residual listening in the viewer: third button "R - difference" =
      loudness-matched difference between original and processed (artifact check)
- [x] 4. Reference matching: match_reference(file, reference) - 1/3-octave
      spectral match EQ (bounded) + loudness match
- [x] 5. Auto de-esser (spectral, speech segments only) + resonance detection
      -> notches; included in the improve rules
- [x] 6. Batch processing: ait batch + MCP improve_folder
- [x] 7. Docs (README/CLAUDE.md), update mcp_smoke, morning report
      (MORNING_REPORT.md) + update memory

## Rules

- Python 3.11, numpy<2, torch<2.9 — don't touch the pins (see CLAUDE.md).
- Never pollute stdout in server code; lazy-load heavy models.
- New MCP tools also go into tests/test_mcp_tools.py EXPECTED and scripts/mcp_smoke.py.
- Test audio lives in upload/ (git-ignored); sessions in ~/AudioImprove/sessions.
- After each feature: uv run pytest -q green -> git commit.
