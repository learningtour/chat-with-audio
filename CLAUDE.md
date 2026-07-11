# Chat with Audio — development notes

Chat-driven audio enhancement tool: MCP server (FastMCP, stdio) + C++ DSP core
(pybind11) + local A/B viewer. See README.md for user documentation.

> Name everywhere: **Chat with Audio** (package `chat_with_audio`, MCP server
> `chat-with-audio`, GitHub `chat-with-audio`, local project folder
> "Chat with Audio").

## Commands

```bash
uv sync --all-extras                              # build (incl. C++) + all deps
uv sync --reinstall-package chat-with-audio # after changes in cpp/
uv run pytest                                     # test suite
uv run python scripts/mcp_smoke.py                # MCP stdio smoke test
uv run ait analyze <file>                         # dev CLI without MCP
uv run ait improve <file> [--profile speech|music] [--denoise-method ai]
uv run ait viewer                                 # viewer on :8471
```

## Architecture map

- `cpp/` — header-only DSP (biquad.hpp, dynamics.hpp) + `bindings.cpp` → module
  `chat_with_audio._dsp`. Arrays are float32 (channels, n); functions return
  new arrays; dynamics use a linked detector across channels.
- `src/chat_with_audio/dsp/__init__.py` — dispatch: native `_dsp` when built,
  otherwise `fallback.py` (scipy; identical signatures, block-based dynamics).
  `spectral_nr.py` = Tier A denoise; `ai_nr.py` = Tier B (DeepFilterNet).
- `analysis.py` → metrics dict + `score_and_issues()`; `improve.py` → profile
  detection + rules → (steps, rationale); `chain.py` → `STEP_REGISTRY` + execution
  (incl. `leveler` and segment-driven `smart_denoise`).
- `segments.py` → speech/music/silence segmentation (level-Otsu primary;
  modulation rhythm as fallback). `refine.py` → iterative measure-and-adjust loop
  (`refine_audio` tool): AI denoising once up front, then adjust leveler/loudness
  until the speech peak and balance are right; silence segments are pushed back
  down afterwards (_duck_silence) because the leveler would otherwise lift them.
- `server.py` — 18 MCP tools; `sessions.py` — session folders under
  `~/AudioImprove/sessions/` (env `AIT_SESSIONS_DIR`; tests isolate this automatically).
- `viewer/server.py` — stdlib http.server on 127.0.0.1:8471 (env `AIT_VIEWER_PORT`);
  `viewer/static/app.js` — A/B player: both buffers always play together,
  switching = gain crossfade.

## Pitfalls

- **stdout is sacred**: the MCP server runs over stdio. Never `print()` in
  server code paths; logging goes to stderr, subprocesses with `capture_output=True`.
- **Python 3.11 + numpy<2 + torch/torchaudio<2.9 are hard pins** (DeepFilterNet
  wheels and the `torchaudio.backend` import). Do not "just upgrade".
- The module is named `dsp/ai_nr.py` (not `ai_denoise.py`) to avoid shadowing
  the function `dsp.ai_denoise()`.
- `normalize_loudness` sets the limiter ceiling 0.3 dB below the true-peak target
  (inter-sample peaks). E2E tests check denoising via the SNR delta, not the
  absolute noise floor (loudness normalization lifts the floor along with it).
- Registration: `.mcp.json` (Claude Code) and the Claude Desktop config use the
  absolute path to `uv` — GUI apps don't have `~/.local/bin` in PATH.
