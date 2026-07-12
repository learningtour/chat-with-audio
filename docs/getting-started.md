# Getting started

## Requirements

- **macOS or Windows** (Linux works too; the viewer and editor integrations
  are tested on macOS)
- **[uv](https://docs.astral.sh/uv/)** — Python and dependency manager
- **ffmpeg** — decode/encode of mp3/m4a/ogg (`brew install ffmpeg` /
  `winget install ffmpeg`)
- **C++ toolchain** for the native DSP core: Xcode Command Line Tools on
  macOS, Visual Studio Build Tools (C++ workload) on Windows. Without it,
  everything still works through the pure-Python fallback — just slower.

Python 3.11 is fetched by uv automatically; you don't need to install it.

## Install

```bash
git clone https://github.com/learningtour/chat-with-audio
cd chat-with-audio
uv sync --all-extras        # builds the C++ core + installs all AI extras
uv run pytest               # 109 tests should pass
uv run python scripts/mcp_smoke.py   # end-to-end MCP check (33 tools)
```

`uv sync` without `--all-extras` installs a lean base (no torch, no
DeepFilterNet, no Whisper, no Demucs). Every AI feature then degrades
gracefully: denoising falls back to spectral gating, intelligibility checks
are skipped, stem tools explain what to install.

### The extras

| Extra | Enables | Brings in |
|---|---|---|
| `ai` | DeepFilterNet speech denoising (Tier B) | torch, torchaudio |
| `asr` | Whisper transcription + word-retention guard | openai-whisper |
| `enhance` | ClearVoice dereverberation | clearvoice |
| `stems` | Stem separation, rebalance, Audition export | demucs |

## Register with Claude

**Claude Code** — the repo ships a `.mcp.json`; opening the project folder in
Claude Code picks it up automatically.

**Claude Desktop** — add to
`~/Library/Application Support/Claude/claude_desktop_config.json`
(`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "chat-with-audio": {
      "command": "/absolute/path/to/uv",
      "args": ["run", "--directory", "/absolute/path/to/chat-with-audio",
               "chat-with-audio-mcp"]
    }
  }
}
```

Use the absolute path to `uv` (`which uv`): GUI apps don't inherit your
shell's PATH. Restart Claude Desktop afterwards.

**Codex CLI** —
`codex mcp add chat-with-audio -- <uv-path> run --directory <project-dir> chat-with-audio-mcp`

## Your first session

Ask Claude:

> Analyze this file: ~/Desktop/interview.wav

You get metrics (LUFS, true peak, SNR, noise floor, spectrum, stereo QC),
0-100 scores, and concrete issues with suggested fixes. Then:

> Make it better.

improve_audio detects speech/music, builds a chain (highpass, hum notches,
denoise, de-esser, EQ, dynamics, loudness) and explains every step. Then:

> Open the viewer.

The A/B viewer starts at <http://127.0.0.1:8471>: space plays, **a/b**
switches original/processed in perfect sync, **r** plays the residual — you
hear exactly what was changed. Sessions are stored under
`~/AudioImprove/sessions/`, each with audio, analyses, chain + rationale,
timeline and a full provenance log.

## The dev CLI (no MCP needed)

```bash
uv run ait analyze recording.wav
uv run ait improve recording.wav --profile speech
uv run ait viewer
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AIT_SESSIONS_DIR` | `~/AudioImprove/sessions` | Session storage |
| `AIT_RECIPES_DIR` | `~/AudioImprove/recipes` | Your saved recipes |
| `AIT_VIEWER_PORT` | `8471` | Viewer port |
