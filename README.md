# Chat with Audio

[![CI](https://github.com/learningtour/chat-with-audio/actions/workflows/ci.yml/badge.svg)](https://github.com/learningtour/chat-with-audio/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)

_Part of **Agentic Production**._

Chat-driven audio enhancement: talk to your local Claude (Desktop or Claude Code)
about a recording and let the toolkit analyze and improve it. A local A/B viewer
lets you compare original and result in perfect sync — hearing *and* seeing.

```
Claude (chat)  ── MCP (stdio) ──>  Python orchestration ──> C++ DSP core (pybind11)
                                        │                    gate · compressor · limiter · EQ
                                        ├─> AI denoise (DeepFilterNet, optional)
                                        ├─> analysis (LUFS, SNR, hum, clipping, spectrum)
                                        └─> sessions ──> A/B viewer (http://127.0.0.1:8471)
```

## What can you ask?

- **"Analyze this file: /path/to/recording.wav"** — metrics, scores and issues.
- **"Make this sound better"** — auto-improve: the tool detects speech/music and
  picks a chain by itself (highpass, hum notches, noise reduction, gate, EQ,
  compression, loudness), with an explanation for every step.
- **"Reduce the noise"** — noise reduction only (spectral gating or DeepFilterNet AI).
- **"Bring the level up without clipping"** — loudness normalization (BS.1770) with
  a true-peak limiter.
- **"Cut 3 dB around 300 Hz and compress lightly"** — an explicit chain via `apply_chain`.
- **"Put the speech at −6 and balance the music"** — `refine_audio`: segments
  speech/music/silence and runs a measure-and-adjust loop until the targets are
  right to the decibel. AI denoising is applied only when the speech SNR is low
  AND Whisper confirms intelligibility doesn't drop; the report includes the
  measurement history, the decisions and a final word-retention check.
- **"Make it even better, take your time"** — `optimize_audio`: runs multiple
  pipeline variants (EQ, leveler, compressor, ClearVoice dereverberation) and
  lets the best one win on an objective score: Whisper word retention/confidence
  plus target deviation. The ranking comes back in the chat.
- **"Transcribe this"** — `transcribe_audio` (Whisper, [asr] extra).
- **"Fix the clips and clicks"** — `repair_audio`: declip (waveform reconstruction)
  and declick; improve_audio applies declip automatically when clipping is detected.
- **"Make it sound like this reference"** — `match_reference`: 1/3-octave match EQ +
  loudness match; keeps episodes/recording days consistent.
- **"Split the stems" / "vocals up 3 dB" / "make a karaoke version"** —
  `separate_stems` and `rebalance_music` (Demucs, [stems] extra).
- **"Do the whole folder"** — `improve_folder`: batch processing (improve/refine/optimize).
- **"Show me what changed"** — `view_audio`: a perceptual panel
  (auditory-scale spectrograms + difference map + level curves) that the AI
  inspects itself to judge what is audible.
- **"This sounds good / this sounds bad"** — `rate_audio`: train your own taste
  model; analyze_audio then scores new audio against your taste.
- **"Set this up in Audition"** — `export_to_audition`: stems + a .sesx
  multitrack session, opened directly in Adobe Audition.
- **"Open the viewer"** / **"What exactly changed?"** — A/B comparison; Claude
  reads the same session data the viewer shows.

## Installation (macOS)

Requires: [uv](https://docs.astral.sh/uv/), ffmpeg (`brew install ffmpeg`), Xcode
Command Line Tools. Python 3.11 is fetched by uv itself.

```bash
cd chat-with-audio
uv sync --all-extras        # builds the C++ core and installs everything (incl. AI denoise)
uv run pytest               # 38 tests
uv run python scripts/mcp_smoke.py   # MCP smoke test
```

`uv sync` (without `--all-extras`) installs the basics without torch/DeepFilterNet;
the tool then falls back to spectral gating automatically.

## Registering with Claude (and Codex)

- **Claude Code**: lives in `.mcp.json` in the project folder (works automatically in this folder).
- **Claude Desktop**: entry `chat-with-audio` in
  `~/Library/Application Support/Claude/claude_desktop_config.json`. Restart
  Claude Desktop after installing; the tools appear under "chat-with-audio".
- **Codex CLI/app**: registered as a global MCP server via
  `codex mcp add chat-with-audio -- <uv-path> run --directory <project-folder> chat-with-audio-mcp`
  (verify with `codex mcp list`). Same 18 tools, same sessions and viewer.

Note: run `uv sync --all-extras` first, otherwise the first server start may
time out while building/downloading.

## The viewer

`open_viewer` (or `uv run ait viewer`) starts it at <http://127.0.0.1:8471>.
Space = play, **a/b = switch between original and processed**, **r = residual**
(you hear exactly what the processing changed — ideal for artifact checking)
while everything keeps playing in sync. Click the waveform to seek. Change the
port with the `AIT_VIEWER_PORT` environment variable.

Sessions live in `~/AudioImprove/sessions/` (override: `AIT_SESSIONS_DIR`), each
with the original, result, analyses, chain + rationale, waveforms and spectrograms.

## Windows

1. Install [uv](https://docs.astral.sh/uv/), ffmpeg (`winget install ffmpeg`) and
   the **Visual Studio Build Tools** (C++ workload) for the native DSP core.
   Everything works without the Build Tools too, via the pure-Python fallback —
   in that case don't remove the C++ build step, it simply fails soft.
2. `uv sync --all-extras` in the project folder (DeepFilterNet ships win_amd64 wheels).
3. Register in `%APPDATA%\Claude\claude_desktop_config.json` with the full path
   to `uv.exe` and the project folder (same shape as the `.mcp.json` here).

## Architecture

| Layer | Location | Role |
|---|---|---|
| C++ DSP core | `cpp/` | biquad EQ (RBJ), noise gate, soft-knee compressor, look-ahead brickwall limiter; exposed via pybind11 as `chat_with_audio._dsp` |
| DSP dispatch | `src/chat_with_audio/dsp/` | native ↔ scipy fallback, spectral gating (`spectral_nr.py`), DeepFilterNet (`ai_nr.py`) |
| Analysis | `analysis.py` | LUFS/LRA (pyloudnorm), true peak, SNR, noise floor, hum, clipping, spectrum, scores + issues |
| Decision logic | `improve.py` | speech/music detection, rules → chain + rationale |
| Segmentation | `segments.py` | speech/music/silence timeline (level Otsu + speech modulation) |
| Refinement loop | `refine.py` | iterative measure → adjust (speech peak, balance, pause floor), Whisper-guarded |
| Optimization | `optimize.py` | variant contest, scored on intelligibility + targets |
| Intelligibility | `asr.py` | Whisper transcription + word retention ([asr] extra) |
| Dereverberation | `dsp/dereverb.py` | ClearVoice MossFormer2 48 kHz, speech segments only ([enhance] extra) |
| Chain | `chain.py` | step registry (incl. leveler, smart_denoise), loudness normalization |
| MCP server | `server.py` | 18 tools over stdio (FastMCP) |
| Viewer | `viewer/` | stdlib http.server + Web Audio A/B player |

Loudness targets: speech −16 LUFS / TP −1.5 dBTP, music −14 LUFS / TP −1.0 dBTP.

## Version pins (deliberate)

- **Python 3.11** and **numpy < 2.0**: DeepFilterNet 0.5.x only ships wheels up
  to cp311 and requires numpy 1.x.
- **torch/torchaudio < 2.9**: DeepFilterNet imports `torchaudio.backend`, which
  was removed in torchaudio 2.9.

After changing the C++ code: `uv sync --reinstall-package chat-with-audio`.
