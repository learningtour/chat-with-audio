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
- **"Fix it only where something is wrong"** — `smart_edit`: AI finds problem
  regions on the timeline — mains hum that comes and goes, noise that rises
  temporarily (AC, traffic), clusters of clipping, a passing low-frequency
  rumble — and applies a targeted mini-chain per region with crossfades.
  Everything outside the regions stays bit-for-bit untouched; the region map
  shows up as a timeline in the viewer. The surgical counterpart to
  improve_audio.
- **"Save this as a preset" / "Do this like my podcast preset"** —
  `save_recipe`, `apply_recipe` and `list_recipes`: keep the chain of a
  session that sounded right as a named recipe, reuse it on new files, and
  share it — a recipe is a small JSON file, and apply_recipe also accepts a
  path to someone else's recipe. Curated presets ship built in.
- **"Make it even better, take your time"** — `optimize_audio`: runs multiple
  pipeline variants (EQ, leveler, compressor, ClearVoice dereverberation) and
  lets the best one win on an objective score: Whisper word retention/confidence
  plus target deviation. The ranking comes back in the chat.
- **"Transcribe this"** — `transcribe_audio` (Whisper, [asr] extra); with
  `word_timestamps=True` you get per-word start/end times.
- **"Haal de uhs eruit en maak de pauzes strakker"** — `edit_speech`: edit the
  recording through its transcript ([asr] extra). Filler words and repeated
  words/false starts are cut, long pauses tightened to a target, named
  passages removed or kept (`keep_text` pulls quotes), and words bleeped or
  redacted with the file's own room tone — every joint crossfaded, the cut
  list on the viewer timeline and exportable as DAW markers. `preview=True`
  shows the plan before anything is rendered.
- **"Make this exactly 25 minutes" / "a semitone up, keep the voice natural"**
  — `retime_audio`: time-stretch without touching pitch (to a tempo factor or
  a target duration), pitch-shift without touching duration (with formant
  preservation so voices keep their timbre — or without, to anonymize a
  voice), and tape-style `varispeed`. Signalsmith Stretch under the hood;
  also available as chain steps for recipes.
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
- **"Is this broadcast-proof?"** — `check_compliance`: pass/fail report
  against EBU R128, ATSC A/85, Netflix 2.0 & 5.1 (dialogue-gated, on 5.1
  detected on the center channel; LFE excluded from loudness), Apple
  Podcasts, Spotify, YouTube or ACX audiobook, including the technical QC
  gates (clipping, dropouts, dead channel, anti-phase, ITU-downmix peak) and
  ADM BWF (Dolby Atmos) recognition.
- **"Master this for European broadcast, 48 kHz 24-bit"** — `master_for`:
  masters to the spec (dialogue-gated specs steer on the detected speech),
  re-verifies, and exports a delivery file with high-quality SRC and bit-depth
  control. The compliance report shows up as a panel in the viewer.
- **"Polish the dialogue"** — the film-post steps: `breath_control` (dim
  breaths, never cut), `deplosive` (p/b-pops fixed on the pop only) and
  `duck_music` (music beds ride down under the speech level) — bundled in the
  built-in `dialogue-polish` recipe.
- **"Give me the region map as markers"** — `export_markers`: what the AI
  found lands as Audition marker CSV / Audacity labels in your DAW.
- **"Fill the gaps with room tone"** — `fill_room_tone`: digital holes
  (dropouts, edit gaps) get filled with the file's own ambience — the
  dialogue editor's classic.
- **"There's a chair squeak at 12.3"** — `spectral_repair`: RX-style
  spectral painting; the damaged time-frequency patch is repainted from its
  context while the programme underneath runs straight through.
- **"Duck the music under the speech"** — `duck_music(mode="stems")`: real
  sidechain ducking for music playing *under* dialogue (Demucs separation,
  vocals-driven gain, no pumping).
- **"QC this file" / "QC this folder"** — `qc_report` (one printable QC
  sheet) and `qc_folder` (a whole directory audited into one index table
  with verdicts).
- **"Sync these recorders"** — `sync_tracks`: the 32-track recorder. Files
  from different devices (lav, boom, field recorder, camera, phone) are
  aligned on the audio itself — sample-accurate GCC-PHAT, confidence per
  track, optional clock-drift correction — and come out as aligned WAVs plus
  an Audition session. A/B in the viewer: unsynced echo soup vs the synced
  sum.
- **"Open the viewer"** / **"What exactly changed?"** — A/B comparison; Claude
  reads the same session data the viewer shows.

## Documentation

Extensive docs live in [`docs/`](docs/README.md): [getting
started](docs/getting-started.md), the full [tool reference](docs/tools.md),
a [workflows cookbook](docs/workflows.md) (podcast, broadcast delivery, film
dialogue, restoration, QC), [delivery compliance](docs/compliance.md),
[smart regions](docs/smart-regions.md), [recipes](docs/recipes.md) and the
[architecture](docs/architecture.md).

## Installation (macOS)

Requires: [uv](https://docs.astral.sh/uv/), ffmpeg (`brew install ffmpeg`), Xcode
Command Line Tools. Python 3.11 is fetched by uv itself.

```bash
cd chat-with-audio
uv sync --all-extras        # builds the C++ core and installs everything (incl. AI denoise)
uv run pytest               # 109 tests
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
  (verify with `codex mcp list`). Same 32 tools, same sessions and viewer.

Note: run `uv sync --all-extras` first, otherwise the first server start may
time out while building/downloading.

## The viewer

`open_viewer` (or `uv run ait viewer`) starts it at <http://127.0.0.1:8471>.
Space = play, **a/b = switch between original and processed**, **r = residual**
(you hear exactly what the processing changed — ideal for artifact checking)
while everything keeps playing in sync. Click the waveform to seek. Change the
port with the `AIT_VIEWER_PORT` environment variable.

Under the waveforms sits a timeline: a content lane (speech/music/silence) and,
for smart_edit sessions, an interventions lane showing exactly where which
problem was treated — click a region to jump there and use **r** to hear what
was removed.

Sessions live in `~/AudioImprove/sessions/` (override: `AIT_SESSIONS_DIR`), each
with the original, result, analyses, chain + rationale, timeline, waveforms and
spectrograms.

## Recipes

A recipe is a saved processing chain as a small JSON file — reusable and
shareable. Built-in presets (distilled from real sessions) ship with the
package; your own live in `~/AudioImprove/recipes/` (override:
`AIT_RECIPES_DIR`). Say "save this as my podcast preset" after a session that
sounded right, apply it to new files with "do this like my podcast preset",
and share the JSON file with anyone — `apply_recipe` also takes a file path.
Steps are validated before anything runs.

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
| Smart regions | `regions.py` | windowed problem detectors (hum/noise/clip/boom) + per-region mini-chains with crossfades |
| Recipes | `recipes.py` | saved chains as shareable JSON; built-in presets + `~/AudioImprove/recipes/` |
| Compliance | `compliance.py` | delivery specs (EBU/ATSC/Netflix/streaming/ACX) + pass/fail checker |
| Dialogue suite | `dsp/dialogue.py` | breath control, plosive repair, music-bed ducking |
| DAW markers | `markers.py` | region map → Audition CSV / Audacity labels / JSON |
| Refinement loop | `refine.py` | iterative measure → adjust (speech peak, balance, pause floor), Whisper-guarded |
| Optimization | `optimize.py` | variant contest, scored on intelligibility + targets |
| Intelligibility | `asr.py` | Whisper transcription + word retention ([asr] extra) |
| Dereverberation | `dsp/dereverb.py` | ClearVoice MossFormer2 48 kHz, speech segments only ([enhance] extra) |
| Chain | `chain.py` | step registry (incl. leveler, smart_denoise), loudness normalization |
| MCP server | `server.py` | 32 tools over stdio (FastMCP) |
| Viewer | `viewer/` | stdlib http.server + Web Audio A/B player |

Loudness targets: speech −16 LUFS / TP −1.5 dBTP, music −14 LUFS / TP −1.0 dBTP.

## Version pins (deliberate)

- **Python 3.11** and **numpy < 2.0**: DeepFilterNet 0.5.x only ships wheels up
  to cp311 and requires numpy 1.x.
- **torch/torchaudio < 2.9**: DeepFilterNet imports `torchaudio.backend`, which
  was removed in torchaudio 2.9.

After changing the C++ code: `uv sync --reinstall-package chat-with-audio`.
