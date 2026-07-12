# Chat with Audio — development notes

Chat-driven audio enhancement tool: MCP server (FastMCP, stdio) + C++ DSP core
(pybind11) + local A/B viewer. See README.md and docs/ for user documentation
(tool reference, workflows, compliance, smart regions, recipes, architecture).

> Name everywhere: **Chat with Audio** (package `chat_with_audio`, MCP server
> `chat-with-audio`, GitHub `chat-with-audio`, local project folder
> "Chat with Audio").

## Commands

```bash
uv sync --all-extras                              # build (incl. C++) + all deps
uv sync --reinstall-package chat-with-audio # after changes in cpp/
uv run pytest                                     # test suite
uv run ruff check .                               # lint (also runs in CI)
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
- `regions.py` → smart problem regions: windowed detectors (hum/noise/clip/boom)
  find where on the timeline something is wrong; per-region mini-chains are
  applied with raised-cosine crossfades, everything outside stays untouched
  (`smart_edit` tool). Noise reference floor is clamped to -80 dB; boom regions
  inside a hum region are dropped (the notch already covers them).
- `recipes.py` → saved chains as shareable JSON (`save_recipe`/`apply_recipe`/
  `list_recipes`); built-ins live in `src/chat_with_audio/recipes/`, user
  recipes in `~/AudioImprove/recipes/` (env `AIT_RECIPES_DIR`; tests isolate
  this automatically). `chain.validate_steps()` guards every load/save.
- `segments.py` → speech/music/silence segmentation (level-Otsu primary;
  modulation rhythm as fallback). `refine.py` → iterative measure-and-adjust loop
  (`refine_audio` tool): AI denoising once up front, then adjust leveler/loudness
  until the speech peak and balance are right; silence segments are pushed back
  down afterwards (_duck_silence) because the leveler would otherwise lift them.
- `compliance.py` → delivery-spec registry (EBU R128, ATSC A/85, Netflix 2.0
  én 5.1 dialogue-gated, streaming, ACX) + pass/fail checker incl. formaat-
  en kanaaleisen; `master_for` schrijft compliance.json (viewer-paneel) en
  exporteert mono → dual-mono bij een 2.0-eis. dialogue_loudness = blok-
  gebaseerde spraak-gated meting (DI-achtig, niét het Dolby-algoritme);
  op 5.1 detectie op het centerkanaal.
- Surround: `SURROUND_LAYOUTS` in analysis.py — 5.1 (SMPTE) krijgt gewogen
  BS.1770 (LFE eruit via loudness_view), per-kanaal-QC en ITU-downmix-piek;
  ADM BWF (Atmos-metadata) wordt herkend via axml/chna-chunks in io.probe.
- `dsp/dialogue.py` → breath_control / deplosive / duck_music (chain steps);
  gain envelopes are smoothed with edge padding — plain convolution would
  drag file edges toward zero.
- `markers.py` → region map → Audition marker CSV + Audacity labels + JSON
  (`export_markers`).
- `dsp/roomtone.py` → room-tone fill: digitale gaten vullen met geshuffelde
  overlap-add van de eigen ambience (`fill_room_tone`); `qcsheet.py` →
  markdown-QC-rapport (`qc_report`, batch: `qc_folder`).
- `dsp/spectral_repair.py` → spectral painting (`spectral_repair`): magnitudes
  interpoleren uit de context, fase phase-vocoder-coherent voortzetten
  (bin-centerfrequenties laten mainlobe-bins driften — gemeten dphi gebruiken).
- `duck_music` heeft twee modi: beds (segmentniveau, licht) en stems
  (Demucs-sidechain voor muziek ónder spraak, [stems]-extra).
- `sync.py` → 32-sporenrecorder (`sync_tracks`): envelope-GCC-PHAT + full-rate
  verfijning, confidence per spoor, klokdrift-meting/-correctie; uitgelijnde
  wavs + .sesx; A/B-sessie = ongesynct vs gesynct mixdown. Valkuil: strak
  periodiek materiaal (metronoom) is inherent dubbelzinnig voor correlatie —
  testsignalen moeten aperiodiek gaten (en recorder-seeds ver van event-seeds,
  anders ontstaat een echte schijncorrelatie).
- `server.py` — 30 MCP tools; `sessions.py` — session folders under
  `~/AudioImprove/sessions/` (env `AIT_SESSIONS_DIR`; tests isolate this
  automatically). Every session writes `timeline.json` (segments + treated
  regions) for the viewer's timeline lane; ids get a `-2` suffix on collision.
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
