# Architecture

```
Claude (chat) ── MCP (stdio) ──> server.py (40 tools, FastMCP)
                                     │
             ┌───────────────────────┼────────────────────────┐
             ▼                       ▼                        ▼
        analysis.py             chain.py                sessions.py
        compliance.py           STEP_REGISTRY           timeline/compliance
        regions.py              21 steps                log.md provenance
        segments.py                 │                        │
             │                      ▼                        ▼
             │             dsp/  (dispatch)           viewer/ (stdlib http)
             │             native C++ ⇄ scipy         A/B + residual player
             ▼             spectral_nr / ai_nr        timeline + compliance
        recipes.py         dialogue / repair /        panels
        markers.py         deess / dereverb / stems
```

## The DSP core

`cpp/` is a header-only C++ core (biquads after RBJ, noise gate, soft-knee
compressor, look-ahead brickwall limiter) bound with pybind11 as
`chat_with_audio._dsp`. Arrays are float32 `(channels, n)`; functions return
new arrays; dynamics use a linked detector across channels.

`dsp/__init__.py` dispatches to the native core when built, otherwise to
`dsp/fallback.py` (scipy implementations with identical signatures). This is
why the base install works without a compiler — just slower.

Denoising is tiered: `spectral_nr.py` (STFT spectral gating with Wiener-like
gains, always available) and `ai_nr.py` (DeepFilterNet, `ai` extra).
Specialised processors: `dialogue.py` (breath control, plosive repair,
music-bed ducking), `repair.py` (declip/declick), `deess.py`,
`dereverb.py` (ClearVoice), `stems.py` (Demucs).

## Analysis & decision layers

- `analysis.py` — metrics (BS.1770 loudness incl. short-term/momentary max,
  true peak 4×, PLR, SNR, spectrum, hum, resonances, clipping, dropouts,
  stereo QC, edge silence), 0-100 scores and issue list.
- `segments.py` — speech/music/silence timeline (level-Otsu primary,
  syllable-rate modulation fallback).
- `regions.py` — problem-region detectors + crossfaded per-region chains
  ([smart regions](smart-regions.md)).
- `improve.py` — the rule engine behind improve_audio; every rule writes its
  own rationale sentence.
- `refine.py` / `optimize.py` — measure-and-adjust loop and the variant
  tournament, both Whisper-guarded when the `asr` extra is present.
- `compliance.py` — delivery specs and the pass/fail checker
  ([compliance](compliance.md)).

## Execution

`chain.py` holds `STEP_REGISTRY` (21 steps) and `run_chain`, which validates
*all* steps upfront (`validate_steps`, shared with recipes), executes them,
and returns the resolved parameters (defaults included) for the session
record. One failed step never leaves you with a half-written session.

## Sessions

Every operation writes a session folder under `~/AudioImprove/sessions/`:

```
20260713-021044-episode-041/
├── original.wav / processed.wav / residual.wav
├── analysis_original.json / analysis_processed.json
├── chain.json          # steps + rationale
├── timeline.json       # segments + treated regions
├── compliance.json     # when master_for ran
├── waveform_*.json / spectrogram_*.png
├── markers/            # after export_markers
└── log.md + log.json   # full provenance: input, analysis, decisions, verification
```

The **residual** is processed minus loudness-matched original: play it (key
**r**) and you hear exactly what the processing changed — the artifact
detector for your ears. Session ids get a `-2` suffix on same-second
collisions.

## The viewer

`viewer/server.py` is a stdlib `http.server` bound to 127.0.0.1 (no
external exposure), serving the static app and session data from disk — the
same JSON the MCP tools return, so chat and viewer never disagree. The app
(`viewer/static/`) is dependency-free vanilla JS: synchronized A/B/R
playback via Web Audio (switching is a gain crossfade, never a restart),
waveforms, timeline lanes, metrics/compliance tables, and one-click "Open
in" for Audition/Logic/Reaper/Audacity/Pro Tools.

## The MCP layer

`server.py` (FastMCP, stdio). Two hard rules: **stdout is sacred** (all
logging to stderr, subprocesses with `capture_output`) and heavy models load
lazily. Tools are thin: parse → orchestrate the layers above → session →
structured result with a human rationale.

## Version pins (deliberate)

Python 3.11 + `numpy<2` (DeepFilterNet wheels stop at cp311 and need
numpy 1.x) and `torch/torchaudio<2.9` (DeepFilterNet imports
`torchaudio.backend`, removed in 2.9). Don't "just upgrade" — CI and the
smoke test will tell you why not.

## Testing

109 pytest tests: DSP correctness, segmentation, region detection with
do-no-harm equality asserts, dialogue suite, compliance pass/fail paths,
recipes round-trips, MCP tool registry and end-to-end flows on synthetic
audio (deterministic seeds, no fixtures from the network). CI runs lint +
tests on ubuntu and macos.
