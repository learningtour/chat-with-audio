# Tool reference

All 25 MCP tools, grouped by job. Every processing tool creates a **session**
(A/B audio, analyses, chain + rationale, timeline, provenance log) and
returns its `session_id` plus a `viewer_url`. File inputs accept wav, mp3,
m4a/aac, flac, ogg and aiff; `out_path` exports to the format matching its
extension.

---

## Analyze & QC

### `analyze_audio(file_path, create_session=True)`
Full quality analysis: loudness (integrated / short-term max / momentary
max, BS.1770), true peak (4× oversampled), PLR, RMS, crest factor, noise
floor, SNR, silence %, LRA, spectrum (band energy, centroid, tilt), hum
detection, narrow resonances, clipping, DC offset, head/tail silence,
digital dropout scan and stereo QC (phase correlation, balance, dead
channel, dual-mono, polarity inversion). Returns metrics, 0-100 scores
(loudness/noise/dynamics/clarity/overall), concrete issues with fix
suggestions, and a taste score once the taste model is trained.

### `check_compliance(file_path, spec="ebu-r128")`
Pass/fail report against a delivery spec — see
[Delivery compliance](compliance.md). Ask: *"Is this broadcast-proof?"*

### `view_audio(session_id | file_path)`
Renders the perceptual comparison panel as an image the AI itself inspects:
auditory-scale spectrograms of A and B, a difference heat map (red = added,
blue = removed), and level curves. This is how the assistant *sees* what you
will hear.

### `transcribe_audio(file_path, model_size="small", language="nl", start_s, end_s)`
Whisper transcription (requires the `asr` extra). Used standalone or as an
intelligibility check: transcribe original and processed, compare.

---

## Enhance (whole file)

### `improve_audio(file_path, profile="auto", target_lufs=None, denoise_method="auto", out_path=None)`
The "make it better" button. Analyzes, detects speech/music, and builds a
chain — declip if needed, highpass, hum notches, denoise (DeepFilterNet for
speech when installed, else spectral gating), soft gate, resonance cuts,
de-esser, tone EQ, leveler/compressor when the balance needs it, loudness
normalization — with a written rationale per step.

### `reduce_noise(file_path, strength_db=12, method="auto", use_gate=True, out_path=None)`
Noise reduction only; loudness untouched. `method`: `spectral` (STFT
gating, music-safe), `ai` (DeepFilterNet, best for speech), `auto`.

### `normalize_loudness(file_path, target_lufs=-16, true_peak_db=-1.5, out_path=None)`
BS.1770 loudness normalization with a true-peak-safe limiter. Speech −16,
music −14 are common targets; for delivery specs use `master_for`.

### `refine_audio(file_path, speech_peak_db=-6, music_gap_db=2, max_iterations=5, denoise="auto", asr_check=True)`
Iterative measure-and-adjust loop until speech peaks and speech/music
balance hit their targets to the decibel. AI denoising is only applied when
speech SNR is low AND Whisper confirms intelligibility doesn't drop. The
report contains the measurement history and decisions.

### `optimize_audio(file_path, judge_model="small", ...)`
Tournament mode: multiple pipeline variants (EQ, leveler, compressor,
dereverb combinations) each run the full refine loop and are scored
objectively — Whisper word retention/confidence plus target deviation. Slow
but thorough; returns the full ranking.

### `match_reference(file_path, reference_path, strength=1.0, max_db=6, match_loudness=True)`
"Sound like this reference": 1/3-octave match EQ (bounded per band) plus
loudness match. Keeps episodes and recording days consistent.

---

## Surgical (only where something is wrong)

### `smart_edit(file_path, problems="auto", denoise_method="auto", out_path=None)`
AI finds problem regions on the timeline and treats **only** those, with
crossfades; everything outside stays bit-for-bit untouched. Detectors: `hum`
(intermittent mains hum → notches), `noise` (noise floor rising above the
file's cleanest ambience → denoise), `clip` (clip clusters → declip),
`boom` (low-frequency rumble → low cut). See
[Smart regions](smart-regions.md). The region map becomes a timeline in the
viewer and can be exported as DAW markers.

### `repair_audio(file_path, declip=True, declick=True, out_path=None)`
Restoration only: declip (spline waveform reconstruction, also for 32-bit
float capsule overload) and declick (impulse repair). Nothing else changes.

---

## Music & stems (require the `stems` extra)

### `separate_stems(file_path, out_dir=None)`
Demucs separation into vocals / drums / bass / other as WAVs — DAW-ready.

### `rebalance_music(file_path, vocals_db=0, drums_db=0, bass_db=0, other_db=0, target_lufs=None)`
Remix per stem: *"vocals up 3 dB"*, *"karaoke version"* (`vocals_db=-60`).
Peak-guarded remix, optional loudness target, A/B session.

### `export_to_audition(session_id | file_path, source="original", include_mix=True, open_app=True)`
Stems + a `.sesx` multitrack session, opened directly in Adobe Audition.

---

## Deliver

### `master_for(file_path, spec="ebu-r128", out_path=None, sample_rate=None, bit_depth=None)`
Master to a delivery spec and re-verify: loudness to target (dialogue-gated
specs steer on the detected speech), true-peak limiting under the spec
ceiling, then a fresh compliance check — the report lands in the session and
the viewer. `out_path` + `sample_rate=48000` + `bit_depth=24` produces a
broadcast delivery file (high-quality polyphase SRC).

### `export_markers(session_id, out_dir=None, include_segments=False)`
The AI region map as DAW markers: Adobe Audition marker CSV, an Audacity
label track and `markers.json`. *"Hum here, noise there"* becomes navigable
markers inside your editor.

---

## Recipes (reuse & share)

### `list_recipes()`
Built-in presets + your own from `~/AudioImprove/recipes/`.

### `save_recipe(name, session_id | steps, description="")`
Keep the chain of a session that sounded right as a named, shareable JSON
recipe. Surgical (region) sessions are refused — regions are file-specific.

### `apply_recipe(file_path, recipe, out_path=None)`
Run a recipe by name or by path to someone's shared recipe file. Steps are
validated before anything runs. See [Recipes](recipes.md).

---

## Sessions, viewer, batch

### `list_sessions(session_id=None)`
All sessions, or one session's full data: both analyses, chain + rationale,
deltas, timeline, compliance — the same data the viewer shows.

### `open_viewer(session_id=None)`
Starts (if needed) and opens the local A/B viewer.

### `improve_folder(dir_path, mode="improve", profile="auto", out_dir=None)`
Batch: every audio file in a folder through `improve` | `refine` |
`optimize`. Each file becomes its own session.

### `rate_audio(label, session_id | file_path, note="")`
Train the taste model: label results `good`/`bad`; from 2+2 examples,
`analyze_audio` scores new material against your taste.

---

## Chain steps (for `apply_chain` and recipes)

`apply_chain(file_path, steps)` runs an explicit list of steps; the same
step objects live inside recipes. Available types:

| Step | Purpose | Key parameters (defaults) |
|---|---|---|
| `highpass` / `lowpass` | rumble / hiss cut | `freq`, `q` (0.707) |
| `notch` | hum, whistle | `freq`, `q` (30) |
| `eq` | biquad chain | `bands`: list of `{type, freq, gain_db, q}` |
| `gain` | static gain | `gain_db` |
| `declip` / `declick` | restoration | `max_gap_ms` (4) / `threshold` (6) |
| `denoise` | broadband NR | `strength_db` (12), `method` (spectral\|ai) |
| `smart_denoise` | segment-aware NR | per-kind strengths |
| `deess` | sibilance (speech segments) | `strength_db` (8), `sensitivity` (2.2) |
| `dereverb` | ClearVoice, speech segments | — |
| `breath_control` | dim breaths, don't cut | `reduction_db` (10) |
| `deplosive` | p/b-pop repair, pop-local | `cutoff_hz` (120), `sensitivity_db` (6) |
| `duck_music` | music beds under speech level | `gap_db` (6) |
| `band_duck` | dynamic low-band taming | `low_hz`, `high_hz`, `max_cut_db` |
| `pause_duck` | broadcast silence in pauses | `duck_db` (20) |
| `gate` | noise gate | `threshold_db`, `range_db` |
| `compressor` | soft-knee dynamics | `threshold_db`, `ratio` (3) |
| `leveler` | gain riding to a common level | `target_db` (−18), bounds |
| `limiter` | look-ahead brickwall | `ceiling_db` (−1.5) |
| `loudness_normalize` | BS.1770 + TP limiter | `target_lufs`, `true_peak_db` |
