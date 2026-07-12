# Tool reference

All 30 MCP tools, grouped by job. Every processing tool creates a **session**
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
channel, dual-mono, polarity inversion). **5.1 files** additionally get
weighted surround loudness (LFE excluded), per-channel levels, dead-channel
detection and the ITU stereo-downmix true peak. Returns metrics, 0-100 scores
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

### `qc_report(file_path, spec=None, out_path=None)`
One readable markdown QC sheet per file — the report a facility wants to see
before accepting a delivery: file info, loudness measurements, technical QC,
findings with severity, and (with `spec`) the delivery compliance check.
Saved into the session, exportable, returned inline.

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

### `match_room(file_path, reference_path, mix=0.35, rt60=None, eq_strength=1.0, out_path=None)`
ADR/room-match: make a dry (studio) line sit in the same room as a scene
reference. Two moves: match-EQ toward the reference's spectral colour
(mic + room), then convolution with a synthesized room whose decay time is
*measured* from the reference (Schroeder-style RT60 estimate on speech
offsets; pass `rt60` to override). Honest note: this matches colour and
decay, not exact reflection patterns — for ADR fitting that is usually
exactly enough. Too much room? Re-run with a lower `mix`.

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

### `fill_room_tone(file_path, out_path=None)`
The dialogue editor's classic: samples the file's own quietest real ambience
and fills digital gaps (dropouts, edit holes, ADR joins) with shuffled,
overlap-added pieces of it — continuous room, never a loop. Only
exact-silence gaps are filled; everything else stays bit-for-bit untouched.

### `spectral_repair(file_path, start_s, end_s, low_hz=None, high_hz=None, out_path=None)`
RX-style spectral painting: point at a cough, chair squeak, tick or thump
(time range up to 5 s, optional frequency band) and the patch is repainted
from its context — per-bin magnitudes interpolated between the left and
right neighbourhood, phase continued coherently (phase-vocoder style) so
tonal content runs straight through the repair. Outside the patch:
bit-for-bit untouched. Find the spot with `view_audio` (vertical
streaks/blobs in the spectrogram) or by ear in the viewer. For damage over
or next to programme — not for conjuring back lost words.

---

## Edit (text-first)

### `edit_speech(file_path, remove_fillers=True, remove_doubles=True, tighten_pauses_to_s=None, remove_text=None, bleep_text=None, bleep_style="tone", language="nl", model_size="small", apply=True, crossfade_ms=12, out_path=None)`
Text-based dialogue editing: *"haal de uhs eruit en maak de pauzes
strakker."* Transcribes with word-level timestamps (Whisper, `asr` extra),
then edits on the transcript:

- `remove_fillers` — filler sounds (eh/uh/ehm/uhm…, per-language lexicon)
- `remove_doubles` — immediate word doubles ("ik ik ga"): the first
  instance goes, the final delivery stays
- `tighten_pauses_to_s` — shorten every inter-word pause to this maximum;
  the head and tail of each pause survive, so breathing room stays real
- `remove_text=["frase", …]` — cut phrases by transcript text (all
  occurrences, punctuation/case-insensitive)
- `bleep_text=["naam", …]` — redact words with a 1 kHz bleep
  (`bleep_style="mute"` silences instead); length-neutral

Every joint gets a raised-cosine crossfade. Cuts shorten the file, so the
A/B in the viewer drifts after the first cut — the timeline lane shows the
cuts on the original timeline, and `export_markers` turns the cut list into
DAW markers. `apply=False` returns the edit plan only (with transcript
context per cut) so you can confirm before committing; the result also
reports phrases that were *not* found in the transcript.

---

## Multitrack (the 32-track recorder)

### `sync_tracks(file_paths | dir_path, reference=None, correct_drift=False, out_dir=None)`
Link multiple recorders: up to **32 tracks** (lav, boom, field recorder,
camera audio, phone) are aligned on the audio content itself —
envelope-GCC-PHAT for the coarse offset, full-rate refinement for sample
accuracy. Every track gets a confidence score (a file without shared audio
is flagged, never silently misplaced), and `correct_drift` measures and
corrects clock drift between recorders (ppm). Output: aligned WAVs on one
common timeline, an Audition `.sesx` with all tracks in place, and an A/B
session where A is the unsynced sum (echo soup) and B the synced sum — you
*hear* the sync. Known limitation: strictly periodic material (metronomes,
click tracks) is inherently ambiguous for correlation-based sync.

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
broadcast delivery file (high-quality polyphase SRC). Specs with a format
requirement fill those in automatically: *"master this for Netflix as
delivery.wav"* comes out dialogue-gated at −27 LKFS, TP ≤ −2 dBTP, 48 kHz /
24-bit PCM — and the re-check is run on the actual delivery file.

### `export_markers(session_id, out_dir=None, include_segments=False)`
The AI region map as DAW markers: Adobe Audition marker CSV, an Audacity
label track and `markers.json`. *"Hum here, noise there"* becomes navigable
markers inside your editor.

### `codec_preview(file_path, codecs=["mp3","ogg","opus"])`
What will lossy compression do to this master? Encode → decode through the
real codecs (libsndfile — no ffmpeg needed) and measure: loudness shift,
true-peak overshoot (**codec overs** — the reason streaming specs demand
−1…−2 dBTP) and residual level, with a verdict per codec. If a codec
clips, the fix is `master_for` with a lower true-peak ceiling.

### `write_bwf_metadata(file_path, description, originator, timecode="HH:MM:SS:FF", fps=25, project, scene, take, note, coding_history)`
Broadcast-WAV metadata, written in place: **bext** (EBU 3285 — originator,
date/time, TimeReference from timecode, coding history) and **iXML**
(project/scene/take). This is what turns bare PCM into a *broadcast wav*
that Avid/Audition/Resolve read. Audio data stays bit-for-bit untouched.

### `export_podcast_mp3(file_path, out_path, title, artist="", album="", chapters=None)`
Podcast MP3 with ID3v2.3 **chapters** (CHAP/CTOC): players like Apple
Podcasts and Overcast show the titles and make the timeline clickable.
`chapters=[{"start_s": 0, "end_s": 62.5, "title": "Intro"}, …]` — let the
chat derive them from the transcript or session segments.

### `delivery_package(file_path | session_id, spec=None, out_dir=None, name=None, include_mp3=False)`
The folder you actually send: master + `qc_report.md` (+ spec check +
`compliance.json`), DAW markers if the session has a region map, optional
MP3 listening copy, and `checksums.md5` + `manifest.json` so the receiving
side can verify the delivery (`md5sum -c checksums.md5`).

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

### `qc_folder(dir_path, spec=None, out_path=None)`
Batch QC: audit a whole directory (incoming deliveries, an archive) in one
command — per file the full analysis + findings and optionally the
delivery-spec check, summarised as a markdown index table with verdicts.
Unreadable files become error rows instead of breaking the batch.

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
| `duck_music` | music under speech level | `gap_db` (6), `mode`: `beds` (between speech) \| `stems` (sidechain via Demucs, under speech) |
| `band_duck` | dynamic low-band taming | `low_hz`, `high_hz`, `max_cut_db` |
| `pause_duck` | broadcast silence in pauses | `duck_db` (20) |
| `gate` | noise gate | `threshold_db`, `range_db` |
| `compressor` | soft-knee dynamics | `threshold_db`, `ratio` (3) |
| `leveler` | gain riding to a common level | `target_db` (−18), bounds |
| `limiter` | look-ahead brickwall | `ceiling_db` (−1.5) |
| `loudness_normalize` | BS.1770 + TP limiter | `target_lufs`, `true_peak_db` |
| `expander` | soft gate: push quiet down | `threshold_db` (−45), `ratio` (2), `range_db` (24) |
| `multiband_compressor` | LR4 bands, per-band comp | `crossovers` ([200, 2000]), `threshold_db`, `ratio` |
| `transient_shaper` | attack/sustain colour, no threshold | `attack_db`, `sustain_db` |
| `tilt_eq` | tilt spectrum around a pivot | `tilt_db` (+ = brighter), `pivot_hz` (650) |
| `trim` | head/tail cut, or auto to modulation | `start_s`/`end_s`, or `to_modulation` + `threshold_db` (−60), `pad_s` (0.5) |
| `insert_silence` | insert gap / head offset | `at_s` (0), `duration_s` (1) |
| `polarity_invert` | flip phase | `channel` (all\|left\|right\|index) |
| `sample_delay` | delay one channel (mic-pair align) | `channel`, `samples` or `ms` (negative = advance) |
| `to_mono` / `dual_mono` | downmix / L=R delivery | `mode`/`source` (sum\|left\|right) |
| `swap_channels` | swap L and R | — |
| `mid_side` | width & M/S gains | `width` (1.0; 0 = mono), `mid_db`, `side_db` |
| `bass_mono` | mono the low end (LR4 split) | `freq` (120) |
| `tone_slate` | broadcast leader: ref tone + gap | `tone_s` (10), `level_db` (−18), `freq` (1000), `gap_s` (1) |
| `two_pop` | sync pop before programme | `offset_s` (2), `pop_ms` (42), `level_db` (−18) |
| `convolve_ir` | convolution reverb: IR file or synthesized room | `ir_path` (else synth), `mix` (0.3), `rt60` (0.4), `damping` (0.35), `predelay_ms` (8), `keep_tail` (False) |
| `saturate` | tape/soft/hard character & futz | `drive_db` (6), `mode` (tape\|soft\|hard), `mix` (1.0) |
| `delay` | slapback/echo (feedback comb) | `time_ms` (120), `feedback` (0.3), `mix` (0.25) |
| `time_stretch` | duration without pitch (phase vocoder, peak-locked) | `rate` (1.25 = 25% faster/shorter; ~0.5–2.0 usable) |
| `pitch_shift` | pitch without duration | `semitones` (±24), `preserve_formants` (False; True keeps the voice character — no Mickey Mouse) |
| `varispeed` | tape-style: tempo + pitch together | `rate` (1.05 = 5% faster and higher) |

Note on `trim`, `insert_silence`, `tone_slate`, `two_pop`, `time_stretch`
and `varispeed`: these change the file's length/offset, so put them at the
very end of a chain (after loudness), and expect the viewer A/B to drift
past the edit point.

16-bit export is TPDF-dithered automatically (high-passed dither, so the
quantization noise lands where the ear is least sensitive); pass
`dither=False` to `io.save_wav` for bit-exact test paths.
