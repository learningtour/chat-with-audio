# Gap analysis & build plan

*Written July 12, 2026, against Serge's full post-production taxonomy (32
categories, ~900 operations). Status at time of writing: 30 MCP tools, 110
tests. This document is the continuation plan — any future session picks up
here. Scope decision by Serge: live production (his ch. 29) is out of scope
for the tool; capture it as knowledge, not features.*

## How to read this

Per category: **✓ covered** (exists and tested), **± partial**, **✗ gap**
(worth building), **— out of scope** (not this tool's job; the chat + a DAW
do it better, or it's creative human work). The phases below turn the gaps
into an ordered build plan.

## Category status (Serge's numbering)

| # | Category | Status | Notes |
|---|---|---|---|
| 1 | Prep & organisation | ± | SRC/bit depth/format conversion ✓, audio-from-video ✓ (ffmpeg path), tech check ✓ (qc_report/qc_folder). Gaps: batch rename, session archive/consolidate. — : DAW project organisation (tracks, colors, templates, handles) |
| 2 | Synchronisation | ± | Waveform auto-sync ✓ (32-track, two-pass, drift ✓). Gaps: BWF timecode read/sync, simple shift/offset tool, transient alignment for mic pairs. — : lipsync-to-picture (no video) |
| 3 | Select & edit | ✗ | The big one: **text-based editing** (Whisper word timestamps → cut list), filler-word removal, pause shortening, silence strip. ✓ already: breath dimming, room-tone insert. — : comping, beat/phrase music edits |
| 4 | Repair & restoration | ✓ | Denoise (2 tiers), hum+harmonics, declip/declick, spectral repair, dropouts→room tone, dereverb, de-ess, resonances, DC, source separation. Gaps: noise-profile-from-selection, codec-artifact repair, wow/flutter (archival) |
| 5 | Volume & dynamics | ± | Comp/gate/limiter/leveler/loudness ✓, ducking ✓ (2 modes). Gaps: **expander, multiband comp, transient shaper, parallel/upward comp** as chain steps |
| 6 | EQ & tone | ± | Biquad set ✓, match-EQ ✓, auto-EQ rules ✓. Gaps: **dynamic EQ, mid-side EQ, tilt-EQ step**, formant correction; futz-EQ (telephone/radio) belongs in recipes (phase D) |
| 7 | Speech & dialogue | ± | De-ess/deplosive/breath ✓, leveling ✓, isolate ✓. Gaps: boom-lav auto-mix (aligned sum + spectral match), ADR room-match (needs phase D), **word/pause edits (phase A)**. — : ADR recording |
| 8 | Vocal (music) editing | — | Tuning/comping/harmonies = music-production DAW work. Only pitch/formant primitives (phase B) as enablers |
| 9 | Musical timing & pitch | ✗ | **No time/pitch engine at all.** Phase B: time-stretch, pitch-shift, varispeed, formant-preserve |
| 10 | Stereo & spatial | ✗ | Gaps (all cheap): to-mono/dual-mono, channel swap/remap, polarity invert, sample delay, M/S width, pseudo-stereo, bass-mono. Meters exist ✓ |
| 11 | Surround & immersive | ± | 5.1 ✓ (weighted loudness, QC, downmix TP, netflix-5.1). Gaps: 7.1, LtRt downmix render, channel remap. — : Atmos object authoring (Dolby toolchain; ADM recognition ✓) |
| 12 | Reverb & acoustics | ✗ | **No reverb engine.** Phase D: convolution reverb (IR), room-match for ADR, dereverb ✓ exists |
| 13 | Delay & echo | ✗ | Phase D: simple delay step (slapback/tempo) — low priority for post |
| 14 | Modulation FX | — | Music sound-design; not post-critical |
| 15 | Distortion & character | ± | Phase D: saturation + speaker-sim/futz for worldizing; rest — |
| 16 | Creative/experimental | — | Sound-design artistry; the chat can improvise via apply_chain |
| 17 | AV sound design | — | Creative work; **futzing/worldizing** is the tool-shaped part (phase D) |
| 18 | Foley | — | Performance work; foley cleanup = existing repair tools |
| 19 | Music production | — | Except stems ✓, karaoke ✓ |
| 20 | Mixing | ± | A/B ✓, mono/downmix checks ✓, stems export ✓ (sync/Audition). Gaps: M&E / DME stems (phase F), mix-minus. — : full mixing UX |
| 21 | Broadcast | ✓ | R128/A85/dialog-gated ✓, compliance report ✓. Cheap adds: OP-59/ARIB specs, **tone & slate / two-pop generator**, head/tail trim to first/last modulation |
| 22 | Radio & podcast | ± | Podcast recipes ✓, double-ender sync ✓. Gaps (phase A/E): filler words, **bleep/redact**, chapter markers + ID3, silence shortening |
| 23 | Mastering | ± | Loudness/TP ✓, album consistency via match_reference ✓. Gaps: **dither + noise shaping on bit reduction (correctness!)**, M/S. — : DDP/ISRC/vinyl |
| 24 | Loudness & metering | ✓ | Full set incl. momentary/LRA/TP/correlation/dropouts/hum. Gaps: THD, spectrogram-in-viewer exists ✓ |
| 25 | Phase & polarity | ✗ | All cheap (phase C): polarity invert, sample delay, mic-pair alignment (reuse sync refine) |
| 26 | Noise & silence | ± | Gate ✓, room tone ✓. Gap: scene noise matching (floor match between takes) |
| 27 | Online video & social | ± | Loudness targets ✓. Gaps: **codec preview** (AAC/MP3 roundtrip + diff report), loopable audio check |
| 28 | Accessibility & language | ± | Clean dialogue via separation ±. Gaps: bleep/redact (phase A), voice anonymize (needs phase B pitch). — : dubbing/AD production |
| 29 | Live production | — | **Out of scope by decision.** Action: ingest this chapter as knowledge (knowledge-ingest MCP, not connected in the build session — do this from a session that has it) |
| 30 | Conversion & delivery | ± | WAV/FLAC/MP3/AAC/OGG ✓, SRC/bit depth ✓, poly-wav ✓. Gaps (phase E): **BEXT/iXML + timecode metadata, ID3/chapters, AC-3/E-AC-3 via ffmpeg, checksums, delivery package bundler** |
| 31 | Quality control | ✓ | qc_report/qc_folder cover most; add codec preview + downmix render checks |
| 32 | Automation & workflow | ± | Batch ✓ (improve/qc/sync folders), recipes ✓, transcription ✓. Gaps: text-based editing (phase A), batch rename, session archive. — : AAF/OMF/EDL interchange, watch folders |

## The build plan (phases, in order of value)

**Phase A — Text-first dialogue editing (the killer feature).**
Whisper word-level timestamps → `edit_speech` tool: remove filler words
("eh/uh", doubles), shorten pauses to a target, delete/keep spans by
transcript text, **bleep/redact** named words, export the cut list as
markers. Crossfades + room-tone fill at every joint (both exist). This
single phase covers the biggest cluster of Serge's ch. 3/7/22/28 items and
is pure post-production chat magic: *"haal de uhs eruit en maak de pauzes
strakker"*.

**Phase B — Time & pitch engine.**
One good dependency (evaluate: signalsmith-stretch python bindings, else
librosa/pyrubberband) → chain steps `time_stretch`, `pitch_shift`
(formant-preserve option), `varispeed`. Unlocks ch. 9 basics, voice
anonymization, pause lengthening, music shorten/extend approximations.

**Phase C — Cheap utility steps (one batch, many list hits).**
`trim` (head/tail/to-first-modulation, frame offset, insert silence),
`polarity_invert`, `sample_delay`, `expander`, `multiband_compressor`,
`transient_shaper`, `mid_side` (width, M/S gains), `to_mono`/`dual_mono`/
channel remap, bass-mono, **dither + noise shaping** in save_wav on bit
reduction, tone & slate / two-pop generator, OP-59 + ARIB specs.

**Phase D — Convolution & futz.**
Convolution engine (IR wav in → reverb step), ADR/room-matching (measure
target room tail → match), futz recipes: telephone, radio/portofoon,
megaphone, other-room (worldizing light), speaker sim. Makes ADR-match and
scene-match real.

**Phase E — Delivery & metadata.**
BEXT/iXML chunk writer (timecode, originator) for broadcast WAV; ID3 +
chapter markers for podcast; AC-3/E-AC-3 encoders via ffmpeg; checksum
(md5) + `delivery_package` bundler (masters + stems + qc_report + loudness
report in one folder); codec preview (encode→decode→diff + verdict).

**Phase F — Stems & versions.**
M&E / DME stems via separation (honest best-effort labeling), mix-minus,
boom-lav auto-mix on the synced recorder set, undipped/dipped stem exports
from sync sessions.

**Continuous:** every phase = feature + tests + commit; counts updated in
docs; demo session in the viewer; honest notes in the report.

## Standing out-of-scope list (don't re-litigate)

DAW editing UX (comping, beat editing, arrangement), music tuning/harmony
production, Atmos object authoring/IAB, AAF/OMF/EDL interchange,
video-side lipsync, ADR/foley *recording*, live production (ch. 29 →
knowledge base), creative sound design as a service — the chat improvises
these with apply_chain where possible.

## Open follow-ups from earlier sessions

Serge: real-material validation of the whole new suite; musical "ZONDER
rumble" verdict; open the .sesx in Audition. Tool: README hero screenshot,
PyPI + version bump, Windows run, viewer cache-busting, session
search/cleanup, taste-link, two-pass smart_edit.
