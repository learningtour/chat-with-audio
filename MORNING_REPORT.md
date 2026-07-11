# Morning report — overnight build, July 12/13, 2026

Good morning! Tonight's assignment: make it a beautiful project, clean up the
code on GitHub, expand the capabilities, let people reuse what worked, make
the interface more professional, and process *parts* of the audio chosen by
smart AI instead of effects over the whole file. All five done, all pushed,
**53 tests green**, lint clean, CI running on GitHub.

## The headline: smart_edit — surgery instead of mastering

The new `smart_edit` tool (the 22-tool set now) finds problem regions on the
timeline and treats *only* those, with crossfades; everything outside stays
bit-for-bit untouched (the tests assert exact equality):

| Detector | Finds | Fix, only there |
|---|---|---|
| **hum** | mains hum that comes and goes (fridge, dimmer) | notches at 50/60 Hz + harmonics |
| **noise** | noise floor rising above the file's cleanest ambience (AC, traffic) | DeepFilterNet on speech, spectral gating elsewhere |
| **clip** | clusters of clipped peaks | declip around the damage |
| **boom** | low-frequency rumble dominating a stretch (passing truck) | highpass + lowshelf cut |

Ask it in the chat: *"fix it only where something is wrong"*. The region map
appears in the result and as a timeline in the viewer.

## Recipes — reuse what worked

"Save this as my podcast preset" now works: `save_recipe` keeps the chain of
a session that sounded right as a small JSON file, `apply_recipe` runs it on
new files (also from a shared file path — recipes are made to be passed
around), `list_recipes` shows everything. Four curated presets ship built in,
including **podcast-speech**, distilled from the previous night's winner
("calm-no-ai", the chain the Whisper jury preferred).

## The viewer got a design pass

"Chat with Audio" branding, refined dark studio theme, round play button,
segmented A/B/R control, score badges (before → after), type chips in the
session list, keyboard-hints footer — and a **timeline** under the waveforms:
a content lane (speech/music/silence) plus an interventions lane showing
exactly where smart_edit treated what, color-coded, click to jump there.
Verified live in the browser: A/B/R via click and keyboard, timeline seeks,
old sessions unaffected, console clean.

## Repo hygiene

MIT LICENSE, ruff lint (configured and fully clean), GitHub Actions CI
(uv sync + ruff + pytest on ubuntu and macos), pyproject metadata (v0.2.0),
README badges and updated docs.

## Honest notes

- Live testing caught three real bugs, all fixed tonight: two sessions on the
  same file within one second overwrote each other; digital-silence files made
  the noise detector report absurd "+163 dB" excesses (reference floor now
  clamps at -80 dB); and the boom detector double-treated hum regions (50 Hz
  *is* low-frequency energy — now suppressed when a hum region covers it).
- The detectors are calibrated on synthetic material and one demo file. Real
  recordings will sharpen the thresholds — feed smart_edit your archive.
- A demo session pair ("nachtdemo-interview") is in the viewer: one surgical,
  one with the podcast-speech recipe, on the same file with hum, noise and
  rumble planted at known spots.

## Where everything lives

- Viewer: http://127.0.0.1:8471 ("open the viewer" in the chat) — the
  nachtdemo session shows the new timeline
- Recipes: `~/AudioImprove/recipes/` + built-ins via list_recipes
- Roadmap + status: `NIGHT_ROADMAP.md`; user docs: `README.md`
- Still open from before: your verdict on the musical "WITHOUT rumble"
  version (session 20260711-100317) and opening the Audition .sesx

## Ideas for a next session (not built, thought through)

1. smart_edit for sibilance/plosive regions (the de-esser is global-per-frame
   now; region-level would let it act only where a mic position changed).
2. Recipe "taste link": rate_audio labels feeding recipe suggestions —
   "files like this usually sound right with podcast-speech".
3. Two-pass smart_edit: fix narrowband problems (hum/clip) first, re-measure,
   then judge broadband (noise/boom) on the intermediate — fewer overlapping
   treatments.
4. A screenshot in the README (the new viewer is worth showing off).
