# Recipes

A recipe is a saved processing chain as a small JSON file — reusable,
versionable, shareable. The natural flow: a session sounds right → *"save
this as my podcast preset"* → apply it to every next file, or hand the JSON
to a colleague.

## Format

```json
{
  "format": "chat-with-audio/recipe@1",
  "name": "my-show",
  "description": "House sound for The Show: calm chain, no AI post.",
  "created": "2026-07-13 02:10:44",
  "source_session": "20260713-021044-episode-041",
  "steps": [
    {"type": "highpass", "freq": 80},
    {"type": "deess"},
    {"type": "leveler", "target_db": -18, "max_boost_db": 20, "max_cut_db": 18},
    {"type": "compressor", "threshold_db": -10, "ratio": 3},
    {"type": "loudness_normalize", "target_lufs": -16, "true_peak_db": -1.5}
  ]
}
```

- `steps` uses exactly the `apply_chain` step vocabulary — see the
  [tool reference](tools.md#chain-steps-for-apply_chain-and-recipes).
- Steps are **validated before anything runs**, on save *and* on load: an
  unknown step type or parameter is rejected with the list of valid options.
  A shared recipe can't silently do something else than it claims.
- `source_session` records provenance when the recipe was saved from a
  session.

## Locations & precedence

| Where | What |
|---|---|
| `src/chat_with_audio/recipes/` (in the package) | Built-in presets |
| `~/AudioImprove/recipes/` (`AIT_RECIPES_DIR`) | Your recipes |

A user recipe with the same name overrides the built-in — so you can fork
`podcast-speech` into your own house variant under the same name.

## Built-in presets

| Recipe | When |
|---|---|
| `podcast-speech` | The calm speech chain (distilled from the variant that won the Whisper word-retention jury: no AI post-processing) |
| `dialogue-polish` | Film dialogue pass: plosive repair, breath dimming, de-esser, −16 LUFS |
| `broadcast-quiet-pauses` | Broadcast silence between sentences without gate artifacts |
| `noisy-speech-rescue` | Heavier chain for genuinely noisy speech (segment-aware denoise) |
| `music-master` | Careful music master: subsonic cut + −14 LUFS / −1 dBTP |
| `futz-telephone` | Narrowband phone line: 300–3400 Hz, presence peak, tape drive, firm comp |
| `futz-walkie` | Portofoon/walkie: even narrower, hard clip, crushed dynamics, gate |
| `futz-megaphone` | Horn resonance at 1.2 kHz, hard clip, a whiff of outdoor slap |
| `futz-other-room` | Worldizing light: voice from the room next door (damped highs + room + −6 dB) |
| `futz-small-speaker` | Transistor radio / laptop / smart speaker: no lows, tinny presence, soft drive |

## Sharing

A recipe **is** its file. Send `my-show.json` to anyone;
they run `apply_recipe(file, recipe="/path/to/my-show.json")` — validation
happens on load. There is no registry, no account, no lock-in: recipes are
plain text you can diff, review and version-control.

## What doesn't fit in a recipe

Surgical `smart_edit` sessions can't be saved as recipes on purpose: their
chains reference regions of *that specific file*. The regions are
re-detected per file — that's the whole point. `save_recipe` explains this
when you try.
