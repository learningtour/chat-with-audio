# Morning report — overnight build, July 10/11, 2026

Good morning! Overnight the toolkit grew from "improving audio" into a
full-fledged chat-driven audio studio with **15 MCP tools**. Everything is
tested (33 tests green) and committed per feature.

## New tonight

| Feature | Ask in the chat | Detail |
|---|---|---|
| **Declip + declick** | "fix the clips" | Waveform reconstruction via splines; improve now declips automatically (your musical file has 70 real clips!) |
| **Stem separation** | "split the stems" | Demucs AI: vocals / drums / bass / other as separate wavs for your DAW |
| **Rebalance / karaoke** | "vocals up 3 dB", "make a karaoke version" | Per-stem gains, dynamics-safe gain staging, A/B session |
| **Residual listening** | button **R · difference** in the viewer (key r) | Hear exactly what the processing changed — the artifact detector for your ears; also works on all old sessions |
| **Reference matching** | "make this sound like <reference>" | 1/3-octave match EQ (bounded) + loudness match; for consistent episodes |
| **De-esser** | automatic on speech | Spectral, only attenuates frames where s-sounds genuinely spike |
| **Resonance detection** | automatic | Narrow peaks (boxy room resonances) are detected and removed surgically |
| **Batch processing** | "do the whole folder" | improve_folder: improve/refine/optimize per file |
| **Whisper-medium referee** | optimize_audio(judge_model="medium") | Stricter intelligibility jury for overnight runs |

## The overnight run on your test file — with a surprise

The deep run with the **stricter Whisper-medium jury** flips the ranking:

| Variant | Retention (medium jury) |
|---|---|
| **calm-no-ai** (winner, session 20260711-000738) | **75%** |
| basic-no-ai | 66% |
| dereverb-deess-calm | 53% |
| dereverb variants | 38-47% |

Last night's small jury preferred dereverb; the medium jury (much stronger in
Dutch) hears that dereverb artifacts cost words. Lesson: **the calm DSP chain
without AI post-processing is the best on this material** — and the quality of
the jury partly determines the outcome. Both sessions are in the viewer;
listen for yourself and let your ears decide. You can now also record your
verdict with rate_audio (which trains the taste model).

## Built later in the night (your 3 directions)

1. **view_audio** — a perceptual panel that I as an AI can inspect myself:
   auditory-scale spectrograms, difference map (red = added, blue = removed),
   level curves. Self-test done: I could point out the leveling, ducking and
   highpass in it directly.
2. **rate_audio + taste model** — label 'good'/'bad'; from 2+2 examples onward
   every analysis gets a taste_score explaining which properties deviate from
   your 'good' examples.
3. **export_to_audition** — stems (Demucs) + .sesx multitrack session for
   Adobe Audition 2024 (found on this Mac). A demo is ready in the session
   folder of the overnight winner (audition/), not opened yet.

## Where everything lives

- Viewer: http://127.0.0.1:8471 ("open the viewer" in the chat)
- Sessions: `~/AudioImprove/sessions/` — newest on top in the viewer
- Roadmap + status: `NIGHT_ROADMAP.md`; user docs: `README.md`
- Note: restarting Claude Desktop + a one-time approval in Claude Code are
  still needed to see the tools there (the config is ready)

## Honest notes

- The R button (residual) has been verified via HTTP and a syntax check, but
  not yet listened to in a real browser (the Chrome extension wasn't connected
  during the night).
- Stem separation on the musical file treats spoken dialogue as "vocals" —
  that's correct Demucs behavior, but takes some getting used to.
- Dereverb (ClearVoice) only runs on speech segments; music dereverb is
  deliberately off (it wrecks the mix).

## Ideas for the next session (not built, but thought through)

1. Preference memory: "I like B better" → the tool learns your taste as a preset.
2. Speech super-resolution (ClearVoice SR model) for old/dull recordings.
3. Multitrack export (stems + improved mix) as a session zip for production houses.
4. Windows test + publishing the GitHub repo if you want — the trade press, remember.
