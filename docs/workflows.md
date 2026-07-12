# Workflows cookbook

Real jobs, phrased the way you'd say them in the chat. Every step is a tool
call Claude makes for you; you can always inspect the result in the viewer
before moving on.

---

## Podcast episode, start to finish

1. *"Analyze episode-042.wav"* — check the diagnosis: SNR, hum, sibilance,
   balance.
2. *"Fix it only where something is wrong"* — `smart_edit` removes the AC
   noise that starts halfway, the fridge hum in the intro, nothing else.
3. *"Apply my podcast preset"* — `apply_recipe("podcast-speech")`: the calm
   speech chain (highpass, de-esser, leveler, light compression, −16 LUFS).
4. *"Check it for Apple Podcasts"* — `check_compliance(spec="apple-podcast")`
   → pass/fail per criterion.
5. *"Export as mp3"* — done.

First episode sounded right? *"Save this as my show preset"* — every next
episode is step 3 with your own recipe.

## Broadcast delivery (EBU R128)

1. *"Master this for European broadcast, 48 kHz 24-bit"* →
   `master_for(spec="ebu-r128", out_path="delivery.wav", sample_rate=48000,
   bit_depth=24)`.
2. The compliance report (also in the viewer as the *Aflever-check* panel)
   shows integrated loudness −23 ±0.5, true peak ≤ −1 dBTP, LRA advisory,
   plus the technical gates: clipping, dropouts, dead channel, anti-phase.
3. Not passing because of a technical gate? The report says which tool fixes
   it (`repair_audio` for clipping, `smart_edit` for hum/noise...). Fix,
   then master again.

For US television use `spec="atsc-a85"`; for Netflix-style dialogue-gated
delivery use `spec="netflix-2.0"` (loudness is steered on the detected
speech, not the whole mix).

## Film / documentary dialogue pass

1. *"Analyze take-07.wav"* — look at the issues list first.
2. *"Repair the clips and clicks"* — `repair_audio` if the location sound
   ran hot.
3. *"Polish the dialogue"* — `apply_recipe("dialogue-polish")`: plosive
   repair, breath dimming (10 dB, never cut), de-esser, −16 LUFS.
4. *"There's a chair squeak at 12.3"* — `spectral_repair(12.2, 12.6,
   1500, 6000)`: the squeak's time-frequency patch is repainted from its
   surroundings; the dialogue below runs straight through.
5. *"Fill that edit gap with room tone"* — `fill_room_tone`: digital holes
   get the file's own ambience.
6. *"What exactly changed?"* — press **r** in the viewer: the residual is
   exactly what was removed. If you hear consonants in the residual,
   something was too aggressive — say so and it gets re-run milder.
7. *"Give me the region map as markers"* — `export_markers` → import the CSV
   in Audition; every treated region is a cue for manual review.

## Rescue: noisy location recording

1. *"This is barely usable, take your time"* — `optimize_audio` runs a
   variant tournament (with/without AI denoise, dereverb, leveling) and lets
   Whisper word-retention pick the winner: the version where you understand
   the *most words*, not the one that merely measures quietest.
2. Check the ranking in the report; listen to the top two in the viewer.
3. `rate_audio("good")` on the keeper — the taste model learns what you
   accept.

## Music: quick master and stem work

- *"Master this for Spotify"* — `master_for(spec="spotify")` (−14 LUFS,
  TP ≤ −1).
- *"Vocals 2 dB up, bass 1 dB down"* — `rebalance_music` (Demucs under the
  hood, peak-guarded remix).
- *"Karaoke version"* — `rebalance_music(vocals_db=-60)`.
- *"Make this live recording sound like the studio track"* —
  `match_reference(live.wav, studio.wav)`.

## Multi-recorder shoot (interview, podcast on location, multicam)

Everyone pressed record at a different moment: the lav, the boom, the Zoom
on the table, the camera. Drop them all in one folder and say:

1. *"Sync these recorders"* → `sync_tracks(dir_path=...)`: offsets are found
   on the audio itself (sample-accurate), each track gets a confidence score,
   and a file that doesn't belong is flagged instead of misplaced.
2. Long event? Add `correct_drift=True` — recorder clocks drift apart over
   an hour; the tool measures it in ppm and corrects.
3. Check the A/B in the viewer: A is the unsynced pile-up, B the synced sum.
4. Open the generated `.sesx` in Audition — all tracks aligned on one
   timeline, ready to mix. Or take the aligned WAVs into any DAW.

## Consistency across an archive / season

1. Pick your reference episode.
2. *"Make every file in this folder sound like the reference"* — loop of
   `match_reference`, or *"improve the whole folder"* → `improve_folder`
   (modes: improve / refine / optimize).
3. Spot-check sessions in the viewer; every file has its own A/B.

## Incoming file QC (before you accept a delivery)

*"QC this file"* → `qc_report` catches: clipping, digital dropouts (with
positions), dead or anti-phase channels (mono fold-down kills those on TV
and phones), dual-mono masquerading as stereo, DC offset, absurd head/tail
silence, hum, and loudness/true-peak violations — as one printable sheet.
Two minutes instead of a full listen-through.

A whole batch at once: *"QC this folder against EBU R128"* → `qc_folder`
returns one index table with a verdict per file and writes it as markdown —
the morning pile of incoming deliveries, keured before your first coffee.

Music under the dialogue too loud in a delivered mix? *"Duck the music
under the speech"* → `duck_music(mode="stems")` separates the vocals and
sidechain-ducks the accompaniment — fast attack, slow release, no pumping.
