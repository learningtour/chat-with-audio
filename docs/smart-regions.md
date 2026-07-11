# Smart regions

`smart_edit` is the surgical counterpart to `improve_audio`: instead of
processing the whole file, the AI locates *where* on the timeline something
is wrong and treats only those stretches. Everything outside a region (plus
its padding/fade margin) stays **bit-for-bit untouched** — the test suite
asserts exact equality.

## The detectors

All detectors are windowed measurements merged into regions; each region
carries its measured severity and gets its own mini-chain.

| Kind | What it finds | Window | The targeted fix |
|---|---|---|---|
| `hum` | Mains hum (50/60 Hz + harmonics) that comes and goes — fridge, dimmer, ground loop | 4 s / hop 2 s, Welch prominence of the fundamental + harmonics | Notch filters at f₀, 2f₀, 3f₀ — only there |
| `noise` | Noise floor rising above the file's cleanest ambience — AC kicking in, traffic | 2 s / hop 1 s, 10th-percentile frame level per window vs the file's global floor | Denoise only there: DeepFilterNet when the region touches speech (and the `ai` extra is present), spectral gating otherwise; strength follows the measured excess (6–18 dB) |
| `clip` | Clusters of clipped peaks | Sample-exact, clustered with ≤ 0.5 s gaps | Declip (spline reconstruction) around the damage |
| `boom` | Low-frequency rumble dominating a stretch — passing truck, table bump | 1 s / hop 0.5 s, 30–160 Hz level vs the file's own low-band median | Highpass + lowshelf cut, depth follows the excess |

## Calibration details that matter

These came out of live testing, not theory:

- **The reference floor is clamped to −80 dB.** Digitally silent material
  (generated audio, hard-gated edits) would otherwise drag the "cleanest
  ambience" to −200 dB and make every real noise floor look like a +160 dB
  problem.
- **Continuous music doesn't false-positive the noise detector.** A window
  needs measurable pauses (10th percentile well below the 90th) or must be
  quiet-but-noisy overall before its floor is trusted.
- **Boom regions fully inside a hum region are dropped.** 50 Hz hum *is*
  low-frequency energy; the notch already handles it, treating it twice
  would be pointless.
- **A dropout must interrupt the waveform mid-cycle.** Gates, fades and
  sentence endings land on zero crossings; only abrupt mid-signal cuts count
  (this lives in the analyzer, but smart_edit's region map builds on it).

## Applying with crossfades

Each region is padded (≥ 250 ms), processed through its mini-chain, and
blended back with raised-cosine crossfades (default 80 ms). Regions apply
sequentially on the running result, so overlapping regions of different
kinds compose. A region whose chain fails is skipped with a warning — one
bad region never blocks the rest.

## Where the region map goes

- **The result** — every region with timestamps, diagnosis, severity and the
  exact steps applied; the rationale reads like an edit log:
  *"0:02–0:12: mains hum around 50 Hz (+26 dB) — notch filters only here."*
- **The viewer** — the *interventions* timeline lane, color-coded per kind;
  click a region to jump there, press **r** to hear what was removed.
- **Your DAW** — `export_markers` turns the map into Audition marker CSV and
  an Audacity label track.

## Honest limitations

- Detectors are tuned conservatively: they prefer missing a mild problem
  over mangling a good recording. Thresholds are documented in
  `regions.py` and easy to adjust.
- `noise` regions treat the *stretch* where the floor is elevated; they
  don't chase noise under continuous loud music (nothing reliable to
  measure there).
- Sibilance and plosives are handled by their own tools (`deess`,
  `deplosive`) because they're frame-scale, not region-scale, phenomena.
