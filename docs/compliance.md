# Delivery compliance

`check_compliance` measures a file against a delivery specification and
returns a pass/fail report per criterion; `master_for` masters toward the
spec and re-verifies. The report is stored as `compliance.json` in the
session and rendered as the *Aflever-check* panel in the viewer.

## Supported specs

| Spec id | Intended for | Loudness | True peak | Extras |
|---|---|---|---|---|
| `ebu-r128` | European broadcast | −23 LUFS ±0.5 (integrated) | ≤ −1 dBTP | LRA ≤ 20 LU (advisory) |
| `atsc-a85` | US television | −24 LKFS ±2 (integrated) | ≤ −2 dBTP | |
| `op-59` | Free TV Australia | −24 LKFS ±1 (integrated) | ≤ −2 dBTP | |
| `arib-tr-b32` | Japanese broadcast | −24 LKFS ±1 (integrated) | ≤ −1 dBTP | |
| `netflix-2.0` | Netflix-style non-theatrical 2.0 | −27 LKFS ±2 (**dialogue-gated**) | ≤ −2 dBTP | delivery format: 48 kHz / ≥24-bit PCM WAV, 2 ch |
| `netflix-5.1` | Netflix-style non-theatrical 5.1 | −27 LKFS ±2 (**dialogue-gated**, LFE excluded) | ≤ −2 dBTP | delivery format: 48 kHz / ≥24-bit PCM WAV, 6 ch (SMPTE) |
| `apple-podcast` | Apple Podcasts | −16 LUFS ±1 | ≤ −1 dBTP | |
| `spotify` | Spotify normalization target | −14 LUFS ±1 | ≤ −1 dBTP | |
| `youtube` | YouTube normalization target | −14 LUFS ±1 | ≤ −1 dBTP | |
| `acx-audiobook` | ACX / Audible audiobooks | — | — | RMS −23…−18 dB, sample peak ≤ −3 dB, noise floor ≤ −60 dB |

Streaming targets (Spotify, YouTube) are *normalization* targets rather than
hard requirements — platforms turn louder material down — but hitting them
means your master plays back exactly as you mixed it.

## The universal technical gates

Every spec additionally runs the QC gates that any facility checks before
accepting a delivery:

- **Clipping** — zero clip events (fix: `repair_audio`)
- **Digital dropouts** — zero mid-signal exact-silence gaps (reported with
  positions)
- **Channels** — no dead channel in a stereo file
- **Polarity** — channels not anti-phase (mono fold-down would cancel)
- **Head/tail silence** — ≤ 1 s (advisory; trim before delivery)

## How measurements work

- **Integrated loudness** — BS.1770 (pyloudnorm), gated, over the full file.
  **5.1 surround** is measured with the BS.1770 channel weights: L/R/C at
  1.0, the surrounds at 1.41 (+1.5 dB), and the **LFE excluded** — exactly
  what a broadcast loudness meter does.
- **Dialogue-gated loudness** — the official Netflix protocol uses Dolby
  Dialogue Intelligence (licensed Dolby technology, not something we can
  ship). Our measurement works *in the same spirit*: 400 ms BS.1770 blocks,
  gated to the blocks that contain detected dialogue, energy-averaged. On
  5.1 the dialogue is detected on the **center channel** (where dialogue
  lives) and measured over the weighted full mix. Good enough to steer a
  master and to flag problems; the distributor's own QC remains the final
  word — the docs and the tool say so rather than pretending.
- **True peak** — 4× oversampled peak (inter-sample peaks included), dBTP.
- **Momentary / short-term max** — 400 ms / 3 s windows, reported in
  `analyze_audio` for metering; not gated criteria in the current specs.

## Mastering to a spec

`master_for` chooses the correct strategy per spec:

- **Integrated specs** — `loudness_normalize` to target with the limiter
  ceiling 0.3 dB under the spec's true-peak maximum (inter-sample safety).
- **Dialogue-gated specs** — measure speech loudness, apply the static gain
  that puts *dialogue* on target, then a true-peak limiter. Music and
  effects keep their relation to the dialogue (as intended by the spec).
- **ACX** — static gain to the middle of the RMS window with a peak-guard
  limiter for the −3 dB sample-peak requirement.

Delivery files: `out_path` plus `sample_rate` (e.g. 48000) and `bit_depth`
(16 / 24 / 32-float) produce the classic 48 kHz/24-bit broadcast WAV via
high-quality polyphase sample-rate conversion.

```text
"Master this for European broadcast as delivery.wav, 48 kHz 24-bit"
→ master_for(file, spec="ebu-r128", out_path="delivery.wav",
             sample_rate=48000, bit_depth=24)
```

The result includes the fresh compliance report; if a technical gate still
fails (say, dropouts in the source), the report names the tool that fixes it.

## Format requirements (Netflix)

Specs that prescribe a delivery *format* — Netflix wants 48 kHz / ≥24-bit
PCM WAV — get two extra normative checks: **Sample rate** and
**Leveringsformaat** (codec + bit depth, read from the actual file).
`check_compliance` verifies the file you point it at; `master_for` verifies
the actual delivery file it wrote — and when you give it an `out_path`
without explicit `sample_rate`/`bit_depth`, it fills in the spec's format
automatically:

```text
"Master this for Netflix as delivery.wav"
→ master_for(file, spec="netflix-2.0", out_path="delivery.wav")
   # → dialogue-gated to −27 LKFS, TP-limited ≤ −2 dBTP,
   #   exported at 48 kHz / 24-bit PCM, then re-checked — all criteria
```

Without an `out_path` the report honestly flags the source sample rate as
non-compliant and tells you the one argument that fixes it. A mono source
mastered against a 2.0 spec is exported as **dual-mono stereo**
automatically — standard delivery practice.

## Surround (5.1) and Dolby Atmos

5.1 files (6-channel SMPTE WAV) get full treatment: weighted loudness (LFE
excluded), per-channel levels, dead-channel detection, silent-LFE note, and
the true peak of the **ITU stereo downmix** — the classic QC catch (a 5.1
mix that clips when a TV folds it down to stereo). `master_for` masters 5.1
dialogue-gated and delivers 6-channel 48 kHz/24-bit.

**Dolby Atmos, honestly:** the object layer (IAB / ADM metadata, binaural
renders) requires the licensed Dolby toolchain and cannot be validated here.
What Chat with Audio does do: it **recognises ADM BWF masters** (the
axml/chna chunks that carry Atmos metadata) and says so in the report, and
it fully QCs the bed and downmix deliverables that accompany an Atmos
delivery.
