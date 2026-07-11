# Night roadmap — July 13/14, 2026

Assignment from Serge: "Judge the software as a sound designer — is it
broadcast and Hollywood proof? Make the tool 200% better; it should become
the industry standard for AI audio editing. It's too small and experimental
for that now. Also write extensive English documentation. You have 8 hours."

## The sound designer's verdict (drives tonight's plan)

Musical and clever, but not yet a broadcast/Hollywood tool. Missing, in order
of professional pain:

1. **Delivery compliance** — no EBU R128 / ATSC A/85 / Netflix / streaming
   spec checking, no dialogue-gated loudness, no momentary/short-term
   ceilings, no pass/fail QC report. A mixer cannot deliver with this alone.
2. **Stereo & technical QC** — no phase correlation, no dead-channel /
   dual-mono / polarity detection, no dropout scan, no head/tail silence
   check. These are the first things a QC department runs.
3. **Dialogue editing depth** — no breath control, no plosive repair, no
   music-under-dialogue ducking. That's the daily bread of film post.
4. **DAW interoperability** — the AI region map stays in the viewer; a
   pro wants it as markers inside Audition/Audacity/Pro Tools.
5. **Delivery-grade I/O** — no sample-rate conversion or bit-depth control
   on export (48 kHz / 24-bit is the broadcast lingua franca).
6. **Documentation** — a README is not industry-standard docs.

## Status

- [x] 1. Pro metering & QC: momentary/short-term LUFS max, PLR, stereo QC
      (correlation, dual-mono, polarity, balance), dropout scan, head/tail
      silence — into metrics + issues, with tests
- [x] 2. Compliance suite: spec registry (EBU R128, ATSC A/85, Netflix
      dialogue-gated, Spotify, Apple, YouTube, ACX audiobook),
      check_compliance (pass/fail per criterion), master_for (master to spec,
      re-verify, delivery export 48 kHz/24-bit) + viewer panel
- [x] 3. Dialogue suite: breath_control, deplosive and duck_music chain
      steps + dialogue-polish recipe, with tests
- [x] 4. DAW export: export_markers (Audition CSV, Audacity labels, JSON)
      from the region timeline; delivery I/O (SRC + bit depth)
- [x] 5. Extensive English docs under docs/ (getting started, tool
      reference, workflows cookbook, compliance guide, smart regions,
      recipe format, architecture) + README links
- [x] 6. Morning report, memory, everything pushed, CI green

Bugs found by building and fixed on the spot: dropout detector vs
gating/fades (zero-crossing boundary rule), breath floor vs digital
silence (clamped at -75 dB), envelope smoothing dragging file edges to
zero (edge padding).

## Rules (unchanged)

- Python 3.11, numpy<2, torch<2.9 — don't touch the pins (see CLAUDE.md).
- Never pollute stdout in server code; lazy-load heavy models.
- New MCP tools also go into tests/test_mcp_tools.py EXPECTED and scripts/mcp_smoke.py.
- After each feature: uv run pytest -q green -> git commit.
