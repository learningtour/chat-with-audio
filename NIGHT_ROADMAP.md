# Nachtroadmap — 10/11 juli 2026

Opdracht van Serge: "Bouw vannacht door. Super geavanceerde audiotoepassingen,
baanbrekend, elk productiehuis wil dit downloaden." Werkwijze: elke feature af
(implementatie + tests + commit) voordat de volgende begint; repo blijft groen.

## Status

- [x] 1. Spectrale reparatie: declip + declick (dsp/repair.py, chain-steps,
      MCP-tool repair_audio, auto-declip in improve bij clip_events > 0)
- [x] 2. Stem-separatie (Demucs htdemucs): separate_stems + rebalance_music
      (incl. karaoke = vocals -60 dB); A/B-sessie voor rebalance
- [x] 3. Residu-beluistering in de viewer: derde knop "R - verschil" =
      loudness-gematcht verschil tussen origineel en bewerking (artefact-check)
- [ ] 4. Reference matching: match_reference(file, reference) - 1/3-octaaf
      spectrale match-EQ (begrensd) + loudness match
- [ ] 5. Auto-de-esser (spectraal, alleen spraaksegmenten) + resonantiedetectie
      -> notches; opnemen in improve-regels
- [ ] 6. Batchverwerking: ait batch + MCP improve_folder
- [ ] 7. Docs (README/CLAUDE.md), mcp_smoke bijwerken, ochtendrapport
      (MORNING_REPORT.md) + memory bijwerken

## Regels

- Python 3.11, numpy<2, torch<2.9 — niet aan pinnen tornen (zie CLAUDE.md).
- Nooit stdout vervuilen in servercode; zware modellen lazy laden.
- Nieuwe MCP-tools ook in tests/test_mcp_tools.py EXPECTED en scripts/mcp_smoke.py.
- Testaudio staat in upload/ (git-ignored); sessies in ~/AudioImprove/sessions.
- Na elke feature: uv run pytest -q groen -> git commit.
