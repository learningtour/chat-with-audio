# Night roadmap — July 12/13, 2026

Assignment from Serge: "Make it a beautiful project. Clean up the code on
GitHub and keep building steadily. Expand the project's capabilities. Let
people reuse the skills you've used. Make the interface more professional.
And make it possible to process *parts* of the audio — not effects over the
whole file, and not time ranges given by the user, but regions chosen by
smart AI." Method unchanged: finish each feature (implementation + tests +
commit) before starting the next; the repo stays green.

## Status

- [x] 1. Repo hygiene: MIT license, ruff lint (config + fixes), GitHub
      Actions CI (uv sync + ruff + pytest on ubuntu/macos), pyproject
      metadata (v0.2.0), README badges
- [x] 2. Smart region edits: regions.py detects problem regions on the
      timeline (hum, noise, clipping, boom) and smart_edit applies a
      targeted mini-chain per region with raised-cosine crossfades —
      everything outside the regions stays bit-for-bit untouched;
      timeline.json feeds the viewer timeline
- [x] 3. Reusable recipes: save a chain that worked as a named, shareable
      JSON recipe (save_recipe / apply_recipe / list_recipes) + four curated
      built-in presets distilled from real sessions
- [x] 4. Viewer professionalization: design pass, content/interventions
      timeline lanes under the waveforms, "Chat with Audio" branding,
      verified live in the browser (incl. click-to-seek and A/B/R)
- [x] 5. Docs (README/CLAUDE.md), morning report, memory update, all pushed

Bonus fixes that fell out of live testing: session-id collision (two
sessions on the same file within one second overwrote each other), noise
reference floor clamped to -80 dB, boom-inside-hum suppression.

## Rules (unchanged from the previous night)

- Python 3.11, numpy<2, torch<2.9 — don't touch the pins (see CLAUDE.md).
- Never pollute stdout in server code; lazy-load heavy models.
- New MCP tools also go into tests/test_mcp_tools.py EXPECTED and scripts/mcp_smoke.py.
- Test audio lives in upload/ (git-ignored); sessions in ~/AudioImprove/sessions.
- After each feature: uv run pytest -q green -> git commit.
