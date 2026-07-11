# Night roadmap — July 12/13, 2026

Assignment from Serge: "Make it a beautiful project. Clean up the code on
GitHub and keep building steadily. Expand the project's capabilities. Let
people reuse the skills you've used. Make the interface more professional.
And make it possible to process *parts* of the audio — not effects over the
whole file, and not time ranges given by the user, but regions chosen by
smart AI." Method unchanged: finish each feature (implementation + tests +
commit) before starting the next; the repo stays green.

## Status

- [ ] 1. Repo hygiene: MIT license, ruff lint (config + fixes), GitHub
      Actions CI (uv sync + pytest), pyproject metadata, README badges
- [ ] 2. Smart region edits: regions.py detects problem regions on the
      timeline (noise, hum, clipping, mud, harshness) and smart_edit applies
      a targeted mini-chain per region with crossfades — nothing touches the
      parts that are already fine; regions.json feeds the viewer timeline
- [ ] 3. Reusable recipes: save a chain that worked as a named, shareable
      JSON recipe (save_recipe / apply_recipe / list_recipes) + curated
      built-in presets distilled from real sessions
- [ ] 4. Viewer professionalization: design pass, region/segment timeline
      lane under the waveforms, "Chat with Audio" branding, verified live
      in the browser
- [ ] 5. Docs (README/CLAUDE.md), morning report, memory update, all pushed

## Rules (unchanged from the previous night)

- Python 3.11, numpy<2, torch<2.9 — don't touch the pins (see CLAUDE.md).
- Never pollute stdout in server code; lazy-load heavy models.
- New MCP tools also go into tests/test_mcp_tools.py EXPECTED and scripts/mcp_smoke.py.
- Test audio lives in upload/ (git-ignored); sessions in ~/AudioImprove/sessions.
- After each feature: uv run pytest -q green -> git commit.
