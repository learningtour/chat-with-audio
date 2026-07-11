# Chat with Audio — ontwikkelnotities

Chat-gestuurde audio-verbetertool: MCP-server (FastMCP, stdio) + C++ DSP-kern
(pybind11) + lokale A/B-viewer. Zie README.md voor gebruikersdocumentatie.

> Naam overal: **Chat with Audio** (package `chat_with_audio`, MCP-server
> `chat-with-audio`, GitHub `chat-with-audio`). Alleen de lokale projectmap heet
> nog "Audio Improve Toolkit" — die naam zit in de MCP-registraties (pad) en is
> lokaal, dus onzichtbaar op GitHub.

## Commando's

```bash
uv sync --all-extras                              # build (incl. C++) + alle deps
uv sync --reinstall-package chat-with-audio # na wijziging in cpp/
uv run pytest                                     # testsuite
uv run python scripts/mcp_smoke.py                # MCP stdio-rooktest
uv run ait analyze <bestand>                      # dev-CLI zonder MCP
uv run ait improve <bestand> [--profile speech|music] [--denoise-method ai]
uv run ait viewer                                 # viewer op :8471
```

## Architectuurkaart

- `cpp/` — header-only DSP (biquad.hpp, dynamics.hpp) + `bindings.cpp` → module
  `chat_with_audio._dsp`. Arrays zijn float32 (channels, n); functies geven
  nieuwe arrays terug; dynamics gebruiken een linked detector over kanalen.
- `src/chat_with_audio/dsp/__init__.py` — dispatch: native `_dsp` indien
  gebouwd, anders `fallback.py` (scipy; identieke signaturen, blok-gebaseerde
  dynamics). `spectral_nr.py` = Tier A denoise; `ai_nr.py` = Tier B (DeepFilterNet).
- `analysis.py` → metrics-dict + `score_and_issues()`; `improve.py` → profiel-
  detectie + regels → (steps, rationale); `chain.py` → `STEP_REGISTRY` + uitvoering
  (incl. `leveler` en segment-gestuurde `smart_denoise`).
- `segments.py` → spraak/muziek/stilte-segmentatie (primair niveau-Otsu; modulatie-
  ritme als fallback). `refine.py` → iteratieve meet-en-bijstuur-lus (`refine_audio`-
  tool): AI-ontruising eenmalig vooraf, dan leveler/loudness bijsturen tot
  spraakpiek en balans kloppen; stiltesegmenten worden na afloop teruggedrukt
  (_duck_silence) omdat de leveler ze anders meetilt.
- `server.py` — 18 MCP-tools; `sessions.py` — sessiemappen onder
  `~/AudioImprove/sessions/` (env `AIT_SESSIONS_DIR`; tests isoleren dit automatisch).
- `viewer/server.py` — stdlib http.server op 127.0.0.1:8471 (env `AIT_VIEWER_PORT`);
  `viewer/static/app.js` — A/B-speler: beide buffers spelen altijd samen,
  wisselen = gain-crossfade.

## Valkuilen

- **stdout is heilig**: de MCP-server draait over stdio. Nooit `print()` in
  servercodepaden; logging gaat naar stderr, subprocessen met `capture_output=True`.
- **Python 3.11 + numpy<2 + torch/torchaudio<2.9 zijn harde pins** (DeepFilterNet-
  wheels en `torchaudio.backend`-import). Niet "even upgraden".
- De module heet `dsp/ai_nr.py` (niet `ai_denoise.py`) om schaduwwerking met de
  functie `dsp.ai_denoise()` te vermijden.
- `normalize_loudness` zet de limiter-ceiling 0.3 dB onder het true-peak-target
  (inter-sample pieken). E2E-tests toetsen ontruising via SNR-delta, niet via de
  absolute ruisvloer (loudness-normalisatie tilt de vloer mee omhoog).
- Registratie: `.mcp.json` (Claude Code) en Claude Desktop-config gebruiken het
  absolute pad naar `uv` — GUI-apps hebben `~/.local/bin` niet in PATH.
