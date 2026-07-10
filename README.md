# Audio Improve Toolkit

Chat-gestuurde audio-verbetering: praat met je lokale Claude (Desktop of Claude Code)
over een opname en laat de toolkit hem analyseren en verbeteren. Een lokale
A/B-viewer laat je origineel en resultaat gesynchroniseerd vergelijken — horen én zien.

```
Claude (chat)  ── MCP (stdio) ──>  Python-orchestratie ──> C++ DSP-kern (pybind11)
                                        │                    gate · compressor · limiter · EQ
                                        ├─> AI-denoise (DeepFilterNet, optioneel)
                                        ├─> analyse (LUFS, SNR, brom, clipping, spectrum)
                                        └─> sessies ──> A/B-viewer (http://127.0.0.1:8471)
```

## Wat kun je vragen?

- **"Analyseer dit bestand: /pad/naar/opname.wav"** — metrics, scores en issues.
- **"Maak dit geluid beter"** — auto-improve: de tool detecteert spraak/muziek en
  kiest zelf een keten (highpass, brom-notches, ruisonderdrukking, gate, EQ,
  compressie, loudness), met uitleg per stap.
- **"Verminder de ruis"** — alleen ruisonderdrukking (spectral gating of DeepFilterNet-AI).
- **"Trek het level op zonder te clippen"** — loudness-normalisatie (BS.1770) met
  true-peak-limiter.
- **"Knip 3 dB rond 300 Hz weg en comprimeer licht"** — expliciete keten via `apply_chain`.
- **"Open de viewer"** / **"Wat is er precies veranderd?"** — A/B-vergelijking; Claude
  leest dezelfde sessiedata als de viewer toont.

## Installatie (macOS)

Vereist: [uv](https://docs.astral.sh/uv/), ffmpeg (`brew install ffmpeg`), Xcode
Command Line Tools. Python 3.11 wordt door uv zelf opgehaald.

```bash
cd "Audio Improve Toolkit"
uv sync --all-extras        # bouwt de C++-kern en installeert alles (incl. AI-denoise)
uv run pytest               # 21 tests
uv run python scripts/mcp_smoke.py   # MCP-rooktest
```

`uv sync` (zonder `--all-extras`) installeert de basis zonder torch/DeepFilterNet;
de tool valt dan automatisch terug op spectral gating.

## Registratie bij Claude

- **Claude Code**: staat in `.mcp.json` in de projectmap (werkt automatisch in deze map).
- **Claude Desktop**: entry `audio-improve` in
  `~/Library/Application Support/Claude/claude_desktop_config.json`. Herstart
  Claude Desktop na installatie; de tools verschijnen onder "audio-improve".

Let op: draai éérst `uv sync --all-extras`, anders kan de eerste serverstart
time-outen op het bouwen/downloaden.

## De viewer

`open_viewer` (of `uv run ait viewer`) start hem op <http://127.0.0.1:8471>.
Spatie = afspelen, **b/a of de knoppen = wisselen tussen origineel en bewerking**
terwijl beide synchroon doorlopen. Klik in de golfvorm om te zoeken. Poort
aanpassen: omgevingsvariabele `AIT_VIEWER_PORT`.

Sessies staan in `~/AudioImprove/sessions/` (override: `AIT_SESSIONS_DIR`), elk met
origineel, resultaat, analyses, keten + rationale, golfvormen en spectrogrammen.

## Windows

1. Installeer [uv](https://docs.astral.sh/uv/), ffmpeg (`winget install ffmpeg`) en
   **Visual Studio Build Tools** (C++ workload) voor de native DSP-kern.
   Zonder Build Tools werkt alles ook, maar dan via de pure-Python fallback —
   verwijder in dat geval de C++-buildstap niet, hij faalt gewoon zacht.
2. `uv sync --all-extras` in de projectmap (DeepFilterNet heeft win_amd64-wheels).
3. Registreer in `%APPDATA%\Claude\claude_desktop_config.json` met het volledige
   pad naar `uv.exe` en de projectmap (zelfde vorm als `.mcp.json` hier).

## Architectuur

| Laag | Locatie | Rol |
|---|---|---|
| C++ DSP-kern | `cpp/` | biquad EQ (RBJ), noise gate, soft-knee compressor, look-ahead brickwall limiter; via pybind11 als `audio_improve_toolkit._dsp` |
| DSP-dispatch | `src/audio_improve_toolkit/dsp/` | native ↔ scipy-fallback, spectral gating (`spectral_nr.py`), DeepFilterNet (`ai_nr.py`) |
| Analyse | `analysis.py` | LUFS/LRA (pyloudnorm), true peak, SNR, ruisvloer, brom, clipping, spectrum, scores + issues |
| Beslislogica | `improve.py` | spraak/muziek-detectie, regels → keten + rationale |
| Keten | `chain.py` | stap-registry, uitvoering, loudness-normalisatie |
| MCP-server | `server.py` | 7 tools over stdio (FastMCP) |
| Viewer | `viewer/` | stdlib http.server + Web Audio A/B-speler |

Loudness-targets: spraak −16 LUFS / TP −1.5 dBTP, muziek −14 LUFS / TP −1.0 dBTP.

## Versiepinnen (bewust)

- **Python 3.11** en **numpy < 2.0**: DeepFilterNet 0.5.x levert alleen wheels
  t/m cp311 en vereist numpy 1.x.
- **torch/torchaudio < 2.9**: DeepFilterNet importeert `torchaudio.backend`, dat
  in torchaudio 2.9 is verwijderd.

Na wijzigingen aan de C++-code: `uv sync --reinstall-package audio-improve-toolkit`.
