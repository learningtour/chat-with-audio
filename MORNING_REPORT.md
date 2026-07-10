# Ochtendrapport — nachtbouw 10/11 juli 2026

Goedemorgen! De toolkit is vannacht uitgegroeid van "audio verbeteren" naar een
volwaardige chat-gestuurde audiostudio met **15 MCP-tools**. Alles is getest
(33 tests groen) en per feature gecommit.

## Nieuw vannacht

| Feature | Vraag het in de chat | Detail |
|---|---|---|
| **Declip + declick** | "herstel de clips" | Golfvorm-reconstructie via splines; improve doet declip nu automatisch (jouw musical-bestand heeft 70 echte clips!) |
| **Stem-separatie** | "splits de stems" | Demucs AI: vocals / drums / bass / other als losse wav's voor je DAW |
| **Rebalance / karaoke** | "zang 3 dB erbij", "maak een karaoke-versie" | Per-stem gains, dynamiek-veilige gain-staging, A/B-sessie |
| **Residu-beluistering** | knop **R · verschil** in de viewer (toets r) | Hoor exact wat de bewerking veranderde — de artefact-detector voor je oren; werkt ook op alle oude sessies |
| **Reference matching** | "laat dit klinken als <referentie>" | 1/3-octaaf match-EQ (begrensd) + loudness-match; voor consistente afleveringen |
| **De-esser** | automatisch bij spraak | Spectraal, dempt alleen frames waar s-klanken echt uitschieten |
| **Resonantiedetectie** | automatisch | Smalle pieken (dozige room-resonanties) worden gedetecteerd en gericht weggenomen |
| **Batchverwerking** | "doe de hele map" | improve_folder: improve/refine/optimize per bestand |
| **Whisper-medium scheidsrechter** | optimize_audio(judge_model="medium") | Strengere verstaanbaarheidsjury voor nachtruns |

## De nachtrun op jouw testbestand

Bij het schrijven van dit rapport draaide de diepe optimalisatie
(Whisper-medium, 9 varianten incl. de-esser/dereverb-combinaties) nog — de
uitslag staat in de nieuwste sessie in de viewer en in de chat.

## Waar alles staat

- Viewer: http://127.0.0.1:8471 (`open de viewer` in de chat)
- Sessies: `~/AudioImprove/sessions/` — nieuwste bovenaan in de viewer
- Roadmap + status: `NIGHT_ROADMAP.md`; gebruikersdocs: `README.md`
- Let op: Claude Desktop herstarten + eenmalige goedkeuring in Claude Code
  zijn nog steeds nodig om de tools daar te zien (config staat klaar)

## Eerlijke aantekeningen

- De R-knop (residu) is via HTTP en syntaxcheck geverifieerd, maar nog niet in
  een echte browser beluisterd (Chrome-extensie was 's nachts niet verbonden).
- Stem-separatie op het musical-bestand behandelt gesproken dialoog als
  "vocals" — dat is correct gedrag van Demucs, maar even wennen.
- Dereverb (ClearVoice) draait alleen op spraaksegmenten; muziek-dereverb staat
  bewust uit (sloopt de mix).

## Ideeën voor de volgende sessie (niet gebouwd, wel doordacht)

1. Voorkeuren-geheugen: "ik vind B beter" → de tool leert jouw smaak als preset.
2. Spraak-superresolutie (ClearVoice SR-model) voor oude/doffe opnames.
3. Meersporen-export (stems + verbeterde mix) als sessie-zip voor productiehuizen.
4. Windows-test + GitHub-repo publiceren als je dat wilt — de vakbladen, weet je nog.
