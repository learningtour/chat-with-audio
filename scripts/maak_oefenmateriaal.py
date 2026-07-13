"""Genereer oefenmateriaal voor de trainingshandleiding.

Elk bestand heeft één bekend, gedocumenteerd gebrek (plus een eindtoets met
alles door elkaar), zodat een cursist kan oefenen met een gecontroleerd
antwoord: ANTWOORDEN.md in de doelmap zegt precies wat er mis is en welke
aanpak past. De generator is deterministisch (vaste seeds, gescreend zodat de
detectoren precies de bedoelde gebreken vinden — tests/test_oefenmateriaal.py
bewaakt dat dit zo blijft).

Gebruik:  uv run python scripts/maak_oefenmateriaal.py [doelmap]
Standaarddoelmap: ./oefenmateriaal
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

SR = 44100


def spraakachtig(dur: float, seed: int, level: float = 0.15) -> np.ndarray:
    """Spraakachtig programma: harmonische 'stem' (200 Hz met vibrato) +
    gebandpaste 'medeklinkers', zinsritme-envelope en een echte ambience-vloer
    op -55 dB (digitale stilte zou elke vloer-meting scheeftrekken)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    env = np.clip(np.sin(2 * np.pi * 0.33 * t)
                  + 0.5 * np.sin(2 * np.pi * 1.1 * t + 0.7), 0, None)
    env = env / (env.max() + 1e-9)
    stem = sum(np.sin(2 * np.pi * 200 * h * t
                      + 0.3 * np.sin(2 * np.pi * 4.5 * t)) / h for h in (1, 2, 3, 4))
    sos = butter(2, [1500, 6000], btype="band", fs=SR, output="sos")
    consonant = sosfilt(sos, rng.standard_normal(t.shape[0])) * 0.5
    x = (stem * 0.5 + consonant) * env * level
    bed = rng.standard_normal(t.shape[0]) * 10 ** (-55 / 20)
    return (x + bed).astype(np.float32)


def oefening_brom(d: Path) -> str:
    t = np.arange(int(30 * SR)) / SR
    x = spraakachtig(30, seed=11)
    brom = 0.018 * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
                    + 0.3 * np.sin(2 * np.pi * 150 * t))
    x = x + (brom * ((t >= 8) & (t < 21))).astype(np.float32)
    sf.write(str(d / "oefening-01-brom.wav"), x, SR, subtype="PCM_24")
    return ("**oefening-01-brom.wav** — netbrom (50 Hz + harmonischen) die bij "
            "8 s áán springt en bij 21 s weer uit (denk: koelkast). Aanpak: "
            "`smart_edit` — chirurgisch, alleen dáár notchen. De brom tilt in de "
            "spreekpauzes ook de gemeten ruisvloer op, dus verwacht dat er naast "
            "de brom- ook een ruisregio wordt gemeld: dat hoort zo. Check in het "
            "resultaat de tweede meetpas (`verification`).")


def oefening_ruis(d: Path) -> str:
    rng = np.random.default_rng(155)
    x = spraakachtig(30, seed=55)
    x = x + (rng.standard_normal(x.shape[0]) * 0.02).astype(np.float32)
    sf.write(str(d / "oefening-02-ruis.wav"), x, SR, subtype="PCM_24")
    return ("**oefening-02-ruis.wav** — constante brede ruisvloer (lage SNR) "
            "over het hele bestand; dit is geen regio-klus maar een heel-bestand-"
            "klus. Aanpak: `analyze_audio` (kijk naar SNR en ruisvloer), dan "
            "`reduce_noise` of `improve_audio`. Luister het residu (R) af: zit "
            "er spraak in wat is weggehaald?")


def oefening_clipping(d: Path) -> str:
    x = spraakachtig(30, seed=33, level=0.3)
    t = np.arange(x.shape[0]) / SR
    zone = (t >= 10) & (t < 14)
    x[zone] = np.clip(x[zone] * 6.0, -0.999, 0.999)
    sf.write(str(d / "oefening-03-clipping.wav"), x, SR, subtype="PCM_24")
    return ("**oefening-03-clipping.wav** — tussen 10 en 14 s is de opname "
            "overstuurd (afgekapte toppen). Aanpak: `repair_audio` of "
            "`smart_edit` (declip alleen rond de schade); kijk met `view_audio` "
            "hoe clipping eruitziet in het spectrogram (brede verticale energie).")


def oefening_dreun(d: Path) -> str:
    x = spraakachtig(30, seed=9)
    t = np.arange(x.shape[0]) / SR
    for start in (6.0, 18.0):
        zone = (t >= start) & (t < start + 5)
        x[zone] += (0.35 * np.sin(2 * np.pi * 35 * t[zone])
                    * np.hanning(zone.sum())).astype(np.float32)
    sf.write(str(d / "oefening-04-dreun.wav"), x, SR, subtype="PCM_24")
    return ("**oefening-04-dreun.wav** — twee keer een laagfrequente dreun "
            "(passerende vrachtwagen, ±35 Hz) rond 6-11 s en 18-23 s. Aanpak: "
            "`smart_edit` (boom-detector). Hoor in A/B dat de stem zelf niet "
            "dunner wordt; de dreun verhoogt in de pauzes ook de vloer, dus een "
            "extra ruisregio-melding is normaal.")


def oefening_pauzes(d: Path) -> str:
    stukken = []
    rng = np.random.default_rng(200)
    for i, (dur, pauze) in enumerate([(3, 2.8), (4, 3.5), (3, 2.2), (4, 0.0)]):
        stukken.append(spraakachtig(dur, seed=(100, 101, 102, 104)[i]))
        if pauze:
            n = int(pauze * SR)
            stukken.append((rng.standard_normal(n) * 10 ** (-55 / 20))
                           .astype(np.float32))
    x = np.concatenate(stukken)
    sf.write(str(d / "oefening-05-pauzes.wav"), x, SR, subtype="PCM_24")
    return ("**oefening-05-pauzes.wav** — prima spraak, maar de pauzes tussen "
            "de zinnen zijn tergend lang (2-3,5 s). Aanpak: `edit_speech` met "
            "`tighten_pauses_to_s=0.6` — eerst met `apply=False` het plan "
            "bekijken. (Stopwoorden knippen oefen je op een eigen spraakmemo: "
            "synthetische spraak heeft geen echte 'uhs'.)")


def oefening_te_stil(d: Path) -> str:
    x = spraakachtig(30, seed=77) * 10 ** (-22 / 20)
    sf.write(str(d / "oefening-06-te-stil.wav"), x, SR, subtype="PCM_24")
    return ("**oefening-06-te-stil.wav** — technisch schoon, maar veel te "
            "stil voor publicatie. Aanpak: `normalize_loudness` naar -16 LUFS "
            "(podcast); check daarna met `check_compliance` en probeer "
            "`codec_preview` om te zien of de master lossy-proof is.")


def eindtoets(d: Path) -> str:
    x = spraakachtig(45, seed=105) * 10 ** (-10 / 20)
    t = np.arange(x.shape[0]) / SR
    rng = np.random.default_rng(206)
    x += (0.015 * np.sin(2 * np.pi * 50 * t) * ((t >= 5) & (t < 15))).astype(np.float32)
    x += (rng.standard_normal(x.shape[0]) * 0.005 * ((t >= 25) & (t < 38))
          ).astype(np.float32)
    zone = (t >= 40) & (t < 42)
    x[zone] = np.clip(x[zone] * 40.0, -0.999, 0.999)
    sf.write(str(d / "eindtoets.wav"), x, SR, subtype="PCM_24")
    return ("**eindtoets.wav** — alles door elkaar: brom van 5-15 s, een "
            "opkomende ruisvloer rond 25-38 s (subtiel — luister in de "
            "pauzes), zware clipping rond 40-42 s, en de spraak zelf veel te "
            "stil. Let op: de clip-knal trekt de integrated loudness-meting "
            "omhoog — óók een leerpunt: eerst repareren, dan pas normaliseren. "
            "Doel: één schone, publicatieklare file op -16 LUFS. Verwachte "
            "route: `smart_edit` (chirurgisch) → `normalize_loudness` → "
            "`qc_report`. Laat de chat het plan uitleggen vóór hij het doet.")


def main() -> None:
    doel = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("oefenmateriaal")
    doel.mkdir(parents=True, exist_ok=True)
    antwoorden = [
        "# Oefenmateriaal — antwoorden\n",
        "Elk bestand heeft één bekend gebrek; de eindtoets combineert ze.",
        "Niet spieken vóór je zelf hebt geanalyseerd!\n",
    ]
    for maak in (oefening_brom, oefening_ruis, oefening_clipping,
                 oefening_dreun, oefening_pauzes, oefening_te_stil, eindtoets):
        antwoorden.append("- " + maak(doel))
    (doel / "ANTWOORDEN.md").write_text("\n".join(antwoorden) + "\n")
    print(f"Oefenmateriaal staat in {doel.resolve()}/ "
          f"({len(list(doel.glob('*.wav')))} bestanden + ANTWOORDEN.md)")


if __name__ == "__main__":
    main()
