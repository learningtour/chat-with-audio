"""Signaalgeneratie (fase C): lineup-toon en two-pop voor broadcast-levering.

Een tape/file-levering begint klassiek met een leader: referentietoon
(1 kHz, -18 dBFS voor EBU-huizen) om de keten uit te lijnen, dan stilte met
een two-pop — één frame 1 kHz — exact 2 seconden vóór het eerste programma-
frame, zodat beeld en geluid te synchroniseren zijn. `leader()` bouwt die
kop vóór bestaand programma; de posities komen terug voor de regiokaart.
"""

from __future__ import annotations

import numpy as np

POP_OFFSET_S = 2.0   # two-pop staat per definitie 2 s voor programma-start
POP_LEN_S = 1.0 / 24.0  # één filmframe bij 24 fps


def tone(sr: int, seconds: float, freq: float = 1000.0, level_db: float = -18.0,
         channels: int = 1, fade_ms: float = 5.0) -> np.ndarray:
    """Sinus-referentietoon; level_db = piekniveau in dBFS, korte fades tegen
    klikken."""
    if seconds <= 0:
        return np.zeros((channels, 0), dtype=np.float32)
    t = np.arange(int(seconds * sr)) / sr
    amp = 10.0 ** (level_db / 20.0)
    y = amp * np.sin(2 * np.pi * freq * t)
    fade = min(int(fade_ms / 1000 * sr), y.shape[0] // 2)
    if fade > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade))
        y[:fade] *= ramp
        y[-fade:] *= ramp[::-1]
    return np.repeat(y[None, :], channels, axis=0).astype(np.float32)


def leader(x: np.ndarray, sr: int, tone_s: float = 10.0, tone_db: float = -18.0,
           tone_hz: float = 1000.0, gap_s: float = 3.0,
           two_pop: bool = True) -> tuple[np.ndarray, dict]:
    """Zet [toon][stilte met two-pop][programma] voor bestaand materiaal.

    De pop (één 24fps-frame 1 kHz) begint exact POP_OFFSET_S vóór het
    programma; gap_s moet daarvoor ruimte laten. Geeft (audio, info) met de
    toon/pop/programma-posities in seconden voor de regiokaart.
    """
    x2 = x[None, :] if x.ndim == 1 else x
    ch = x2.shape[0]
    if tone_s < 0 or gap_s < 0:
        raise ValueError("tone_s en gap_s moeten >= 0 zijn.")
    if two_pop and gap_s < POP_OFFSET_S + 0.25:
        raise ValueError(f"gap_s moet minstens {POP_OFFSET_S + 0.25:.2f} s zijn "
                         "voor een two-pop (die staat 2 s voor het programma).")
    head = tone(sr, tone_s, freq=tone_hz, level_db=tone_db, channels=ch)
    gap = np.zeros((ch, int(gap_s * sr)), dtype=np.float32)
    program_start = tone_s + gap_s
    info: dict = {"tone": {"start_s": 0.0, "end_s": round(tone_s, 3),
                           "freq": tone_hz, "level_db": tone_db} if tone_s else None,
                  "program_start_s": round(program_start, 3)}
    if two_pop:
        pop = tone(sr, POP_LEN_S, freq=1000.0, level_db=tone_db, channels=ch,
                   fade_ms=2.0)
        pos = gap.shape[1] - int(POP_OFFSET_S * sr)
        gap[:, pos:pos + pop.shape[1]] = pop[:, :max(0, gap.shape[1] - pos)]
        info["two_pop"] = {"start_s": round(tone_s + pos / sr, 3),
                           "end_s": round(tone_s + pos / sr + POP_LEN_S, 3)}
    y = np.concatenate([head, gap, x2.astype(np.float32)], axis=1)
    return np.ascontiguousarray(y), info
