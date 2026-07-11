"""Klik-bursts in stilte verwijderen, maar woordfinale plosieven laten staan."""

import numpy as np

from audio_improve_toolkit.dsp.repair import declick


def test_burst_clicks_in_silence_removed_plosive_kept(sr):
    rng = np.random.default_rng(3)
    n = sr * 4
    x = rng.normal(0, 10 ** (-60 / 20), n).astype(np.float32)  # stiltevloer

    # 'spraak' 0.5-1.5 s (aanhoudend), met een plosief-achtige burst vlak erna
    t = np.arange(sr) / sr
    x[int(0.5 * sr):int(1.5 * sr)] += (0.3 * np.sin(2 * np.pi * 220 * t)
                                       * (0.6 + 0.4 * np.sin(2 * np.pi * 5 * t))
                                       ).astype(np.float32)
    plos = int(1.52 * sr)  # 30 ms na spraak-einde: beschermzone
    x[plos:plos + int(0.006 * sr)] += (0.4 * rng.normal(0, 1, int(0.006 * sr))
                                       ).astype(np.float32)

    # echte klik-bursts diep in de stilte (8 en 14 ms)
    clicks = []
    for tt, dur in ((2.4, 0.008), (3.2, 0.014)):
        a = int(tt * sr)
        b = a + int(dur * sr)
        x[a:b] += (0.8 * rng.normal(0, 1, b - a)).astype(np.float32)
        clicks.append((a, b))

    y, count = declick(x[None, :], sr)
    y = y[0]

    for a, b in clicks:
        assert np.abs(y[a:b]).max() < 0.02, "klik-burst moet weg zijn"
    # plosief in de beschermzone blijft staan
    assert np.abs(y[plos:plos + int(0.006 * sr)]).max() > 0.1
    # spraak zelf vrijwel onaangetast (reparatie-energie < -35 dB t.o.v. signaal)
    sp = slice(int(0.6 * sr), int(1.4 * sr))
    d = (y[sp] - x[sp]).astype(np.float64)
    rel = (10 * np.log10((d**2).mean() + 1e-20)
           - 10 * np.log10((x[sp].astype(np.float64) ** 2).mean()))
    assert rel < -35
    assert count >= 2
