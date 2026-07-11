import numpy as np

from chat_with_audio import chain


def _band_rms_db(x, sr, lo=60.0, hi=170.0):
    from chat_with_audio import dsp

    band = dsp.eq(x, sr, [("highpass", lo, 0.0, 0.707)] * 2
                         + [("lowpass", hi, 0.0, 0.707)] * 2)
    return 10 * np.log10(np.mean(np.asarray(band, np.float64) ** 2) + 1e-20)


def test_band_duck_tames_boom_keeps_mids(sr):
    rng = np.random.default_rng(9)
    t4, t6 = np.arange(sr * 4) / sr, np.arange(sr * 6) / sr
    # 'spraak': zachte AM-ruisband; 'muziek': 1 kHz + dreunende 120 Hz-pulsen
    speech = (rng.normal(0, 1, sr * 4) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4 * t4)))
              * 10 ** (-30 / 20)).astype(np.float32)
    mid = 0.25 * np.sin(2 * np.pi * 1000 * t6)
    boom = (0.7 * np.sin(2 * np.pi * 120 * t6)
            * (np.sin(2 * np.pi * 2 * t6) > 0)).astype(np.float32)
    x = np.concatenate([speech, (mid + boom).astype(np.float32)])[None, :]

    y, _ = chain.run_chain(x, sr, [{"type": "band_duck", "max_cut_db": 12.0}])

    muz = slice(int(4.5 * sr), None)
    spr = slice(0, int(3.5 * sr))
    # dreunband in de muziek flink omlaag
    assert _band_rms_db(x[:, muz], sr) - _band_rms_db(y[:, muz], sr) >= 5.0
    # 1 kHz-inhoud vrijwel onaangetast
    from chat_with_audio import dsp

    mid_x = dsp.eq(x[:, muz], sr, [("highpass", 500, 0.0, 0.707)] * 2)
    mid_y = dsp.eq(y[:, muz], sr, [("highpass", 500, 0.0, 0.707)] * 2)
    d = (10 * np.log10(np.mean(mid_x.astype(np.float64) ** 2) + 1e-20)
         - 10 * np.log10(np.mean(mid_y.astype(np.float64) ** 2) + 1e-20))
    assert abs(d) < 1.0
    # spraakdeel onaangetast (music_only)
    np.testing.assert_allclose(y[:, spr], x[:, spr], atol=5e-4)
