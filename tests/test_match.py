import numpy as np

from chat_with_audio import match


def _pink(sr, dur=6.0, seed=1):
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1, int(sr * dur))
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(white.shape[0], 1 / sr)
    spec[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spec, n=white.shape[0])
    return (0.1 * x / np.abs(x).max()).astype(np.float32)[None, :]


def test_match_reference_brings_spectrum_closer(sr):
    src = _pink(sr, seed=1)
    from chat_with_audio import dsp

    # referentie = zelfde soort signaal, maar met een duidelijk andere klankkleur
    ref = dsp.eq(_pink(sr, seed=2), sr, [("highshelf", 4000.0, 6.0, 0.707),
                                         ("lowshelf", 150.0, -4.0, 0.707)])
    y, info = match.match_reference(src, sr, ref, sr, match_loudness=False)
    assert len(info["eq_bands"]) >= 4

    def banddiff(a, b):
        la, lb = match._band_levels_db(a, sr), match._band_levels_db(b, sr)
        v = ~np.isnan(la) & ~np.isnan(lb)
        d = (la - lb)[v]
        return float(np.std(d - d.mean()))

    assert banddiff(y, ref) < banddiff(src, ref) * 0.55  # duidelijk dichterbij


def test_match_loudness(sr):
    src = _pink(sr, seed=3)
    ref = (0.02 * _pink(sr, seed=4) / 0.1).astype(np.float32)  # veel zachtere referentie
    y, info = match.match_reference(src, sr, ref, sr, match_loudness=True)
    from chat_with_audio.analysis import measure_lufs

    assert abs(measure_lufs(y, sr) - measure_lufs(ref, sr)) < 1.5
