import numpy as np

from chat_with_audio import analysis, dsp
from chat_with_audio.dsp.deess import deess


def _band_rms_db(x, sr, lo, hi):
    from scipy.signal import welch

    f, p = welch(x, fs=sr, nperseg=4096)
    sel = (f >= lo) & (f <= hi)
    return 10 * np.log10(float(p[sel].mean()) + 1e-20)


def test_deess_tames_sibilance_keeps_vowel(sr):
    rng = np.random.default_rng(8)
    t = np.arange(sr * 4) / sr
    vowel = 0.25 * np.sin(2 * np.pi * 300 * t)
    sib = np.zeros_like(vowel)
    burst = rng.normal(0, 1, int(sr * 0.15))
    burst = dsp.eq(burst.astype(np.float32), sr, [("highpass", 6000.0, 0.0, 0.9),
                                                  ("lowpass", 8500.0, 0.0, 0.9)])
    for start_s in (0.5, 1.5, 2.5, 3.3):
        a = int(start_s * sr)
        sib[a:a + burst.shape[0]] += 0.5 * burst
    x = (vowel + sib).astype(np.float32)[None, :]

    y = deess(x, sr, strength_db=10.0)
    assert _band_rms_db(y[0], sr, 6000, 8500) < _band_rms_db(x[0], sr, 6000, 8500) - 3
    assert abs(_band_rms_db(y[0], sr, 250, 400) - _band_rms_db(x[0], sr, 250, 400)) < 0.5


def test_resonance_detected(sr):
    rng = np.random.default_rng(9)
    noise = rng.normal(0, 0.05, sr * 5).astype(np.float32)
    x = dsp.eq(noise, sr, [("peaking", 1000.0, 14.0, 12.0)])[None, :]
    m = analysis.analyze(x, sr)
    freqs = [r["freq"] for r in m["resonances"]]
    assert any(abs(f - 1000) < 120 for f in freqs), f"resonantie niet gevonden: {freqs}"
