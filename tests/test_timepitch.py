"""Tijd- en toonhoogtemotor (fase B): duur- en pitchverhoudingen, formantbehoud.

Meetprincipe: sinussen en synthetische klinkers met bekende frequenties; na
stretch/shift moet de dominante frequentie (en bij formantbehoud de
omhullende-piek) op de verwachte plek liggen.
"""

import numpy as np
import pytest
from scipy.signal import lfilter

from chat_with_audio import chain
from chat_with_audio.dsp import timepitch

SR = 44100


def tone(freq=440.0, dur=2.0, amp=0.3):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(x, sr=SR):
    seg = np.asarray(x, dtype=np.float64)
    seg = seg * np.hanning(seg.shape[0])
    spec = np.abs(np.fft.rfft(seg))
    return float(np.fft.rfftfreq(seg.shape[0], 1 / sr)[int(np.argmax(spec))])


def run(x, steps):
    y, _ = chain.run_chain(x, SR, steps)
    return y


# ---------------------------------------------------------------- time stretch

@pytest.mark.parametrize("rate", [0.8, 1.25])
def test_time_stretch_changes_duration_keeps_pitch(rate):
    x = tone(440.0, 2.0)
    y = run(x, [{"type": "time_stretch", "rate": rate}])[0]
    assert y.shape[0] == pytest.approx(x.shape[0] / rate, rel=0.02)
    mid = y[y.shape[0] // 4: 3 * y.shape[0] // 4]
    assert dominant_freq(mid) == pytest.approx(440.0, abs=3.0)


def test_time_stretch_rate_one_is_identity():
    x = tone()
    y = run(x, [{"type": "time_stretch", "rate": 1.0}])[0]
    np.testing.assert_allclose(y, x, atol=1e-7)


def test_time_stretch_preserves_level():
    x = tone(440.0, 2.0)
    y = run(x, [{"type": "time_stretch", "rate": 1.3}])[0]
    rms_x = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    mid = y[y.shape[0] // 4: 3 * y.shape[0] // 4]
    rms_y = np.sqrt(np.mean(mid.astype(np.float64) ** 2))
    assert 20 * abs(np.log10(rms_y / rms_x)) < 1.5  # binnen 1.5 dB


def test_time_stretch_range_guard():
    with pytest.raises(ValueError):
        timepitch.time_stretch(tone(), SR, rate=10.0)


# ---------------------------------------------------------------- pitch shift

@pytest.mark.parametrize("semi,factor", [(4, 2 ** (4 / 12)), (-5, 2 ** (-5 / 12))])
def test_pitch_shift_moves_pitch_keeps_duration(semi, factor):
    x = tone(440.0, 2.0)
    y = run(x, [{"type": "pitch_shift", "semitones": semi}])[0]
    assert y.shape[0] == x.shape[0]
    mid = y[y.shape[0] // 4: 3 * y.shape[0] // 4]
    assert dominant_freq(mid) == pytest.approx(440.0 * factor, rel=0.01)


def vowel(f0=110.0, formants=(700.0, 1200.0), dur=2.0):
    """Synthetische klinker: pulstrein door resonante filters (formanten)."""
    n = int(dur * SR)
    x = np.zeros(n)
    period = int(SR / f0)
    x[::period] = 1.0
    for fc in formants:
        r = 0.995
        w = 2 * np.pi * fc / SR
        b, a = [1 - r], [1, -2 * r * np.cos(w), r * r]
        x = lfilter(b, a, x)
    x = x / np.max(np.abs(x)) * 0.5
    return x.astype(np.float32)


def envelope_peak_hz(x, sr=SR):
    """Piek van de cepstraal-gladde omhullende (formantligging)."""
    seg = np.asarray(x, dtype=np.float64)
    n_fft = 4096
    frames = [seg[i:i + n_fft] * np.hanning(n_fft)
              for i in range(0, seg.shape[0] - n_fft, n_fft // 2)]
    mag = np.abs(np.fft.rfft(np.stack(frames), axis=1)).mean(axis=0)
    env = timepitch._cepstral_envelope(mag[None, :], lifter=60)[0]
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs > 200) & (freqs < 2000)
    return float(freqs[band][int(np.argmax(env[band]))])


def test_pitch_shift_formant_preserve_keeps_envelope():
    x = vowel()
    base = envelope_peak_hz(x)
    plain = timepitch.pitch_shift(x, SR, 7.0)[0]
    kept = timepitch.pitch_shift(x, SR, 7.0, preserve_formants=True)[0]
    drift_plain = abs(envelope_peak_hz(plain) - base)
    drift_kept = abs(envelope_peak_hz(kept) - base)
    # zonder behoud schuift de omhullende flink mee omhoog; met behoud blijft
    # hij bij het origineel — en in elk geval ruim dichter dan zonder
    assert drift_plain > 60.0  # sanity: er wás echt drift
    assert drift_kept < drift_plain * 0.6


# ---------------------------------------------------------------- varispeed

def test_varispeed_couples_duration_and_pitch():
    x = tone(440.0, 2.0)
    y = run(x, [{"type": "varispeed", "rate": 1.05}])[0]
    assert y.shape[0] == pytest.approx(x.shape[0] / 1.05, rel=0.01)
    assert dominant_freq(y) == pytest.approx(440.0 * 1.05, rel=0.005)


def test_varispeed_stereo_shape():
    x = np.vstack([tone(440.0, 1.0), tone(880.0, 1.0)])
    y = run(x, [{"type": "varispeed", "rate": 0.95}])
    assert y.shape[0] == 2
    assert y.shape[1] == pytest.approx(x.shape[1] / 0.95, rel=0.01)
