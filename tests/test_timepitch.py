"""Time & pitch engine: stretch-/shift-/varispeed-semantiek + formantbehoud."""

import numpy as np
import pytest

from chat_with_audio.chain import run_chain, validate_steps
from chat_with_audio.dsp import timepitch


def _peak_freq(mono, sr, fmin=50.0):
    spec = np.abs(np.fft.rfft(mono.astype(np.float64)))
    f = np.fft.rfftfreq(mono.shape[0], 1.0 / sr)
    spec[f < fmin] = 0.0
    return float(f[spec.argmax()])


@pytest.fixture
def sine(sr):
    t = np.arange(sr * 3) / sr
    return (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def vowel(sr):
    """Kunstklinker: f0 110 Hz met een vaste formant rond 800 Hz."""
    t = np.arange(sr * 2) / sr
    x = np.zeros_like(t)
    rng = np.random.default_rng(3)
    for k in range(1, 40):
        f = 110.0 * k
        if f > 4000:
            break
        amp = np.exp(-0.5 * ((f - 800.0) / 250.0) ** 2) + 0.02
        x += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return (0.25 * x / np.abs(x).max()).astype(np.float32)


def _envelope_peak(mono, sr):
    """Frequentie van de gladgemaakte spectrale piek (formantligging)."""
    spec = np.abs(np.fft.rfft(mono.astype(np.float64)))
    f = np.fft.rfftfreq(mono.shape[0], 1.0 / sr)
    width = max(3, int(120.0 * mono.shape[0] / sr))  # ~120 Hz gladstrijken
    kernel = np.ones(width) / width
    smooth = np.convolve(spec, kernel, mode="same")
    smooth[(f < 200) | (f > 4000)] = 0.0
    return float(f[smooth.argmax()])


def test_time_stretch_changes_length_not_pitch(sine, sr):
    y = timepitch.time_stretch(sine, sr, 1.25)
    assert y.ndim == 2 and y.dtype == np.float32
    assert y.shape[1] / sine.shape[0] == pytest.approx(1.25, rel=0.02)
    assert _peak_freq(y[0], sr) == pytest.approx(440, abs=5)

    y2 = timepitch.time_stretch(sine, sr, 0.8)
    assert y2.shape[1] / sine.shape[0] == pytest.approx(0.8, rel=0.02)
    assert _peak_freq(y2[0], sr) == pytest.approx(440, abs=5)


def test_pitch_shift_changes_pitch_not_length(sine, sr):
    y = timepitch.pitch_shift(sine, sr, 12.0, preserve_formants=False)
    assert y.shape[1] == sine.shape[0]
    assert _peak_freq(y[0], sr) == pytest.approx(880, abs=10)

    y2 = timepitch.pitch_shift(sine, sr, -12.0, preserve_formants=False)
    assert _peak_freq(y2[0], sr) == pytest.approx(220, abs=8)


def test_pitch_shift_preserves_formants(vowel, sr):
    up = 2.0 ** (7.0 / 12.0)
    plain = timepitch.pitch_shift(vowel, sr, 7.0, preserve_formants=False)
    kept = timepitch.pitch_shift(vowel, sr, 7.0, preserve_formants=True)
    assert kept.shape[1] == vowel.shape[0]
    # zonder formantbehoud schuift de klankkleur mee omhoog...
    assert _envelope_peak(plain[0], sr) > 800.0 * (1 + up) / 2 * 0.9
    # ...met formantbehoud blijft de envelop-piek bij de originele formant
    assert abs(_envelope_peak(kept[0], sr) - 800.0) < 200.0
    # en de pitch is wél verschoven: harmonischen liggen nu op k*110*up
    spec = np.abs(np.fft.rfft(kept[0].astype(np.float64)))
    f = np.fft.rfftfreq(kept.shape[1], 1.0 / sr)
    h1 = 110.0 * up
    band = spec[(f > h1 - 15) & (f < h1 + 15)].max()
    off = spec[(f > 110.0 * 1.15) & (f < h1 - 20)].max()
    assert band > 3 * off


def test_varispeed_couples_length_and_pitch(sine, sr):
    y = timepitch.varispeed(sine, sr, factor=1.25)
    assert y.shape[1] / sine.shape[0] == pytest.approx(0.8, rel=0.01)
    assert _peak_freq(y[0], sr) == pytest.approx(550, abs=6)

    y2 = timepitch.varispeed(sine, sr, semitones=12.0)
    assert y2.shape[1] / sine.shape[0] == pytest.approx(0.5, rel=0.01)
    assert _peak_freq(y2[0], sr) == pytest.approx(880, abs=10)


def test_stereo_and_noop_paths(sine, sr):
    st = np.stack([sine, 0.5 * sine])
    y = timepitch.time_stretch(st, sr, 1.1)
    assert y.shape[0] == 2
    assert np.array_equal(timepitch.time_stretch(st, sr, 1.0), st)
    assert np.array_equal(timepitch.pitch_shift(st, sr, 0.0), st)
    assert np.array_equal(timepitch.varispeed(st, sr, factor=1.0), st)


def test_validation_errors(sine, sr):
    with pytest.raises(ValueError, match="factor"):
        timepitch.time_stretch(sine, sr, 10.0)
    with pytest.raises(ValueError, match="semitones"):
        timepitch.pitch_shift(sine, sr, 60.0)
    with pytest.raises(ValueError, match="factor óf semitones"):
        timepitch.varispeed(sine, sr, factor=1.2, semitones=3.0)
    with pytest.raises(ValueError, match="factor of semitones"):
        timepitch.varispeed(sine, sr)


def test_chain_steps_registered(sine, sr):
    steps = [{"type": "time_stretch", "factor": 1.2},
             {"type": "pitch_shift", "semitones": -2, "preserve_formants": False},
             {"type": "varispeed", "factor": 1.2}]
    y, resolved = run_chain(sine, sr, steps)
    # 1.2 langer en daarna 1.2 sneller is netto weer de oorspronkelijke duur
    assert y.shape[1] / sine.shape[0] == pytest.approx(1.0, rel=0.03)
    assert [r["type"] for r in resolved] == ["time_stretch", "pitch_shift", "varispeed"]
    with pytest.raises(ValueError, match="Onbekende parameter"):
        validate_steps([{"type": "pitch_shift", "cents": 100}])
