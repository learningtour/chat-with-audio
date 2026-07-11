"""DSP-tests, geparametriseerd over de native (C++) en fallback-backend."""

import numpy as np
import pytest

from chat_with_audio.dsp import fallback

try:
    from chat_with_audio import _dsp as native
except ImportError:
    native = None

IMPLS = [pytest.param(fallback, id="fallback")]
if native is not None:
    IMPLS.append(pytest.param(native, id="native"))


@pytest.fixture(params=IMPLS)
def impl(request):
    return request.param


def _sine(sr, freq=440.0, dur=2.0, amp=1.0):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :]


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def test_apply_gain(impl, sr):
    x = _sine(sr, amp=0.1)
    y = impl.apply_gain(x, 6.0)
    assert y.shape == x.shape
    np.testing.assert_allclose(_rms_db(y) - _rms_db(x), 6.0, atol=0.01)


def test_limiter_respects_ceiling(impl, sr):
    x = _sine(sr, amp=1.0)  # 0 dBFS
    y = impl.limiter(x, float(sr), -6.0, 60.0, 5.0)
    assert float(np.abs(y).max()) <= 10 ** (-6 / 20) + 1e-4


def test_compressor_reduces_level_difference(impl, sr):
    t = np.arange(sr * 2) / sr
    sine = np.sin(2 * np.pi * 440 * t)
    amp = np.where((t % 1.0) < 0.5, 0.8, 0.08)  # om en om luid (-2 dB) en zacht (-22 dB)
    x = (sine * amp).astype(np.float32)[None, :]
    y = impl.compressor(x, float(sr), -20.0, 4.0, 5.0, 80.0, 2.0, 0.0)
    loud = slice(int(sr * 0.25), int(sr * 0.45))   # ruim na de attack
    soft = slice(int(sr * 0.75), int(sr * 0.95))   # ruim na de release
    diff_x = _rms_db(x[:, loud]) - _rms_db(x[:, soft])
    diff_y = _rms_db(y[:, loud]) - _rms_db(y[:, soft])
    assert diff_y < diff_x - 6


def test_gate_attenuates_quiet_tail(impl, sr):
    loud = _sine(sr, amp=0.5, dur=1.0)
    rng = np.random.default_rng(2)
    tail = rng.normal(0, 10 ** (-60 / 20), sr * 2).astype(np.float32)[None, :]
    x = np.concatenate([loud, tail], axis=1)
    y = impl.noise_gate(x, float(sr), -40.0, 5.0, 120.0, 50.0, 12.0)
    # laatste seconde (ver na de release) moet ~range_db stiller zijn
    before = _rms_db(x[:, -sr:])
    after = _rms_db(y[:, -sr:])
    assert after <= before - 10


def test_peaking_eq_gain_at_center(impl, sr):
    x = _sine(sr, freq=440.0, amp=0.1, dur=3.0)
    y = impl.biquad_chain(x, float(sr), [("peaking", 440.0, 6.0, 1.0)])
    # steady-state (laatste 2 s) vergelijken
    g = _rms_db(y[:, sr:]) - _rms_db(x[:, sr:])
    assert abs(g - 6.0) < 0.5


@pytest.mark.skipif(native is None, reason="native backend niet gebouwd")
def test_native_matches_fallback_filters(sr):
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.1, (2, sr)).astype(np.float32)
    bands = [("highpass", 80.0, 0.0, 0.707), ("peaking", 1000.0, 3.0, 1.0),
             ("highshelf", 8000.0, 2.0, 0.707)]
    yn = native.biquad_chain(x, float(sr), bands)
    yf = fallback.biquad_chain(x, float(sr), bands)
    assert np.max(np.abs(yn - yf)) < 2e-3


def test_spectral_denoise_lowers_noise_floor(sr, noisy_bursts):
    from chat_with_audio import dsp

    x = noisy_bursts[None, :]
    y = dsp.spectral_denoise(x, sr, reduction_db=12)
    # ruisvloer in een stil stuk (seconde 3-4 valt in een pauze)
    seg = slice(int(sr * 3.2), int(sr * 3.8))
    assert _rms_db(y[:, seg]) < _rms_db(x[:, seg]) - 6
