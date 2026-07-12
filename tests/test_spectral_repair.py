"""Spectral repair painting: schade weg, programma intact, buiten de patch
bit-voor-bit onaangetast."""

import numpy as np
import pytest

from chat_with_audio.dsp.spectral_repair import spectral_repair


def _tone_with_squeak(sr):
    """10 s doorlopende 300 Hz-toon (-20 dB) met een 'stoelpiep' (chirp
    2->4 kHz, 0.25 s) op 3.0-3.25 s."""
    t = np.arange(sr * 10) / sr
    base = 10 ** (-20 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 300 * t)
    seg = (t >= 3.0) & (t < 3.25)
    tt = t[seg] - 3.0
    chirp = 0.15 * np.sin(2 * np.pi * (2000 * tt + (2000 / 0.5) * tt**2))
    x = base.copy()
    x[seg] += chirp
    return x.astype(np.float32)[None, :]


def _band_db(sig, sr, a, b, lo, hi):
    seg = sig[0, int(a * sr):int(b * sr)].astype(np.float64)
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(seg.size, 1 / sr)
    sel = (freqs >= lo) & (freqs <= hi)
    return 20 * np.log10(np.sqrt(np.mean(spec[sel] ** 2)) + 1e-12)


def test_squeak_removed_tone_preserved(sr):
    x = _tone_with_squeak(sr)
    y = spectral_repair(x, sr, 2.95, 3.30, low_hz=1500, high_hz=5000)

    # de piep is weg (band 2-4.5 kHz in het patchvenster flink omlaag)
    assert (_band_db(x, sr, 2.95, 3.30, 1900, 4500)
            - _band_db(y, sr, 2.95, 3.30, 1900, 4500)) > 12
    # de toon eronder loopt gewoon door (300 Hz-band vrijwel onveranderd)
    assert abs(_band_db(x, sr, 2.95, 3.30, 250, 350)
               - _band_db(y, sr, 2.95, 3.30, 250, 350)) < 1.0
    # ruim buiten de patch: bit-voor-bit onaangetast
    assert np.array_equal(x[:, :int(2.7 * sr)], y[:, :int(2.7 * sr)])
    assert np.array_equal(x[:, int(3.6 * sr):], y[:, int(3.6 * sr):])


def test_full_band_repair_fills_from_context(sr):
    x = _tone_with_squeak(sr)
    y = spectral_repair(x, sr, 3.0, 3.25)  # volledige band
    # de toon loopt coherent door de patch: zelfde niveau als een even lang
    # schoon venster elders (vensterlengtes gelijk houden voor de band-RMS)
    lvl_patch = _band_db(y, sr, 3.02, 3.23, 250, 350)
    lvl_ctx = _band_db(x, sr, 2.02, 2.23, 250, 350)
    assert abs(lvl_patch - lvl_ctx) < 1.5
    # en de rms-envelope in de patch is vlak op toonniveau (-20 dB)
    mid = y[0, int(3.05 * sr):int(3.20 * sr)].astype(np.float64)
    assert abs(10 * np.log10(np.mean(mid**2)) - (-20.0)) < 1.0


def test_repair_is_deterministic(sr):
    x = _tone_with_squeak(sr)
    y1 = spectral_repair(x, sr, 2.95, 3.30, low_hz=1500, high_hz=5000)
    y2 = spectral_repair(x, sr, 2.95, 3.30, low_hz=1500, high_hz=5000)
    assert np.array_equal(y1, y2)


def test_repair_validates_input(sr):
    x = _tone_with_squeak(sr)
    with pytest.raises(ValueError, match="te kort"):
        spectral_repair(x, sr, 3.0, 3.001)
    with pytest.raises(ValueError, match="5 s"):
        spectral_repair(x, sr, 1.0, 9.0)
    with pytest.raises(ValueError, match="high_hz"):
        spectral_repair(x, sr, 3.0, 3.2, low_hz=4000, high_hz=1000)
