"""Room-tone fill: gaten gevuld met eigen ambience, rest onaangetast."""

import numpy as np

from chat_with_audio.dsp import roomtone


def _recording_with_gap(sr, rng, gap=(4.0, 4.3)):
    """10 s 'locatieopname': spraakzinnen + doorlopende room tone (-52 dB),
    met een digitaal gat (exacte nullen) op gap."""
    t = np.arange(sr * 10) / sr
    speech = (0.1 * np.sin(2 * np.pi * 300 * t)
              * (np.sin(2 * np.pi * 5.0 * t) > 0)
              * (np.sin(2 * np.pi * 0.25 * t) > -0.3))
    tone = rng.normal(0, 10 ** (-52 / 20), t.size)
    x = (speech + tone).astype(np.float32)
    a, b = int(gap[0] * sr), int(gap[1] * sr)
    x[a:b] = 0.0
    return x[None, :], a, b


def _rms_db(seg):
    return 10 * np.log10(np.mean(np.asarray(seg, dtype=np.float64) ** 2) + 1e-20)


def test_gap_filled_at_ambience_level(sr):
    rng = np.random.default_rng(4)
    x, a, b = _recording_with_gap(sr, rng)
    y, info = roomtone.fill_room_tone(x, sr)
    assert len(info["filled"]) == 1
    assert abs(info["filled"][0]["start_s"] - 4.0) < 0.05

    mid = y[0, a + int(0.05 * sr):b - int(0.05 * sr)]
    assert not np.any(np.abs(mid) < 1e-9), "gat mag geen exacte nullen meer bevatten"
    level = _rms_db(mid)
    assert abs(level - (-52.0)) < 4.0, f"vulling op ambienceniveau, kreeg {level:.1f}"
    # buiten het gat: bit-voor-bit onaangetast
    assert np.array_equal(x[:, :a], y[:, :a])
    assert np.array_equal(x[:, b:], y[:, b:])


def test_fill_is_deterministic(sr):
    rng = np.random.default_rng(4)
    x, _, _ = _recording_with_gap(sr, rng)
    y1, _ = roomtone.fill_room_tone(x, sr)
    y2, _ = roomtone.fill_room_tone(x, sr)
    assert np.array_equal(y1, y2)


def test_no_gaps_is_noop(sr):
    rng = np.random.default_rng(4)
    t = np.arange(sr * 6) / sr
    x = (0.05 * np.sin(2 * np.pi * 300 * t)
         + rng.normal(0, 10 ** (-52 / 20), t.size)).astype(np.float32)[None, :]
    y, info = roomtone.fill_room_tone(x, sr)
    assert info["filled"] == []
    assert "geen digitale gaten" in info["reason"]
    assert np.array_equal(x, y)


def test_no_donor_reports_reason(sr):
    """Puur synthetisch materiaal zonder ambience: gat, maar geen donor."""
    t = np.arange(sr * 6) / sr
    x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    a = int((3.0 + 1 / (4 * 440)) * sr)
    x[a:a + int(0.1 * sr)] = 0.0
    y, info = roomtone.fill_room_tone(x[None, :], sr)
    assert info["filled"] == []
    assert "room tone" in info["reason"]
    assert np.array_equal(x[None, :], y)


def test_natural_quiet_is_not_filled(sr):
    """Natuurlijke stilte (met room tone erin) is geen gat."""
    rng = np.random.default_rng(4)
    t = np.arange(sr * 8) / sr
    speech = 0.1 * np.sin(2 * np.pi * 300 * t) * ((t < 3) | (t > 5))
    tone = rng.normal(0, 10 ** (-52 / 20), t.size)  # ambience overal aanwezig
    x = (speech + tone).astype(np.float32)[None, :]
    y, info = roomtone.fill_room_tone(x, sr)
    assert info["filled"] == []
    assert np.array_equal(x, y)
