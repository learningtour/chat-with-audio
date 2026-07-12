"""Dialoog-suite: ademreductie, plosief-reparatie en muziekbed-ducking."""

import numpy as np

from chat_with_audio.dsp import dialogue


def _rms_db(seg):
    return 10 * np.log10(np.mean(np.asarray(seg, dtype=np.float64) ** 2) + 1e-20)


def _speech_with_breaths(sr, rng):
    """8 s: spraakzinnen (300 Hz, -18 dB) met een adem (ruis, -36 dB, 0.4 s)
    vlak voor elke zin; ademposities: 1.6-2.0 en 4.6-5.0 s."""
    t = np.arange(sr * 8) / sr
    x = np.zeros_like(t)
    for start in (2.0, 5.0):
        zin = (t >= start) & (t < start + 2.0)
        x += 10 ** (-18 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 300 * t) * zin
    breath = rng.normal(0, 10 ** (-36 / 20), t.size)
    for start in (1.6, 4.6):
        m = (t >= start) & (t < start + 0.4)
        x += breath * m
    return x.astype(np.float32)[None, :]


def test_breath_control_dims_breaths_not_speech(sr):
    rng = np.random.default_rng(9)
    x = _speech_with_breaths(sr, rng)
    y, n = dialogue.breath_control(x, sr, reduction_db=10.0)
    assert n == 2, f"verwacht 2 adems, kreeg {n}"
    a, b = int(1.68 * sr), int(1.92 * sr)
    assert _rms_db(x[0, a:b]) - _rms_db(y[0, a:b]) > 6, "adem moet ~10 dB zakken"
    s, e = int(2.3 * sr), int(3.7 * sr)
    assert abs(_rms_db(x[0, s:e]) - _rms_db(y[0, s:e])) < 0.5, "spraak blijft staan"


def test_breath_control_leaves_clean_speech_alone(sr):
    t = np.arange(sr * 6) / sr
    x = (0.1 * np.sin(2 * np.pi * 300 * t)
         * (np.sin(2 * np.pi * 5.0 * t) > 0)).astype(np.float32)[None, :]
    y, n = dialogue.breath_control(x, sr)
    assert n == 0
    assert np.array_equal(x, y)


def test_deplosive_fixes_pop_only_there(sr):
    t = np.arange(sr * 8) / sr
    speech = 0.1 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 5.0 * t) > 0)
    pop = 0.25 * np.sin(2 * np.pi * 60 * t) * ((t >= 3.0) & (t < 3.08))
    x = (speech + pop).astype(np.float32)[None, :]
    y, n = dialogue.deplosive(x, sr)
    assert n >= 1

    def low_energy(sig, a, b):
        seg = sig[0, int(a * sr):int(b * sr)].astype(np.float64)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(seg.size, 1 / sr)
        return 20 * np.log10(spec[(freqs > 40) & (freqs < 100)].max() + 1e-12)

    assert low_energy(x, 2.95, 3.15) - low_energy(y, 2.95, 3.15) > 8
    # ver van de pop: bit-voor-bit onaangetast
    assert np.array_equal(x[:, :int(2.5 * sr)], y[:, :int(2.5 * sr)])
    assert np.array_equal(x[:, int(4.0 * sr):], y[:, int(4.0 * sr):])


def test_duck_music_rides_bed_under_speech_level(sr):
    t8 = np.arange(sr * 8) / sr
    speech = (10 ** (-20 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 300 * t8)
              * (np.sin(2 * np.pi * 5.0 * t8) > 0)
              * (np.sin(2 * np.pi * 0.25 * t8) > -0.6))  # echte zinspauzes
    music = (10 ** (-12 / 20) * (np.sin(2 * np.pi * 220 * t8)
             + 0.6 * np.sin(2 * np.pi * 331 * t8)
             + 0.4 * np.sin(2 * np.pi * 495 * t8)) / 2.0)
    x = np.concatenate([speech, music]).astype(np.float32)[None, :]
    y, info = dialogue.duck_music(x, sr, gap_db=6.0)
    assert info["ducked"], info
    bed = info["ducked"][0]
    assert bed["cut_db"] > 3
    before = _rms_db(x[0, int(10 * sr):int(14 * sr)])
    after = _rms_db(y[0, int(10 * sr):int(14 * sr)])
    assert before - after > 3, "bed moet hoorbaar zakken"
    # spraak (ver voor de segmentgrens) onaangetast
    assert np.array_equal(x[:, :int(7.0 * sr)], y[:, :int(7.0 * sr)])


def test_sidechain_gain_ducks_and_releases(sr):
    """Vocals aan: gain naar duck-niveau (snelle attack); pauze: terug naar
    1.0 met trage release."""
    t = np.arange(sr * 6) / sr
    vocals = (0.2 * np.sin(2 * np.pi * 300 * t)
              * ((t >= 1.0) & (t < 3.0))).astype(np.float32)[None, :]
    g = dialogue.sidechain_gain(vocals, sr, duck_db=6.0,
                                attack_ms=15.0, release_ms=250.0)
    target = 10 ** (-6 / 20)
    # ruim binnen het actieve stuk: op duck-niveau
    assert abs(g[int(2.0 * sr)] - target) < 0.02
    # vóór de inzet: (vrijwel) open
    assert g[int(0.5 * sr)] > 0.97
    # attack is snel: 100 ms na de inzet al vrijwel op duck-niveau
    assert g[int(1.1 * sr)] < target + 0.06
    # release is traag: 100 ms na het einde nog duidelijk gedoken
    assert g[int(3.1 * sr)] < 0.8
    # maar na 1.5 s weer open
    assert g[int(4.5 * sr)] > 0.9
    # envelope is glad: geen sprongen
    assert np.abs(np.diff(g)).max() < 0.002


def test_duck_music_rejects_unknown_mode(sr):
    import pytest

    x = np.zeros((1, sr), dtype=np.float32)
    with pytest.raises(ValueError, match="beds|stems"):
        dialogue.duck_music(x, sr, mode="magisch")


def test_dialogue_steps_available_in_chain(sr):
    from chat_with_audio.chain import STEP_REGISTRY, run_chain

    assert {"breath_control", "deplosive", "duck_music"} <= set(STEP_REGISTRY)
    t = np.arange(sr * 4) / sr
    x = (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)[None, :]
    y, resolved = run_chain(x, sr, [{"type": "breath_control"},
                                    {"type": "deplosive"}])
    assert len(resolved) == 2
    assert y.shape == x.shape


def test_dialogue_polish_recipe_exists():
    from chat_with_audio import recipes

    rec = recipes.load_recipe("dialogue-polish")
    types = [s["type"] for s in rec["steps"]]
    assert "breath_control" in types and "deplosive" in types
