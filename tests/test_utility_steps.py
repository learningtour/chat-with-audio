"""Gereedschapsstappen (fase C): trim, kanalen, fase, dynamiek, M/S, leader,
dither. Elke stap wordt via de keten (STEP_REGISTRY) aangeroepen zodat ook de
registratie en parametervalidatie meelopen.
"""

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import chain, io
from chat_with_audio.dsp import utility

SR = 44100


def run(x, steps, sr=SR):
    y, _ = chain.run_chain(x, sr, steps)
    return y


@pytest.fixture
def stereo():
    """2 s stereo: L = 440 Hz, R = 880 Hz, -20 dBFS RMS."""
    t = np.arange(2 * SR) / SR
    amp = 10 ** (-20 / 20) * np.sqrt(2)
    return np.vstack([amp * np.sin(2 * np.pi * 440 * t),
                      amp * np.sin(2 * np.pi * 880 * t)]).astype(np.float32)


def rms_db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


# ---------------------------------------------------------------- trim & tijd

def test_trim_explicit_window(stereo):
    y = run(stereo, [{"type": "trim", "start_s": 0.5, "end_s": 1.5}])
    assert y.shape == (2, SR)


def test_trim_to_modulation():
    x = np.zeros(4 * SR, dtype=np.float32)
    x[SR:2 * SR] = 0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    y = run(x, [{"type": "trim", "to_modulation": True, "pad_s": 0.2}])
    # 1 s programma + ~0.2 s pad aan beide kanten
    assert y.shape[1] == pytest.approx(1.4 * SR, rel=0.05)


def test_trim_empty_window_raises(stereo):
    with pytest.raises(ValueError):
        run(stereo, [{"type": "trim", "start_s": 1.5, "end_s": 0.5}])


def test_insert_silence_shifts_programme(stereo):
    y = run(stereo, [{"type": "insert_silence", "at_s": 0.0, "duration_s": 0.5}])
    assert y.shape[1] == stereo.shape[1] + int(0.5 * SR)
    assert np.max(np.abs(y[:, : int(0.5 * SR)])) == 0.0
    np.testing.assert_allclose(y[:, int(0.5 * SR):], stereo, atol=1e-7)


# ---------------------------------------------------------------- fase & kanalen

def test_polarity_invert_roundtrip(stereo):
    y = run(stereo, [{"type": "polarity_invert"}])
    np.testing.assert_allclose(y, -stereo, atol=1e-7)
    z = run(stereo, [{"type": "polarity_invert", "channel": "left"}])
    np.testing.assert_allclose(z[0], -stereo[0], atol=1e-7)
    np.testing.assert_allclose(z[1], stereo[1], atol=1e-7)


def test_sample_delay_length_preserving(stereo):
    y = run(stereo, [{"type": "sample_delay", "channel": "right", "samples": 100}])
    assert y.shape == stereo.shape
    np.testing.assert_allclose(y[1, 100:200], stereo[1, 0:100], atol=1e-7)
    assert np.max(np.abs(y[1, :100])) == 0.0


def test_sample_delay_ms_and_negative(stereo):
    y = utility.sample_delay(stereo, SR, "left", ms=-1.0)
    d = int(0.001 * SR)
    np.testing.assert_allclose(y[0, :100], stereo[0, d:d + 100], atol=1e-7)


def test_to_mono_and_dual_mono(stereo):
    m = run(stereo, [{"type": "to_mono"}])
    assert m.shape[0] == 1
    np.testing.assert_allclose(m[0], stereo.mean(axis=0), atol=1e-7)
    dm = run(stereo, [{"type": "dual_mono", "source": "left"}])
    assert dm.shape[0] == 2
    np.testing.assert_allclose(dm[0], dm[1], atol=0)
    np.testing.assert_allclose(dm[0], stereo[0], atol=1e-7)


def test_swap_channels(stereo):
    y = run(stereo, [{"type": "swap_channels"}])
    np.testing.assert_allclose(y[0], stereo[1], atol=0)
    np.testing.assert_allclose(y[1], stereo[0], atol=0)


def test_swap_requires_stereo():
    with pytest.raises(ValueError):
        utility.swap_channels(np.zeros(100, dtype=np.float32), SR)


def test_mid_side_width_zero_is_mono(stereo):
    y = run(stereo, [{"type": "mid_side", "width": 0.0}])
    np.testing.assert_allclose(y[0], y[1], atol=1e-7)
    np.testing.assert_allclose(y[0], stereo.mean(axis=0), atol=1e-6)


def test_mid_side_width_one_is_identity(stereo):
    y = run(stereo, [{"type": "mid_side", "width": 1.0}])
    np.testing.assert_allclose(y, stereo, atol=1e-6)


def test_bass_mono_monos_low_keeps_high():
    t = np.arange(2 * SR) / SR
    low = 0.1 * np.sin(2 * np.pi * 60 * t)
    high = 0.1 * np.sin(2 * np.pi * 3000 * t)
    x = np.vstack([low + high, -low + high]).astype(np.float32)  # laag in tegenfase
    y = run(x, [{"type": "bass_mono", "freq": 200}])
    # tegenfase-laag valt weg in de mono-som; hoog blijft
    mono_sum = y.mean(axis=0)
    spec = np.abs(np.fft.rfft(y[0, SR:]))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    e_low = spec[np.abs(freqs - 60).argmin()]
    e_high = spec[np.abs(freqs - 3000).argmin()]
    assert e_low < e_high / 50  # laag is (vrijwel) weg na mono-som van tegenfase
    assert rms_db(mono_sum) > -30  # hoog overleeft


# ---------------------------------------------------------------- dynamiek

def test_expander_pushes_quiet_down_keeps_loud():
    t = np.arange(2 * SR) / SR
    x = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    x[SR:] *= 10 ** (-40 / 20)  # tweede seconde 40 dB zachter (ruisbed)
    y = run(x, [{"type": "expander", "threshold_db": -30, "ratio": 3,
                 "range_db": 24}])[0]
    loud_delta = rms_db(y[: SR // 2]) - rms_db(x[: SR // 2])
    quiet_delta = rms_db(y[SR + SR // 4:]) - rms_db(x[SR + SR // 4:])
    assert abs(loud_delta) < 1.0       # boven drempel: ongemoeid
    assert quiet_delta < -10.0         # onder drempel: flink omlaag


def test_multiband_keeps_flat_signal_flat(stereo):
    # neutraal ingesteld (drempel boven signaal): LR4-som moet vlak blijven
    y = run(stereo, [{"type": "multiband_compressor", "threshold_db": 0.0,
                      "ratio": 2.0}])
    assert abs(rms_db(y) - rms_db(stereo)) < 0.2


def test_multiband_compresses_loud_band_only():
    t = np.arange(2 * SR) / SR
    low = 0.4 * np.sin(2 * np.pi * 80 * t)     # luide bas
    high = 0.02 * np.sin(2 * np.pi * 5000 * t)  # zachte hoog
    x = (low + high).astype(np.float32)
    y = run(x, [{"type": "multiband_compressor", "crossovers": [500],
                 "threshold_db": -18, "ratio": 4}])[0]
    spec_x = np.abs(np.fft.rfft(x[SR // 2: SR]))
    spec_y = np.abs(np.fft.rfft(y[SR // 2: SR]))
    freqs = np.fft.rfftfreq(SR // 2, 1 / SR)
    i80, i5k = np.abs(freqs - 80).argmin(), np.abs(freqs - 5000).argmin()
    low_delta = 20 * np.log10(spec_y[i80] / spec_x[i80])
    high_delta = 20 * np.log10(spec_y[i5k] / spec_x[i5k])
    assert low_delta < -3.0        # bas gecomprimeerd
    assert abs(high_delta) < 1.0   # hoog ongemoeid


def _burst_train():
    """Kloppend signaal: 200 ms exponentieel uitstervende bursts op 0.5 s."""
    x = np.zeros(2 * SR, dtype=np.float32)
    t = np.arange(int(0.2 * SR)) / SR
    for k in range(4):
        i = int(k * 0.5 * SR)
        x[i:i + t.shape[0]] = 0.3 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 20)
    return x


def _attack_sustain_ratio(x):
    """Energie in de aanzet (eerste 15 ms) vs de staart (100-200 ms) van burst 2."""
    i = int(0.5 * SR)
    attack = x[i:i + int(0.015 * SR)]
    sustain = x[i + int(0.10 * SR): i + int(0.20 * SR)]
    return rms_db(attack) - rms_db(sustain)


def test_transient_shaper_attack_vs_sustain():
    x = _burst_train()
    base = _attack_sustain_ratio(x)
    ya = run(x, [{"type": "transient_shaper", "attack_db": 6.0}])[0]
    ys = run(x, [{"type": "transient_shaper", "sustain_db": 6.0}])[0]
    assert _attack_sustain_ratio(ya) > base + 2.0   # aanzet naar voren
    assert _attack_sustain_ratio(ys) < base - 2.0   # staart/kamer omhoog


def test_tilt_eq_brightens(stereo):
    y = run(stereo, [{"type": "tilt_eq", "tilt_db": 6.0}])
    # L draagt 440 Hz (onder pivot), R 880: kijk naar spectrale balans L
    spec_x = np.abs(np.fft.rfft(stereo[0]))
    spec_y = np.abs(np.fft.rfft(y[0]))
    freqs = np.fft.rfftfreq(stereo.shape[1], 1 / SR)
    i = np.abs(freqs - 440).argmin()
    assert 20 * np.log10(spec_y[i] / spec_x[i]) < -1.0  # onder pivot: omlaag


# ---------------------------------------------------------------- leader

def test_tone_slate_prepends_reference_tone(stereo):
    y = run(stereo, [{"type": "tone_slate", "tone_s": 2.0, "level_db": -18,
                      "gap_s": 0.5}])
    assert y.shape[1] == stereo.shape[1] + int(2.5 * SR)
    tone = y[0, int(0.5 * SR): int(1.5 * SR)]
    assert rms_db(tone) == pytest.approx(-18.0, abs=0.3)
    gap = y[:, int(2.0 * SR): int(2.5 * SR)]
    assert np.max(np.abs(gap)) == 0.0


def test_two_pop_at_offset(stereo):
    y = run(stereo, [{"type": "two_pop", "offset_s": 2.0}])
    assert y.shape[1] == stereo.shape[1] + 2 * SR
    pop = y[0, : int(0.042 * SR)]
    assert np.max(np.abs(pop)) > 0.05
    silence = y[:, int(0.05 * SR): 2 * SR]
    assert np.max(np.abs(silence)) == 0.0
    np.testing.assert_allclose(y[:, 2 * SR:], stereo, atol=1e-7)


# ---------------------------------------------------------------- dither

def test_save_wav_16bit_dithers(tmp_path):
    # -60 dBFS toon: zonder dither is het kwantisatieresidu sterk gecorreleerd
    # met het signaal (vervorming); met TPDF-dither wordt het ruis.
    t = np.arange(SR) / SR
    x = (10 ** (-60 / 20) * np.sin(2 * np.pi * 997 * t)).astype(np.float32)
    p16 = io.save_wav(tmp_path / "d.wav", x, SR, subtype="PCM_16")
    y, _ = sf.read(str(p16), dtype="float64")
    info = sf.info(str(p16))
    assert info.subtype == "PCM_16"
    resid = y - x.astype(np.float64)
    # dither aanwezig: residu is geen pure kwantisatietrap (meer dan 3 niveaus)
    levels = np.unique(np.rint(resid * 32768.0))
    assert levels.size > 3
    # en het residu blijft klein: < 2 LSB piek
    assert np.max(np.abs(resid)) < 2.5 / 32768.0


def test_save_wav_16bit_dither_can_be_disabled(tmp_path):
    t = np.arange(SR) / SR
    x = (10 ** (-60 / 20) * np.sin(2 * np.pi * 997 * t)).astype(np.float32)
    p = io.save_wav(tmp_path / "nd.wav", x, SR, subtype="PCM_16", dither=False)
    y, _ = sf.read(str(p), dtype="float64")
    levels = np.unique(np.rint((y - x.astype(np.float64)) * 32768.0))
    assert levels.size <= 3  # kale kwantisatie: residu is de trap zelf


def test_compliance_new_specs_listed():
    from chat_with_audio import compliance

    ids = {s["id"] for s in compliance.list_specs()}
    assert {"op-59", "arib-tr-b32"} <= ids
    assert compliance.SPECS["op-59"]["true_peak_max"] == -2.0
    assert compliance.SPECS["arib-tr-b32"]["true_peak_max"] == -1.0
