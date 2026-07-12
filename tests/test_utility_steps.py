"""Fase C-batch: utility-steps, dynamiek-uitbreidingen, dither, toon & two-pop."""

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import io
from chat_with_audio.chain import run_chain
from chat_with_audio.dsp import dynamics_plus, generate, utility

# ------------------------------------------------------------------- utility

def test_trim_explicit_auto_and_pad(sr):
    t = np.arange(sr * 6) / sr
    x = np.zeros_like(t, dtype=np.float32)
    a, b = int(2.0 * sr), int(4.0 * sr)
    x[a:b] = 0.2 * np.sin(2 * np.pi * 300 * t[a:b]).astype(np.float32)

    y = utility.trim(x, sr, start_s=1.0, end_s=5.0)
    assert y.shape[1] == 4 * sr

    y2 = utility.trim(x, sr, to_modulation=True, keep_s=0.25)
    assert y2.shape[1] / sr == pytest.approx(2.5, abs=0.05)  # 2 s + 2x marge
    assert float(np.abs(y2[:, :int(0.2 * sr)]).max()) < 1e-6  # marge is stilte

    y3 = utility.trim(x, sr, start_s=2.0, end_s=4.0, pad_head_s=0.5, pad_tail_s=0.25)
    assert y3.shape[1] == int(2.75 * sr)
    assert float(np.abs(y3[:, :int(0.5 * sr)]).max()) == 0.0

    with pytest.raises(ValueError, match="niets over"):
        utility.trim(x, sr, start_s=5.0, end_s=5.0)


def test_polarity_and_sample_delay(sr):
    t = np.arange(sr) / sr
    x = np.stack([np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 200 * t)]
                 ).astype(np.float32)
    y = utility.polarity_invert(x)
    assert np.array_equal(y, -x)
    y1 = utility.polarity_invert(x, channels=[1])
    assert np.array_equal(y1[0], x[0]) and np.array_equal(y1[1], -x[1])

    d = utility.sample_delay(x, sr, samples=100, channel=1)
    assert d.shape == x.shape
    assert np.array_equal(d[0], x[0])
    assert np.array_equal(d[1, 100:], x[1, :-100])
    assert float(np.abs(d[1, :100]).max()) == 0.0
    dm = utility.sample_delay(x, sr, ms=-10.0)  # alles 10 ms eerder
    lead = int(0.010 * sr)
    assert np.array_equal(dm[0, :-lead], x[0, lead:])
    with pytest.raises(ValueError, match="samples óf ms"):
        utility.sample_delay(x, sr)


def test_channel_map_modes(sr):
    x = np.stack([np.ones(100), -np.ones(100)]).astype(np.float32)
    mono = utility.channel_map(x, mode="to_mono")
    assert mono.shape == (1, 100) and float(np.abs(mono).max()) < 1e-7
    dm = utility.channel_map(x[:1], mode="dual_mono")
    assert dm.shape == (2, 100) and np.array_equal(dm[0], dm[1])
    sw = utility.channel_map(x, mode="swap")
    assert np.array_equal(sw[0], x[1]) and np.array_equal(sw[1], x[0])
    ordered = utility.channel_map(x, order=[1, 1, 0])
    assert ordered.shape == (3, 100)
    with pytest.raises(ValueError, match="mode"):
        utility.channel_map(x, mode="kwak")


def test_mid_side_width(sr):
    t = np.arange(sr) / sr
    m = np.sin(2 * np.pi * 300 * t)
    s = 0.3 * np.sin(2 * np.pi * 700 * t)
    x = np.stack([m + s, m - s]).astype(np.float32)
    narrow = utility.mid_side(x, width=0.0)
    assert np.allclose(narrow[0], narrow[1], atol=1e-6)  # mono
    wide = utility.mid_side(x, width=2.0)
    side_orig = float(((x[0] - x[1]) ** 2).mean())
    side_wide = float(((wide[0] - wide[1]) ** 2).mean())
    assert side_wide == pytest.approx(4.0 * side_orig, rel=0.01)
    mono_in = utility.mid_side(x[:1], width=2.0)  # mono blijft onaangetast
    assert mono_in.shape == (1, sr)


def test_bass_mono(sr):
    t = np.arange(sr * 2) / sr
    low_l = np.sin(2 * np.pi * 60 * t)
    low_r = np.sin(2 * np.pi * 60 * t + np.pi / 2)  # laag uit fase
    high = 0.3 * np.sin(2 * np.pi * 3000 * t)
    x = np.stack([low_l + high, low_r - high]).astype(np.float32)
    y = utility.bass_mono(x, sr, freq=120.0)
    # laag is mono geworden: verschilenergie onder 120 Hz vrijwel weg
    diff = (y[0] - y[1]).astype(np.float64)
    spec = np.abs(np.fft.rfft(diff))
    f = np.fft.rfftfreq(diff.shape[0], 1.0 / sr)
    assert spec[f < 100].max() < 0.01 * spec[(f > 2900) & (f < 3100)].max()
    # en de som (mono-inhoud) is niet aangetast
    assert np.allclose(y[0] + y[1], x[0] + x[1], atol=1e-4)


# ------------------------------------------------------------- dynamics_plus

def test_expander_pulls_quiet_down(sr):
    t = np.arange(sr * 4) / sr
    loud = (np.abs(np.sin(2 * np.pi * 0.25 * t)) > 0.7).astype(np.float64)
    # pauzes op -54 dB piek: ruim onder de drempel van -45
    x = (np.sin(2 * np.pi * 300 * t) * (0.3 * loud + 0.002 * (1 - loud))
         ).astype(np.float32)
    y = dynamics_plus.expander(x, sr, threshold_db=-45.0, ratio=3.0,
                               range_db=24.0)

    def _rms_db(sig, mask):
        return 10 * np.log10((sig[0, mask].astype(np.float64) ** 2).mean() + 1e-20)

    quiet = np.repeat(loud < 0.5, 1)
    x2 = x[None, :]
    drop_quiet = _rms_db(x2, quiet) - _rms_db(y, quiet)
    drop_loud = _rms_db(x2, ~quiet) - _rms_db(y, ~quiet)
    assert drop_quiet > 8.0        # pauzes zakken duidelijk weg
    assert abs(drop_loud) < 1.0    # programma blijft staan


def test_multiband_compresses_only_hot_band(sr):
    t = np.arange(sr * 3) / sr
    low = 0.5 * np.sin(2 * np.pi * 100 * t)     # heet laag
    high = 0.02 * np.sin(2 * np.pi * 5000 * t)  # rustig hoog
    x = (low + high).astype(np.float32)
    y = dynamics_plus.multiband_compressor(x, sr, crossovers=[400.0, 2500.0],
                                           threshold_db=-18.0, ratio=4.0)

    def _band_energy_db(sig, lo, hi):
        spec = np.abs(np.fft.rfft(sig[0].astype(np.float64))) ** 2
        f = np.fft.rfftfreq(sig.shape[1], 1.0 / sr)
        return 10 * np.log10(float(spec[(f > lo) & (f < hi)].sum()) + 1e-20)

    x2 = x[None, :]
    low_drop = _band_energy_db(y, 80, 120) - _band_energy_db(x2, 80, 120)
    high_drop = _band_energy_db(y, 4800, 5200) - _band_energy_db(x2, 4800, 5200)
    assert low_drop < -5.0          # hete band is gecomprimeerd
    assert abs(high_drop) < 1.0     # rustige band vrijwel onaangetast

    # onbewerkt (drempel onhaalbaar hoog) reconstrueren de banden het origineel
    y0 = dynamics_plus.multiband_compressor(x, sr, threshold_db=0.0, ratio=1.0)
    assert np.abs(y0[0] - x).max() < 1e-3

    with pytest.raises(ValueError, match="banden"):
        dynamics_plus.multiband_compressor(x, sr, crossovers=[200.0],
                                           threshold_db=[-20, -20, -20])


def test_transient_shaper_attack_and_sustain(sr):
    # slagen met natrilling: aanzet + exponentiële decay (tau 80 ms)
    n = sr * 3
    x = np.zeros(n, dtype=np.float64)
    for start in range(sr // 2, n - sr, sr // 2):
        seg = np.arange(int(0.3 * sr))
        x[start:start + seg.size] += (0.4 * np.exp(-seg / (0.08 * sr))
                                      * np.sin(2 * np.pi * 800 * seg / sr))
    x = x.astype(np.float32)

    def _win_gain_db(y, t0, t1):
        a, b = int((0.5 + t0) * sr), int((0.5 + t1) * sr)
        return 10 * np.log10(((y[0, a:b].astype(np.float64) ** 2).mean() + 1e-20)
                             / ((x[a:b].astype(np.float64) ** 2).mean() + 1e-20))

    sharper = dynamics_plus.transient_shaper(x, sr, attack_db=6.0)
    assert _win_gain_db(sharper, 0.0, 0.010) > 4.0    # aanzet omhoog
    assert _win_gain_db(sharper, 0.15, 0.30) < 1.5    # staart blijft staan

    drier = dynamics_plus.transient_shaper(x, sr, sustain_db=-12.0)
    assert _win_gain_db(drier, 0.15, 0.30) < -8.0     # staart droger
    assert _win_gain_db(drier, 0.0, 0.010) > -1.0     # aanzet blijft staan


# ----------------------------------------------------------- dither (io.py)

def test_dither_fixes_truncation_distortion(tmp_path, sr):
    t = np.arange(sr * 2) / sr
    x = (10 ** (-90 / 20) * np.sin(2 * np.pi * 997 * t)).astype(np.float32)

    def _thd_db(sig):
        spec = np.abs(np.fft.rfft(sig.astype(np.float64))) ** 2
        f = np.fft.rfftfreq(sig.shape[0], 1.0 / sr)

        def band(freq):
            return float(spec[(f > freq - 5) & (f < freq + 5)].sum())

        return 10 * np.log10(sum(band(997.0 * k) for k in range(2, 9))
                             / (band(997.0) + 1e-30) + 1e-30)

    p_plain = tmp_path / "plain.wav"
    sf.write(str(p_plain), x, sr, subtype="PCM_16")  # kale conversie: truncatie
    plain, _ = io.load_audio(p_plain)
    p_dith = tmp_path / "dither.wav"
    io.save_wav(p_dith, x, sr, subtype="PCM_16")  # auto-dither bij PCM_16
    dith, _ = io.load_audio(p_dith)

    # rond de LSB geeft kale kwantisatie zware harmonische vervorming;
    # dither decorreleert de fout (gemeten: ~-8 dB THD vs ~-31 dB)
    assert _thd_db(plain[0]) > _thd_db(dith[0]) + 15.0
    corr = float(np.dot(dith[0].astype(np.float64), x.astype(np.float64))
                 / (np.linalg.norm(dith[0]) * np.linalg.norm(x) + 1e-20))
    assert corr > 0.5  # het signaal zelf blijft lineair behouden

    # hoogdoorlaat-dither: ditherruis zit vooral in het hoog (de witte
    # afrondingsruis blijft — vandaar geen extremere verhouding)
    spec = np.abs(np.fft.rfft(dith[0].astype(np.float64)))
    f = np.fft.rfftfreq(dith.shape[1], 1.0 / sr)
    lowband = spec[(f > 100) & (f < 2000)].mean()
    highband = spec[(f > 15000) & (f < 20000)].mean()
    assert highband > 1.5 * lowband


def test_dither_off_and_24bit_untouched(tmp_path, sr):
    t = np.arange(sr) / sr
    x = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "no_dither.wav"
    io.save_wav(p, x, sr, subtype="PCM_16", dither=False)
    y, _ = io.load_audio(p)
    err = np.abs(y[0] - x).max()
    assert err < 1.5 / 32768  # gewone kwantisatie, geen ditherruis erbovenop

    p24 = tmp_path / "p24.wav"
    io.save_wav(p24, x, sr)  # PCM_24: geen dither-pad
    assert io.probe(p24)["bit_depth"] == 24


# ------------------------------------------------------ toon & two-pop

def test_leader_layout_and_levels(sr):
    t = np.arange(sr * 2) / sr
    prog = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    y, info = generate.leader(prog, sr, tone_s=4.0, tone_db=-18.0, gap_s=3.0)
    assert y.shape[1] == int((4.0 + 3.0 + 2.0) * sr)
    assert info["program_start_s"] == pytest.approx(7.0)
    # toonpiek op -18 dBFS
    tone_peak_db = 20 * np.log10(float(np.abs(y[0, :4 * sr]).max()))
    assert tone_peak_db == pytest.approx(-18.0, abs=0.2)
    # pop begint exact 2 s voor het programma en is één 24fps-frame lang
    pop = info["two_pop"]
    assert info["program_start_s"] - pop["start_s"] == pytest.approx(2.0, abs=0.01)
    a = int((pop["start_s"] + 0.005) * sr)
    assert float(np.abs(y[0, a:a + 100]).max()) > 0.05
    before = int((pop["start_s"] - 0.1) * sr)
    assert float(np.abs(y[0, before:before + 100]).max()) == 0.0
    # programma zelf bit-voor-bit intact
    assert np.array_equal(y[:, int(7.0 * sr):], prog[None, :])

    with pytest.raises(ValueError, match="gap_s"):
        generate.leader(prog, sr, gap_s=1.0)


# ------------------------------------------------------------- chain-koppeling

def test_phase_c_steps_in_chain(sr):
    t = np.arange(sr * 2) / sr
    x = np.stack([0.3 * np.sin(2 * np.pi * 200 * t),
                  0.3 * np.sin(2 * np.pi * 200 * t + 0.5)]).astype(np.float32)
    y, resolved = run_chain(x, sr, [
        {"type": "trim", "start_s": 0.5},
        {"type": "polarity_invert"},
        {"type": "mid_side", "width": 1.2},
        {"type": "bass_mono", "freq": 100},
        {"type": "channel_map", "mode": "to_mono"},
        {"type": "expander", "threshold_db": -60},
        {"type": "transient_shaper", "attack_db": 2},
        {"type": "multiband_compressor", "threshold_db": -20},
        {"type": "sample_delay", "samples": 10},
    ])
    assert y.shape == (1, int(1.5 * sr))
    assert len(resolved) == 9
