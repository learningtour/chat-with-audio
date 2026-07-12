"""Ruimte & karakter (fase D): synthetische IR, convolutie, saturatie, delay,
RT60-schatting, futz-recepten en de match_room-tool.
"""

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import chain, recipes
from chat_with_audio.dsp import space

SR = 44100


def rms_db(x):
    return 10 * np.log10(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-20)


def speech_like(dur=8.0, seed=3):
    """Spraakachtig: 300 ms ruisbursts met pauzes — offsets voor RT60-meting."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur * SR), dtype=np.float32)
    t = 0.4
    while t < dur - 1.0:
        n = int(0.3 * SR)
        i = int(t * SR)
        burst = rng.standard_normal(n) * 0.25
        env = np.minimum(np.arange(n), n // 10) / (n // 10)
        x[i:i + n] = (burst * env * env[::-1]).astype(np.float32)
        t += 0.3 + float(rng.uniform(0.45, 0.7))
    return x


# ---------------------------------------------------------------- synth IR

def test_synth_ir_decays_and_is_normalized():
    ir = space.synth_ir(SR, rt60=0.5)
    assert np.sum(ir**2) == pytest.approx(1.0, rel=1e-6)
    n = ir.shape[0]
    assert n >= int(0.5 * 1.5 * SR)
    early = rms_db(ir[: n // 4])
    late = rms_db(ir[3 * n // 4:])
    assert early - late > 20  # exponentieel verval


def test_synth_ir_damping_kills_highs_faster():
    ir = space.synth_ir(SR, rt60=0.8, damping=0.6)
    n = ir.shape[0]
    late = ir[n // 2:]
    spec = np.abs(np.fft.rfft(late))
    freqs = np.fft.rfftfreq(late.shape[0], 1 / SR)
    lo = spec[(freqs > 150) & (freqs < 500)].mean()
    hi = spec[(freqs > 4000) & (freqs < 8000)].mean()
    assert 20 * np.log10(lo / hi) > 6  # staart is donkerder dan de kop


# ---------------------------------------------------------------- convolve

def test_convolve_ir_mix_zero_is_dry():
    x = speech_like(2.0)
    y, _ = chain.run_chain(x, SR, [{"type": "convolve_ir", "mix": 0.0}])
    np.testing.assert_allclose(y[0], x, atol=1e-6)


def test_convolve_ir_adds_tail_energy_in_gaps():
    x = speech_like(4.0)
    y, _ = chain.run_chain(x, SR, [{"type": "convolve_ir", "mix": 0.5,
                                    "rt60": 0.6}])
    assert y.shape[1] == x.shape[0]  # lengte-neutraal zonder keep_tail
    # direct na een burst-offset moet er nu galm staan waar stilte was
    i = int(0.7 * SR)  # net na de eerste burst (0.4-0.7 s)
    gap_before = rms_db(x[i + int(0.05 * SR): i + int(0.2 * SR)])
    gap_after = rms_db(y[0, i + int(0.05 * SR): i + int(0.2 * SR)])
    assert gap_after > gap_before + 10


def test_convolve_ir_keep_tail_extends():
    x = speech_like(2.0)
    y, _ = chain.run_chain(x, SR, [{"type": "convolve_ir", "mix": 0.4,
                                    "rt60": 0.5, "keep_tail": True}])
    assert y.shape[1] > x.shape[0]


def test_convolve_ir_from_file(tmp_path):
    ir = space.synth_ir(SR, rt60=0.3)
    p = tmp_path / "ir.wav"
    sf.write(str(p), ir.astype(np.float32), SR)
    x = speech_like(2.0)
    y, _ = chain.run_chain(x, SR, [{"type": "convolve_ir", "ir_path": str(p),
                                    "mix": 0.5}])
    assert y.shape[1] == x.shape[0]
    assert not np.allclose(y[0], x, atol=1e-4)


# ---------------------------------------------------------------- saturate & delay

def test_saturate_adds_harmonics_keeps_level():
    t = np.arange(2 * SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y, _ = chain.run_chain(x, SR, [{"type": "saturate", "drive_db": 12}])
    assert abs(rms_db(y[0]) - rms_db(x)) < 1.5
    spec = np.abs(np.fft.rfft(y[0] * np.hanning(y.shape[1])))
    freqs = np.fft.rfftfreq(y.shape[1], 1 / SR)
    f0 = spec[np.abs(freqs - 440).argmin()]
    h3 = spec[np.abs(freqs - 1320).argmin()]
    assert 20 * np.log10(h3 / f0) > -40  # duidelijke 3e harmonische


def test_saturate_mode_guard():
    with pytest.raises(ValueError):
        space.saturate(np.zeros(100, dtype=np.float32), SR, mode="warp")


def test_delay_produces_taps():
    x = np.zeros(SR, dtype=np.float32)
    x[1000] = 1.0
    y = space.delay(x, SR, time_ms=100.0, feedback=0.5, mix=0.5)[0]
    d = int(0.1 * SR)
    assert y[1000] == pytest.approx(0.5, abs=0.01)          # dry deel
    assert y[1000 + d] == pytest.approx(0.5, abs=0.01)      # tap 1
    assert y[1000 + 2 * d] == pytest.approx(0.25, abs=0.01)  # tap 2 (feedback)


# ---------------------------------------------------------------- RT60

def test_estimate_rt60_recovers_synthetic_room():
    dry = speech_like(10.0)
    for rt in (0.3, 0.8):
        wet, _ = chain.run_chain(dry, SR, [{"type": "convolve_ir", "mix": 0.85,
                                            "rt60": rt, "damping": 0.0}])
        est = space.estimate_rt60(wet[0], SR)
        assert est is not None
        assert est == pytest.approx(rt, rel=0.5)  # goede orde van grootte
    # en de droge opname meet aantoonbaar korter dan de natte
    est_dry = space.estimate_rt60(dry, SR)
    if est_dry is not None:
        assert est_dry < 0.25


def test_estimate_rt60_none_on_flat_signal():
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(4 * SR) * 0.1).astype(np.float32)
    assert space.estimate_rt60(x, SR) is None


# ---------------------------------------------------------------- futz & tool

def test_futz_recipes_load_and_run(tmp_path):
    listing = recipes.list_recipes()
    names = {r["name"] for r in listing}
    assert {"futz-telephone", "futz-walkie", "futz-megaphone",
            "futz-other-room", "futz-small-speaker"} <= names
    x = speech_like(2.0)
    p = tmp_path / "v.wav"
    sf.write(str(p), x, SR)
    from chat_with_audio import server

    res = server.apply_recipe(str(p), "futz-telephone")
    y, _ = sf.read(res["output_path"])
    # smalband: energie onder 200 en boven 5k moet flink weg zijn
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.shape[0], 1 / SR)
    mid = spec[(freqs > 500) & (freqs < 2500)].mean()
    low = spec[freqs < 150].mean()
    high = spec[freqs > 6000].mean()
    assert 20 * np.log10(mid / low) > 12
    assert 20 * np.log10(mid / high) > 12


def test_match_room_tool(tmp_path):
    from chat_with_audio import server

    dry = speech_like(6.0, seed=5)
    room, _ = chain.run_chain(speech_like(8.0, seed=9), SR,
                              [{"type": "convolve_ir", "mix": 0.8, "rt60": 0.7,
                                "damping": 0.0}])
    p_dry = tmp_path / "adr.wav"
    p_ref = tmp_path / "scene.wav"
    sf.write(str(p_dry), dry, SR)
    sf.write(str(p_ref), room[0], SR)
    res = server.match_room(str(p_dry), str(p_ref), mix=0.5)
    assert res["used_rt60_s"] > 0.2  # er is echt kamer gemeten of gezet
    y, _ = sf.read(res["output_path"])
    # de gematchte ADR moet nu meetbaar meer staart hebben dan de droge bron
    est = space.estimate_rt60(y.T if y.ndim == 2 else y, SR)
    assert est is not None and est > 0.2
