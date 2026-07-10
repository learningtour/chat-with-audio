import numpy as np

from audio_improve_toolkit import analysis


def test_sine_levels(sr):
    t = np.arange(sr * 5) / sr
    x = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)[None, :]  # peak -20 dB
    m = analysis.analyze(x, sr)
    assert abs(m["rms_db"] - (-23.0)) < 0.3          # sine-RMS = piek - 3.01 dB
    assert abs(m["sample_peak_db"] - (-20.0)) < 0.1
    assert abs(m["true_peak_dbtp"] - (-20.0)) < 0.5
    assert abs(m["crest_factor_db"] - 3.0) < 0.3
    assert m["clip_events"] == 0
    assert m["lufs_integrated"] is not None


def test_clipping_detected(sr):
    t = np.arange(sr) / sr
    x = np.clip(1.5 * np.sin(2 * np.pi * 440 * t), -1, 1).astype(np.float32)[None, :]
    m = analysis.analyze(x, sr)
    assert m["clipped_samples"] > 100
    assert m["clip_events"] > 0
    _, issues = analysis.score_and_issues(m)
    assert any(i["code"] == "clipping" for i in issues)


def test_hum_detected(sr):
    rng = np.random.default_rng(7)
    t = np.arange(sr * 4) / sr
    hum = sum(0.05 / h * np.sin(2 * np.pi * 50 * h * t) for h in (1, 2, 3))
    x = (hum + rng.normal(0, 0.001, t.shape[0])).astype(np.float32)[None, :]
    m = analysis.analyze(x, sr)
    assert m["hum"]["detected"]
    assert m["hum"]["freq"] == 50.0


def test_noise_floor_and_snr(sr, noisy_bursts):
    m = analysis.analyze(noisy_bursts[None, :], sr)
    assert m["noise_floor_db"] < -38          # ruis op -45 dBFS in de pauzes
    assert m["snr_db"] > 8
    assert m["silence_pct"] > 5               # pauzes aanwezig
