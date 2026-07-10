"""E2E: synthetisch ruizig signaal -> improve -> aantoonbaar beter."""

import numpy as np

from audio_improve_toolkit import analysis, chain, improve, io, sessions


def test_improve_end_to_end(tmp_path, sr, noisy_wav):
    x, file_sr = io.load_audio(noisy_wav)
    assert file_sr == sr

    m0 = analysis.analyze(x, sr)
    profile, steps, rationale = improve.build_improve_chain(m0, profile="music")
    assert steps[-1]["type"] == "loudness_normalize"
    assert any(s["type"] == "denoise" for s in steps), "lage SNR moet denoise triggeren"
    assert len(rationale) >= 3

    y, resolved = chain.run_chain(x, sr, steps)
    m1 = analysis.analyze(y, sr)

    assert abs(m1["lufs_integrated"] - (-14.0)) <= 1.0
    assert m1["true_peak_dbtp"] <= -0.8
    # loudness-normalisatie tilt alles op; SNR is de eerlijke maat voor ontruising
    assert m1["snr_db"] >= m0["snr_db"] + 8

    session = sessions.create_session(noisy_wav, x, sr, m0, y, m1, resolved,
                                      rationale, profile)
    d = sessions.session_path(session["session_id"])
    for name in ("original.wav", "processed.wav", "analysis_original.json",
                 "analysis_processed.json", "chain.json", "session.json",
                 "waveform_original.json", "waveform_processed.json",
                 "spectrogram_original.png", "spectrogram_processed.png"):
        assert (d / name).exists(), f"{name} ontbreekt in de sessiemap"

    loaded = sessions.load_session(session["session_id"])
    assert loaded["deltas"]["lufs_integrated"] > 10


def test_leveler_balances_levels(sr):
    t = np.arange(sr * 8) / sr
    sine = np.sin(2 * np.pi * 440 * t)
    amp = np.where(t < 4, 10 ** (-40 / 20), 10 ** (-10 / 20))  # zacht -> luid, 30 dB gat
    x = (sine * amp).astype(np.float32)[None, :]
    y, _ = chain.run_chain(x, sr, [{"type": "leveler", "target_db": -18,
                                    "max_boost_db": 20, "max_cut_db": 18,
                                    "floor_db": -50}])
    rms = lambda s: 10 * np.log10(np.mean(np.asarray(s, dtype=np.float64) ** 2) + 1e-20)
    gap_in = rms(x[:, 5 * sr:7 * sr]) - rms(x[:, sr:3 * sr])
    gap_out = rms(y[:, 5 * sr:7 * sr]) - rms(y[:, sr:3 * sr])
    assert gap_in > 25
    assert abs(gap_out) < 8


def test_normalize_loudness_no_clip(sr):
    rng = np.random.default_rng(5)
    t = np.arange(sr * 6) / sr
    x = (0.02 * np.sin(2 * np.pi * 300 * t)
         + 0.01 * rng.normal(0, 1, t.shape[0])).astype(np.float32)[None, :]
    y, info = chain.normalize_loudness(x, sr, target_lufs=-14.0, true_peak_db=-1.0)
    m = analysis.analyze(y, sr)
    assert abs(m["lufs_integrated"] - (-14.0)) <= 1.0
    assert m["true_peak_dbtp"] <= -0.8
    assert info["gain_db"] > 10
