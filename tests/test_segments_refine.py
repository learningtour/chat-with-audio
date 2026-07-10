"""Segmentatie + iteratieve verfijning op synthetisch spraak/muziek-materiaal."""

import numpy as np

from audio_improve_toolkit import refine
from audio_improve_toolkit.segments import classify_segments


def _theater(sr):
    """10 s: 4 s zachte 'spraak' (AM-ruisband, -35 dB), 1 s stilte, 5 s luide 'muziek' (-8 dB)."""
    rng = np.random.default_rng(11)
    t_sp = np.arange(sr * 4) / sr
    speech = (rng.normal(0, 1, sr * 4) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 4 * t_sp)))
              * 10 ** (-35 / 20)).astype(np.float32)
    gap = (rng.normal(0, 10 ** (-65 / 20), sr)).astype(np.float32)
    t_mu = np.arange(sr * 5) / sr
    music = (sum(np.sin(2 * np.pi * f * t_mu) for f in (220, 277, 330, 440))
             * 10 ** (-8 / 20) / 4).astype(np.float32)
    return np.concatenate([speech, gap, music])[None, :]


def test_classify_segments_speech_music_silence(sr):
    x = _theater(sr)
    segs = classify_segments(x, sr)
    assert segs[0]["start_s"] == 0.0
    assert abs(segs[-1]["end_s"] - 10.0) < 0.1
    for a, b in zip(segs, segs[1:]):
        assert abs(a["end_s"] - b["start_s"]) < 1e-6  # volledige dekking
    kinds = [s["kind"] for s in segs]
    assert kinds[0] == "speech"
    assert kinds[-1] == "music"


def test_refine_hits_targets(sr):
    x = _theater(sr)
    y, info = refine.refine(x, sr, speech_peak_db=-6.0, music_gap_db=2.0,
                            max_iterations=6, denoise="off", tone=False,
                            asr_check=False)
    rep = info["report"]
    final = rep["final_measurements"]
    assert abs(final["speech_peak_db"] - (-6.0)) <= 1.5
    assert abs(final["music_vs_speech_gap_db"] - 2.0) <= 1.5
    assert final["true_peak_est_db"] <= -1.2
    assert len(rep["iterations"]) >= 1
