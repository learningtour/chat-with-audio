"""Pro-metering & QC: stereo-checks, dropouts, kop/staart-stilte, momentary/PLR."""

import numpy as np

from chat_with_audio import analysis


def _tone(sr, dur_s, freq=440.0, amp=0.1):
    t = np.arange(int(sr * dur_s)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_stereo_qc_healthy(sr):
    left = _tone(sr, 5)
    rng = np.random.default_rng(1)
    right = (left + 0.01 * rng.normal(0, 1, left.size).astype(np.float32))
    m = analysis.analyze(np.stack([left, right]), sr)
    s = m["stereo"]
    assert s["correlation"] > 0.9
    assert s["dead_channel"] is None
    assert not s["polarity_inverted"]
    assert not s["dual_mono"]
    assert abs(s["balance_db"]) < 1.0


def test_stereo_qc_dual_mono_and_issues(sr):
    left = _tone(sr, 5)
    m = analysis.analyze(np.stack([left, left.copy()]), sr)
    assert m["stereo"]["dual_mono"] is True
    _, issues = analysis.score_and_issues(m)
    assert any(i["code"] == "dual_mono" for i in issues)


def test_stereo_qc_polarity_inverted(sr):
    left = _tone(sr, 5)
    m = analysis.analyze(np.stack([left, -left]), sr)
    assert m["stereo"]["polarity_inverted"] is True
    _, issues = analysis.score_and_issues(m)
    assert any(i["code"] == "polarity_inverted" for i in issues)


def test_stereo_qc_dead_channel(sr):
    left = _tone(sr, 5)
    m = analysis.analyze(np.stack([left, np.zeros_like(left)]), sr)
    assert m["stereo"]["dead_channel"] == "right"
    _, issues = analysis.score_and_issues(m)
    assert any(i["code"] == "dead_channel" for i in issues)


def test_mono_has_no_stereo_block(sr):
    m = analysis.analyze(_tone(sr, 3)[None, :], sr)
    assert m["stereo"] is None


def test_dropout_detected_and_positioned(sr):
    x = _tone(sr, 6, amp=0.2)
    # gat begint op een kwartperiode (golfvorm op maximum): abrupt afgekapt,
    # zoals een echte dropout — niet netjes op een nuldoorgang
    a = int((3.0 + 1 / (4 * 440)) * sr)
    x[a:a + int(0.05 * sr)] = 0.0  # 50 ms digitaal gat midden in de toon
    m = analysis.analyze(x[None, :], sr)
    assert m["dropouts"]["count"] == 1
    assert abs(m["dropouts"]["positions_s"][0] - 3.0) < 0.05
    _, issues = analysis.score_and_issues(m)
    assert any(i["code"] == "dropouts" for i in issues)


def test_no_dropouts_in_gated_speech(sr):
    """Normale spraakpauzes (stilte tussen zinnen) zijn geen dropouts."""
    t = np.arange(sr * 8) / sr
    x = (0.1 * np.sin(2 * np.pi * 300 * t)
         * (np.sin(2 * np.pi * 0.25 * t) > 0)).astype(np.float32)
    m = analysis.analyze(x[None, :], sr)
    assert m["dropouts"]["count"] == 0


def test_edge_silence_measured(sr):
    x = np.concatenate([np.zeros(int(2.0 * sr), dtype=np.float32),
                        _tone(sr, 4, amp=0.2),
                        np.zeros(int(1.0 * sr), dtype=np.float32)])
    m = analysis.analyze(x[None, :], sr)
    assert abs(m["lead_silence_s"] - 2.0) < 0.15
    assert abs(m["tail_silence_s"] - 1.0) < 0.15
    _, issues = analysis.score_and_issues(m)
    assert not any(i["code"] == "lead_silence_s" for i in issues)  # < 3 s: geen issue


def test_momentary_and_plr(sr):
    x = _tone(sr, 6, amp=0.1)[None, :]
    m = analysis.analyze(x, sr)
    assert m["lufs_momentary_max"] is not None
    assert m["lufs_momentary_max"] >= m["lufs_integrated"] - 0.5
    assert m["plr_db"] is not None
    assert 0 <= m["plr_db"] < 40
