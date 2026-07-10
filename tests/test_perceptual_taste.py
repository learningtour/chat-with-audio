import numpy as np
import pytest

from audio_improve_toolkit import analysis, training, visuals
from audio_improve_toolkit.audition import write_sesx


@pytest.fixture(autouse=True)
def _isolate_taste(tmp_path, monkeypatch):
    monkeypatch.setenv("AIT_TASTE_DIR", str(tmp_path / "taste"))


def _tone(sr, freq=440.0, amp=0.1, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(sr * 3) / sr
    return (amp * np.sin(2 * np.pi * freq * t)
            + rng.normal(0, noise, t.shape[0])).astype(np.float32)[None, :]


def test_perceptual_panel_renders_png(sr):
    xo = _tone(sr, noise=0.02)
    xp = _tone(sr, amp=0.3, noise=0.002)
    png = visuals.perceptual_panel(xo, sr, xp)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 20_000  # geen leeg plaatje

    png_single = visuals.perceptual_panel(xo, sr, None)
    assert png_single[:8] == b"\x89PNG\r\n\x1a\n"


def test_taste_model_learns_preference(sr):
    # 'goed' = schoon en op niveau; 'slecht' = zacht en ruizig
    for seed in (1, 2):
        m = analysis.analyze(_tone(sr, amp=0.25, noise=0.001, seed=seed), sr)
        training.add_example(m, "good", f"goed-{seed}")
    for seed in (3, 4):
        m = analysis.analyze(_tone(sr, amp=0.01, noise=0.02, seed=seed), sr)
        training.add_example(m, "bad", f"slecht-{seed}")

    like_good = training.score(analysis.analyze(
        _tone(sr, amp=0.22, noise=0.001, seed=5), sr))
    like_bad = training.score(analysis.analyze(
        _tone(sr, amp=0.012, noise=0.025, seed=6), sr))
    assert like_good["taste_score"] > 65
    assert like_bad["taste_score"] < 35
    assert like_bad["largest_deviations"]  # legt uit wat er afwijkt


def test_sesx_is_valid_xml(tmp_path, sr):
    import xml.etree.ElementTree as ET

    f1, f2 = tmp_path / "vocals.wav", tmp_path / "drums.wav"
    f1.touch(), f2.touch()
    p = write_sesx(tmp_path, "testsessie", [("vocals", f1, 48000), ("drums", f2, 48000)], sr)
    tree = ET.parse(p)
    root = tree.getroot()
    assert root.tag == "sesx"
    assert len(root.findall(".//audioTrack")) == 2
    assert len(root.findall(".//file")) == 2
