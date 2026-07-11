"""Slimme regio's: detectie klopt in de tijd, fixes werken alleen dáár en
alles buiten de regio's blijft bit-voor-bit onaangetast."""

import numpy as np

from chat_with_audio import regions


def _speechy(sr: int, dur_s: float, amp_db: float = -20.0) -> np.ndarray:
    """Spraak-achtig testsignaal: 300 Hz met 5 Hz lettergreepritme en pauzes."""
    t = np.arange(int(sr * dur_s)) / sr
    syllables = (np.sin(2 * np.pi * 5.0 * t) > 0).astype(np.float64)
    sentences = (np.sin(2 * np.pi * 0.25 * t) > -0.6).astype(np.float64)
    amp = 10 ** (amp_db / 20) * np.sqrt(2)
    return (amp * np.sin(2 * np.pi * 300 * t) * syllables * sentences)


def test_clean_audio_has_no_regions(sr):
    x = _speechy(sr, 10).astype(np.float32)[None, :]
    assert regions.detect_regions(x, sr) == []


def test_hum_region_detected_and_fixed_only_there(sr):
    t = np.arange(sr * 12) / sr
    base = _speechy(sr, 12)
    hum = 0.02 * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
                  + 0.3 * np.sin(2 * np.pi * 150 * t))
    x = (base + hum * ((t >= 4) & (t < 8))).astype(np.float32)[None, :]

    found = regions.detect_regions(x, sr)
    hums = [r for r in found if r["kind"] == "hum"]
    assert hums, f"geen bromregio gevonden in {found}"
    r = hums[0]
    assert 1.5 <= r["start_s"] <= 5.0, r
    assert 7.0 <= r["end_s"] <= 10.5, r
    assert r["freq"] == 50.0

    planned, rationale = regions.plan_region_fixes(hums, sr)
    assert planned[0]["steps"][0]["type"] == "notch"
    assert any("netbrom" in line for line in rationale)
    y, applied = regions.apply_regions(x, sr, planned)
    assert len(applied) == 1

    def hum_level(sig):
        seg = sig[0, int(5 * sr):int(7 * sr)].astype(np.float64)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(seg.size, 1 / sr)
        return 20 * np.log10(spec[np.abs(freqs - 50.0) < 2.0].max() + 1e-12)

    assert hum_level(x) - hum_level(y) > 12, "brom moet in de regio flink dalen"
    # ruim buiten regio + pad/fade: onaangetast
    safe = int(1.0 * sr)
    assert np.array_equal(x[:, :safe], y[:, :safe])
    assert np.array_equal(x[:, -int(0.5 * sr):], y[:, -int(0.5 * sr):])


def test_noise_region_detected_and_reduced(sr):
    rng = np.random.default_rng(3)
    n = sr * 12
    t = np.arange(n) / sr
    base = _speechy(sr, 12)
    noise = rng.normal(0, 10 ** (-38 / 20), n) * (t >= 6)
    x = (base + noise).astype(np.float32)[None, :]

    found = regions.detect_regions(x, sr)
    noisy = [r for r in found if r["kind"] == "noise"]
    assert noisy, f"geen ruisregio gevonden in {found}"
    assert noisy[0]["start_s"] >= 4.0, noisy

    planned, _ = regions.plan_region_fixes(noisy, sr)
    y, applied = regions.apply_regions(x, sr, planned)
    assert applied

    def lvl(sig, a, b):
        seg = sig[0, int(a * sr):int(b * sr)].astype(np.float64)
        return 10 * np.log10(np.mean(seg ** 2) + 1e-20)

    # in een spraakpauze binnen de ruiszone (zin-envelope uit: 6.4-7.6 s) moet
    # de ruis merkbaar zakken
    assert lvl(x, 6.55, 7.45) - lvl(y, 6.55, 7.45) > 6
    # het schone begin blijft onaangetast
    assert np.array_equal(x[:, :int(4.5 * sr)], y[:, :int(4.5 * sr)])


def test_clip_region_detected_and_repaired(sr):
    t = np.arange(sr * 10) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    hot = (t >= 5.0) & (t < 6.0)
    x[hot] = np.clip(x[hot] * 4.0, -0.9995, 0.9995)
    # np.clip naar exact 0.9995 geeft flat-tops onder 0.999; forceer echte clips
    x[hot] = np.clip(x[hot] * 1.01, -1.0, 1.0)
    x = x[None, :]

    found = regions.detect_regions(x, sr)
    clips = [r for r in found if r["kind"] == "clip"]
    assert clips, f"geen clipregio gevonden in {found}"
    assert 4.5 <= clips[0]["start_s"] <= 5.2, clips
    assert 5.8 <= clips[-1]["end_s"] <= 6.5, clips

    planned, _ = regions.plan_region_fixes(clips, sr)
    y, applied = regions.apply_regions(x, sr, planned)
    assert applied
    from chat_with_audio import analysis

    assert analysis.analyze(y, sr)["clip_events"] < analysis.analyze(x, sr)["clip_events"]
    assert np.array_equal(x[:, :int(4.0 * sr)], y[:, :int(4.0 * sr)])


def test_boom_region_detected_and_tamed(sr):
    t = np.arange(sr * 12) / sr
    base = _speechy(sr, 12)
    rumble = 0.15 * np.sin(2 * np.pi * 70 * t) * ((t >= 3) & (t < 6))
    x = (base + rumble).astype(np.float32)[None, :]

    found = regions.detect_regions(x, sr)
    booms = [r for r in found if r["kind"] == "boom"]
    assert booms, f"geen dreunregio gevonden in {found}"
    r = booms[0]
    assert 2.0 <= r["start_s"] <= 4.0, r
    assert 5.0 <= r["end_s"] <= 7.5, r

    planned, _ = regions.plan_region_fixes(booms, sr)
    y, _applied = regions.apply_regions(x, sr, planned)

    def low_level(sig, a, b):
        seg = sig[0, int(a * sr):int(b * sr)].astype(np.float64)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(seg.size, 1 / sr)
        band = spec[(freqs > 40) & (freqs < 120)]
        return 20 * np.log10(band.max() + 1e-12)

    assert low_level(x, 4.0, 5.0) - low_level(y, 4.0, 5.0) > 6
    assert np.array_equal(x[:, :int(2.0 * sr)], y[:, :int(2.0 * sr)])


def test_apply_regions_smooth_at_boundaries(sr):
    """De crossfade mag geen hoorbare sprong maken op de regiogrenzen."""
    t = np.arange(sr * 8) / sr
    x = (0.1 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)[None, :]
    planned = [{"kind": "noise", "start_s": 3.0, "end_s": 5.0, "severity_db": 10.0,
                "steps": [{"type": "gain", "gain_db": -6.0}]}]
    y, applied = regions.apply_regions(x, sr, planned)
    assert applied
    jumps = np.abs(np.diff(y[0].astype(np.float64)))
    # continu signaal: sprongen blijven in de orde van de sinus zelf
    assert jumps.max() < 0.1 * 2 * np.pi * 330 / sr * 1.5
