"""Stems & versies (fase F): Dugan-automix, mix-minus en DME-export.

De automix-tests gebruiken sporen met bekende actieve vensters: het aandeel
van een spoor moet zijn energie volgen, en de som van de aandelen is 1 (geen
ruisstapeling). DME wordt getest met een gemockte scheiding — de rekenlogica
(D + M&E = mix) is wat hier geverifieerd wordt.
"""

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import automix

SR = 44100


def tone_span(freq, active, dur=4.0, amp=0.3):
    """Sinus die alleen in [start, eind) klinkt."""
    t = np.arange(int(dur * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    mask = (t >= active[0]) & (t < active[1])
    return (x * mask).astype(np.float32)


def band_energy(x, freq, sr=SR):
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64)))
    freqs = np.fft.rfftfreq(x.shape[0], 1 / sr)
    return float(spec[np.abs(freqs - freq).argmin()])


def test_automix_follows_activity():
    boom = tone_span(330, (0.0, 4.0), amp=0.12)
    lav1 = tone_span(440, (0.0, 2.0), amp=0.3)
    lav2 = tone_span(550, (2.0, 4.0), amp=0.3)
    y, info = automix.automix([boom, lav1, lav2], SR, match_to=None)
    first, second = y[0, : 2 * SR], y[0, 2 * SR:]
    assert band_energy(first, 440) > band_energy(first, 550) * 5
    assert band_energy(second, 550) > band_energy(second, 440) * 5
    shares = {t["index"]: t["avg_share"] for t in info["tracks"]}
    assert shares[1] == pytest.approx(shares[2], abs=0.1)  # symmetrische lavs


def test_automix_no_noise_stacking():
    # 3 identieke ruissporen: automix mag niet +4.8 dB stapelen zoals een som
    rng = np.random.default_rng(0)
    base = (rng.standard_normal(2 * SR) * 0.1).astype(np.float32)
    y, _ = automix.automix([base, base, base], SR, match_to=None)
    rms_in = np.sqrt(np.mean(base.astype(np.float64) ** 2))
    rms_out = np.sqrt(np.mean(y[0].astype(np.float64) ** 2))
    assert abs(20 * np.log10(rms_out / rms_in)) < 1.0  # som aandelen = 1


def test_automix_needs_two_tracks():
    with pytest.raises(ValueError):
        automix.automix([tone_span(440, (0, 1))], SR)


def test_automix_match_eq_reports():
    boom = tone_span(330, (0.0, 4.0), amp=0.2)
    lav = tone_span(330, (0.0, 4.0), amp=0.2)
    _y, info = automix.automix([boom, lav], SR, match_to=0)
    assert info["tracks"][0]["match_eq"] == []
    assert isinstance(info["tracks"][1]["match_eq"], list)


def test_mix_minus_excludes_only_target():
    t0 = tone_span(330, (0, 2), dur=2.0)
    t1 = tone_span(440, (0, 2), dur=2.0)
    t2 = tone_span(550, (0, 2), dur=2.0)
    y = automix.mix_minus([t0, t1, t2], exclude=1)
    assert band_energy(y[0], 440) < band_energy(y[0], 330) / 100
    assert band_energy(y[0], 550) > band_energy(y[0], 330) / 3


def test_mix_minus_guards():
    t = tone_span(440, (0, 1), dur=1.0)
    with pytest.raises(ValueError):
        automix.mix_minus([t, t], exclude=5)
    with pytest.raises(ValueError):
        automix.mix_minus([t], exclude=0)


# ---------------------------------------------------------------- MCP-tools

def scene_tracks(tmp_path):
    """Drie 'mics' van dezelfde scène met bekende offsets (aperiodiek!)."""
    rng = np.random.default_rng(11)
    scene = rng.standard_normal(6 * SR) * np.clip(
        np.sin(2 * np.pi * 0.4 * np.arange(6 * SR) / SR), 0, 1) * 0.3
    offsets = [0.0, 0.7, 1.3]
    paths = []
    for i, off in enumerate(offsets):
        pad = np.zeros(int(off * SR), dtype=np.float32)
        sig = np.concatenate([pad, scene.astype(np.float32)])
        sig = sig + rng.standard_normal(sig.shape[0]).astype(np.float32) * 0.005
        p = tmp_path / f"mic{i}.wav"
        sf.write(str(p), sig, SR)
        paths.append(p)
    return paths, offsets


def test_automix_tracks_tool_syncs_and_mixes(tmp_path):
    from chat_with_audio import server

    paths, offsets = scene_tracks(tmp_path)
    res = server.automix_tracks(file_paths=[str(p) for p in paths],
                                match_tone=False)
    assert res["sync"] is not None
    rel = [s["offset_s"] for s in res["sync"]]
    assert rel[1] - rel[0] == pytest.approx(-0.7, abs=0.02)
    assert rel[2] - rel[0] == pytest.approx(-1.3, abs=0.02)
    y, sr = sf.read(res["output_path"])
    assert y.shape[0] > 6 * SR  # gezamenlijke tijdlijn


def test_mix_minus_tool_by_name(tmp_path):
    from chat_with_audio import server

    p = []
    for i, f in enumerate((330, 440, 550)):
        q = tmp_path / f"t{i}.wav"
        sf.write(str(q), tone_span(f, (0, 2), dur=2.0), SR)
        p.append(q)
    res = server.mix_minus([str(q) for q in p], exclude="t1.wav")
    assert res["excluded"] == "t1.wav"
    y, _ = sf.read(res["output_path"])
    assert band_energy(y, 440) < band_energy(y, 330) / 100


def test_export_dme_reconstructs_mix(tmp_path, monkeypatch):
    from chat_with_audio import server
    from chat_with_audio.dsp import stems

    rng = np.random.default_rng(2)
    n = 2 * SR
    vocals = (rng.standard_normal(n) * 0.1).astype(np.float32)
    music = (rng.standard_normal(n) * 0.2).astype(np.float32)
    mix = vocals + music
    p = tmp_path / "mix.wav"
    sf.write(str(p), mix, SR)

    def fake_separate(x, sr):
        v = vocals[None, :]
        return {"vocals": v,
                "drums": (music * 0.5)[None, :],
                "bass": (music * 0.3)[None, :],
                "other": (music * 0.2)[None, :]}

    monkeypatch.setattr(stems, "is_available", lambda: True)
    monkeypatch.setattr(stems, "separate", fake_separate)
    res = server.export_dme(str(p), out_dir=str(tmp_path / "dme"),
                            music_split=True)
    d, _ = sf.read(res["stems"]["dialogue"])
    me, _ = sf.read(res["stems"]["me"])
    orig, _ = sf.read(str(p))
    np.testing.assert_allclose(d + me, orig, atol=1e-4)  # D + M&E = mix
    mu, _ = sf.read(res["stems"]["music"])
    fx, _ = sf.read(res["stems"]["effects"])
    np.testing.assert_allclose(mu + fx, me, atol=1e-4)
