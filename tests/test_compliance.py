"""Compliance-suite: spec-register, pass/fail-checks, dialogue-gated meting,
master_for inclusief leveringsexport (SRC + bit depth)."""

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import analysis, chain, compliance, io, server


def _speechy(sr, dur_s, amp_db=-20.0):
    t = np.arange(int(sr * dur_s)) / sr
    syllables = (np.sin(2 * np.pi * 5.0 * t) > 0).astype(np.float64)
    sentences = (np.sin(2 * np.pi * 0.25 * t) > -0.6).astype(np.float64)
    amp = 10 ** (amp_db / 20) * np.sqrt(2)
    return (amp * np.sin(2 * np.pi * 300 * t) * syllables * sentences)


def test_spec_registry():
    ids = {s["id"] for s in compliance.list_specs()}
    assert {"ebu-r128", "atsc-a85", "netflix-2.0", "apple-podcast",
            "spotify", "youtube", "acx-audiobook"} <= ids
    with pytest.raises(ValueError, match="Onbekende spec"):
        compliance.check({}, "bestaat-niet")


def test_check_pass_and_fail_on_loudness(sr, noisy_bursts):
    x = noisy_bursts[None, :]
    y, _ = chain.normalize_loudness(x, sr, target_lufs=-23.0, true_peak_db=-1.5)
    m = analysis.analyze(y, sr)
    rep = compliance.check(m, "ebu-r128")
    loud = next(c for c in rep["checks"] if c["name"] == "Integrated loudness")
    assert loud["passed"], rep
    tp = next(c for c in rep["checks"] if c["name"] == "True peak")
    assert tp["passed"]

    rep14 = compliance.check(m, "spotify")  # -23 gemeten vs -14 vereist
    assert not rep14["passed"]
    assert "Integrated loudness" in rep14["failed_checks"]


def test_technical_gates_fail_on_dropout(sr):
    t = np.arange(sr * 6) / sr
    x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    a = int((3.0 + 1 / (4 * 440)) * sr)  # abrupt afgekapt op een golftop
    x[a:a + int(0.05 * sr)] = 0.0
    y, _ = chain.normalize_loudness(x[None, :], sr, target_lufs=-23.0, true_peak_db=-1.5)
    m = analysis.analyze(y, sr)
    rep = compliance.check(m, "ebu-r128")
    assert not rep["passed"]
    assert "Dropouts" in rep["failed_checks"]


def test_dialogue_loudness_differs_from_integrated(sr):
    """Zachte spraak + luide muziek: dialogue-gated meet de spraak, niet de mix."""
    speech = _speechy(sr, 10, amp_db=-28.0)
    t = np.arange(int(sr * 6)) / sr
    music = 0.25 * (np.sin(2 * np.pi * 220 * t) + 0.6 * np.sin(2 * np.pi * 331 * t)
                    + 0.4 * np.sin(2 * np.pi * 495 * t))
    x = np.concatenate([speech, music]).astype(np.float32)[None, :]
    from chat_with_audio.segments import classify_segments

    segs = classify_segments(x, sr)
    dlg = compliance.dialogue_loudness(x, sr, segs)
    integrated = analysis.measure_lufs(x, sr)
    assert dlg is not None
    assert dlg < integrated - 3, (dlg, integrated)


def test_master_for_ebu(sr, noisy_wav):
    res = server.master_for(str(noisy_wav), spec="ebu-r128")
    assert res["compliance"]["passed"], res["compliance"]["failed_checks"]
    m = res["compliance"]
    loud = next(c for c in m["checks"] if c["name"] == "Integrated loudness")
    assert abs(loud["measured"] - (-23.0)) <= 0.5


def test_master_for_acx_and_delivery_export(sr, noisy_wav, tmp_path):
    out = tmp_path / "levering.wav"
    res = server.master_for(str(noisy_wav), spec="acx-audiobook",
                            out_path=str(out), sample_rate=48000, bit_depth=24)
    rep = res["compliance"]
    rms = next(c for c in rep["checks"] if c["name"] == "RMS-niveau")
    pk = next(c for c in rep["checks"] if c["name"] == "Sample peak")
    assert rms["passed"] and pk["passed"], rep
    info = sf.info(str(out))
    assert info.samplerate == 48000
    assert info.subtype == "PCM_24"
    assert res["export"]["sample_rate"] == 48000


def test_netflix_format_checks(sr, tmp_path):
    """Netflix eist 48 kHz/24-bit PCM: een 44.1k/16-bit bron faalt op formaat."""
    x = _speechy(sr, 10, amp_db=-28.0).astype(np.float32)
    p = tmp_path / "bron_44k1_16bit.wav"
    sf.write(str(p), x, sr, subtype="PCM_16")
    rep = server.check_compliance(str(p), spec="netflix-2.0")
    names = {c["name"]: c for c in rep["checks"]}
    assert not names["Sample rate"]["passed"]
    assert not names["Leveringsformaat"]["passed"]
    assert "48000" in names["Sample rate"]["hint"]
    # EBU heeft geen formaateis: geen formaatchecks in dat rapport
    rep2 = server.check_compliance(str(p), spec="ebu-r128")
    assert "Sample rate" not in {c["name"] for c in rep2["checks"]}


def test_master_for_netflix_delivers_spec_format(sr, tmp_path):
    """master_for(netflix) zonder expliciet formaat levert 48 kHz/24-bit en
    keurt het échte leveringsbestand goed."""
    x = _speechy(sr, 12, amp_db=-24.0).astype(np.float32)
    p = tmp_path / "dialoog.wav"
    sf.write(str(p), x, sr)
    out = tmp_path / "netflix_master.wav"
    res = server.master_for(str(p), spec="netflix-2.0", out_path=str(out))
    rep = res["compliance"]
    assert rep["passed"], rep["failed_checks"]
    names = {c["name"]: c for c in rep["checks"]}
    assert names["Sample rate"]["passed"]
    assert names["Leveringsformaat"]["passed"]
    loud = names["Dialogue-gated loudness"]
    assert abs(loud["measured"] - (-27.0)) <= 2.0
    info = sf.info(str(out))
    assert info.samplerate == 48000
    assert info.subtype == "PCM_24"
    assert res["export"]["sample_rate"] == 48000


def test_master_for_netflix_without_export_hints_at_format(sr, tmp_path):
    x = _speechy(sr, 10, amp_db=-24.0).astype(np.float32)
    p = tmp_path / "dialoog.wav"
    sf.write(str(p), x, sr)
    res = server.master_for(str(p), spec="netflix-2.0")
    rep = res["compliance"]
    # loudness klopt, maar het 44.1k-sessiebestand is nog geen leveringsformaat
    assert "Sample rate" in rep["failed_checks"]
    assert any("out_path" in r for r in res["rationale"])


def test_check_compliance_tool_reports_specs(noisy_wav):
    rep = server.check_compliance(str(noisy_wav), spec="apple-podcast")
    assert rep["spec"] == "apple-podcast"
    assert {s["id"] for s in rep["available_specs"]} >= {"ebu-r128", "netflix-2.0"}
    assert isinstance(rep["passed"], bool)
    # ruwe testfile op -30ish LUFS haalt -16 +-1 niet
    assert not rep["passed"]
    assert "hint" in rep


def test_resample_quality(sr):
    t = np.arange(sr * 2) / sr
    x = (0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)[None, :]
    y, new_sr = io.resample(x, sr, 48000)
    assert new_sr == 48000
    assert abs(y.shape[1] - 2 * 48000) <= 2
    # de toon blijft een toon: piek in het spectrum op 1 kHz
    spec = np.abs(np.fft.rfft(y[0]))
    freqs = np.fft.rfftfreq(y.shape[1], 1 / 48000)
    assert abs(freqs[np.argmax(spec)] - 1000.0) < 5.0
