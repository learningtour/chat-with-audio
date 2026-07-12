"""5.1-surround: BS.1770-weging, kanaal-QC, downmix, DI-stijl dialogue gating,
netflix-5.1-spec en ADM-BWF-herkenning."""

import numpy as np
import soundfile as sf

from chat_with_audio import analysis, compliance, io, server


def _tone(sr, dur_s, freq=440.0, amp=0.1):
    t = np.arange(int(sr * dur_s)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _five_one(sr, dur_s, fl=0.0, fr=0.0, fc=0.0, lfe=0.0, bl=0.0, br=0.0):
    """SMPTE-volgorde: FL FR FC LFE BL BR; amplitudes per kanaal."""
    amps = [fl, fr, fc, lfe, bl, br]
    return np.stack([_tone(sr, dur_s, amp=a) if a else
                     np.zeros(int(sr * dur_s), dtype=np.float32) for a in amps])


def test_surround_weighting_and_lfe_exclusion(sr):
    front = analysis.measure_lufs(_five_one(sr, 6, fl=0.1, fr=0.1), sr)
    back = analysis.measure_lufs(_five_one(sr, 6, bl=0.1, br=0.1), sr)
    # surroundkanalen wegen +1.5 dB (G = 1.41) t.o.v. het frontpaar
    assert abs((back - front) - 1.5) < 0.2, (front, back)
    # LFE telt niet mee in de loudness
    assert analysis.measure_lufs(_five_one(sr, 6, lfe=0.3), sr) is None


def test_surround_qc_dead_channel_and_downmix(sr):
    x = _five_one(sr, 6, fl=0.1, fr=0.1, fc=0.1, bl=0.1)  # BR dood, LFE stil
    m = analysis.analyze(x, sr)
    s = m["surround"]
    assert s["layout"] == "5.1 (SMPTE)"
    assert s["dead_channels"] == ["BR"]
    assert s["lfe_silent"] is True
    assert s["downmix_true_peak_dbtp"] is not None
    assert m["stereo"] is None  # stereo-QC alleen voor 2-kanaals
    _, issues = analysis.score_and_issues(m)
    assert any(i["code"] == "dead_surround_channel" for i in issues)


def test_downmix_clipping_flagged(sr):
    # alles vol open: de ITU-downmix (FL + 0.708·FC + 0.708·BL) klipt
    x = _five_one(sr, 4, fl=0.9, fr=0.9, fc=0.9, bl=0.9, br=0.9)
    m = analysis.analyze(x, sr)
    assert m["surround"]["downmix_true_peak_dbtp"] > 0
    _, issues = analysis.score_and_issues(m)
    assert any(i["code"] == "downmix_clipping" for i in issues)


def _speechy(sr, dur_s, amp_db=-24.0):
    t = np.arange(int(sr * dur_s)) / sr
    syllables = (np.sin(2 * np.pi * 5.0 * t) > 0).astype(np.float64)
    sentences = (np.sin(2 * np.pi * 0.25 * t) > -0.6).astype(np.float64)
    amp = 10 ** (amp_db / 20) * np.sqrt(2)
    return (amp * np.sin(2 * np.pi * 300 * t) * syllables * sentences)


def _five_one_program(sr, dur_s=12.0):
    """5.1-programma: dialoog op center, muziekbed op de fronten."""
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    fc = _speechy(sr, dur_s).astype(np.float32)
    music = (0.03 * (np.sin(2 * np.pi * 220 * t)
                     + 0.5 * np.sin(2 * np.pi * 331 * t))).astype(np.float32)
    zero = np.zeros(n, dtype=np.float32)
    return np.stack([music, music, fc, zero, zero, zero])


def test_dialogue_loudness_on_center_channel(sr):
    x = _five_one_program(sr)
    dlg = compliance.dialogue_loudness(x, sr)
    integrated = analysis.measure_lufs(x, sr)
    assert dlg is not None
    # gated meting wijkt af van integrated en zit in een plausibel bereik
    assert -45 < dlg < -10
    assert abs(dlg - integrated) < 12


def test_master_for_netflix_51(sr, tmp_path):
    x = _five_one_program(sr)
    p = tmp_path / "mix51.wav"
    sf.write(str(p), x.T, sr, subtype="PCM_24")
    out = tmp_path / "netflix51.wav"
    res = server.master_for(str(p), spec="netflix-5.1", out_path=str(out))
    rep = res["compliance"]
    assert rep["passed"], rep["failed_checks"]
    names = {c["name"]: c for c in rep["checks"]}
    assert names["Kanalen (formaat)"]["passed"]
    assert names["Sample rate"]["passed"]
    info = sf.info(str(out))
    assert info.channels == 6 and info.samplerate == 48000
    assert info.subtype == "PCM_24"


def test_netflix_20_rejects_51_mix(sr, tmp_path):
    x = _five_one_program(sr)
    p = tmp_path / "mix51.wav"
    sf.write(str(p), x.T, sr, subtype="PCM_24")
    rep = server.check_compliance(str(p), spec="netflix-2.0")
    names = {c["name"]: c for c in rep["checks"]}
    assert not names["Kanalen (formaat)"]["passed"]
    assert "netflix-5.1" in names["Kanalen (formaat)"]["hint"]


def test_adm_bwf_detection(sr, tmp_path):
    p = tmp_path / "atmos_adm.wav"
    sf.write(str(p), np.zeros((sr, 2), dtype=np.float32), sr, subtype="PCM_24")
    raw = bytearray(p.read_bytes())
    axml = b"<adm>demo</adm>"
    chunk = b"axml" + len(axml).to_bytes(4, "little") + axml + (b"\x00" * (len(axml) & 1))
    raw += chunk
    riff_size = len(raw) - 8
    raw[4:8] = riff_size.to_bytes(4, "little")
    p.write_bytes(bytes(raw))

    info = io.probe(p)
    assert info["adm_bwf"] is True
    rep = compliance.check({"channels": 2, "sample_rate": sr}, "spotify",
                           file_info=info)
    adm = [c for c in rep["checks"] if "Atmos" in c["name"]]
    assert adm and adm[0]["advisory"]

    # gewone wav: geen ADM
    q = tmp_path / "gewoon.wav"
    sf.write(str(q), np.zeros(sr, dtype=np.float32), sr)
    assert io.probe(q)["adm_bwf"] is False
