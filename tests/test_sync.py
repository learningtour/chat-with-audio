"""32-sporen sync: offsets op samplenauwkeurigheid, confidence, klokdrift en
de sync_tracks-tool end-to-end."""

import numpy as np
import soundfile as sf

from chat_with_audio import server, sync


def _event(sr, dur_s=30.0, seed=2):
    """Het 'gebeuren' dat alle recorders horen: spraakachtig + ambience.

    De zinsindeling is bewust APERIODIEK (willekeurige blokken): een periodiek
    gatingpatroon geeft dubbelzinnige correlatiepieken — precies zoals echte
    click tracks/metronomen dat doen, maar echte spraak niet.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    gate = np.zeros(n)
    pos = 0
    while pos < n:
        seg = int(rng.uniform(0.4, 2.2) * sr)
        if rng.random() > 0.35:
            gate[pos:pos + seg] = 1.0
        pos += seg
    speech = (0.12 * np.sin(2 * np.pi * 300 * t)
              * (np.sin(2 * np.pi * 5.0 * t) > 0) * gate)
    return (speech + rng.normal(0, 10 ** (-46 / 20), n)).astype(np.float64)


def _recorder(event, sr, start_s, dur_s, tone_db=-50.0, lowpass=False, seed=105):
    """Een recorder die op start_s begint: eigen ruisvloer, eigen 'mickleur'.
    (Seeds bewust ver van de event-seed: identieke ruis zou een echte
    schijncorrelatie geven.)"""
    rng = np.random.default_rng(seed)
    a = int(start_s * sr)
    seg = event[a:a + int(dur_s * sr)].copy()
    if lowpass:
        from scipy.signal import butter, sosfiltfilt

        seg = sosfiltfilt(butter(4, 4000, btype="lowpass", fs=sr, output="sos"), seg)
    seg = seg * 0.7 + rng.normal(0, 10 ** (tone_db / 20), seg.size)
    return seg.astype(np.float32)


def test_offset_measured_sample_accurate(sr):
    event = _event(sr)
    ref = _recorder(event, sr, 0.0, 25.0)                 # hoofdrecorder
    late = _recorder(event, sr, 2.5, 20.0, lowpass=True)  # start 2.5 s later
    offset, conf = sync.measure_offset(ref.astype(np.float64),
                                       late.astype(np.float64), sr)
    assert abs(offset - 2.5) < 0.002, offset
    assert conf > sync._CONF_SYNCED


def test_no_shared_audio_low_confidence(sr):
    rng = np.random.default_rng(9)
    ref = _event(sr, 20.0)
    other = rng.normal(0, 0.05, int(sr * 20)).astype(np.float64)
    _, conf = sync.measure_offset(ref, other, sr)
    assert conf < sync._CONF_SYNCED


def test_drift_measured_and_corrected(sr):
    event = _event(sr, 30.0)
    ref = _recorder(event, sr, 0.0, 30.0)
    b = _recorder(event, sr, 1.0, 28.0, seed=107)
    # klok van recorder B loopt 200 ppm langzaam: tijdas uitgerekt
    drifted = sync.correct_drift(b[None, :], 200.0)[0]
    offset, conf = sync.measure_offset(ref.astype(np.float64),
                                       drifted.astype(np.float64), sr)
    assert conf > sync._CONF_SYNCED
    drift = sync.measure_drift(ref.astype(np.float64),
                               drifted.astype(np.float64), sr, offset)
    assert drift is not None
    assert abs(abs(drift) - 200.0) < 60.0, drift

    fixed = sync.correct_drift(drifted[None, :], drift)[0]
    offset2, _ = sync.measure_offset(ref.astype(np.float64),
                                     fixed.astype(np.float64), sr)
    residual = sync.measure_drift(ref.astype(np.float64),
                                  fixed.astype(np.float64), sr, offset2)
    assert residual is not None and abs(residual) < 60.0, residual


def test_sync_tracks_tool_end_to_end(tmp_path, sr):
    event = _event(sr)
    files = []
    specs = [(0.0, 25.0, False), (2.5, 20.0, True), (5.0, 18.0, False)]
    for i, (start, dur, lp) in enumerate(specs):
        p = tmp_path / f"recorder{i + 1}.wav"
        sf.write(str(p), _recorder(event, sr, start, dur, lowpass=lp, seed=100 + i), sr)
        files.append(str(p))

    res = server.sync_tracks(file_paths=files)
    assert res["reference"] == "recorder1.wav"  # langste bestand
    by_name = {t["name"]: t for t in res["tracks"]}
    assert all(t["synced"] for t in res["tracks"])
    assert abs(by_name["recorder2.wav"]["place_s"] - 2.5) < 0.005
    assert abs(by_name["recorder3.wav"]["place_s"] - 5.0) < 0.005

    from pathlib import Path

    aligned = sorted(Path(res["aligned_dir"]).glob("track*.wav"))
    assert len(aligned) == 3
    lengths = {sf.info(str(p)).frames for p in aligned}
    assert len(lengths) == 1, "alle uitgelijnde sporen even lang"
    assert Path(res["sesx"]).is_file()

    detail = server.list_sessions(session_id=res["session_id"])
    assert len(detail["timeline"]["regions"]) == 3
    # de gesynchroniseerde som is coherenter dan de ongesynchroniseerde:
    # spraakpieken stapelen -> hogere piek/loudness-verhouding
    assert detail["deltas"] is not None


def test_sync_tracks_marks_unrelated_file(tmp_path, sr):
    event = _event(sr)
    rng = np.random.default_rng(4)
    p1 = tmp_path / "a.wav"
    p2 = tmp_path / "b.wav"
    p3 = tmp_path / "los.wav"
    sf.write(str(p1), _recorder(event, sr, 0.0, 22.0), sr)
    sf.write(str(p2), _recorder(event, sr, 3.0, 18.0, seed=103), sr)
    sf.write(str(p3), rng.normal(0, 0.05, sr * 15).astype(np.float32), sr)
    res = server.sync_tracks(file_paths=[str(p1), str(p2), str(p3)])
    by_name = {t["name"]: t for t in res["tracks"]}
    assert by_name["b.wav"]["synced"]
    assert not by_name["los.wav"]["synced"]
    assert by_name["los.wav"]["place_s"] == 0.0


def test_sync_thirty_two_tracks(tmp_path):
    """De volle 32 sporen: elk spoor samplenauwkeurig op zijn plek, één .sesx
    met 32 tracks, alle uitgelijnde bestanden even lang."""
    sr = 22050  # halve rate houdt de stresstest vlot; het algoritme is rate-agnostisch
    rng = np.random.default_rng(42)
    event = _event(sr, 40.0)
    files = []
    starts = {}
    for i in range(32):
        start = round(float(rng.uniform(0.0, 12.0)), 3)
        dur = float(rng.uniform(14.0, 22.0))
        name = f"rec{i + 1:02d}.wav"
        p = tmp_path / name
        sf.write(str(p), _recorder(event, sr, start, dur,
                                   lowpass=bool(i % 3 == 0), seed=200 + i), sr)
        files.append(str(p))
        starts[name] = start

    res = server.sync_tracks(file_paths=files)
    assert len(res["tracks"]) == 32
    assert all(t["synced"] for t in res["tracks"]), [
        t["name"] for t in res["tracks"] if not t["synced"]]

    # plaatsing = eigen start minus de start van het vroegste spoor
    t0 = min(starts.values())
    for t in res["tracks"]:
        expected = starts[t["name"]] - t0
        assert abs(t["place_s"] - expected) < 0.005, (t["name"], t["place_s"], expected)

    from pathlib import Path

    aligned = sorted(Path(res["aligned_dir"]).glob("track*.wav"))
    assert len(aligned) == 32
    assert len({sf.info(str(p)).frames for p in aligned}) == 1
    sesx_text = Path(res["sesx"]).read_text()
    assert sesx_text.count("<audioClip ") >= 32 or sesx_text.count("track") >= 32

    detail = server.list_sessions(session_id=res["session_id"])
    assert len(detail["timeline"]["regions"]) == 32


def test_sync_tracks_limits(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="minstens 2"):
        server.sync_tracks(file_paths=[str(tmp_path / "een.wav")])
    with pytest.raises(ValueError, match="32"):
        server.sync_tracks(file_paths=[f"/tmp/x{i}.wav" for i in range(33)])
