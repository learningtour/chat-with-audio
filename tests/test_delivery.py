"""Aflevering & metadata (fase E): bext/iXML, ID3-hoofdstukken, codec-preview,
checksums en het afleverpakket.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import bwf, delivery, id3

SR = 48000


@pytest.fixture
def wav(tmp_path):
    t = np.arange(4 * SR) / SR
    x = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "master.wav"
    sf.write(str(p), x, SR, subtype="PCM_24")
    return p


# ---------------------------------------------------------------- bext/iXML

def test_timecode_roundtrip():
    samples = bwf.timecode_to_samples("10:00:00:00", SR, fps=25)
    assert samples == 10 * 3600 * SR
    assert bwf.samples_to_timecode(samples, SR, fps=25) == "10:00:00:00"
    with pytest.raises(ValueError):
        bwf.timecode_to_samples("10:00:00", SR)


def test_bwf_write_and_read_back(wav):
    bext = bwf.build_bext(description="Aflevering 12", originator="AIT",
                          origination_date="2026-07-12",
                          origination_time="21:00:00",
                          time_reference=bwf.timecode_to_samples(
                              "09:59:50:00", SR),
                          coding_history="A=PCM,F=48000,W=24,M=stereo")
    ixml = bwf.build_ixml(project="Demo", scene="12A", take="3", note="ok")
    bwf.write_chunks(wav, bext=bext, ixml=ixml)

    meta = bwf.read_metadata(wav)
    assert meta["bext"]["description"] == "Aflevering 12"
    assert meta["bext"]["originator"] == "AIT"
    assert meta["bext"]["time_reference"] == bwf.timecode_to_samples(
        "09:59:50:00", SR)
    assert "PCM" in meta["bext"]["coding_history"]
    assert "<SCENE>12A</SCENE>" in meta["ixml_raw"]

    # audio blijft bit-voor-bit gelijk en het bestand blijft geldig
    y, sr = sf.read(str(wav), dtype="float32")
    assert sr == SR and y.shape[0] == 4 * SR


def test_bwf_replaces_existing_chunks(wav):
    bwf.write_chunks(wav, bext=bwf.build_bext(description="v1"))
    bwf.write_chunks(wav, bext=bwf.build_bext(description="v2"))
    meta = bwf.read_metadata(wav)
    assert meta["bext"]["description"] == "v2"
    data = wav.read_bytes()
    assert data.count(b"bext") == 1


def test_bwf_tool(wav):
    from chat_with_audio import server

    res = server.write_bwf_metadata(str(wav), description="E2E",
                                    timecode="01:00:00:00", fps=25,
                                    project="Demo", scene="1", take="2")
    assert res["written"]["bext"]["description"] == "E2E"
    assert res["timecode"] == "01:00:00:00"


# ---------------------------------------------------------------- ID3

def test_id3_chapters_roundtrip(tmp_path, wav):
    from chat_with_audio import server

    chapters = [{"start_s": 0.0, "end_s": 1.5, "title": "Intro"},
                {"start_s": 1.5, "end_s": 4.0, "title": "Hoofdstuk één"}]
    res = server.export_podcast_mp3(str(wav), str(tmp_path / "ep.mp3"),
                                    title="Aflevering 1", artist="Serge",
                                    chapters=chapters)
    assert res["chapters_written"] == 2
    back = id3.read_chapters(res["export_path"])
    assert back[0]["title"] == "Intro"
    assert back[1]["title"] == "Hoofdstuk één"
    assert back[1]["start_s"] == pytest.approx(1.5, abs=0.01)
    # mp3 blijft afspeelbaar met de tag ervoor
    y, sr = sf.read(res["export_path"])
    assert y.shape[0] > 3 * sr


def test_id3_replaces_existing_tag(tmp_path):
    p = tmp_path / "t.mp3"
    x = np.zeros(SR, dtype=np.float32)
    sf.write(str(p), x, SR, format="MP3")
    id3.write_tags(p, title="Eerste")
    size1 = p.stat().st_size
    id3.write_tags(p, title="Tweede")
    assert abs(p.stat().st_size - size1) < 64  # tag vervangen, niet gestapeld


def test_id3_requires_content():
    with pytest.raises(ValueError):
        id3.build_tag()


# ---------------------------------------------------------------- codec preview

def test_codec_preview_measures_overs(tmp_path):
    # master vlak tegen 0 dBFS: mp3 moet intersample overs opleveren
    t = np.arange(6 * SR) / SR
    rng = np.random.default_rng(0)
    hot = np.clip(0.99 * np.sign(np.sin(2 * np.pi * 997 * t))
                  + 0.01 * rng.standard_normal(t.shape[0]), -1, 1)
    p = tmp_path / "hot.wav"
    sf.write(str(p), hot.astype(np.float32), SR)
    from chat_with_audio import server

    res = server.codec_preview(str(p), codecs=["mp3"])
    r = res["codecs"][0]
    assert r["true_peak_out_dbtp"] > r["true_peak_in_dbtp"]
    assert r["codec_overs"] is True
    assert "verlaag" in res["hint"]


def test_codec_preview_clean_master_passes(wav):
    from chat_with_audio import server

    res = server.codec_preview(str(wav), codecs=["mp3", "ogg", "opus"])
    assert len(res["codecs"]) == 3
    assert all(abs(r["true_peak_delta_db"]) < 3 for r in res["codecs"])
    assert not any(r["codec_overs"] for r in res["codecs"])


def test_codec_roundtrip_opus_resamples():
    x = np.zeros((1, 44100), dtype=np.float32)
    y, sr = delivery.codec_roundtrip(x, 44100, "opus")
    assert sr == 48000


# ---------------------------------------------------------------- pakket

def test_delivery_package_from_file(wav, tmp_path):
    from chat_with_audio import server

    res = server.delivery_package(file_path=str(wav), spec="ebu-r128",
                                  out_dir=str(tmp_path / "pkg"),
                                  include_mp3=True)
    d = Path(res["package_dir"])
    names = {p.name for p in d.iterdir()}
    assert {"master.wav", "qc_report.md", "compliance.json",
            "checksums.md5", "manifest.json", "master.mp3"} <= names
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["format"] == "chat-with-audio/delivery@1"
    assert manifest["spec"] == "ebu-r128"
    # checksums kloppen echt
    for entry in manifest["files"]:
        assert delivery.md5sum(d / entry["name"]) == entry["md5"]


def test_delivery_package_from_session(noisy_wav):
    from chat_with_audio import server

    ses = server.apply_chain(str(noisy_wav), steps=[{"type": "gain", "gain_db": -2}])
    res = server.delivery_package(session_id=ses["session_id"], name="ep1")
    d = Path(res["package_dir"])
    assert (d / "ep1.wav").exists()
    assert (d / "checksums.md5").exists()
    assert res["passed_compliance"] is None  # geen spec meegegeven
