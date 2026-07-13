"""Het oefenmateriaal van de trainingshandleiding moet blijven kloppen: elke
oefening moet precies het gedocumenteerde gebrek laten detecteren. Als een
detector-afstelling dit breekt, hoort deze test dat te melden — anders staat
er een handleiding die belooft wat de tool niet meer laat zien."""

import importlib.util
import sys
from pathlib import Path

import pytest

from chat_with_audio import analysis, io
from chat_with_audio.regions import detect_regions
from chat_with_audio.segments import classify_segments

SCRIPT = Path(__file__).parent.parent / "scripts" / "maak_oefenmateriaal.py"


@pytest.fixture(scope="module")
def materiaal(tmp_path_factory):
    d = tmp_path_factory.mktemp("oefen")
    spec = importlib.util.spec_from_file_location("maak_oefenmateriaal", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old_argv = sys.argv
    sys.argv = ["maak_oefenmateriaal", str(d)]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    return d


def kinds(d, name):
    x, sr = io.load_audio(d / name)
    return {r["kind"] for r in detect_regions(x, sr, segments=classify_segments(x, sr))}


def test_alle_bestanden_bestaan(materiaal):
    namen = {p.name for p in materiaal.glob("*")}
    assert {"oefening-01-brom.wav", "oefening-02-ruis.wav",
            "oefening-03-clipping.wav", "oefening-04-dreun.wav",
            "oefening-05-pauzes.wav", "oefening-06-te-stil.wav",
            "eindtoets.wav", "ANTWOORDEN.md"} <= namen


def test_brom_wordt_gedetecteerd(materiaal):
    assert "hum" in kinds(materiaal, "oefening-01-brom.wav")


def test_ruis_meetbaar_in_analyse(materiaal):
    x, sr = io.load_audio(materiaal / "oefening-02-ruis.wav")
    m = analysis.analyze(x, sr)
    assert m["snr_db"] < 20  # hoorbaar slechte SNR: dit is de les


def test_clipping_wordt_gedetecteerd(materiaal):
    assert "clip" in kinds(materiaal, "oefening-03-clipping.wav")


def test_dreun_wordt_gedetecteerd_zonder_valse_brom(materiaal):
    k = kinds(materiaal, "oefening-04-dreun.wav")
    assert "boom" in k and "hum" not in k


def test_pauzes_zijn_lang(materiaal):
    x, sr = io.load_audio(materiaal / "oefening-05-pauzes.wav")
    segs = classify_segments(x, sr)
    stiltes = [s["end_s"] - s["start_s"] for s in segs if s["kind"] == "silence"]
    assert stiltes and max(stiltes) > 2.0


def test_te_stil_is_te_stil(materiaal):
    x, sr = io.load_audio(materiaal / "oefening-06-te-stil.wav")
    m = analysis.analyze(x, sr)
    assert m["lufs_integrated"] < -30


def test_eindtoets_bevat_de_combinatie(materiaal):
    k = kinds(materiaal, "eindtoets.wav")
    assert {"hum", "noise", "clip"} <= k
    x, sr = io.load_audio(materiaal / "eindtoets.wav")
    # de spraak zelf is veel te stil; meet vóór de clip-knal (die domineert
    # anders de integrated loudness — wat op zichzelf al een leerpunt is)
    m = analysis.analyze(x[:, : 39 * sr], sr)
    assert m["lufs_integrated"] < -20


def test_antwoorden_dekken_alle_bestanden(materiaal):
    tekst = (materiaal / "ANTWOORDEN.md").read_text()
    for p in materiaal.glob("*.wav"):
        assert p.name in tekst
