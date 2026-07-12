"""Tweede pas van smart_edit: na de ingreep opnieuw meten en eerlijk melden
of het probleem echt weg is. Bromcase (geen AI-model nodig)."""

import numpy as np
import soundfile as sf
from test_regions import _speechy

from chat_with_audio import server


def hum_wav(tmp_path, sr):
    t = np.arange(sr * 12) / sr
    base = _speechy(sr, 12)
    hum = 0.02 * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
                  + 0.3 * np.sin(2 * np.pi * 150 * t))
    x = (base + hum * ((t >= 4) & (t < 8))).astype(np.float32)
    p = tmp_path / "hum.wav"
    sf.write(str(p), x, sr)
    return p


def test_smart_edit_second_pass_confirms_fix(tmp_path, sr):
    p = hum_wav(tmp_path, sr)
    res = server.smart_edit(str(p), problems="hum", denoise_method="spectral")
    v = res["verification"]
    assert v is not None
    assert v["treated"] >= 1
    # de notch moet de brom echt hebben opgeruimd
    assert v["resolved"] == v["treated"], v
    assert v["remaining"] == []
    assert any("verificatie" in line.lower() for line in res["rationale"])


def test_smart_edit_verify_can_be_disabled(tmp_path, sr):
    p = hum_wav(tmp_path, sr)
    res = server.smart_edit(str(p), problems="hum", denoise_method="spectral",
                            verify=False)
    assert res["verification"] is None
