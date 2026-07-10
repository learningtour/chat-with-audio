import numpy as np
import soundfile as sf

from audio_improve_toolkit import server


def test_improve_folder(tmp_path, sr):
    rng = np.random.default_rng(4)
    d = tmp_path / "batch"
    d.mkdir()
    t = np.arange(sr * 3) / sr
    for i, freq in enumerate((330, 550)):
        sig = (0.05 * np.sin(2 * np.pi * freq * t)
               + rng.normal(0, 0.002, t.shape[0])).astype(np.float32)
        sf.write(str(d / f"file{i}.wav"), sig, sr)
    (d / "notitie.txt").write_text("geen audio")

    res = server.improve_folder(str(d), mode="improve")
    assert res["processed"] == 2
    assert res["failed"] == 0
    assert all("session_id" in r for r in res["results"])
