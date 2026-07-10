import numpy as np
import pytest
import soundfile as sf


@pytest.fixture(autouse=True)
def _isolate_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("AIT_SESSIONS_DIR", str(tmp_path / "sessions"))


@pytest.fixture
def sr():
    return 44100


@pytest.fixture
def noisy_bursts(sr):
    """10 s: 440 Hz-toonstoten (-30 dBFS RMS, 2 s aan / 2 s uit) + witte ruis (-45 dBFS)."""
    rng = np.random.default_rng(42)
    t = np.arange(int(sr * 10)) / sr
    amp = 10 ** (-30 / 20) * np.sqrt(2)
    env = (np.sin(2 * np.pi * 0.25 * t) > 0).astype(np.float32)
    sig = amp * np.sin(2 * np.pi * 440 * t) * env
    noise = rng.normal(0, 10 ** (-45 / 20), t.shape[0])
    return (sig + noise).astype(np.float32)


@pytest.fixture
def noisy_wav(tmp_path, sr, noisy_bursts):
    p = tmp_path / "noisy.wav"
    sf.write(str(p), noisy_bursts, sr)
    return p
