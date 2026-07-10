import numpy as np
import pytest

from audio_improve_toolkit.dsp import stems


@pytest.mark.skipif(not stems.is_available(), reason="[stems]-extra niet geinstalleerd")
def test_separate_returns_four_stems_that_sum_to_input(sr):
    rng = np.random.default_rng(3)
    t = np.arange(int(sr * 1.5)) / sr
    mix = (0.3 * np.sin(2 * np.pi * 220 * t)          # 'bass/other'
           + 0.2 * np.sin(2 * np.pi * 880 * t)
           + 0.05 * rng.normal(0, 1, t.shape[0])).astype(np.float32)[None, :]
    parts = stems.separate(mix, sr)
    assert set(parts) == {"vocals", "drums", "bass", "other"}
    for y in parts.values():
        assert y.shape == mix.shape
    total = sum(parts.values())
    # demucs-stems sommeren bij benadering naar het origineel
    corr = np.corrcoef(total[0], mix[0])[0, 1]
    assert corr > 0.9
