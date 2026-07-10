import numpy as np

from audio_improve_toolkit.dsp import repair


def _clean_sine(sr, dur=2.0, freq=220.0, amp=0.9):
    t = np.arange(int(sr * dur)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_declip_reconstructs_flat_tops(sr):
    clean = _clean_sine(sr)
    clipped = np.clip(clean, -0.7, 0.7)  # zware clipping
    # schaal naar 'vol' bereik zodat flat-tops nabij de piek liggen
    y, fixed = repair.declip(clipped[None, :] / 0.7, sr)
    ref = clean[None, :] / 0.7
    assert fixed > 100
    err_before = float(np.mean((clipped[None, :] / 0.7 - ref) ** 2))
    err_after = float(np.mean((y - ref) ** 2))
    assert err_after < err_before * 0.35  # ruim meer dan de helft van de schade hersteld


def test_declip_leaves_clean_audio_alone(sr):
    clean = _clean_sine(sr, amp=0.5)[None, :]
    y, fixed = repair.declip(clean, sr)
    assert fixed == 0
    np.testing.assert_allclose(y, clean, atol=1e-6)


def test_declick_removes_impulses(sr):
    clean = _clean_sine(sr, amp=0.3)
    dirty = clean.copy()
    idx = np.arange(1000, len(dirty), 9000)[:20]
    dirty[idx] += 0.6  # klikken
    y, fixed = repair.declick(dirty[None, :], sr)
    assert fixed >= 15
    err_before = float(np.max(np.abs(dirty - clean)))
    err_after = float(np.max(np.abs(y[0] - clean)))
    assert err_after < err_before * 0.2
