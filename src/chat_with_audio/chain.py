"""Ketenuitvoering: een lijst stappen (dicts) toepassen op audio.

Elke stap is {"type": <naam>, ...params}. run_chain valideert, voert uit en
geeft de daadwerkelijk gebruikte parameters terug (voor chain.json en de chat).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable

import numpy as np

from chat_with_audio import dsp
from chat_with_audio.analysis import measure_lufs

log = logging.getLogger(__name__)


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float = -16.0,
                       true_peak_db: float = -1.5, max_iter: int = 2) -> tuple[np.ndarray, dict]:
    """Gain naar target-LUFS met een true-peak-veilige limiter erachter.

    De limiter-ceiling ligt 0.3 dB onder het true-peak-target omdat inter-sample
    pieken boven de sample-piek kunnen uitkomen.
    """
    before = measure_lufs(x, sr)
    if before is None:
        return x, {"skipped": "audio te kort of stil voor loudness-meting"}
    ceiling = true_peak_db - 0.3
    applied = 0.0
    y = x
    for _ in range(max_iter):
        cur = measure_lufs(y, sr)
        if cur is None or abs(cur - target_lufs) < 0.5:
            break
        step = target_lufs - cur
        applied += step
        y = dsp.limiter(dsp.gain(y, step), sr, ceiling_db=ceiling)
    after = measure_lufs(y, sr)
    return y, {
        "lufs_before": round(before, 2),
        "lufs_after": round(after, 2) if after is not None else None,
        "gain_db": round(applied, 2),
        "limiter_ceiling_db": round(ceiling, 2),
    }


def _step_highpass(x, sr, freq: float = 80.0, q: float = 0.707):
    return dsp.highpass(x, sr, freq, q)


def _step_lowpass(x, sr, freq: float = 16000.0, q: float = 0.707):
    return dsp.lowpass(x, sr, freq, q)


def _step_notch(x, sr, freq: float, q: float = 30.0):
    return dsp.notch(x, sr, freq, q)


def _step_eq(x, sr, bands: list):
    return dsp.eq(x, sr, bands)


def _step_gain(x, sr, gain_db: float):
    return dsp.gain(x, gain_db)


def _step_declip(x, sr, max_gap_ms: float = 4.0):
    from chat_with_audio.dsp import repair

    y, fixed = repair.declip(x, sr, max_gap_ms=max_gap_ms)
    log.info("declip: %d regio's gereconstrueerd", fixed)
    return y


def _step_declick(x, sr, threshold: float = 6.0):
    from chat_with_audio.dsp import repair

    y, fixed = repair.declick(x, sr, threshold=threshold)
    log.info("declick: %d klikken gerepareerd", fixed)
    return y


def _step_denoise(x, sr, strength_db: float = 12.0, method: str = "spectral"):
    if method == "ai":
        return dsp.ai_denoise(x, sr, strength_db=strength_db)
    return dsp.spectral_denoise(x, sr, reduction_db=strength_db)


def _assemble_segments(x2: np.ndarray, sr: int, process_fn, fade_ms: float = 60.0):
    """Verwerk de tijdlijn per segment (process_fn(chunk, kind) -> chunk of None
    voor 'laat origineel') en smeed alles met crossfades weer aaneen."""
    from chat_with_audio.segments import classify_segments

    n = x2.shape[1]
    segs = classify_segments(x2, sr)
    fade = max(1, int(fade_ms / 1000 * sr))
    pad = max(fade, int(0.3 * sr))

    out = np.zeros_like(x2, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for seg in segs:
        a, b = int(seg["start_s"] * sr), int(seg["end_s"] * sr)
        if b <= a:
            continue
        aa, bb = max(0, a - pad), min(n, b + pad)
        chunk = x2[:, aa:bb]
        proc = process_fn(chunk, seg["kind"])
        if proc is None:
            proc = chunk
        w = np.ones(bb - aa)
        ramp = max(1, min(fade, (bb - aa) // 2))
        if aa > 0:
            w[:ramp] = np.linspace(0.0, 1.0, ramp)
        if bb < n:
            w[-ramp:] = np.minimum(w[-ramp:], np.linspace(1.0, 0.0, ramp))
        out[:, aa:bb] += np.asarray(proc, dtype=np.float64) * w
        wsum[aa:bb] += w

    holes = wsum <= 1e-9
    out[:, holes] = x2[:, holes]
    wsum[holes] = 1.0
    return (out / wsum[None, :]).astype(np.float32)


def _step_smart_denoise(x, sr, speech_strength_db: float = 24.0,
                        music_strength_db: float = 6.0,
                        silence_strength_db: float = 18.0, fade_ms: float = 60.0):
    """Segment-gestuurde ontruising: AI (DeepFilterNet) op spraak, milde spectral
    gating op muziek, stevige reductie op stiltes."""
    x2 = x[None, :] if x.ndim == 1 else x
    ai_ok = dsp.ai_denoise_available()

    def process(chunk, kind):
        if kind == "speech" and speech_strength_db > 0:
            if ai_ok:
                return dsp.ai_denoise(chunk, sr, strength_db=speech_strength_db)
            return dsp.spectral_denoise(chunk, sr,
                                        reduction_db=min(speech_strength_db, 18))
        if kind == "music" and music_strength_db > 0:
            return dsp.spectral_denoise(chunk, sr, reduction_db=music_strength_db)
        if kind == "silence" and silence_strength_db > 0:
            return dsp.spectral_denoise(chunk, sr, reduction_db=silence_strength_db)
        return None

    return _assemble_segments(x2, sr, process, fade_ms)


def _step_deess(x, sr, strength_db: float = 8.0, sensitivity: float = 2.2,
                fade_ms: float = 60.0):
    """De-esser op de spraaksegmenten; muziek blijft onaangeroerd."""
    from chat_with_audio.dsp.deess import deess

    x2 = x[None, :] if x.ndim == 1 else x

    def process(chunk, kind):
        if kind == "speech":
            return deess(chunk, sr, strength_db=strength_db, sensitivity=sensitivity)
        return None

    return _assemble_segments(x2, sr, process, fade_ms)


def _step_dereverb(x, sr, fade_ms: float = 60.0):
    """Dereverberatie (ClearVoice MossFormer2) op de spraaksegmenten; muziek en
    stilte blijven onaangeroerd. Vereist het [enhance]-extra."""
    from chat_with_audio.dsp import dereverb as drv

    if not drv.is_available():
        raise RuntimeError(drv.INSTALL_HINT)
    x2 = x[None, :] if x.ndim == 1 else x

    def process(chunk, kind):
        return drv.dereverb(chunk, sr) if kind == "speech" else None

    return _assemble_segments(x2, sr, process, fade_ms)


def _step_band_duck(x, sr, low_hz: float = 60.0, high_hz: float = 170.0,
                    headroom_db: float = 10.0, threshold_db: float | None = None,
                    max_cut_db: float = 12.0, attack_ms: float = 8.0,
                    release_ms: float = 120.0, music_only: bool = True):
    """Dynamische banddemping (dreun-bestrijding) via parallelle bandaftrek.

    Dempt de band low_hz-high_hz wanneer hij de mix domineert: alles waar de
    band boven (totaalniveau - headroom_db) uitkomt wordt weggeregeld (tot
    max_cut_db). threshold_db zet in plaats daarvan een absolute banddrempel.
    Met music_only blijft spraak volledig onaangetast."""
    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    # Zero-fase bandextractie via FFT-masker: bij parallelle aftrek zou de
    # fasedraaiing van IIR-filters de demping grotendeels opheffen.
    spec = np.fft.rfft(x2, axis=1)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    ramp = 12.0  # Hz overgangszone
    mask = np.clip((freqs - (low_hz - ramp)) / ramp, 0.0, 1.0) \
        * np.clip(((high_hz + ramp) - freqs) / ramp, 0.0, 1.0)
    band = np.fft.irfft(spec * mask[None, :], n=n, axis=1).astype(np.float32)

    block = max(1, int(sr * 0.001))
    nb = (n + block - 1) // block

    def _env_db(sig):
        det = np.abs(sig).max(axis=0)
        padded = np.zeros(nb * block, dtype=det.dtype)
        padded[:det.shape[0]] = det
        return 20 * np.log10(padded.reshape(nb, block).max(axis=1) + 1e-10)

    env_db = _env_db(band)

    # segmentmasker: alleen muziek dempen (spraakwarmte blijft intact)
    active = np.ones(nb, dtype=bool)
    if music_only:
        from chat_with_audio.segments import classify_segments

        active[:] = False
        for seg in classify_segments(x2, sr):
            if seg["kind"] == "music":
                a = int(seg["start_s"] * sr / block)
                b = int(seg["end_s"] * sr / block) + 1
                active[a:b] = True
        if not active.any():
            return x2.astype(np.float32)

    if threshold_db is not None:
        thr = np.full(nb, float(threshold_db))
    else:
        thr = _env_db(x2) - abs(headroom_db)  # band mag niet dicht bij de mix komen

    static_cut = np.where(active, np.clip(env_db - thr, 0.0, abs(max_cut_db)), 0.0)

    aA = float(np.exp(-block / (0.001 * attack_ms * sr))) if attack_ms > 0 else 0.0
    aR = float(np.exp(-block / (0.001 * release_ms * sr))) if release_ms > 0 else 0.0
    cut = np.empty_like(static_cut)
    c = 0.0
    for i, s in enumerate(static_cut):
        c = aA * c + (1 - aA) * s if s > c else aR * c + (1 - aR) * s
        cut[i] = c

    centers = np.arange(nb) * block + block / 2.0
    g = np.interp(np.arange(n), centers, 10.0 ** (-cut / 20.0)).astype(np.float32)
    return (x2 - band * (1.0 - g)[None, :]).astype(np.float32)


def _step_pause_duck(x, sr, duck_db: float = 20.0, speech_floor_db: float = -32.0,
                     pad_ms: float = 100.0, fade_ms: float = 60.0):
    """Uitzend-stilte: alles buiten de spraak omlaag, zonder gate-artefacten.

    Framegebaseerde spraakdetectie (25 ms) met beschermmarge (pad_ms) aan
    weerszijden, daarna zachte fades — woordaanzetten en slotmedeklinkers
    blijven staan waar een klassieke gate ze zou afknabbelen.
    """
    from scipy.ndimage import binary_dilation as _dil

    x2 = x[None, :] if x.ndim == 1 else x
    n = x2.shape[1]
    mono = x2.mean(axis=0).astype(np.float64)
    flen = max(1, int(sr * 0.025))
    nf = max(1, n // flen)
    fr = 10 * np.log10((mono[: nf * flen].reshape(nf, flen) ** 2).mean(axis=1) + 1e-20)
    speech = fr > speech_floor_db
    if not speech.any():
        return x2.astype(np.float32)
    pad_frames = max(1, int(pad_ms / 25.0))
    speech = _dil(speech, iterations=pad_frames)

    g = 10.0 ** (-abs(duck_db) / 20.0)
    env_f = np.where(speech, 1.0, g)
    centers = np.arange(nf) * flen + flen / 2.0
    env = np.interp(np.arange(n), centers, env_f).astype(np.float32)
    # extra gladstrijken zodat de overgang nooit hoorbaar hakt
    from scipy.ndimage import uniform_filter1d

    env = uniform_filter1d(env, size=max(3, int(fade_ms / 1000 * sr)), mode="nearest")
    return (x2 * env[None, :]).astype(np.float32)


def _step_breath_control(x, sr, reduction_db: float = 10.0, max_breath_s: float = 0.9):
    """Ademhalingen detecteren en dempen (niet wegknippen)."""
    from chat_with_audio.dsp.dialogue import breath_control

    y, n = breath_control(x, sr, reduction_db=reduction_db, max_breath_s=max_breath_s)
    log.info("breath_control: %d adem(s)", n)
    return y


def _step_deplosive(x, sr, cutoff_hz: float = 120.0, sensitivity_db: float = 6.0):
    """Plosief-pops (p/b-drukstoten) alleen op de pop zelf highpassen."""
    from chat_with_audio.dsp.dialogue import deplosive

    y, n = deplosive(x, sr, cutoff_hz=cutoff_hz, sensitivity_db=sensitivity_db)
    log.info("deplosive: %d pop(s)", n)
    return y


def _step_duck_music(x, sr, gap_db: float = 6.0, fade_ms: float = 120.0,
                     mode: str = "beds", attack_ms: float = 15.0,
                     release_ms: float = 250.0):
    """Muziek onder het spraakniveau: mode=beds (bedden tussen de spraak) of
    mode=stems (sidechain via Demucs, voor muziek ónder spraak)."""
    from chat_with_audio.dsp.dialogue import duck_music

    y, info = duck_music(x, sr, gap_db=gap_db, fade_ms=fade_ms, mode=mode,
                         attack_ms=attack_ms, release_ms=release_ms)
    log.info("duck_music: %s", info)
    return y


def _step_trim(x, sr, start_s: float | None = None, end_s: float | None = None,
               to_modulation: bool = False, threshold_db: float = -60.0,
               keep_s: float = 0.25, pad_head_s: float = 0.0, pad_tail_s: float = 0.0):
    """Kop/staart snijden (expliciet of tot eerste/laatste modulatie) en/of
    stilte aanzetten."""
    from chat_with_audio.dsp import utility

    return utility.trim(x, sr, start_s=start_s, end_s=end_s,
                        to_modulation=to_modulation, threshold_db=threshold_db,
                        keep_s=keep_s, pad_head_s=pad_head_s, pad_tail_s=pad_tail_s)


def _step_polarity_invert(x, sr, channels: list | None = None):
    from chat_with_audio.dsp import utility

    return utility.polarity_invert(x, channels=channels)


def _step_sample_delay(x, sr, samples: int | None = None, ms: float | None = None,
                       channel: int | None = None):
    from chat_with_audio.dsp import utility

    return utility.sample_delay(x, sr, samples=samples, ms=ms, channel=channel)


def _step_channel_map(x, sr, mode: str | None = None, order: list | None = None):
    from chat_with_audio.dsp import utility

    return utility.channel_map(x, mode=mode, order=order)


def _step_mid_side(x, sr, width: float = 1.0, mid_db: float = 0.0,
                   side_db: float = 0.0):
    from chat_with_audio.dsp import utility

    return utility.mid_side(x, width=width, mid_db=mid_db, side_db=side_db)


def _step_bass_mono(x, sr, freq: float = 120.0):
    from chat_with_audio.dsp import utility

    return utility.bass_mono(x, sr, freq=freq)


def _step_expander(x, sr, threshold_db: float = -45.0, ratio: float = 2.0,
                   attack_ms: float = 5.0, release_ms: float = 120.0,
                   range_db: float = 24.0):
    from chat_with_audio.dsp import dynamics_plus

    return dynamics_plus.expander(x, sr, threshold_db=threshold_db, ratio=ratio,
                                  attack_ms=attack_ms, release_ms=release_ms,
                                  range_db=range_db)


def _step_multiband_compressor(x, sr, crossovers: list | None = None,
                               threshold_db=-24.0, ratio=2.5,
                               attack_ms: float = 15.0, release_ms: float = 150.0,
                               knee_db: float = 6.0, makeup_db: float = 0.0,
                               band_gains_db: list | None = None):
    from chat_with_audio.dsp import dynamics_plus

    return dynamics_plus.multiband_compressor(
        x, sr, crossovers=crossovers, threshold_db=threshold_db, ratio=ratio,
        attack_ms=attack_ms, release_ms=release_ms, knee_db=knee_db,
        makeup_db=makeup_db, band_gains_db=band_gains_db)


def _step_transient_shaper(x, sr, attack_db: float = 0.0, sustain_db: float = 0.0,
                           attack_window_ms: float = 30.0,
                           sustain_release_ms: float = 200.0):
    from chat_with_audio.dsp import dynamics_plus

    return dynamics_plus.transient_shaper(x, sr, attack_db=attack_db,
                                          sustain_db=sustain_db,
                                          attack_window_ms=attack_window_ms,
                                          sustain_release_ms=sustain_release_ms)


def _step_time_stretch(x, sr, factor: float = 1.0):
    """Duur x factor (1.25 = 25% langer), toonhoogte blijft staan."""
    from chat_with_audio.dsp import timepitch

    return timepitch.time_stretch(x, sr, factor)


def _step_pitch_shift(x, sr, semitones: float = 0.0, preserve_formants: bool = True):
    """Toonhoogte +/- semitones, duur exact gelijk; preserve_formants houdt de
    klankkleur (spectrale envelop) op zijn plek."""
    from chat_with_audio.dsp import timepitch

    return timepitch.pitch_shift(x, sr, semitones, preserve_formants=preserve_formants)


def _step_varispeed(x, sr, factor: float | None = None, semitones: float | None = None):
    """Tape-stijl: duur én toonhoogte samen (factor = snelheid, of semitones)."""
    from chat_with_audio.dsp import timepitch

    return timepitch.varispeed(x, sr, factor=factor, semitones=semitones)


def _step_gate(x, sr, threshold_db: float, attack_ms: float = 5.0,
               release_ms: float = 120.0, hold_ms: float = 50.0, range_db: float = 12.0):
    return dsp.noise_gate(x, sr, threshold_db, attack_ms, release_ms, hold_ms, range_db)


def _step_compressor(x, sr, threshold_db: float, ratio: float = 3.0, attack_ms: float = 10.0,
                     release_ms: float = 150.0, knee_db: float = 6.0, makeup_db: float = 0.0):
    return dsp.compressor(x, sr, threshold_db, ratio, attack_ms, release_ms, knee_db, makeup_db)


def _step_leveler(x, sr, target_db: float = -18.0, max_boost_db: float = 20.0,
                  max_cut_db: float = 12.0, floor_db: float | None = None,
                  smooth_s: float = 0.8):
    """Automatische gain-riding: stille passages (spraak) omhoog, luide (muziek)
    omlaag naar een gezamenlijk kortetermijnniveau. Stilte/ruis onder floor_db
    wordt niet opgetild."""
    from scipy.ndimage import gaussian_filter1d

    x2 = x[None, :] if x.ndim == 1 else x
    mono = x2.mean(axis=0).astype(np.float64)
    n = mono.shape[0]
    hop = max(1, int(sr * 0.05))
    half = max(1, int(sr * 0.2))  # 400 ms meetvenster

    cs = np.concatenate([[0.0], np.cumsum(mono**2)])
    centers = np.arange(0, n, hop)
    lo = np.maximum(centers - half, 0)
    hi = np.minimum(centers + half, n)
    level = 10.0 * np.log10((cs[hi] - cs[lo]) / np.maximum(hi - lo, 1) + 1e-20)

    if floor_db is None:
        quiet = np.sort(level)[: max(1, len(level) // 10)]
        floor_db = float(quiet.mean()) + 8.0

    active = level > floor_db
    if not active.any():
        return x2.astype(np.float32)
    gain_db = np.clip(target_db - level, -abs(max_cut_db), abs(max_boost_db))
    idx = np.where(active)[0]
    # inactieve frames (pauzes) volgen hun actieve buren, zodat ruis niet wordt opgepompt
    gain_db = np.interp(np.arange(len(level)), idx, gain_db[idx])
    sigma_frames = max(smooth_s * sr / hop / 2.0, 1.0)
    gain_db = gaussian_filter1d(gain_db, sigma=sigma_frames)

    gains = (10.0 ** (gain_db / 20.0)).astype(np.float64)
    per_sample = np.interp(np.arange(n), centers, gains)
    return (x2 * per_sample[None, :]).astype(np.float32)


def _step_limiter(x, sr, ceiling_db: float = -1.5, release_ms: float = 60.0,
                  lookahead_ms: float = 5.0):
    return dsp.limiter(x, sr, ceiling_db, release_ms, lookahead_ms)


def _step_loudness_normalize(x, sr, target_lufs: float = -16.0, true_peak_db: float = -1.5):
    y, _info = normalize_loudness(x, sr, target_lufs, true_peak_db)
    return y


STEP_REGISTRY = {
    "highpass": _step_highpass,
    "lowpass": _step_lowpass,
    "notch": _step_notch,
    "eq": _step_eq,
    "gain": _step_gain,
    "declip": _step_declip,
    "declick": _step_declick,
    "denoise": _step_denoise,
    "smart_denoise": _step_smart_denoise,
    "band_duck": _step_band_duck,
    "pause_duck": _step_pause_duck,
    "deess": _step_deess,
    "dereverb": _step_dereverb,
    "breath_control": _step_breath_control,
    "deplosive": _step_deplosive,
    "duck_music": _step_duck_music,
    "trim": _step_trim,
    "polarity_invert": _step_polarity_invert,
    "sample_delay": _step_sample_delay,
    "channel_map": _step_channel_map,
    "mid_side": _step_mid_side,
    "bass_mono": _step_bass_mono,
    "expander": _step_expander,
    "multiband_compressor": _step_multiband_compressor,
    "transient_shaper": _step_transient_shaper,
    "time_stretch": _step_time_stretch,
    "pitch_shift": _step_pitch_shift,
    "varispeed": _step_varispeed,
    "gate": _step_gate,
    "compressor": _step_compressor,
    "leveler": _step_leveler,
    "limiter": _step_limiter,
    "loudness_normalize": _step_loudness_normalize,
}


def validate_steps(steps: list[dict]) -> list[tuple[str, Callable, dict]]:
    """Controleer stappen tegen STEP_REGISTRY zonder ze uit te voeren.

    Geeft per stap (type, fn, params) terug; gebruikt door run_chain en door
    recipes (een recept mag nooit stilletjes ongeldige stappen bevatten).
    """
    checked: list[tuple[str, Callable, dict]] = []
    for step in steps:
        step = dict(step)
        stype = step.pop("type", None)
        fn = STEP_REGISTRY.get(stype)
        if fn is None:
            raise ValueError(f"Onbekende stap '{stype}'. Geldig: {sorted(STEP_REGISTRY)}")
        sig = inspect.signature(fn)
        valid = {k for k in sig.parameters if k not in ("x", "sr")}
        unknown = set(step) - valid
        if unknown:
            raise ValueError(f"Onbekende parameter(s) {sorted(unknown)} voor stap "
                             f"'{stype}'. Geldig: {sorted(valid)}")
        checked.append((stype, fn, step))
    return checked


def run_chain(x: np.ndarray, sr: int, steps: list[dict],
              progress=None) -> tuple[np.ndarray, list[dict]]:
    """Voer de stappen uit; geeft (audio, resolved_steps incl. defaults) terug."""
    y = x
    resolved: list[dict] = []
    checked = validate_steps(steps)  # alles vooraf valideren, dan pas rekenen
    for i, (stype, fn, params_in) in enumerate(checked):
        sig = inspect.signature(fn)
        bound = sig.bind(None, sr, **params_in)
        bound.apply_defaults()
        params = {k: v for k, v in bound.arguments.items() if k not in ("x", "sr")}
        log.info("stap %d/%d: %s %s", i + 1, len(checked), stype, params)
        if progress:
            progress(i, len(checked), stype)
        y = fn(y, sr, **params_in)
        resolved.append({"type": stype, **params})
    return y, resolved
