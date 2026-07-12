"""Whisper-transcriptie als meetbare verstaanbaarheidscheck (optioneel [asr]-extra).

Gebruik: transcribeer origineel en bewerking en vergelijk met word_retention().
Zakt de retentie, dan heeft de bewerking spraakdetail gesloopt — een hard signaal
waarop de verfijnlus of Claude kan bijsturen.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

INSTALL_HINT = ("Whisper is niet geinstalleerd. Installeer met: uv sync --all-extras "
                "(in de projectmap).")

_MODELS: dict = {}


def is_available() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except Exception:
        return False


def _model(size: str):
    if size not in _MODELS:
        if not is_available():
            raise RuntimeError(INSTALL_HINT)
        import whisper

        log.info("Whisper-model '%s' laden...", size)
        _MODELS[size] = whisper.load_model(size)
    return _MODELS[size]


def _prep(x: np.ndarray, sr: int) -> np.ndarray:
    """(channels, n) audio -> mono 16 kHz float32 in [-1, 1] voor Whisper."""
    x2 = x[None, :] if x.ndim == 1 else x
    mono = x2.mean(axis=0).astype(np.float64)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // g, sr // g)
    peak = np.abs(mono).max()
    if peak > 1.0:  # 32-bit float headroom: Whisper verwacht [-1, 1]
        mono = mono / peak * 0.9
    return mono.astype(np.float32)


def transcribe(x: np.ndarray, sr: int, model_size: str = "small",
               language: str = "nl") -> dict:
    """Transcribeer (channels, n) audio; geeft tekst + segmenten met zekerheid."""
    result = _model(model_size).transcribe(_prep(x, sr), language=language, fp16=False)
    return {
        "text": result["text"].strip(),
        "language": result.get("language", language),
        "segments": [{
            "start": round(float(s["start"]), 2),
            "end": round(float(s["end"]), 2),
            "text": s["text"].strip(),
            "avg_logprob": round(float(s["avg_logprob"]), 3),
            "no_speech_prob": round(float(s["no_speech_prob"]), 3),
        } for s in result["segments"]],
    }


def transcribe_words(x: np.ndarray, sr: int, model_size: str = "small",
                     language: str = "nl") -> dict:
    """Transcriptie met woord-timestamps — de basis voor tekstgestuurd knippen
    (textedit.plan_edits / de edit_speech-tool)."""
    result = _model(model_size).transcribe(_prep(x, sr), language=language,
                                           fp16=False, word_timestamps=True)
    words = [{"word": str(w["word"]).strip(),
              "start": round(float(w["start"]), 3),
              "end": round(float(w["end"]), 3),
              "probability": round(float(w.get("probability", 0.0)), 3)}
             for s in result["segments"] for w in (s.get("words") or [])]
    return {"text": result["text"].strip(),
            "language": result.get("language", language),
            "words": words}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ɏ']+", text.lower())


def word_retention(reference_text: str, hypothesis_text: str) -> float:
    """Welk deel van de referentiewoorden komt terug in de hypothese (0-1)."""
    ref, hyp = _words(reference_text), _words(hypothesis_text)
    if not ref:
        return 1.0
    bag: dict[str, int] = {}
    for w in hyp:
        bag[w] = bag.get(w, 0) + 1
    hit = 0
    for w in ref:
        if bag.get(w, 0) > 0:
            bag[w] -= 1
            hit += 1
    return round(hit / len(ref), 3)
