"""Aflevering: codec-preview (encode→decode→meetbaar verschil), checksums en
het complete afleverpakket (master + rapporten + manifest in één map).

Codec-preview beantwoordt de vraag "wat doet de mp3/AAC-compressie straks met
mijn master?" vóór publicatie: loudness-verschuiving, true-peak-overshoot
(codec overs — dé reden dat streamingdiensten -1 à -2 dBTP eisen) en waar in
het spectrum de codec het meest weggooit.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from chat_with_audio import analysis

CODECS = {
    "mp3": {"format": "MP3", "subtype": None, "ext": ".mp3"},
    "ogg": {"format": "OGG", "subtype": "VORBIS", "ext": ".ogg"},
    "opus": {"format": "OGG", "subtype": "OPUS", "ext": ".opus"},
}

_OPUS_RATES = {8000, 12000, 16000, 24000, 48000}


def codec_roundtrip(x: np.ndarray, sr: int, codec: str) -> tuple[np.ndarray, int]:
    """Encodeer en decodeer via libsndfile; geeft (audio, sr) van de decode."""
    spec = CODECS.get(codec)
    if spec is None:
        raise ValueError(f"Onbekende codec '{codec}'. Beschikbaar: {sorted(CODECS)}")
    x2 = x[None, :] if x.ndim == 1 else x
    use_sr = sr
    if codec == "opus" and sr not in _OPUS_RATES:
        from chat_with_audio import io as audio_io

        x2, use_sr = audio_io.resample(x2, sr, 48000)
    with tempfile.NamedTemporaryFile(suffix=spec["ext"], delete=True) as f:
        sf.write(f.name, x2.T, use_sr, format=spec["format"],
                 subtype=spec["subtype"])
        y, y_sr = sf.read(f.name, dtype="float32", always_2d=True)
    return y.T, y_sr


def codec_report(x: np.ndarray, sr: int, codecs: list[str]) -> list[dict]:
    """Meet per codec wat de compressie met het signaal doet."""
    x2 = x[None, :] if x.ndim == 1 else x
    lufs_in = analysis.measure_lufs(x2, sr)
    tp_in = analysis._true_peak_db(x2, sr)
    out = []
    for codec in codecs:
        y, y_sr = codec_roundtrip(x2, sr, codec)
        n = min(x2.shape[1], y.shape[1])
        lufs_out = analysis.measure_lufs(y, y_sr)
        tp_out = analysis._true_peak_db(y, y_sr)
        overs = tp_out > -0.3
        report = {
            "codec": codec,
            "true_peak_in_dbtp": round(tp_in, 2),
            "true_peak_out_dbtp": round(tp_out, 2),
            "true_peak_delta_db": round(tp_out - tp_in, 2),
            "lufs_delta": (round(lufs_out - lufs_in, 2)
                           if lufs_in is not None and lufs_out is not None else None),
            "codec_overs": bool(overs),
        }
        if y_sr == sr and n > sr:  # residu alleen zinvol zonder resample
            resid = y[:, :n].astype(np.float64) - x2[:, :n].astype(np.float64)
            sig_p = float(np.mean(x2[:, :n].astype(np.float64) ** 2)) + 1e-20
            report["residual_db"] = round(10 * np.log10(
                float(np.mean(resid**2)) / sig_p + 1e-20), 1)
        report["verdict"] = (
            "codec overs: true peak komt boven -0.3 dBTP uit — master naar "
            "-1.5 à -2 dBTP vóór lossy export" if overs else "ok")
        out.append(report)
    return out


def md5sum(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_checksums(paths: list[Path], out_file: Path) -> None:
    lines = [f"{md5sum(p)}  {p.name}" for p in sorted(paths, key=lambda p: p.name)]
    out_file.write_text("\n".join(lines) + "\n")


def write_manifest(out_dir: Path, entries: list[dict], meta: dict) -> Path:
    manifest = {"format": "chat-with-audio/delivery@1", **meta, "files": entries}
    p = out_dir / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return p
