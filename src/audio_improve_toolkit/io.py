"""Audio-I/O: soundfile voor wav/flac/ogg/mp3, ffmpeg als vangnet en voor export.

Let op: alle subprocessen draaien met capture_output zodat er nooit iets op
stdout belandt (het MCP stdio-transport zou daarop breken).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


class AudioIOError(RuntimeError):
    pass


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise AudioIOError(
            f"{name} niet gevonden. Installeer ffmpeg (macOS: brew install ffmpeg, "
            f"Windows: winget install ffmpeg) en herstart daarna de MCP-server.")
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise AudioIOError(f"{Path(cmd[0]).name} faalde: " + " | ".join(tail))
    return proc


def load_audio(path: str | Path, mono: bool = False) -> tuple[np.ndarray, int]:
    """Laad audio als float32 (channels, n). Valt terug op ffmpeg-decodering."""
    path = Path(path).expanduser()
    if not path.exists():
        raise AudioIOError(f"Bestand niet gevonden: {path}")
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        log.info("soundfile kan %s niet lezen; decoderen via ffmpeg", path.name)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "decoded.wav"
            decode_to_wav(path, tmp)
            data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    x = np.ascontiguousarray(data.T)  # (channels, n)
    if x.shape[1] == 0:
        raise AudioIOError(f"Bestand bevat geen audio: {path}")
    if mono and x.shape[0] > 1:
        x = x.mean(axis=0, keepdims=True)
    return x, int(sr)


def save_wav(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_24") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x2 = x[None, :] if x.ndim == 1 else x
    # 32-bit float opnames kunnen boven 0 dBFS pieken; dan als FLOAT wegschrijven
    # zodat de headroom behouden blijft in plaats van hard af te kappen.
    if x2.size and float(np.abs(x2).max()) > 0.999:
        subtype = "FLOAT"
    sf.write(str(path), x2.T, sr, subtype=subtype)
    return path


def decode_to_wav(src: str | Path, dst: str | Path, sr: int | None = None) -> Path:
    cmd = [_tool("ffmpeg"), "-y", "-i", str(src), "-map", "0:a:0"]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd += ["-c:a", "pcm_f32le", str(dst)]
    _run(cmd)
    return Path(dst)


_ENCODERS = {
    ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
    ".aac": ["-c:a", "aac", "-b:a", "192k"],
    ".flac": ["-c:a", "flac"],
    ".ogg": ["-c:a", "libvorbis", "-q:a", "6"],
}


def encode_wav_to(wav_path: str | Path, out_path: str | Path) -> Path:
    """Exporteer een wav naar het formaat dat bij de extensie van out_path hoort."""
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in ("", ".wav"):
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".wav")
        shutil.copyfile(wav_path, out_path)
        return out_path
    args = _ENCODERS.get(ext)
    if args is None:
        raise AudioIOError(f"Onbekend exportformaat '{ext}'. "
                           f"Ondersteund: wav, {', '.join(k[1:] for k in _ENCODERS)}")
    _run([_tool("ffmpeg"), "-y", "-i", str(wav_path), *args, str(out_path)])
    return out_path


def probe(path: str | Path) -> dict:
    """Containerinfo (formaat, codec, duur, bitrate) via ffprobe; sf.info als vangnet."""
    path = Path(path).expanduser()
    try:
        proc = _run([_tool("ffprobe"), "-v", "error", "-print_format", "json",
                     "-show_format", "-show_streams", str(path)])
        info = json.loads(proc.stdout)
        audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
        fmt = info.get("format", {})
        return {
            "format": fmt.get("format_name"),
            "codec": audio.get("codec_name"),
            "sample_rate": int(audio.get("sample_rate", 0) or 0),
            "channels": int(audio.get("channels", 0) or 0),
            "duration_s": round(float(fmt.get("duration", 0) or 0), 2),
            "bit_rate": int(fmt.get("bit_rate", 0) or 0),
        }
    except Exception:
        try:
            i = sf.info(str(path))
            return {"format": i.format, "codec": i.subtype, "sample_rate": i.samplerate,
                    "channels": i.channels, "duration_s": round(i.duration, 2), "bit_rate": 0}
        except Exception:
            return {}
