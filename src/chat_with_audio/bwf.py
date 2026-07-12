"""Broadcast-WAV-metadata: bext- en iXML-chunks lezen en schrijven.

De bext-chunk (EBU 3285, v1) draagt originator, datum/tijd, timecode
(TimeReference = samples sinds middernacht) en coding history; iXML draagt
project/scene/take. Beide worden ná de fmt-chunk ingevoegd; bestaande
bext/iXML-chunks worden vervangen. Pure stdlib — geen mutagen nodig.
"""

from __future__ import annotations

import struct
from pathlib import Path

BEXT_V1_SIZE = 602  # vaste velden t/m Reserved; CodingHistory komt erachteraan


def _fix(s: str, n: int) -> bytes:
    return s.encode("ascii", "replace")[:n].ljust(n, b"\x00")


def timecode_to_samples(tc: str, sr: int, fps: float = 25.0) -> int:
    """"HH:MM:SS:FF" -> samples sinds middernacht (bext TimeReference)."""
    parts = tc.strip().split(":")
    if len(parts) != 4:
        raise ValueError(f"Timecode '{tc}' moet HH:MM:SS:FF zijn.")
    h, m, s, f = (int(p) for p in parts)
    if not (0 <= f < fps and 0 <= s < 60 and 0 <= m < 60):
        raise ValueError(f"Timecode '{tc}' buiten bereik (fps {fps}).")
    seconds = h * 3600 + m * 60 + s + f / fps
    return int(round(seconds * sr))


def samples_to_timecode(samples: int, sr: int, fps: float = 25.0) -> str:
    seconds = samples / sr
    h = int(seconds // 3600)
    m = int(seconds % 3600 // 60)
    s = int(seconds % 60)
    f = int(round((seconds - int(seconds)) * fps)) % int(fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def build_bext(description: str = "", originator: str = "",
               originator_reference: str = "", origination_date: str = "",
               origination_time: str = "", time_reference: int = 0,
               coding_history: str = "") -> bytes:
    body = (
        _fix(description, 256)
        + _fix(originator, 32)
        + _fix(originator_reference, 32)
        + _fix(origination_date, 10)
        + _fix(origination_time, 8)
        + struct.pack("<II", time_reference & 0xFFFFFFFF,
                      (time_reference >> 32) & 0xFFFFFFFF)
        + struct.pack("<H", 1)      # BWF versie 1
        + b"\x00" * 64              # UMID (leeg)
        + b"\x00" * 190             # Reserved (v1)
    )
    assert len(body) == BEXT_V1_SIZE
    hist = coding_history.encode("ascii", "replace")
    if hist and not hist.endswith(b"\r\n"):
        hist += b"\r\n"
    return body + hist


def build_ixml(project: str = "", scene: str = "", take: str = "",
               tape: str = "", note: str = "", fps: float = 25.0) -> bytes:
    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    rate = f"{int(fps * 1000)}/1000" if fps != int(fps) else f"{int(fps)}/1"
    xml = ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
           "<BWFXML><IXML_VERSION>1.61</IXML_VERSION>"
           f"<PROJECT>{esc(project)}</PROJECT>"
           f"<SCENE>{esc(scene)}</SCENE>"
           f"<TAKE>{esc(take)}</TAKE>"
           f"<TAPE>{esc(tape)}</TAPE>"
           f"<NOTE>{esc(note)}</NOTE>"
           f"<SPEED><TIMECODE_RATE>{rate}</TIMECODE_RATE>"
           "<TIMECODE_FLAG>NDF</TIMECODE_FLAG></SPEED>"
           "</BWFXML>")
    return xml.encode("utf-8")


def _iter_chunks(data: bytes):
    """Yield (chunk_id, start_of_header, payload_size) over een RIFF-bestand."""
    pos = 12  # na 'RIFF'<size>'WAVE'
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        yield cid, pos, size
        pos += 8 + size + (size & 1)  # chunks zijn word-aligned


def write_chunks(path: str | Path, bext: bytes | None = None,
                 ixml: bytes | None = None) -> None:
    """Voeg bext/iXML toe aan een wav (in place); bestaande worden vervangen."""
    path = Path(path)
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path.name} is geen RIFF/WAVE-bestand "
                         "(RF64/W64 wordt nog niet ondersteund).")
    keep: list[bytes] = []
    fmt_block = None
    for cid, pos, size in _iter_chunks(data):
        block = data[pos:pos + 8 + size + (size & 1)]
        if cid == b"fmt ":
            fmt_block = block
        elif cid in (b"bext", b"iXML"):
            continue  # vervangen
        else:
            keep.append(block)
    if fmt_block is None:
        raise ValueError(f"{path.name}: geen fmt-chunk gevonden.")

    def chunk(cid: bytes, payload: bytes) -> bytes:
        pad = b"\x00" if len(payload) & 1 else b""
        return cid + struct.pack("<I", len(payload)) + payload + pad

    inserts = b""
    if bext is not None:
        inserts += chunk(b"bext", bext)
    if ixml is not None:
        inserts += chunk(b"iXML", ixml)
    body = fmt_block + inserts + b"".join(keep)
    out = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
    path.write_bytes(out)


def read_metadata(path: str | Path) -> dict:
    """Lees bext/iXML uit een wav; lege dict als er niets is."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {}
    result: dict = {}
    for cid, pos, size in _iter_chunks(data):
        payload = data[pos + 8:pos + 8 + size]
        if cid == b"bext" and size >= BEXT_V1_SIZE:
            def s(a, b, _p=payload):
                return _p[a:b].split(b"\x00", 1)[0].decode("ascii", "replace")
            lo, hi = struct.unpack("<II", payload[338:346])
            result["bext"] = {
                "description": s(0, 256),
                "originator": s(256, 288),
                "originator_reference": s(288, 320),
                "origination_date": s(320, 330),
                "origination_time": s(330, 338),
                "time_reference": (hi << 32) | lo,
                "version": struct.unpack("<H", payload[346:348])[0],
                "coding_history": payload[BEXT_V1_SIZE:].split(b"\x00", 1)[0]
                .decode("ascii", "replace").strip(),
            }
        elif cid == b"iXML":
            result["ixml_raw"] = payload.decode("utf-8", "replace")
    return result
