"""ID3v2.3-tags met hoofdstukmarkeringen (CHAP/CTOC) voor podcast-mp3's.

Pure stdlib. Hoofdstukken zijn het verschil tussen een mp3 en een
podcastaflevering: spelers (Apple Podcasts, Overcast, Pocket Casts) tonen
de titels en maken de tijdlijn klikbaar. build_tag maakt de tagbytes;
write_tags zet ze vóór de mp3-frames (een bestaande ID3v2 wordt vervangen).
"""

from __future__ import annotations

import struct
from pathlib import Path


def _synchsafe(n: int) -> bytes:
    return bytes([(n >> 21) & 0x7F, (n >> 14) & 0x7F, (n >> 7) & 0x7F, n & 0x7F])


def _text_frame(fid: str, text: str) -> bytes:
    # encoding 0x01 = UTF-16 met BOM (v2.3-veilig voor niet-latin tekens)
    payload = b"\x01" + text.encode("utf-16")  # utf-16 levert BOM + LE
    return fid.encode("ascii") + struct.pack(">I", len(payload)) + b"\x00\x00" + payload


def _chap_frame(element_id: str, start_ms: int, end_ms: int, title: str) -> bytes:
    sub = _text_frame("TIT2", title)
    payload = (element_id.encode("ascii") + b"\x00"
               + struct.pack(">IIII", start_ms, end_ms, 0xFFFFFFFF, 0xFFFFFFFF)
               + sub)
    return b"CHAP" + struct.pack(">I", len(payload)) + b"\x00\x00" + payload


def _ctoc_frame(child_ids: list[str]) -> bytes:
    payload = (b"toc\x00"
               + b"\x03"  # top-level + geordend
               + bytes([len(child_ids)])
               + b"".join(c.encode("ascii") + b"\x00" for c in child_ids))
    return b"CTOC" + struct.pack(">I", len(payload)) + b"\x00\x00" + payload


def build_tag(title: str | None = None, artist: str | None = None,
              album: str | None = None,
              chapters: list[dict] | None = None) -> bytes:
    """chapters: [{"start_s": float, "end_s": float, "title": str}, ...]"""
    frames = b""
    if title:
        frames += _text_frame("TIT2", title)
    if artist:
        frames += _text_frame("TPE1", artist)
    if album:
        frames += _text_frame("TALB", album)
    if chapters:
        ids = [f"ch{i}" for i in range(len(chapters))]
        frames += _ctoc_frame(ids)
        for cid, ch in zip(ids, chapters, strict=True):
            frames += _chap_frame(cid, int(float(ch["start_s"]) * 1000),
                                  int(float(ch["end_s"]) * 1000),
                                  str(ch.get("title", cid)))
    if not frames:
        raise ValueError("Niets te taggen: geef titel, artiest of hoofdstukken.")
    return b"ID3\x03\x00\x00" + _synchsafe(len(frames)) + frames


def strip_tag(data: bytes) -> bytes:
    """Verwijder een bestaande ID3v2-tag aan het begin van mp3-data."""
    if data[:3] != b"ID3" or len(data) < 10:
        return data
    size = ((data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14
            | (data[8] & 0x7F) << 7 | (data[9] & 0x7F))
    return data[10 + size:]


def write_tags(mp3_path: str | Path, title: str | None = None,
               artist: str | None = None, album: str | None = None,
               chapters: list[dict] | None = None) -> None:
    """Schrijf/vervang de ID3v2.3-tag van een mp3 (in place)."""
    p = Path(mp3_path)
    audio = strip_tag(p.read_bytes())
    p.write_bytes(build_tag(title, artist, album, chapters) + audio)


def read_chapters(mp3_path: str | Path) -> list[dict]:
    """Lees CHAP-frames terug (voor tests en verificatie)."""
    data = Path(mp3_path).read_bytes()
    if data[:3] != b"ID3":
        return []
    size = ((data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14
            | (data[8] & 0x7F) << 7 | (data[9] & 0x7F))
    pos, end = 10, 10 + size
    out = []
    while pos + 10 <= end:
        fid = data[pos:pos + 4]
        fsize = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        if fid == b"CHAP":
            payload = data[pos + 10:pos + 10 + fsize]
            eid, rest = payload.split(b"\x00", 1)
            start_ms, end_ms = struct.unpack(">II", rest[:8])
            title = ""
            sub = rest[16:]
            if sub[:4] == b"TIT2":
                tlen = struct.unpack(">I", sub[4:8])[0]
                traw = sub[10:10 + tlen]
                title = traw[1:].decode("utf-16") if traw[:1] == b"\x01" else \
                    traw[1:].decode("latin-1")
            out.append({"id": eid.decode(), "start_s": start_ms / 1000,
                        "end_s": end_ms / 1000, "title": title})
        if fid.strip(b"\x00") == b"" or fsize == 0 and fid == b"\x00\x00\x00\x00":
            break
        pos += 10 + fsize
    return out
