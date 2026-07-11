"""Adobe Audition-export: stems als multitrack-sessie (.sesx) + losse wav's.

De .sesx is een minimale, best-effort sessie (Audition's XML-formaat); de losse
wav's staan er altijd naast, dus slepen in een eigen sessie kan ook.
"""

from __future__ import annotations

import glob
import logging
import subprocess
from pathlib import Path
from xml.sax.saxutils import quoteattr

log = logging.getLogger(__name__)

_TRACK_HUES = {"vocals": 210, "drums": 30, "bass": 120, "other": 280, "mix": 0}


def find_audition() -> str | None:
    hits = sorted(glob.glob("/Applications/Adobe Audition*/Adobe Audition*.app")
                  + glob.glob("/Applications/Adobe Audition*.app"), reverse=True)
    return hits[0] if hits else None


def write_sesx(out_dir: Path, name: str, tracks: list[tuple[str, Path, int]],
               sr: int) -> Path:
    """tracks: (naam, wav-pad, duur_in_samples). Best-effort minimale sessie."""
    file_xml, track_xml = [], []
    for i, (tname, path, dur) in enumerate(tracks):
        tid = 10000 + i * 10
        file_xml.append(f'    <file absolutePath={quoteattr(str(path))} id="{i}" '
                        f'mediaHandler="AmioWav" relativePath={quoteattr(path.name)}/>')
        track_xml.append(f"""    <audioTrack automationLaneOpenState="false" id="{tid}" index="{i + 1}" select="false" visible="true">
      <trackParameters trackHeight="100" trackHue="{_TRACK_HUES.get(tname, 200)}" isSelected="false">
        <name>{tname}</name>
      </trackParameters>
      <trackAudioParameters audioChannelType="stereo" automationMode="1" monitoring="false" recordArmed="false" solo="false" soloSafe="false">
        <trackOutput outputID="10000" type="trackID"/>
        <trackInput inputID="1"/>
      </trackAudioParameters>
      <audioClip clipAutoCrossfade="true" crossFadeHeadClipID="-1" crossFadeTailClipID="-1" endPoint="{dur}" fileID="{i}" hue="-1" id="{tid + 1}" lockedInTime="false" looped="false" mute="false" name={quoteattr(tname)} offline="false" select="false" sourceInPoint="0" sourceOutPoint="{dur}" startPoint="0" zOrder="{i}"/>
    </audioTrack>""")
    sesx = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE sesx>
<sesx version="1.9">
  <session appBuild="0" appVersion="24.0" audioChannelType="stereo" bitDepth="32" duration="{max((d for _, _, d in tracks), default=0)}" sampleRate="{sr}">
    <name>{name}</name>
    <tracks>
      <masterTrack automationLaneOpenState="false" id="10000" index="0" select="false" visible="true">
        <trackParameters trackHeight="100" trackHue="-1" isSelected="false">
          <name>Master</name>
        </trackParameters>
        <trackAudioParameters audioChannelType="stereo" automationMode="1" monitoring="false" recordArmed="false" solo="false" soloSafe="false">
          <trackOutput outputID="-1" type="hardwareOutput"/>
        </trackAudioParameters>
      </masterTrack>
{chr(10).join(track_xml)}
    </tracks>
  </session>
  <files>
{chr(10).join(file_xml)}
  </files>
</sesx>
"""
    out = out_dir / f"{name}.sesx"
    out.write_text(sesx)
    return out


def open_in_audition(paths: list[Path]) -> bool:
    app = find_audition()
    if not app:
        return False
    subprocess.run(["open", "-a", app, *[str(p) for p in paths]],
                   capture_output=True)
    return True
