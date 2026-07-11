"""Recepten: bewaarde bewerkingsketens die je kunt hergebruiken en delen.

Een recept is een klein JSON-bestand (naam, beschrijving, stappen, herkomst).
Eigen recepten staan in ~/AudioImprove/recipes/ (env AIT_RECIPES_DIR); de
toolkit levert daarnaast ingebouwde presets mee die uit echte sessies zijn
gedestilleerd. Delen is het bestand doorgeven: apply_recipe accepteert ook
een pad naar een recept-JSON van iemand anders.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from chat_with_audio.chain import validate_steps

FORMAT = "chat-with-audio/recipe@1"


def recipes_dir() -> Path:
    root = os.environ.get("AIT_RECIPES_DIR")
    d = Path(root).expanduser() if root else Path.home() / "AudioImprove" / "recipes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def builtin_dir() -> Path:
    return Path(__file__).parent / "recipes"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")[:48]


def list_recipes() -> list[dict]:
    """Alle recepten (ingebouwd + eigen); een eigen recept met dezelfde naam
    gaat vóór het ingebouwde."""
    seen: dict[str, dict] = {}
    for base, builtin in ((builtin_dir(), True), (recipes_dir(), False)):
        if not base.is_dir():
            continue
        for f in sorted(base.glob("*.json")):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            name = data.get("name") or f.stem
            seen[name] = {"name": name,
                          "description": data.get("description", ""),
                          "steps": len(data.get("steps") or []),
                          "builtin": builtin,
                          "source_session": data.get("source_session"),
                          "path": str(f)}
    return sorted(seen.values(), key=lambda r: r["name"])


def load_recipe(ref: str) -> dict:
    """Laad een recept op naam (list_recipes) of op pad naar een JSON-bestand."""
    p = Path(str(ref)).expanduser()
    if p.suffix == ".json" and p.is_file():
        data = json.loads(p.read_text())
    else:
        slug = _slug(str(ref))
        for base in (recipes_dir(), builtin_dir()):
            f = base / f"{slug}.json"
            if f.is_file():
                data = json.loads(f.read_text())
                break
        else:
            known = ", ".join(r["name"] for r in list_recipes()) or "(geen)"
            raise FileNotFoundError(f"Recept '{ref}' niet gevonden. "
                                    f"Beschikbaar: {known}")
    steps = data.get("steps") or []
    if not steps:
        raise ValueError(f"Recept '{ref}' bevat geen stappen.")
    validate_steps(steps)  # nooit stilletjes een kapot/rommel-recept toepassen
    data.setdefault("name", _slug(str(ref)) or "recept")
    return data


def save_recipe(name: str, steps: list[dict], description: str = "",
                source_session: str | None = None) -> dict:
    """Bewaar een keten als recept in de eigen receptenmap; geeft metadata terug."""
    slug = _slug(name)
    if not slug:
        raise ValueError("Lege of ongeldige receptnaam.")
    if not steps:
        raise ValueError("Een recept zonder stappen heeft geen zin.")
    validate_steps(steps)
    data = {"format": FORMAT, "name": slug, "description": description,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_session": source_session, "steps": steps}
    f = recipes_dir() / f"{slug}.json"
    f.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return {"name": slug, "path": str(f), "description": description,
            "steps": len(steps)}
