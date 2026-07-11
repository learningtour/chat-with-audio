"""Recepten: opslaan/laden/delen, validatie en de ingebouwde presets."""

import pytest

from chat_with_audio import recipes


def test_builtin_recipes_present_and_valid():
    names = {r["name"] for r in recipes.list_recipes()}
    assert {"podcast-speech", "broadcast-quiet-pauses",
            "music-master", "noisy-speech-rescue"} <= names
    for name in names:
        rec = recipes.load_recipe(name)  # laadt én valideert de stappen
        assert rec["steps"], name
        assert rec["description"], name


def test_save_load_roundtrip_and_share_by_path():
    steps = [{"type": "highpass", "freq": 100}, {"type": "gain", "gain_db": -3}]
    rec = recipes.save_recipe("Mijn Test!", steps, description="testje")
    assert rec["name"] == "mijn-test"

    loaded = recipes.load_recipe("mijn-test")
    assert loaded["steps"] == steps
    assert loaded["description"] == "testje"

    # delen: laden op pad werkt ook (recept-JSON van iemand anders)
    shared = recipes.load_recipe(rec["path"])
    assert shared["steps"] == steps


def test_save_rejects_invalid_steps():
    with pytest.raises(ValueError, match="Onbekende stap"):
        recipes.save_recipe("fout", [{"type": "region", "kind": "hum"}])
    with pytest.raises(ValueError, match="Onbekende parameter"):
        recipes.save_recipe("fout", [{"type": "highpass", "frq": 80}])
    with pytest.raises(ValueError):
        recipes.save_recipe("", [{"type": "highpass"}])


def test_user_recipe_overrides_builtin():
    steps = [{"type": "gain", "gain_db": 1}]
    recipes.save_recipe("music-master", steps, description="eigen versie")
    assert recipes.load_recipe("music-master")["steps"] == steps
    listing = {r["name"]: r for r in recipes.list_recipes()}
    assert listing["music-master"]["builtin"] is False


def test_load_unknown_recipe_lists_alternatives():
    with pytest.raises(FileNotFoundError, match="podcast-speech"):
        recipes.load_recipe("bestaat-niet")
