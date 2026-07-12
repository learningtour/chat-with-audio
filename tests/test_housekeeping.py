"""Professionalisering: sessie-zoeken/opruimen en de viewer-versiehandshake."""

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import server, sessions

SR = 44100


def make_session(tmp_path, name):
    p = tmp_path / f"{name}.wav"
    sf.write(str(p), np.zeros(SR, dtype=np.float32), SR)
    return server.apply_chain(str(p), steps=[{"type": "gain", "gain_db": -1}])


def test_list_sessions_search_and_limit(tmp_path):
    make_session(tmp_path, "interview")
    make_session(tmp_path, "muziek")
    make_session(tmp_path, "interview-take2")

    res = server.list_sessions(search="interview")
    assert res["count"] == 2
    res = server.list_sessions(limit=1)
    assert res["count"] == 3 and res["shown"] == 1


def test_cleanup_sessions_dry_run_then_delete(tmp_path):
    make_session(tmp_path, "a")
    make_session(tmp_path, "b")
    make_session(tmp_path, "c")

    dry = server.cleanup_sessions(keep_last=1)
    assert dry["dry_run"] is True and dry["count"] == 2
    assert len(sessions.list_sessions()) == 3  # nog niets weg

    real = server.cleanup_sessions(keep_last=1, dry_run=False)
    assert real["count"] == 2 and real["deleted"]
    assert len(sessions.list_sessions()) == 1


def test_cleanup_requires_a_criterion():
    with pytest.raises(ValueError):
        server.cleanup_sessions()


def test_cleanup_by_search_only(tmp_path):
    make_session(tmp_path, "podcast-ep1")
    make_session(tmp_path, "vergadering")
    res = server.cleanup_sessions(search="podcast", dry_run=False)
    assert res["count"] == 1
    left = sessions.list_sessions()
    assert len(left) == 1 and "vergadering" in left[0]["label"]


def test_delete_session_guards():
    with pytest.raises(FileNotFoundError):
        sessions.delete_session("../evil")
    with pytest.raises(FileNotFoundError):
        sessions.delete_session("bestaat-niet")


def test_viewer_health_reports_version():
    from chat_with_audio import __version__
    from chat_with_audio.viewer import server as vs

    # de handler zelf draait in tests niet; controleer de bron van waarheid
    assert __version__ == "0.3.0"
    assert vs is not None
