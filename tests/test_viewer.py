"""Viewer-server: endpoints, guards en de versiehandshake — over echte HTTP."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from chat_with_audio import __version__, server
from chat_with_audio.viewer.server import Handler


@pytest.fixture
def viewer_port():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()
    t.join(timeout=5)


def get(port, path):
    return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5)


def test_health_reports_package_version(viewer_port):
    data = json.loads(get(viewer_port, "/health").read())
    assert data == {"status": "ok", "version": __version__}


def test_index_and_static(viewer_port):
    html = get(viewer_port, "/").read().decode()
    assert "app.js" in html and "Blinde luistertest" in html
    js = get(viewer_port, "/static/app.js").read().decode()
    assert "enterBlind" in js


def test_static_traversal_blocked(viewer_port):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(viewer_port, "/static/../server.py")
    assert e.value.code == 404


def test_sessions_api_lists_created_session(viewer_port, tmp_path):
    sr = 44100
    p = tmp_path / "x.wav"
    sf.write(str(p), np.zeros(sr, dtype=np.float32), sr)
    res = server.apply_chain(str(p), steps=[{"type": "gain", "gain_db": -1}])
    lst = json.loads(get(viewer_port, "/api/sessions").read())
    assert [s["session_id"] for s in lst] == [res["session_id"]]
    detail = json.loads(
        get(viewer_port, f"/api/sessions/{res['session_id']}").read())
    assert detail["has_processed"] is True
    wav = get(viewer_port, f"/files/{res['session_id']}/processed.wav").read()
    assert wav[:4] == b"RIFF"


def test_files_whitelist(viewer_port, tmp_path):
    sr = 44100
    p = tmp_path / "y.wav"
    sf.write(str(p), np.zeros(sr, dtype=np.float32), sr)
    res = server.apply_chain(str(p), steps=[{"type": "gain", "gain_db": -1}])
    with pytest.raises(urllib.error.HTTPError) as e:
        get(viewer_port, f"/files/{res['session_id']}/session.json")
    assert e.value.code == 403


def test_shutdown_endpoint_stops_server(viewer_port):
    req = urllib.request.Request(
        f"http://127.0.0.1:{viewer_port}/api/shutdown", data=b"{}",
        headers={"Content-Type": "application/json"}, method="POST")
    data = json.loads(urllib.request.urlopen(req, timeout=5).read())
    assert data == {"stopping": True}
