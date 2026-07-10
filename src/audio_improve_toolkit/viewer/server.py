"""Viewer-webserver: stdlib http.server, alleen op 127.0.0.1.

Endpoints:
  GET /                      -> static/index.html
  GET /static/<naam>         -> app-bestanden
  GET /health                -> {"status": "ok"}
  GET /api/sessions          -> sessieoverzicht
  GET /api/sessions/<id>     -> volledige sessiedata incl. waveforms
  GET /files/<id>/<naam>     -> audio/afbeeldingen uit de sessiemap (whitelist)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from audio_improve_toolkit import sessions

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
                ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml"}
ALLOWED_SESSION_FILES = {
    "original.wav": "audio/wav",
    "processed.wav": "audio/wav",
    "spectrogram_original.png": "image/png",
    "spectrogram_processed.png": "image/png",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # geen request-spam
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def do_GET(self):  # noqa: N802
        try:
            self._route(self.path.split("?", 1)[0])
        except BrokenPipeError:
            pass
        except Exception as exc:
            log.exception("viewer-fout voor %s", self.path)
            try:
                self._error(500, str(exc))
            except Exception:
                pass

    def _route(self, path: str) -> None:
        if path == "/" or path == "/index.html":
            self._static("index.html")
        elif path == "/health":
            self._json({"status": "ok"})
        elif path == "/api/sessions":
            self._json(sessions.list_sessions())
        elif path.startswith("/api/sessions/"):
            sid = path.rsplit("/", 1)[1]
            try:
                data = sessions.load_session(sid)
            except FileNotFoundError as exc:
                return self._error(404, str(exc))
            for name in ("waveform_original", "waveform_processed"):
                f = sessions.session_path(sid) / f"{name}.json"
                if f.exists():
                    data[name] = json.loads(f.read_text())
            self._json(data)
        elif path.startswith("/static/"):
            self._static(path[len("/static/"):])
        elif path.startswith("/files/"):
            parts = path.split("/")
            if len(parts) != 4:
                return self._error(404, "onbekend pad")
            _, _, sid, name = parts
            ctype = ALLOWED_SESSION_FILES.get(name)
            if ctype is None or "/" in sid or ".." in sid:
                return self._error(403, "niet toegestaan")
            f = sessions.session_path(sid) / name
            if not f.is_file():
                return self._error(404, f"{name} niet gevonden in sessie {sid}")
            self._send(200, f.read_bytes(), ctype)
        else:
            self._error(404, "onbekend pad")

    def _static(self, name: str) -> None:
        f = (STATIC / name).resolve()
        if not str(f).startswith(str(STATIC.resolve())) or not f.is_file():
            return self._error(404, "bestand niet gevonden")
        self._send(200, f.read_bytes(), STATIC_TYPES.get(f.suffix, "application/octet-stream"))


def main(port: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s")
    port = port or int(os.environ.get("AIT_VIEWER_PORT", "8471"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    log.info("viewer draait op http://127.0.0.1:%d/ (sessies: %s)",
             port, sessions.sessions_dir())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
