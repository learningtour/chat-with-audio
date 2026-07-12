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

import glob
import json
import logging
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chat_with_audio import sessions

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
                ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml"}
# Editors voor de "Open in ..."-knop; alleen deze whitelist kan gestart worden.
EDITORS = {
    "audition": ("Adobe Audition", ["/Applications/Adobe Audition*/Adobe Audition*.app",
                                     "/Applications/Adobe Audition*.app"]),
    "protools": ("Pro Tools", ["/Applications/Pro Tools.app"]),
    "audacity": ("Audacity", ["/Applications/Audacity.app"]),
    "logic": ("Logic Pro", ["/Applications/Logic Pro*.app"]),
    "reaper": ("REAPER", ["/Applications/REAPER*.app"]),
}


def _find_app(patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = sorted(glob.glob(pat), reverse=True)
        if hits:
            return hits[0]
    return None


def available_editors() -> list[dict]:
    return [{"key": k, "name": name, "installed": _find_app(pats) is not None}
            for k, (name, pats) in EDITORS.items()]


ALLOWED_SESSION_FILES = {
    "original.wav": "audio/wav",
    "processed.wav": "audio/wav",
    "residual.wav": "audio/wav",
    "log.md": "text/plain; charset=utf-8",
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
            from chat_with_audio import __version__

            self._json({"status": "ok", "version": __version__})
        elif path == "/api/editors":
            self._json(available_editors())
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
            data["has_log"] = (sessions.session_path(sid) / "log.md").is_file()
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

    def do_POST(self):  # noqa: N802
        try:
            # CSRF-bescherming: alleen requests vanaf de viewer zelf
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if origin and origin.split("//")[-1] != host:
                return self._error(403, "verboden origin")
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/open-in-editor":
                self._open_in_editor(body)
            elif self.path == "/api/shutdown":
                # nette zelf-herstart na een upgrade: de MCP-server vraagt een
                # verouderde viewer te stoppen en start dan een verse
                import threading

                self._json({"stopping": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._error(404, "onbekend pad")
        except BrokenPipeError:
            pass
        except Exception as exc:
            log.exception("viewer-fout voor %s", self.path)
            try:
                self._error(500, str(exc))
            except Exception:
                pass

    def _open_in_editor(self, body: dict) -> None:
        key = str(body.get("editor", "audition"))
        sid = str(body.get("session_id", ""))
        if "/" in sid or ".." in sid or not sid:
            return self._error(403, "ongeldige sessie")
        info = EDITORS.get(key)
        if info is None:
            return self._error(400, f"onbekende editor '{key}'")
        app = _find_app(info[1])
        if app is None:
            return self._error(404, f"{info[0]} niet gevonden in /Applications")
        d = sessions.session_path(sid)
        f = d / "processed.wav"
        if not f.is_file():
            f = d / "original.wav"
        if not f.is_file():
            return self._error(404, "geen audio in deze sessie")
        if body.get("dry_run"):
            return self._json({"dry_run": True, "app": app, "file": str(f)})
        if sys.platform == "win32":
            os.startfile(str(f))  # noqa: S606 - Windows kent geen 'open -a'
        else:
            subprocess.run(["open", "-a", app, str(f)], capture_output=True)
        self._json({"opened": True, "app": info[0], "file": f.name})

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
