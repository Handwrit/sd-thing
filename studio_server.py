#!/usr/bin/env python3
"""Local studio HTTP server: HTML, output listing, and cheap JPEG thumbs."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required. pip install pillow")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
THUMB_DIR = ROOT / "output_thumbs"
SETTINGS_DIR = ROOT / "studio_settings"
INDEX_HTML = ROOT / "index.html"
MAX_SETTINGS_BODY = 512_000
THUMB_SIZE = (160, 96)
SAFE_PNG = re.compile(r"^[A-Za-z0-9._-]+\.png$", re.IGNORECASE)
SAFE_JPG = re.compile(r"^[A-Za-z0-9._-]+\.jpg$", re.IGNORECASE)
SAFE_MEDIA = re.compile(r"^[A-Za-z0-9._-]+\.(png|webm|avi)$", re.IGNORECASE)

_queue: list[Path] = []
_queued: set[str] = set()
_qlock = threading.Lock()
_wake = threading.Event()
_stop = threading.Event()


def png_name_ok(name: str) -> bool:
    if len(name) < 5 or name[0] == "." or ".." in name:
        return False
    return bool(SAFE_PNG.fullmatch(name))


def media_name_ok(name: str) -> bool:
    if len(name) < 5 or name[0] == "." or ".." in name:
        return False
    return bool(SAFE_MEDIA.fullmatch(name))


def is_video_name(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".webm") or lower.endswith(".avi")


def media_type(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".webm"):
        return "video/webm"
    if lower.endswith(".avi"):
        return "video/x-msvideo"
    return "image/png"


def jpg_name_ok(name: str) -> bool:
    if len(name) < 5 or name[0] == "." or ".." in name:
        return False
    return bool(SAFE_JPG.fullmatch(name))


def thumb_path_for(png_name: str) -> Path:
    return THUMB_DIR / (Path(png_name).stem + ".jpg")


def thumb_stale(png: Path, jpg: Path) -> bool:
    if not jpg.is_file():
        return True
    try:
        return png.stat().st_mtime > jpg.stat().st_mtime
    except OSError:
        return True


def enqueue_thumb(png: Path) -> None:
    stem = png.stem
    with _qlock:
        if stem in _queued:
            return
        _queued.add(stem)
        _queue.append(png)
    _wake.set()


def scan_outputs() -> list[dict]:
    entries: list[tuple[float, str]] = []
    if OUTPUT_DIR.is_dir():
        for path in OUTPUT_DIR.iterdir():
            if not path.is_file() or not media_name_ok(path.name):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if not is_video_name(path.name):
                jpg = thumb_path_for(path.name)
                if thumb_stale(path, jpg):
                    enqueue_thumb(path)
            entries.append((mtime, path.name))
    entries.sort(key=lambda item: (-item[0], item[1]))
    files = []
    for _, name in entries:
        if is_video_name(name):
            files.append({
                "name": name,
                "thumb": "",
                "ready": False,
                "video": True,
            })
            continue
        jpg = thumb_path_for(name)
        files.append({
            "name": name,
            "thumb": "/thumbs/" + jpg.name,
            "ready": jpg.is_file() and not thumb_stale(OUTPUT_DIR / name, jpg),
        })
    return files


def write_thumb(png: Path) -> None:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    dest = thumb_path_for(png.name)
    tmp = dest.with_suffix(".jpg.tmp")
    with Image.open(png) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail(THUMB_SIZE, Image.Resampling.BILINEAR)
        rgb.save(tmp, "JPEG", quality=70)
    tmp.replace(dest)


def thumb_worker() -> None:
    while not _stop.is_set():
        _wake.wait(timeout=2.0)
        _wake.clear()
        if _stop.is_set():
            return
        scan_outputs()
        while not _stop.is_set():
            with _qlock:
                if not _queue:
                    break
                png = _queue.pop(0)
                _queued.discard(png.stem)
            if not png.is_file():
                continue
            dest = thumb_path_for(png.name)
            if not thumb_stale(png, dest):
                continue
            try:
                write_thumb(png)
            except Exception as exc:
                print(f"thumb failed {png.name}: {exc}", file=sys.stderr)


def clean_setting_piece(text: str, limit: int) -> str:
    text = str(text).replace("\x00", "")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("/", " ").replace("\\", " ")
    text = text.replace("..", ".")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def setting_name(model: str, i2i: bool, note: str) -> str:
    model_piece = clean_setting_piece(model, 120) or "model"
    note_piece = clean_setting_piece(note, 80)
    if not note_piece:
        return ""
    if i2i:
        return f"[{model_piece}][i2i][{note_piece}]"
    return f"[{model_piece}][{note_piece}]"


def setting_path(name: str) -> Path | None:
    if not name or name.startswith(".") or "/" in name or "\\" in name or ".." in name:
        return None
    if len(name) > 240:
        return None
    return SETTINGS_DIR / (name + ".json")


def json_setting_value(value, depth: int = 0) -> bool:
    if depth > 6:
        return False
    if isinstance(value, dict):
        if len(value) > 400:
            return False
        return all(isinstance(key, str) and len(key) < 200 and json_setting_value(item, depth + 1) for key, item in value.items())
    if isinstance(value, list):
        return len(value) < 400 and all(json_setting_value(item, depth + 1) for item in value)
    if isinstance(value, str):
        return len(value) <= 20000
    return isinstance(value, (int, float, bool)) or value is None


def list_settings() -> list[dict]:
    entries: list[tuple[float, str]] = []
    if SETTINGS_DIR.is_dir():
        for path in SETTINGS_DIR.iterdir():
            if not path.is_file() or path.suffix != ".json" or path.name.startswith("."):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                name = payload.get("name")
                if not isinstance(name, str) or setting_path(name) != path:
                    name = path.stem
                mtime = path.stat().st_mtime
            except (OSError, json.JSONDecodeError, UnicodeError):
                continue
            entries.append((mtime, name))
    entries.sort(key=lambda item: (-item[0], item[1]))
    return [{"name": name} for _, name in entries]


def add_cors(handler: BaseHTTPRequestHandler) -> None:
    origin = handler.headers.get("Origin") or "*"
    handler.send_header("Access-Control-Allow-Origin", origin)
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "*")
    handler.send_header("Access-Control-Expose-Headers", "*")


class StudioHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        add_cors(self)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        guessed, _ = mimetypes.guess_type(path.name)
        self._send(200, data, guessed or content_type)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        add_cors(self)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/", "/index.html"):
            if not INDEX_HTML.is_file():
                self._send(404, b"index.html missing", "text/plain")
                return
            self._send_file(INDEX_HTML, "text/html; charset=utf-8")
            return

        if path == "/outputs":
            files = scan_outputs()
            body = json.dumps({"files": files}).encode("utf-8")
            self._send(200, body, "application/json")
            return

        if path == "/settings":
            body = json.dumps({"settings": list_settings()}).encode("utf-8")
            self._send(200, body, "application/json")
            return

        if path.startswith("/settings/"):
            name = unquote(path[len("/settings/"):])
            dest = setting_path(name)
            if dest is None or not dest.is_file():
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            self._send_file(dest, "application/json")
            return

        if path.startswith("/thumbs/"):
            name = path[len("/thumbs/"):]
            if "/" in name or not jpg_name_ok(name):
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            jpg = THUMB_DIR / name
            png = OUTPUT_DIR / (Path(name).stem + ".png")
            if png.is_file() and thumb_stale(png, jpg):
                enqueue_thumb(png)
            if not jpg.is_file():
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            self._send_file(jpg, "image/jpeg")
            return

        if path.startswith("/files/"):
            name = path[len("/files/"):]
            if "/" in name or not media_name_ok(name):
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            media = OUTPUT_DIR / name
            if not media.is_file():
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            content_type = media_type(name)
            if content_type.startswith("video/"):
                self._send(200, media.read_bytes(), content_type)
            else:
                self._send_file(media, content_type)
            return

        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if unquote(parsed.path) != "/settings":
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            length = -1
        if length <= 0 or length > MAX_SETTINGS_BODY:
            self._send(400, b'{"error":"bad body"}', "application/json")
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return
        if not isinstance(payload, dict):
            self._send(400, b'{"error":"invalid json"}', "application/json")
            return
        note = payload.get("note")
        model = payload.get("model")
        form = payload.get("form")
        designer = payload.get("designer", {})
        i2i = payload.get("i2i") is True
        if not isinstance(note, str) or not isinstance(model, str) or not isinstance(form, dict):
            self._send(400, b'{"error":"missing fields"}', "application/json")
            return
        if designer is None:
            designer = {}
        if not isinstance(designer, dict) or not json_setting_value(form) or not json_setting_value(designer):
            self._send(400, b'{"error":"settings too large"}', "application/json")
            return
        form.pop("seed", None)
        name = setting_name(model, i2i, note)
        dest = setting_path(name)
        if dest is None:
            self._send(400, b'{"error":"type a name"}', "application/json")
            return
        stored = {
            "name": name,
            "model": clean_setting_piece(model, 120) or "model",
            "i2i": i2i,
            "note": clean_setting_piece(note, 80),
            "form": form,
            "designer": designer,
        }
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(dest)
        body = json.dumps({"name": name}).encode("utf-8")
        self._send(200, body, "application/json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve index.html and JPEG thumbs for output/")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    args = parser.parse_args()

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    worker = threading.Thread(target=thumb_worker, name="thumbs", daemon=True)
    worker.start()
    scan_outputs()
    _wake.set()

    httpd = ThreadingHTTPServer((args.bind, args.port), StudioHandler)
    print(f"studio listening on http://{args.bind}:{args.port}", flush=True)
    print(f"outputs {OUTPUT_DIR}", flush=True)
    print(f"thumbs  {THUMB_DIR}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        _stop.set()
        _wake.set()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
