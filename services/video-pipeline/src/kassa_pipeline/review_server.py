"""Local review server: serves the fact-correction page, timelines as JSON and clip videos."""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from kassa_pipeline.facts import FactTimeline, clip_id

MAX_BODY = 5 * 1024 * 1024
CHUNK = 64 * 1024
PAGE = Path(__file__).parent / "data" / "review.html"
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def _sort_key(clip: str) -> tuple:
    m = re.search(r"-(\d+)$", clip)
    return (int(m.group(1)) if m else float("inf"), clip)


class _Handler(BaseHTTPRequestHandler):
    facts_dir: Path
    videos_dir: Path
    timeout = 10  # a stalled client must not pin a handler thread

    def handle(self) -> None:
        try:
            super().handle()
        except (ConnectionError, TimeoutError):
            pass  # client went away or stalled; nothing to answer

    def _local_names(self) -> set[str]:
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _guard(self, post: bool = False) -> bool:
        """Reject DNS-rebinding (Host), cross-site (Origin) and non-JSON POSTs; True when allowed."""
        names = self._local_names()
        if self.headers.get("Host") not in names:
            self._error(403, "forbidden host")
            return False
        if not post:
            return True
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{n}" for n in names}:
            self._error(403, "forbidden origin")
            return False
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            self._error(415, "Content-Type must be application/json")
            return False
        return True

    def log_message(self, format, *args):  # noqa: A002 - silence per-request logging
        pass

    # --- lookups: names come only from existing files, never joined from the URL ---
    def _fact_files(self) -> dict[str, Path]:
        return {p.stem: p for p in self.facts_dir.glob("*.json")}

    def _video_files(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for p in sorted(self.videos_dir.iterdir()):
            if p.suffix.lower() == ".mov":
                found.setdefault(clip_id(p.stem), p)
        return found

    # --- responses ---
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, msg: str) -> None:
        self._json(status, {"error": msg})

    def _route(self) -> list[str]:
        return [unquote(s) for s in urlsplit(self.path).path.split("/")[1:]]

    # --- GET ---
    def do_GET(self) -> None:
        if not self._guard():
            return
        parts = self._route()
        if parts == [""]:
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif parts == ["api", "clips"]:
            self._list_clips()
        elif len(parts) == 3 and parts[:2] == ["api", "clips"]:
            path = self._fact_files().get(parts[2])
            if path is None:
                return self._error(404, "unknown clip")
            try:
                self._json(200, FactTimeline.load(path).to_dict())
            except (OSError, ValueError) as exc:
                self._error(500, f"unreadable timeline: {exc}")
        elif len(parts) == 2 and parts[0] == "video":
            path = self._video_files().get(parts[1])
            if path is None:
                return self._error(404, "unknown video")
            self._send_video(path)
        else:
            self._error(404, "not found")

    def _list_clips(self) -> None:
        items = []
        for clip, path in self._fact_files().items():
            try:
                t = FactTimeline.load(path)
                items.append({"clip": clip, "status": t.status, "duration": t.duration})
            except (OSError, ValueError):
                items.append({"clip": clip, "status": "invalid", "duration": None})
        items.sort(key=lambda i: _sort_key(i["clip"]))
        self._json(200, items)

    def _send_video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        header = self.headers.get("Range")
        m = _RANGE_RE.match(header.strip()) if header else None
        if m:
            first, last = m.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            elif last:  # suffix range: the final N bytes
                start = max(size - int(last), 0)
            else:
                m = None
            if m and (start >= size or start > end):
                return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
            status = 206 if m else 200
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/quicktime")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    chunk = f.read(min(CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the player seeked or closed; normal for video

    # --- POST ---
    def do_POST(self) -> None:
        if not self._guard(post=True):
            return
        parts = self._route()
        if not (len(parts) == 3 and parts[:2] == ["api", "clips"]):
            return self._error(404, "not found")
        path = self._fact_files().get(parts[2])
        if path is None:
            return self._error(404, "unknown clip")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "bad Content-Length")
        if length < 0:
            return self._error(400, "bad Content-Length")
        if length > MAX_BODY:
            left = length  # drain so the client can read the answer; stop at EOF or timeout
            try:
                while left > 0:
                    chunk = self.rfile.read(min(CHUNK, left))
                    if not chunk:
                        break
                    left -= len(chunk)
            except OSError:
                pass
            self.close_connection = True
            return self._error(413, f"body larger than {MAX_BODY} bytes")
        try:
            raw = self.rfile.read(length)
        except OSError:
            self.close_connection = True
            return
        try:
            timeline = FactTimeline.from_dict(json.loads(raw))
            if timeline.clip != parts[2]:
                raise ValueError(f"clip {timeline.clip!r} in body does not match {parts[2]!r} in URL")
        except ValueError as exc:  # includes JSON decode errors
            return self._error(400, str(exc))
        timeline.save(path)
        self._json(200, {"ok": True})


def make_server(facts_dir: Path, videos_dir: Path, port: int = 8765) -> ThreadingHTTPServer:
    """Review server bound to 127.0.0.1 only."""
    facts_dir, videos_dir = Path(facts_dir), Path(videos_dir)
    for d in (facts_dir, videos_dir):
        if not d.is_dir():
            raise ValueError(f"directory does not exist: {d}")
    handler = type("ReviewHandler", (_Handler,), {"facts_dir": facts_dir, "videos_dir": videos_dir})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    return srv
