import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from kassa_pipeline.facts import FactTimeline
from kassa_pipeline.review_server import make_server


@pytest.fixture
def server(tmp_path):
    facts_dir = tmp_path / "facts"
    videos_dir = tmp_path / "videos"
    facts_dir.mkdir()
    videos_dir.mkdir()
    for n, status in ((10, "reviewed"), (2, "draft")):
        t = FactTimeline.empty(f"X-{n}", 4)
        t.status = status
        t.save(facts_dir / f"X-{n}.json")
    (facts_dir / "broken.json").write_text("{not json", encoding="utf-8")
    (videos_dir / "Х-2 (2).MOV").write_bytes(bytes(range(256)) * 4)  # 1024 bytes, Cyrillic Х
    srv = make_server(facts_dir, videos_dir, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    srv.base = f"http://127.0.0.1:{srv.server_address[1]}"
    srv.facts_dir = facts_dir
    yield srv
    srv.shutdown()
    srv.server_close()


def call(url, data=None, headers=None):
    headers = dict(headers or {})
    if data is not None:
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_list_clips_sorted_with_status(server):
    status, _, body = call(server.base + "/api/clips")
    assert status == 200
    assert json.loads(body) == [
        {"clip": "X-2", "status": "draft", "duration": 4},
        {"clip": "X-10", "status": "reviewed", "duration": 4},
        {"clip": "broken", "status": "invalid", "duration": None},
    ]


def test_get_clip_returns_json(server):
    status, _, body = call(server.base + "/api/clips/X-2")
    assert status == 200
    assert json.loads(body) == FactTimeline.load(server.facts_dir / "X-2.json").to_dict()
    assert call(server.base + "/api/clips/X-99")[0] == 404


def test_post_valid_saves_file(server):
    d = json.loads(call(server.base + "/api/clips/X-2")[2])
    d["states"]["screen"][0] = "work"
    d["status"] = "reviewed"
    status, _, body = call(server.base + "/api/clips/X-2", json.dumps(d).encode())
    assert status == 200 and json.loads(body) == {"ok": True}
    saved = FactTimeline.load(server.facts_dir / "X-2.json")
    assert saved.states["screen"][0] == "work" and saved.status == "reviewed"


def test_post_invalid_keeps_file(server):
    path = server.facts_dir / "X-2.json"
    before = path.read_bytes()
    d = json.loads(call(server.base + "/api/clips/X-2")[2])
    d["states"]["screen"] = d["states"]["screen"][:-1]
    status, _, body = call(server.base + "/api/clips/X-2", json.dumps(d).encode())
    assert status == 400 and "screen" in json.loads(body)["error"]
    assert path.read_bytes() == before
    assert call(server.base + "/api/clips/X-2", b"{nope")[0] == 400
    d["events"] = [{"t": 1, "type": "room_exit", "t_end": "abc"}]
    assert call(server.base + "/api/clips/X-2", json.dumps(d).encode())[0] == 400
    assert path.read_bytes() == before


def test_post_clip_mismatch_rejected(server):
    path = server.facts_dir / "X-2.json"
    before = path.read_bytes()
    d = json.loads(call(server.base + "/api/clips/X-2")[2])
    d["clip"] = "X-10"
    status, _, body = call(server.base + "/api/clips/X-2", json.dumps(d).encode())
    assert status == 400 and "error" in json.loads(body)
    assert path.read_bytes() == before


def test_post_too_large_is_413(server):
    status, _, _ = call(server.base + "/api/clips/X-2", b" " * (5 * 1024 * 1024 + 1))
    assert status == 413


def test_video_range_request_returns_206(server):
    status, h, body = call(server.base + "/video/X-2", headers={"Range": "bytes=10-19"})
    assert status == 206
    assert h["Content-Range"] == "bytes 10-19/1024"
    assert h["Content-Type"] == "video/quicktime"
    assert body == bytes(range(10, 20))
    status, h, body = call(server.base + "/video/X-2", headers={"Range": "bytes=1000-"})
    assert status == 206 and len(body) == 24
    status, h, body = call(server.base + "/video/X-2")
    assert status == 200 and len(body) == 1024 and h["Accept-Ranges"] == "bytes"
    assert call(server.base + "/video/X-2", headers={"Range": "bytes=5000-"})[0] == 416
    assert call(server.base + "/video/X-10")[0] == 404


def test_path_traversal_is_404(server):
    assert call(server.base + "/api/clips/..%2Fetc")[0] == 404
    assert call(server.base + "/video/..%2F..%2Fetc%2Fpasswd")[0] == 404
    assert call(server.base + "/nope")[0] == 404


def test_index_serves_html(server):
    status, h, body = call(server.base + "/")
    assert status == 200 and h["Content-Type"].startswith("text/html")
    assert b"<video" in body
    assert "batch (optional: blank = latest received batch)".encode() in body


def raw(server, method, path, headers=None, body=None):
    port = server.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Host": f"127.0.0.1:{port}"}
    h.update(headers or {})
    conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for k, v in h.items():
        conn.putheader(k, v)
    conn.endheaders(body)
    r = conn.getresponse()
    out = r.status, r.read()
    conn.close()
    return out


def test_post_short_body_with_huge_length_does_not_hang(server):
    before = threading.active_count()
    sock = socket.create_connection(("127.0.0.1", server.server_address[1]))
    sock.sendall(
        b"POST /api/clips/X-2 HTTP/1.0\r\nHost: 127.0.0.1:%d\r\nContent-Type: application/json\r\n"
        b"Content-Length: 99999999\r\n\r\nabc" % server.server_address[1]
    )
    sock.close()
    deadline = time.time() + 2
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= before, "handler thread still running"


def test_post_negative_length_is_400(server):
    status, _ = raw(server, "POST", "/api/clips/X-2", {"Content-Type": "application/json", "Content-Length": "-5"})
    assert status == 400


def test_post_requires_json_content_type(server):
    body = call(server.base + "/api/clips/X-2")[2]
    for ctype in ("text/plain", "application/x-www-form-urlencoded", None):
        h = {"Content-Length": str(len(body))}
        if ctype:
            h["Content-Type"] = ctype
        assert raw(server, "POST", "/api/clips/X-2", h, body)[0] == 415
    h = {"Content-Length": str(len(body)), "Content-Type": "application/json; charset=utf-8"}
    assert raw(server, "POST", "/api/clips/X-2", h, body)[0] == 200


def test_origin_must_be_local(server):
    port = server.server_address[1]
    body = call(server.base + "/api/clips/X-2")[2]
    base = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    assert raw(server, "POST", "/api/clips/X-2", {**base, "Origin": "http://evil.example"}, body)[0] == 403
    assert raw(server, "POST", "/api/clips/X-2", {**base, "Origin": f"http://127.0.0.1:{port + 1}"}, body)[0] == 403
    for origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
        assert raw(server, "POST", "/api/clips/X-2", {**base, "Origin": origin}, body)[0] == 200


def test_host_must_be_local(server):
    port = server.server_address[1]
    for path in ("/", "/api/clips", "/api/clips/X-2", "/video/X-2"):
        assert raw(server, "GET", path, {"Host": "evil.example"})[0] == 403
        assert raw(server, "GET", path, {"Host": f"evil.example:{port}"})[0] == 403
        assert raw(server, "GET", path, {"Host": f"localhost:{port}"})[0] == 200
    body = call(server.base + "/api/clips/X-2")[2]
    h = {"Host": "evil.example", "Content-Length": str(len(body)), "Content-Type": "application/json"}
    assert raw(server, "POST", "/api/clips/X-2", h, body)[0] == 403


def test_make_server_requires_existing_dirs(tmp_path):
    with pytest.raises(ValueError, match="nofacts"):
        make_server(tmp_path / "nofacts", tmp_path, port=0)
    with pytest.raises(ValueError, match="novideos"):
        make_server(tmp_path, tmp_path / "novideos", port=0)
