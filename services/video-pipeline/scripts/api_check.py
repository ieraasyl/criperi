"""Check the public API the way the jury will: auth, upload, poll, and validate the JSON schema.

    python scripts/api_check.py <base_url> <api_key_file> <video.mov>
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

try:  # python.org builds of Python have no CA bundle
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass


def call(method, url, key=None, body=None, ctype=None):
    req = urllib.request.Request(url, data=body, method=method)
    if key:
        req.add_header("X-API-Key", key)
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def multipart(path):
    b = uuid.uuid4().hex
    data = open(path, "rb").read()
    name = path.rsplit("/", 1)[-1]
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: video/quicktime\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def validate(res):
    errs = []
    if res.get("status") != "done":
        return [f"status is {res.get('status')}"]
    if res.get("verdict") not in ("violation", "no_violation"):
        errs.append("verdict must be violation|no_violation")
    v = res.get("violations")
    if not isinstance(v, list):
        return errs + ["violations must be a list"]
    if res.get("verdict") == "no_violation" and v:
        errs.append("no_violation must have an empty violations list")
    if res.get("verdict") == "violation" and not v:
        errs.append("violation must list at least one violation")
    for i, x in enumerate(v):
        for k, t in (("type", str), ("timecode_sec", (int, float)), ("description", str), ("evidence_frames", list)):
            if not isinstance(x.get(k), t):
                errs.append(f"violations[{i}].{k} missing or wrong type")
        ef = x.get("evidence_frames", [])
        if not 1 <= len(ef) <= 3:
            errs.append(f"violations[{i}] must have 1-3 evidence frames, has {len(ef)}")
        for j, f in enumerate(ef):
            try:
                raw = base64.b64decode(f["image_base64"])
                assert raw[:2] == b"\xff\xd8", "not a JPEG"
                float(f["timestamp_sec"])
            except Exception as e:
                errs.append(f"violations[{i}].evidence_frames[{j}]: {e}")
    return errs


if __name__ == "__main__":
    base, keyfile, video = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
    key = open(keyfile).read().strip()
    print("auth without key ->", call("POST", f"{base}/api/v1/analyze", None, *multipart(video))[0], "(expect 401)")
    print("auth with bad key ->", call("POST", f"{base}/api/v1/analyze", "wrong", *multipart(video))[0], "(expect 401)")
    t0 = time.time()
    code, r = call("POST", f"{base}/api/v1/analyze", key, *multipart(video))
    print("POST /analyze ->", code, r)
    jid = r["job_id"]
    while True:
        code, res = call("GET", f"{base}/api/v1/jobs/{jid}", key)
        if res.get("status") != "processing" or time.time() - t0 > 480:
            break
        time.sleep(5)
    print(f"final status: {res.get('status')} after {time.time() - t0:.0f}s (limit 300 s)")
    errs = validate(res)
    print("schema check:", "OK" if not errs else errs)
    short = json.loads(json.dumps(res))
    for x in short.get("violations", []):
        for f in x["evidence_frames"]:
            f["image_base64"] = f["image_base64"][:24] + f"...({len(f['image_base64'])} chars)"
    print(json.dumps(short, ensure_ascii=False, indent=1))
