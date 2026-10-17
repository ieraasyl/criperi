"""Live view of the extract stage: each record streamed from Modal is drawn on the local copy of the video
as soon as it arrives. Nothing but OpenCV and numpy runs here; the models run on the Modal GPU.
"""

import json
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PANEL_W = 560
BG = (26, 20, 18)  # BGR
FG, DIM = (238, 232, 230), (150, 136, 130)
GREEN, RED, AMBER, BLUE, CYAN, MAGENTA, GRAY = (
    (120, 220, 80), (90, 90, 255), (60, 190, 255), (255, 160, 110), (255, 230, 0), (255, 80, 230), (110, 96, 90),
)
CLS_COL = {"cash": RED, "cash_bag": RED, "id_card": CYAN, "personal": AMBER, "food_drink": AMBER,
           "phone": MAGENTA, "paper": GRAY}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, xy, scale=0.5, col=FG, th=1):
    cv2.putText(img, s, (int(xy[0]), int(xy[1])), FONT, scale, col, th, cv2.LINE_AA)


def _rect(img, b, col, th=1):
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)


class Viewer:
    def __init__(self, video_path: Path, window: bool, out_mp4: Path | None, scale: float = 1.25):
        self.cap = cv2.VideoCapture(str(video_path))
        self.idx, self.frame = -1, None
        self.name = video_path.name
        self.window = window
        self.scale = scale
        self.title = f"Extract on Modal - {self.name}"
        self.out_mp4 = out_mp4
        self.ff = None
        self.meta: dict = {}
        self.layouts: list[dict] = []
        self.stages: dict[str, float] = {}
        self.ms_hist: dict[str, deque] = {}
        self.t_start = time.time()
        self.first_frame_wall = None
        self.status = "uploading the video to Modal"
        self.seen_objects: dict[int, str] = {}

    # ---------------------------------------------------------------- source frames
    def _seek(self, i: int):
        while self.idx < i:
            if not self.cap.grab():
                return
            self.idx += 1
        ok, bgr = self.cap.retrieve()
        if ok:
            self.frame = bgr

    def _first(self):
        if self.frame is None:
            self._seek(0)
        return self.frame

    # ---------------------------------------------------------------- drawing
    def _layout_at(self, t: float) -> dict | None:
        cur = None
        for lay in self.layouts:
            if t + 1e-6 >= lay["start"]:
                cur = lay
        return cur

    def _video(self, rec: dict | None):
        img = self._first().copy()
        if rec is None:
            return img
        f = rec["facts"]
        lay = self._layout_at(rec["t"])
        if lay:
            if lay.get("workstation"):
                _rect(img, lay["workstation"], (0, 110, 0))
            if lay.get("safe"):
                _rect(img, lay["safe"], RED if f["safe"] == "open" else (160, 120, 120), 2 if f["safe"] == "open" else 1)
                _text(img, f"safe {f['safe']} {f['safe_prob']:.2f}", (lay["safe"][0], lay["safe"][1] - 4), 0.4, FG)
            if lay.get("laptop"):
                col = AMBER if f["screen"] == "work_screen" else GREEN
                _rect(img, lay["laptop"], col)
                _text(img, f"laptop {f['screen']}", (lay["laptop"][0], lay["laptop"][1] - 4), 0.4, col)
            if lay.get("counter"):
                _rect(img, lay["counter"], (180, 180, 180))
            for d in lay.get("doors", [])[:2]:
                _rect(img, d, (200, 200, 90))
        objs = {o["track"]: o for o in rec["objects"]}
        for o in rec["objects"]:
            col = CLS_COL.get(o["cls"], GRAY)
            _rect(img, o["box"], col, 2 if o["held_by"] else 1)
            _text(img, f"{o['cls']}#{o['track']} {o['score']:.2f}", (o["box"][0], o["box"][1] - 3), 0.38, col)
        for p in rec["people"]:
            col = GREEN if p["role"] == "cashier" else BLUE
            _rect(img, p["box"], col, 2)
            _text(img, f"#{p['track']} {p['role']} | {p['zone']}", (p["box"][0], max(12, p["box"][1] - 5)), 0.42, col)
            for side, pt in p["hands"].items():
                if pt is None:
                    continue
                c = CYAN if side == "left" else MAGENTA
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, c, -1)
                held = p["holding"].get(side)
                if held is not None and held in objs:
                    ob = objs[held]["box"]
                    cv2.line(img, (int(pt[0]), int(pt[1])), (int((ob[0] + ob[2]) / 2), int((ob[1] + ob[3]) / 2)), c, 2)
        return img

    def _panel(self, H: int, ev: dict | None):
        pan = np.full((H, PANEL_W, 3), BG, np.uint8)
        x, y = 16, 30
        _text(pan, "Stage 1  Extract  (Modal, L40S GPU)", (x, y), 0.65, FG, 2)
        y += 24
        _text(pan, self.name, (x, y), 0.5, DIM)
        y += 26
        models = self.meta.get("models", {})
        for k in ("person", "pose", "objects", "classes"):
            if k in models:
                _text(pan, f"{k:8s} {models[k]}", (x, y), 0.42, DIM)
                y += 17
        y += 8
        for st in ("decode", "layout"):
            done = st in self.stages
            cv2.circle(pan, (x + 6, y - 5), 6, GREEN if done else GRAY, -1)
            _text(pan, f"{st}" + (f"   {self.stages[st]:.1f} s" if done else ""), (x + 20, y), 0.5, FG)
            y += 22
        if ev is None:
            _text(pan, self.status, (x, y + 10), 0.55, AMBER)
            return pan

        k, n, rec = ev["k"], ev["n"], ev["rec"]
        wall = time.time() - (self.first_frame_wall or time.time())
        speed = rec["t"] / wall if wall > 0.5 else 0.0
        cv2.circle(pan, (x + 6, y - 5), 6, AMBER, -1)
        _text(pan, f"frames  {k + 1}/{n}   t = {rec['t']:5.1f} s", (x + 20, y), 0.5, FG)
        y += 20
        _text(pan, f"speed   {speed:4.2f} x real time   wall {ev['wall']:.0f} s", (x + 20, y), 0.45, DIM)
        y += 8
        cv2.rectangle(pan, (x, y), (PANEL_W - x, y + 8), (60, 50, 46), -1)
        cv2.rectangle(pan, (x, y), (x + int((PANEL_W - 2 * x) * (k + 1) / n), y + 8), GREEN, -1)
        y += 30

        _text(pan, "model time per frame (ms, avg of 10)", (x, y), 0.5, FG)
        y += 20
        for key, v in ev["ms"].items():
            self.ms_hist.setdefault(key, deque(maxlen=10)).append(v)
        for key in ("people", "pose", "detect", "classify", "obj_track", "states", "total"):
            if key in self.ms_hist:
                avg = float(np.mean(self.ms_hist[key]))
                _text(pan, f"{key:10s}", (x, y), 0.45, DIM)
                bar = int(min(avg / 400.0, 1.0) * 250)
                cv2.rectangle(pan, (x + 110, y - 10), (x + 110 + bar, y - 2), BLUE if key != "total" else AMBER, -1)
                _text(pan, f"{avg:6.1f}", (x + 370, y), 0.45, FG)
                y += 18
        y += 12

        f = rec["facts"]
        _text(pan, "facts at this frame (provisional roles)", (x, y), 0.5, FG)
        y += 22
        held = ", ".join(f"#{a} {b}" for a, b in f["cash_in_hand"]) or "-"
        rows = [
            ("cashier", f["cashier_zone"], GREEN if f["cashier_at_workplace"] else AMBER),
            ("people in room", str(f["people_in_room"]), FG),
            ("visitors at desk", str(f["visitors_at_desk"]), FG),
            ("cash on desk", "YES" if f["cash_on_desk"] else "no", RED if f["cash_on_desk"] else FG),
            ("cash in hand", held, AMBER if f["cash_in_hand"] else FG),
            ("id card seen", "yes" if f["id_card_seen"] else "no", CYAN if f["id_card_seen"] else FG),
            ("items on desk", ", ".join(f["items_on_desk"]) or "-", AMBER if f["items_on_desk"] else FG),
            ("safe", f"{f['safe']} ({f['safe_prob']:.2f})", RED if f["safe"] == "open" else FG),
            ("laptop screen", f["screen"], AMBER if f["screen"] == "work_screen" else FG),
        ]
        for a, b, c in rows:
            _text(pan, a, (x, y), 0.45, DIM)
            _text(pan, b, (x + 170, y), 0.45, c)
            y += 18
        y += 12
        _text(pan, "hands", (x, y), 0.5, FG)
        y += 20
        for p in rec["people"]:
            hold = {s: p["holding"][s] for s in ("left", "right")}
            desc = "  ".join(f"{s[0].upper()}:{'-' if v is None else '#' + str(v)}" for s, v in hold.items())
            _text(pan, f"#{p['track']} {p['role']:8s} {desc}", (x, y), 0.45, GREEN if p["role"] == "cashier" else BLUE)
            y += 18
        for o in rec["objects"]:
            self.seen_objects[o["track"]] = o["cls"]
        y += 10
        _text(pan, f"object tracks so far: {len(self.seen_objects)}", (x, y), 0.5, FG)
        y += 20
        line = ", ".join(f"#{t} {c}" for t, c in sorted(self.seen_objects.items())[-12:])
        for chunk in [line[i : i + 62] for i in range(0, len(line), 62)][:3]:
            _text(pan, chunk, (x, y), 0.42, DIM)
            y += 17
        return pan

    def show(self, ev: dict | None):
        rec = ev["rec"] if ev else None
        if rec is not None:
            self._seek(rec["i"])
        vid = self._video(rec)
        if self.scale != 1.0:
            vid = cv2.resize(vid, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)
        H = vid.shape[0] + vid.shape[0] % 2
        if H != vid.shape[0]:
            vid = cv2.copyMakeBorder(vid, 0, 1, 0, 0, cv2.BORDER_CONSTANT)
        img = np.hstack([vid, self._panel(H, ev)])
        if self.window:
            cv2.imshow(self.title, img)
            cv2.waitKey(1)
        if self.out_mp4 is not None and ev is not None:
            self._write(img)
        return img

    def _write(self, img):
        if self.ff is None:
            h, w = img.shape[:2]
            fps = self.meta.get("sample_fps", 4)
            self.ff = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                 "-movflags", "+faststart", str(self.out_mp4)],
                stdin=subprocess.PIPE,
            )
        self.ff.stdin.write(img.tobytes())

    def close(self, hold_sec: float = 4.0):
        if self.ff is not None:
            self.ff.stdin.close()
            self.ff.wait()
        if self.window:
            end = time.time() + hold_sec
            while time.time() < end:
                if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
                    break
            cv2.destroyAllWindows()
        self.cap.release()


def watch(video_path: Path, events: Iterable[dict], out_dir: Path, window: bool = True, log=print) -> dict | None:
    """Consume the event stream, draw every frame record live, save the final timeline JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    v = Viewer(video_path, window, out_dir / f"{video_path.stem}_extract_live.mp4")
    v.show(None)
    final = None
    last_log = 0.0
    for ev in events:
        typ = ev["type"]
        if typ == "stage":
            v.stages[ev["stage"]] = ev["sec"]
            if "meta" in ev:
                v.meta = ev["meta"]
                v.status = "detecting the scene layout"
                log(f"  decode  {ev['sec']:5.1f} s  {ev['meta']['frames']} frames at {ev['meta']['sample_fps']} fps; models {ev['meta'].get('models')}")
            if "layouts" in ev:
                v.layouts = ev["layouts"]
                v.status = "running the models frame by frame"
                log(f"  layout  {ev['sec']:5.1f} s  {len(ev['layouts'])} camera segment(s)")
            v.show(None)
        elif typ == "frame":
            if v.first_frame_wall is None:
                v.first_frame_wall = time.time()
            v.show(ev)
            if time.time() - last_log > 2.0 or ev["k"] + 1 == ev["n"]:
                last_log = time.time()
                f = ev["rec"]["facts"]
                log(f"  frame {ev['k'] + 1:4d}/{ev['n']}  t={ev['rec']['t']:5.1f}s  {ev['ms']['total']:6.1f} ms  "
                    f"people={f['people_in_room']} cashier={f['cashier_zone']} cash_on_desk={f['cash_on_desk']} items={f['items_on_desk']}")
        elif typ == "final":
            final = ev["result"]
            log(f"  final   {final['timings']}  saved on Modal: {ev.get('saved_to')}")
    v.close()
    if final is not None:
        p = out_dir / f"{video_path.stem}_extract.json"
        p.write_text(json.dumps(final, ensure_ascii=False), "utf8")
        log(f"timeline: {p}")
        log(f"live recording: {v.out_mp4}")
    return final
