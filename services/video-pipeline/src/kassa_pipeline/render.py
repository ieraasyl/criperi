"""Renders an annotated replay: the video with what each pipeline stage sees, plus a live panel.

The panel shows the stage timings of the real run, the per-second state, state strips with a moving
cursor, and an event log that fills in as the video reaches each timecode.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kassa_pipeline import events as E
from kassa_pipeline.people import zone_of
from kassa_pipeline.pipeline import AnalysisResult
from kassa_pipeline.report import _FONTS, TYPE_TITLE
from pathlib import Path as _P

PANEL_W = 660
BG = (18, 20, 26)
FG = (230, 232, 238)
DIM = (130, 136, 150)
GREEN, RED, AMBER, BLUE, GRAY = (80, 220, 120), (255, 90, 90), (255, 190, 60), (110, 160, 255), (90, 96, 110)

STAGES = [
    ("decode", "0 Декодирование, 4 к/с"),
    ("layout", "1 Раскладка сцены (стол, сейф, ноутбук)"),
    ("people", "2 Люди: детекция, трекинг, роли, поза"),
    ("objects", "3 Объекты: наличные, вещи, экран, сейф"),
    ("timeline", "4 Состояние по секундам"),
    ("events", "5 Правила событий"),
    ("report", "8-9 Кадры, описание, норма"),
]
ZONE_RU = {
    "cashier_seat": "сидит на рабочем месте",
    "standing_at_seat": "встала у рабочего места",
    "at_desk": "у стола",
    "near_safe": "у сейфа",
    "doorway": "в дверях",
    "elsewhere": "вне стола",
    "absent": "нет в кадре",
}
SCREEN_RU = {"work_screen": "рабочий экран", "locked": "заблокирован", "off": "выключен", "unknown": "не виден"}
SCREEN_COL = {"work_screen": AMBER, "locked": GREEN, "off": GRAY, "unknown": GRAY}


def _font(size: int):
    for p in _FONTS:
        if _P(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F18, F15, F13, F22 = _font(18), _font(15), _font(13), _font(22)


def _draw_video(frame, res: AnalysisResult, t: float, active: list[E.Candidate]):
    img = frame.copy()
    lay = res.layouts.at(t)
    sec = min(int(t), len(res.timeline.seconds) - 1)
    st = res.timeline.seconds[sec]

    def rect(b, col, th=1):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, th)

    if lay.workstation:
        rect(lay.workstation, (0, 110, 0), 1)
    if lay.safe:
        rect(lay.safe, (200, 60, 60), 1)
    if lay.laptop:
        rect(lay.laptop, (255, 140, 60) if st.screen == "work_screen" else (120, 200, 120), 1)
    for o in res.objects.objects.get(sec, []):
        col = (0, 165, 255) if o.in_hand else ((0, 255, 255) if o.on_desk else (160, 160, 160))
        rect(o.box, col, 1)
        cv2.putText(img, o.cat, (int(o.box[0]), int(o.box[1]) - 3), 0, 0.4, col, 1, cv2.LINE_AA)
    for tr in res.people.tracks:
        o = tr.at(t, tol=0.35)
        if o is None:
            continue
        z = zone_of(o.t, o.box, res.layouts, res.people).name
        col = (80, 220, 120) if tr.role == "cashier" else (255, 160, 90)
        rect(o.box, col, 2)
        label = ("cashier" if tr.role == "cashier" else f"visitor {tr.id}") + f" | {z}"
        cv2.putText(img, label, (int(o.box[0]), max(12, int(o.box[1]) - 5)), 0, 0.45, col, 1, cv2.LINE_AA)
        for w in o.wrists:
            cv2.circle(img, (int(w[0]), int(w[1])), 3, (255, 60, 60), -1)
    if active:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 60, 60), 6)
    return img


def _strip(d: ImageDraw.ImageDraw, x0, y0, w, h, values, colors, label, n, t):
    d.text((x0, y0 - 18), label, font=F13, fill=DIM)
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(30, 33, 42))
    for i, v in enumerate(values):
        c = colors(v)
        if c:
            d.rectangle([x0 + w * i / n, y0, x0 + w * (i + 1) / n - 1, y0 + h], fill=c)
    cx = x0 + w * min(t / n, 1.0)
    d.line([cx, y0 - 3, cx, y0 + h + 3], fill=(255, 255, 255), width=2)


def _panel(res: AnalysisResult, t: float, H: int, name: str, shown: list[E.Candidate], finished: bool):
    pil = Image.new("RGB", (PANEL_W, H), BG)
    d = ImageDraw.Draw(pil)
    x = 18
    d.text((x, 12), "Конвейер анализа видео", font=F22, fill=FG)
    d.text((x, 40), f"{name}   t = {t:5.1f} с", font=F15, fill=DIM)

    y = 72
    for key, title in STAGES:
        sec = res.timings.get(key)
        d.ellipse([x, y + 4, x + 12, y + 16], fill=GREEN if sec is not None else GRAY)
        d.text((x + 22, y), title, font=F15, fill=FG)
        if sec is not None:
            d.text((PANEL_W - 70, y), f"{sec:5.1f} с", font=F15, fill=DIM)
        y += 21

    n = len(res.timeline.seconds)
    sec_i = min(int(t), n - 1)
    st = res.timeline.seconds[sec_i]
    y += 6
    d.text((x, y), "Сейчас", font=F18, fill=FG)
    y += 26
    rows = [
        ("Кассир", ZONE_RU.get(st.cashier_zone, st.cashier_zone), GREEN if st.cashier_at_workplace else AMBER),
        ("Посетители", str(len(st.visitors)) + (" (у стола)" if st.client_at_desk else ""), FG),
        ("Наличные на столе", "да" if st.cash_on_desk else "нет", RED if st.cash_on_desk else FG),
        ("Личные вещи на столе", "да" if st.personal_on_desk else "нет", RED if st.personal_on_desk else FG),
        ("Еда / напиток", "да" if st.food_on_desk else "нет", AMBER if st.food_on_desk else FG),
        ("Экран ноутбука", SCREEN_RU.get(st.screen, st.screen), SCREEN_COL.get(st.screen, FG)),
        ("Сейф", {"open": "открыт", "closed": "закрыт"}.get(st.safe, "не виден"), RED if st.safe == "open" else FG),
    ]
    for k, v, c in rows:
        d.text((x, y), k, font=F15, fill=DIM)
        d.text((x + 230, y), v, font=F15, fill=c)
        y += 20

    y += 8
    d.text((x, y), "Состояние по времени", font=F18, fill=FG)
    y += 44
    W, h = PANEL_W - 2 * x, 14
    S = res.timeline.seconds
    strips = [
        ("кассир на рабочем месте", [s.cashier_at_workplace for s in S], lambda v: GREEN if v else None),
        ("посетитель у стола", [s.client_at_desk for s in S], lambda v: BLUE if v else None),
        ("наличные на столе", [s.cash_on_desk for s in S], lambda v: RED if v else None),
        ("личные вещи / еда на столе", [s.personal_on_desk or s.food_on_desk for s in S], lambda v: AMBER if v else None),
        ("экран ноутбука", [s.screen for s in S], lambda v: {"work_screen": AMBER, "locked": GREEN}.get(v)),
        ("сейф открыт", [s.safe == "open" for s in S], lambda v: RED if v else None),
    ]
    for label, vals, colf in strips:
        _strip(d, x, y, W, h, vals, colf, label, n, t)
        y += 32
    # violations strip
    d.text((x, y - 18), "нарушения (кандидаты)", font=F13, fill=DIM)
    d.rectangle([x, y, x + W, y + h], fill=(30, 33, 42))
    for c in res.candidates:
        ok = c.score >= 0.5
        d.rectangle([x + W * c.t_start / n, y, max(x + W * c.t_end / n, x + W * c.t_start / n + 3), y + h], fill=RED if ok else GRAY)
    cx = x + W * min(t / n, 1.0)
    d.line([cx, y - 3, cx, y + h + 3], fill=(255, 255, 255), width=2)
    y += 26

    d.text((x, y), "Журнал событий", font=F18, fill=FG)
    y += 26
    if not shown:
        d.text((x, y), "нарушений пока не обнаружено", font=F15, fill=DIM)
        y += 22
    for c in shown[-3:]:
        d.text((x, y), f"{c.timecode:5.1f} с  {c.type}  {TYPE_TITLE[c.type]}", font=F15, fill=RED)
        y += 20
        d.text((x + 52, y), f"уверенность {c.score:.2f}", font=F13, fill=DIM)
        y += 18

    by = H - 46
    verdict = res.verdict if finished else None
    if verdict == "violation":
        d.rectangle([x, by, PANEL_W - x, by + 34], fill=(120, 30, 36))
        d.text((x + 12, by + 6), f"ВЕРДИКТ: обнаружены признаки нарушения ({len(res.violations)})", font=F18, fill=FG)
    elif verdict:
        d.rectangle([x, by, PANEL_W - x, by + 34], fill=(26, 90, 50))
        d.text((x + 12, by + 6), "ВЕРДИКТ: нарушение не обнаружено", font=F18, fill=FG)
    else:
        d.rectangle([x, by, PANEL_W - x, by + 34], fill=(34, 38, 48))
        d.text((x + 12, by + 6), "анализ воспроизводится…", font=F18, fill=DIM)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def iter_replay(res: AnalysisResult, fps: int = 15):
    """Yield composed BGR frames (video + panel) at `fps`, then hold the verdict for 3 s."""
    cap = cv2.VideoCapture(str(res.video.path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = res.video.height
    H = h + (h % 2)
    name = res.video.path.name
    cands = [c for c in res.candidates if c.score >= 0.5]
    step, nxt, i, last, last_t = src_fps / fps, 0.0, 0, None, 0.0

    def compose(bgr, t, active, shown, finished):
        frame = _draw_video(bgr, res, t, active)
        if H != h:
            frame = cv2.copyMakeBorder(frame, 0, H - h, 0, 0, cv2.BORDER_CONSTANT)
        return np.hstack([frame, _panel(res, t, H, name, shown, finished)])

    while cap.grab():
        if i + 1e-6 >= nxt:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            t = i / src_fps
            last, last_t = bgr, t
            active = [c for c in cands if c.t_start <= t <= c.t_end]
            shown = [c for c in cands if c.timecode <= t]
            yield compose(bgr, t, active, shown, t >= res.video.duration - 1.0 / fps - 0.05)
            nxt += step
        i += 1
    cap.release()
    if last is not None:
        end = compose(last, res.video.duration, [], cands, True)
        for _ in range(fps * 3):
            yield end


def render_replay(res: AnalysisResult, out_path: str | Path, fps: int = 15) -> Path:
    out_path = Path(out_path)
    H = res.video.height + (res.video.height % 2)
    ff = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{res.video.width + PANEL_W}x{H}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    for frame in iter_replay(res, fps):
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    return out_path


def play_window(res: AnalysisResult, fps: int = 15) -> None:
    """Play the replay in an OpenCV window at real-time speed. Press q or Esc to close."""
    import time

    title = f"kassa pipeline - {res.video.path.name}"
    t0 = time.time()
    for k, frame in enumerate(iter_replay(res, fps)):
        cv2.imshow(title, frame)
        wait = max(1, int((t0 + (k + 1) / fps - time.time()) * 1000))
        if cv2.waitKey(wait) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
