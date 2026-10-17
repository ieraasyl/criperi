"""The case's answer: verdict, and per violation a type, timecode, description and 1-3 evidence frames.

Nothing here is generated freely: the description is a template chosen by what the scan saw
(Finding.kind), and the title and norm come from the hand-verified rule base (data/rules_kb.json).
"""

import base64
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from kassa_pipeline import geometry as g
from kassa_pipeline.video import read_frame_at
from kassa_pipeline.vlm_scan import Stage1

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
ACT = {"NBRK-231": "НБРК № 231", "NBRK-14": "НБРК № 14", "AFR-53": "АРРФР № 53"}

# What is visible, then why it may be a violation. {t} is the timecode in seconds.
SEEN = {
    "cash_away": ("В {t} с кассир покидает рабочее место, а на столе остаются наличные (купюры или сумка с деньгами).",
                  "Ранее принятые наличные должны храниться в запираемом устройстве, а не на столешнице."),
    "cash_new_client": ("В {t} с входит новый клиент, а на столе кассира лежат наличные от предыдущей операции.",
                        "На рабочем месте допускаются только деньги текущего клиента; остальные должны быть убраны."),
    "cash_no_client": ("С {t} с на столе лежат наличные, клиента нет, с деньгами никто не работает.",
                       "Наличные вне операции должны находиться в запираемом устройстве."),
    "cash_prev_visit": ("В {t} с, когда клиент подходит к кассе, на столе уже лежат наличные от предыдущей операции.",
                        "На рабочем месте допускаются только деньги текущего клиента; остальные должны быть убраны."),
    "cash_after_visit": ("В {t} с, после ухода клиента, на столе остаются наличные.",
                         "Принятые наличные должны быть убраны в запираемое устройство."),
    "personal_items": ("С {t} с на рабочем столе кассира лежат личные вещи (сумка, одежда).",
                       "Личные вещи работников должны храниться вне кассового узла."),
    "food": ("С {t} с на рабочем столе кассира находится еда или напиток.",
             "Это может указывать на нарушение требования хранить личные вещи вне кассового узла."),
    "laptop_away": ("В {t} с кассир уходит с рабочего места, на экране ноутбука остаётся рабочее окно (экран не заблокирован).",
                    "Оставленный без блокировки рабочий компьютер даёт доступ к информационной системе посторонним."),
    "door_open": ("С {t} с дверь кабинета кассы остаётся открытой.",
                  "Касса должна быть изолирована закрывающимся входом."),
    "door_open_person": ("С {t} с дверь кабинета кассы остаётся открытой, в проёме находится человек.",
                         "Касса должна быть изолирована закрывающимся входом."),
}
MONEY = ("cash", "cash_bag")
FOCUS = {  # object classes outlined on the evidence frames, per kind
    "cash_away": MONEY, "cash_new_client": MONEY, "cash_no_client": MONEY, "cash_prev_visit": MONEY, "cash_after_visit": MONEY,
    "personal_items": ("personal",), "food": ("food_drink",),
}
OBJ_TEXT = {"cash": "наличные", "cash_bag": "сумка с деньгами", "personal": "личная вещь", "food_drink": "еда/напиток"}


@lru_cache(maxsize=None)
def _font(size: int):
    for p in _FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


@lru_cache(maxsize=None)
def rule_base() -> dict:
    return json.loads(resources.files("kassa_pipeline.data").joinpath("rules_kb.json").read_text("utf8"))


def rule(rule_id: str) -> dict:
    kb = rule_base()
    r = next(r for r in kb["rules"] if r["id"] == rule_id)
    doc = kb["documents"][r["doc"]]
    s = r["strength"].lower()
    return {
        "id": r["id"], "title": r["title_ru"], "act": ACT.get(r["doc"], r["doc"]), "document": doc["short_ru"],
        "point": r["point"], "quote": r["quote_ru"], "url": doc["url"],
        "indirect": s.startswith(("indirect", "hypothesis", "partly")) or "weak" in s,
    }


def type_label(rule_id: str) -> str:
    r = rule(rule_id)
    return f"{r['title']} ({r['act']}, {r['point']})"


def describe(rule_id: str, kind: str, t: float) -> str:
    r = rule(rule_id)
    seen, why = SEEN.get(kind, ("", ""))
    parts = [seen.format(t=f"{t:.0f}"), why, f"Источник: {r['document']}, {r['point']}: «{r['quote']}»"]
    if r["indirect"]:
        parts.append("Связь с нормой косвенная; результат является подсказкой для проверки человеком.")
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------- evidence frames


def _label(img: np.ndarray, box, text: str, color: tuple[int, int, int]) -> np.ndarray:
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    box = [int(v) for v in box]
    d.rectangle(box, outline=color, width=3)
    font = _font(max(14, img.shape[0] // 45))
    tw = d.textlength(text, font=font)
    y = max(0, box[1] - font.size - 4)
    d.rectangle([box[0], y, box[0] + tw + 8, y + font.size + 4], fill=color)
    d.text((box[0] + 4, y), text, fill=(0, 0, 0), font=font)
    return np.array(pil)


def annotate(img: np.ndarray, t: float, kind: str, s1: Stage1) -> np.ndarray:
    rec = s1.record(t) or {}
    lay = s1.layout(t)
    out = img
    for o in rec.get("objects", []):
        if o["cls"] in FOCUS.get(kind, ()) and o["on_desk"]:
            out = _label(out, o["box"], OBJ_TEXT[o["cls"]], (255, 200, 0))
    if kind == "laptop_away" and lay.get("laptop"):
        out = _label(out, lay["laptop"], "экран не заблокирован", (255, 80, 80))
    if kind.startswith("door") and lay.get("doors"):
        out = _label(out, g.union([tuple(d) for d in lay["doors"][:2]]), "дверь открыта", (255, 80, 80))
    for p in rec.get("people", []):
        cashier = p["role"] == "cashier"
        out = _label(out, p["box"], "кассир" if cashier else "посетитель", (80, 220, 120) if cashier else (120, 160, 255))
    return out


@dataclass
class Evidence:
    timestamp_sec: float
    jpeg: bytes

    def to_api(self) -> dict:
        return {"timestamp_sec": round(self.timestamp_sec, 2), "image_base64": base64.b64encode(self.jpeg).decode()}


def evidence_frames(video_path: Path, duration: float, t: float, kind: str, s1: Stage1) -> list[Evidence]:
    """The moment itself, the moment before and one a little later."""
    times = [max(0.0, t - 2.0), min(t + 0.5, duration - 0.2), min(t + 3.0, duration - 0.2)]
    out, seen = [], set()
    for x in times:
        if round(x, 1) in seen:
            continue
        seen.add(round(x, 1))
        try:
            img = annotate(read_frame_at(video_path, x), x, kind, s1)
        except ValueError:
            continue
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append(Evidence(x, buf.tobytes()))
    return out


def to_api(scan_result: dict, stage1: dict, video_path: Path, threshold: float, with_evidence: bool = True) -> dict:
    """The case's JSON: findings at or above the threshold become violations, ordered by time."""
    s1 = Stage1(stage1)
    viols = []
    for f in sorted((f for f in scan_result["findings"] if f["confidence"] >= threshold), key=lambda f: f["t"]):
        ev = evidence_frames(video_path, scan_result["duration"], f["t"], f.get("kind", ""), s1) if with_evidence else []
        viols.append({
            "type": type_label(f["rule"]),
            "timecode_sec": round(f["t"], 2),
            "description": describe(f["rule"], f.get("kind", ""), f["t"]),
            "evidence_frames": [e.to_api() for e in ev],
        })
    return {"verdict": "violation" if viols else "no_violation", "violations": viols}
