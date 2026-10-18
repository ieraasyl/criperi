"""The case's answer: verdict, and per violation a type, timecode, description and 1-3 evidence frames.

Nothing here is generated freely: the type is the fixed case class (data/violation_catalog.json, V01-V21),
the description is a template per class, and the norm comes from the hand-verified rule base
(data/rules_kb.json). A class without an open norm says so.
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
from kassa_pipeline.vlm_scan import Stage1, catalog

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
ACT = {"NBRK-231": "НБРК № 231", "NBRK-14": "НБРК № 14", "AFR-53": "АРРФР № 53"}

# What is visible, per case class (violation_catalog.json). {t} is the timecode in seconds.
SEEN = {
    "V01": "В {t} с кассир выходит из кабинета, а наличные или кассовая сумка остаются на открытом рабочем столе.",
    "V02": "В {t} с в кабинет входит следующий клиент, а деньги предыдущего клиента остаются на открытом столе.",
    "V03": "Около {t} с партии денег разных клиентов объединяются до завершения их отдельной обработки.",
    "V04": "В {t} с кассир принимает деньги или сумку раньше, чем клиент предъявил документ.",
    "V05": "Около {t} с открытая наличность принята и убрана без наблюдаемого первичного пересчёта на большом счётчике.",
    "V06": "В {t} с первичный пересчёт денег выполняется, когда клиента уже нет у кассы.",
    "V07": "Около {t} с отклонённые счётчиком купюры приняты без проверки на втором (малом) аппарате.",
    "V08": "Около {t} с отклонённая часть принята без пересчёта её суммы.",
    "V09": "Около {t} с отклонённые купюры объединены с проверенными до вторичной проверки и пересчёта.",
    "V10": "Около {t} с операция завершена без ввода записи или печати на ноутбуке.",
    "V11": "В {t} с операция закончена, а принятые деньги остаются на открытом рабочем столе.",
    "V12": "С {t} с кассир закончила работу с сейфом, а дверца сейфа остаётся открытой.",
    "V13": "В {t} с кассир выходит из кабинета, а ключ от сейфа остаётся в замке или на столе.",
    "V14": "В {t} с кассир выходит из кабинета, оставив ноутбук открытым.",
    "V15": "С {t} с на рабочем столе лежит личная вещь, пока идёт обслуживание или на столе находятся деньги.",
    "V16": "С {t} с одежда лежит на рабочем столе или помещена в сейф.",
    "V17": "С {t} с на рабочем месте находится еда или напиток.",
    "V18": "С {t} с дверь кабинета остаётся открытой после прохода людей.",
    "V19": "С {t} с посетитель находится на стороне кассира, у её рабочего места.",
    "V20": "С {t} с в кабинете одновременно находятся не менее трёх человек.",
    "V21": "В {t} с кассир выходит из кабинета, не завершив обслуживание клиента.",
}
MONEY = ("cash", "cash_bag")
FOCUS = {"V01": MONEY, "V02": MONEY, "V03": MONEY, "V11": MONEY, "V15": ("personal",), "V16": ("personal",), "V17": ("food_drink",)}
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


def type_label(code: str) -> str:
    c = catalog()[code]
    if not c["rule"]:
        return f"{code}. {c['title_ru']}"
    r = rule(c["rule"])
    return f"{code}. {c['title_ru']} ({r['act']}, {r['point']})"


def describe(code: str, t: float) -> str:
    c = catalog()[code]
    parts = [SEEN.get(code, "").format(t=f"{t:.0f}")]
    if c["rule"]:
        r = rule(c["rule"])
        parts.append(f"Источник: {r['document']}, {r['point']}: «{r['quote']}»")
        if r["indirect"]:
            parts.append("Связь с нормой косвенная; результат является подсказкой для проверки человеком.")
    else:
        parts.append("Открытой нормы в актах adilet.zan.kz не найдено; класс задан правилами кейса.")
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


def annotate(img: np.ndarray, t: float, code: str, s1: Stage1) -> np.ndarray:
    rec = s1.record(t) or {}
    lay = s1.layout(t)
    out = img
    for o in rec.get("objects", []):
        if o["cls"] in FOCUS.get(code, ()) and o["on_desk"]:
            out = _label(out, o["box"], OBJ_TEXT[o["cls"]], (255, 200, 0))
    if code == "V14" and lay.get("laptop"):
        out = _label(out, lay["laptop"], "экран не заблокирован", (255, 80, 80))
    if code == "V12" and lay.get("safe"):
        out = _label(out, lay["safe"], "сейф открыт", (255, 80, 80))
    if code == "V18" and lay.get("doors"):
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


def evidence_frames(video_path: Path, duration: float, t: float, code: str, s1: Stage1) -> list[Evidence]:
    """The moment itself, the moment before and one a little later."""
    times = [max(0.0, t - 2.0), min(t + 0.5, duration - 0.2), min(t + 3.0, duration - 0.2)]
    out, seen = [], set()
    for x in times:
        if round(x, 1) in seen:
            continue
        seen.add(round(x, 1))
        try:
            img = annotate(read_frame_at(video_path, x), x, code, s1)
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
        ev = evidence_frames(video_path, scan_result["duration"], f["t"], f["code"], s1) if with_evidence else []
        viols.append({
            "type": type_label(f["code"]),
            "timecode_sec": round(f["t"], 2),
            "description": describe(f["code"], f["t"]),
            "evidence_frames": [e.to_api() for e in ev],
        })
    return {"verdict": "violation" if viols else "no_violation", "violations": viols}
