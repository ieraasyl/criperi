"""Stages 8-9: evidence frames, templated description and rule citation.

Nothing here is generated freely: text comes from templates filled with timeline facts, and the
citation is looked up in the hand-verified rule base (data/rules_kb.json). A VLM may later add one
sentence of observation, but it never chooses or words the rule.
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
from kassa_pipeline.events import RULE_OF, TYPE_TITLE, Candidate
from kassa_pipeline.layout import LayoutTimeline
from kassa_pipeline.objects import ObjectsResult
from kassa_pipeline.people import PeopleResult
from kassa_pipeline.video import Video, read_frame_at

ZONE_RU = {
    "doorway": "направляется к выходу",
    "elsewhere": "выходит из зоны рабочего места",
    "absent": "её нет в кадре",
    "near_safe": "находится у сейфа",
    "at_desk": "у стола",
    "cashier_seat": "на рабочем месте",
    "standing_at_seat": "встаёт со своего места",
}

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


@lru_cache(maxsize=None)
def _font(size: int):
    for p in _FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


@lru_cache(maxsize=None)
def rule_base() -> dict:
    return json.loads(resources.files("kassa_pipeline.data").joinpath("rules_kb.json").read_text("utf8"))


def rule_for(type_id: str) -> dict | None:
    kb = rule_base()
    rid = RULE_OF.get(type_id)
    rule = next((r for r in kb["rules"] if r["id"] == rid), None)
    if rule is None:
        return None
    doc = kb["documents"][rule["doc"]]
    return {
        "id": rule["id"],
        "document": doc["short_ru"],
        "point": rule["point"],
        "quote": rule["quote_ru"],
        "url": doc["url"],
        "indirect": str(rule.get("strength", "")).lower().startswith(("indirect", "weak")),
        "caveat": rule.get("caveat"),
    }


def type_label(type_id: str) -> str:
    r = rule_for(type_id)
    title = TYPE_TITLE[type_id]
    if r is None:
        return title
    short = r["document"].split("(")[0].strip()
    return f"{title} ({r['point']}, {short})"


def _fmt(t: float) -> str:
    return f"{t:.0f}".replace(".", ",")


def observation(c: Candidate) -> tuple[str, str]:
    """(what is visible, why it may indicate a violation)."""
    t, f = _fmt(c.timecode), c.facts
    if c.type == "T1":
        mode = f.get("mode")
        if mode == "cashier_away":
            zone = ZONE_RU.get(f.get("cashier_zone", ""), "отходит от стола")
            return (
                f"В {t} с кассир покидает рабочее место ({zone}); на столе остаются наличные "
                f"(купюры или сумка с деньгами), они видны около {f.get('seconds', 1)} с.",
                "Ранее принятые наличные должны храниться в запираемом устройстве, а не на столешнице.",
            )
        if mode == "new_visitor_arrives_with_cash_on_desk":
            return (
                f"В {t} с в помещение входит новый посетитель, а на столе кассира лежат наличные, "
                "оставшиеся от предыдущей операции.",
                "На рабочем месте допускаются только деньги текущего клиента; остальные должны быть убраны.",
            )
        return (
            f"С {t} с на столе лежат наличные, клиента у стола нет.",
            "Наличные вне операции должны находиться в запираемом устройстве.",
        )
    if c.type == "T2":
        return (
            f"С {t} с на рабочем столе кассира лежат личные вещи (сумка, одежда).",
            "Личные вещи работников должны храниться вне кассового узла.",
        )
    if c.type == "T3":
        return (
            f"С {t} с на рабочем столе кассира находится еда или напиток.",
            "Это может указывать на нарушение требования хранить личные вещи вне кассового узла.",
        )
    if c.type == "T9":
        return (
            f"С {t} с на рабочем столе кассира виден мобильный телефон.",
            "Наличие фото- и видеофункции на видео проверить нельзя; требуется проверка сотрудником.",
        )
    if c.type == "T4":
        return (
            f"В {t} с рядом с рабочим местом кассира, со стороны кассира, находится посторонний человек.",
            "Кассовый узел должен быть изолирован; доступ в рабочую зону кассира ограничен.",
        )
    if c.type == "T5":
        return (
            f"В {t} с кассир отходит от стола, на экране ноутбука остаётся рабочее окно (экран не заблокирован).",
            "Оставленный без блокировки рабочий компьютер даёт доступ к информационной системе посторонним.",
        )
    if c.type == "T6":
        return (
            f"В {t} с кассир выходит из рабочей зоны, пока клиент остаётся у стола.",
            "Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента.",
        )
    if c.type == "T7":
        return (
            f"С {t} с дверца сейфа открыта, кассира в помещении нет.",
            "Наличные и ценности должны храниться в запертых устройствах.",
        )
    return ("", "")


def describe(c: Candidate) -> str:
    seen, why = observation(c)
    r = rule_for(c.type)
    parts = [seen, why]
    if r:
        parts.append(f"Источник: {r['document']}, {r['point']}: «{r['quote']}»")
        if r["indirect"]:
            parts.append(
                "Связь с нормой косвенная; результат является подсказкой для проверки человеком."
            )
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------- evidence frames


def _label(img: np.ndarray, box: g.Box, text: str, color: tuple[int, int, int]) -> np.ndarray:
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.rectangle(box, outline=color, width=3)
    font = _font(max(14, img.shape[0] // 45))
    tw = d.textlength(text, font=font)
    y = max(0, box[1] - font.size - 4)
    d.rectangle([box[0], y, box[0] + tw + 8, y + font.size + 4], fill=color)
    d.text((box[0] + 4, y), text, fill=(0, 0, 0), font=font)
    return np.array(pil)


_OBJ_TEXT = {
    "cash": "наличные",
    "cash_bag": "сумка с деньгами",
    "personal": "личная вещь",
    "food_drink": "еда/напиток",
    "phone": "телефон",
}
_FOCUS = {
    "T1": ("cash", "cash_bag"),
    "T2": ("personal",),
    "T3": ("food_drink",),
    "T9": ("phone",),
}


def annotate(
    img: np.ndarray,
    t: float,
    c: Candidate,
    layouts: LayoutTimeline,
    people: PeopleResult,
    objects: ObjectsResult,
) -> np.ndarray:
    out = img
    sec = int(t)
    for o in objects.objects.get(sec, []):
        if o.cat in _FOCUS.get(c.type, ()):
            out = _label(out, tuple(int(v) for v in o.box), _OBJ_TEXT[o.cat], (255, 200, 0))
    lay = layouts.at(t)
    if c.type == "T5" and lay.laptop:
        out = _label(out, tuple(int(v) for v in lay.laptop), "экран не заблокирован", (255, 80, 80))
    for tr in people.tracks:
        o = tr.at(t, tol=0.5)
        if o is None:
            continue
        if tr.role == "cashier":
            out = _label(out, tuple(int(v) for v in o.box), "кассир", (80, 220, 120))
        elif c.type in ("T4", "T6", "T1"):
            out = _label(out, tuple(int(v) for v in o.box), "посетитель", (120, 160, 255))
    return out


@dataclass
class Evidence:
    timestamp_sec: float
    jpeg: bytes

    def to_api(self) -> dict:
        return {"timestamp_sec": round(self.timestamp_sec, 2), "image_base64": base64.b64encode(self.jpeg).decode()}


def evidence_frames(
    video: Video,
    c: Candidate,
    layouts: LayoutTimeline,
    people: PeopleResult,
    objects: ObjectsResult,
    n: int = 3,
) -> list[Evidence]:
    """Main frame just after the timecode, plus the moment before and a later one."""
    dur = video.duration
    main = min(c.timecode + 0.5, max(c.t_end - 0.1, c.timecode), dur - 0.2)
    times = [main]
    if n >= 2:
        times.insert(0, max(0.0, c.timecode - 2.0))
    if n >= 3:
        times.append(min(dur - 0.2, max(main + 2.5, min(c.t_end, dur - 0.2))))
    seen, out = set(), []
    for t in times:
        key = round(t, 1)
        if key in seen:
            continue
        seen.add(key)
        try:
            img = read_frame_at(video.path, t)
        except ValueError:
            continue
        img = annotate(img, t, c, layouts, people, objects)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out.append(Evidence(t, buf.tobytes()))
    return out
