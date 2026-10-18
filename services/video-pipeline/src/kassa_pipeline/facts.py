"""FactTimeline: per-second states and point events consumed by the scenario rules."""

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

UNKNOWN = "unknown"
_YES_NO = ("yes", "no", UNKNOWN)
STATE_VALUES: dict[str, tuple] = {
    "laptop_lid": ("open", "closed", "partial", UNKNOWN),
    "screen": ("work", "locked", "off", UNKNOWN),
    "safe_door": ("open", "closed", UNKNOWN),
    "desk_cash": _YES_NO,
    "desk_bag": _YES_NO,
    "desk_personal": _YES_NO,
    "desk_docs": _YES_NO,
    "people_inside": (),  # int >= 0 or None
}
EVENT_TYPES: tuple[str, ...] = (
    "room_entry", "room_exit", "id_presented", "id_returned", "cash_received", "bag_received",
    "bag_opened", "machine_count", "cash_stored", "cash_returned", "safe_access_start",
    "safe_access_end",
)
STATUSES = ("draft", "reviewed")
ORIGINS = ("vlm", "cv", "human")

_CLIP_RE = re.compile(r"[XxХх]\s*-\s*(\d+)")
_ACTOR_RE = re.compile(r"cashier|client-\d+|other-\d+")


def clip_id(name: str) -> str:
    """Canonical clip id: "Х-20", "X-20.mov", "X-20 (2)" -> "X-20"; other names -> stem."""
    m = _CLIP_RE.search(name)
    if m:
        return f"X-{m.group(1)}"
    return Path(name).stem


@dataclass
class Event:
    t: float
    type: str
    who: str | None = None
    batch: str | None = None
    from_: str | None = None  # JSON key "from"
    t_end: float | None = None
    origin: str = "human"
    note: str | None = None

    def to_dict(self) -> dict:
        d = {"t": self.t, "type": self.type, "who": self.who, "batch": self.batch,
             "from": self.from_, "t_end": self.t_end, "origin": self.origin, "note": self.note}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(t=d["t"], type=d["type"], who=d.get("who"), batch=d.get("batch"),
                   from_=d.get("from"), t_end=d.get("t_end"), origin=d.get("origin", "human"),
                   note=d.get("note"))


@dataclass
class FactTimeline:
    clip: str
    duration: float
    status: str = "draft"
    zones: dict[str, list[float]] = field(default_factory=dict)
    states: dict[str, list] = field(default_factory=dict)
    state_origin: dict[str, list[str]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    observable: list[str] = field(default_factory=lambda: list(EVENT_TYPES))

    @property
    def n_seconds(self) -> int:
        return math.ceil(self.duration)

    def state(self, name: str, sec: int) -> str | int | None:
        """Value of a state at a whole second; out of range or missing -> unknown (None for people_inside)."""
        missing = None if name == "people_inside" else UNKNOWN
        arr = self.states.get(name)
        if arr is None or not 0 <= sec < len(arr):
            return missing
        return arr[sec]

    def events_of(self, type: str, who: str | None = None) -> list[Event]:
        found = [e for e in self.events if e.type == type and (who is None or e.who == who)]
        return sorted(found, key=lambda e: e.t)

    def to_dict(self) -> dict:
        return {
            "clip": self.clip,
            "duration": self.duration,
            "status": self.status,
            "zones": {k: list(v) for k, v in self.zones.items()},
            "states": {k: list(v) for k, v in self.states.items()},
            "state_origin": {k: list(v) for k, v in self.state_origin.items()},
            "events": [e.to_dict() for e in self.events],
            "observable": list(self.observable),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactTimeline":
        """Build and validate; raises ValueError with a readable message."""
        try:
            facts = cls(
                clip=str(d["clip"]),
                duration=float(d["duration"]),
                status=d.get("status", "draft"),
                zones={k: list(v) for k, v in d.get("zones", {}).items()},
                states={k: list(v) for k, v in d.get("states", {}).items()},
                state_origin={k: list(v) for k, v in d.get("state_origin", {}).items()},
                events=[Event.from_dict(e) for e in d.get("events", [])],
                observable=list(d.get("observable", EVENT_TYPES)),
            )
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError(f"malformed timeline: {exc!r}") from exc
        facts.validate()
        return facts

    def validate(self) -> None:
        if not (math.isfinite(self.duration) and self.duration > 0):
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        n = self.n_seconds
        for name, arr in self.states.items():
            if name not in STATE_VALUES:
                raise ValueError(f"unknown state {name!r}")
            if len(arr) != n:
                raise ValueError(f"{name}: expected {n} values (one per second), got {len(arr)}")
            for i, v in enumerate(arr):
                if name == "people_inside":
                    ok = v is None or (isinstance(v, int) and not isinstance(v, bool) and v >= 0)
                else:
                    ok = v in STATE_VALUES[name]
                if not ok:
                    raise ValueError(f"{name}[{i}]: invalid value {v!r}")
        for name, arr in self.state_origin.items():
            if name not in self.states:
                raise ValueError(f"state_origin {name!r} has no matching state")
            if len(arr) != len(self.states[name]):
                raise ValueError(f"state_origin {name}: expected {len(self.states[name])} values, got {len(arr)}")
            for i, v in enumerate(arr):
                if v not in ORIGINS:
                    raise ValueError(f"state_origin {name}[{i}]: invalid origin {v!r}")
        for name, box in self.zones.items():
            if len(box) not in (0, 4) or not all(isinstance(x, (int, float)) for x in box):
                raise ValueError(f"zone {name!r} must be 4 numbers or empty, got {box!r}")
        for e in self.events:
            if e.type not in EVENT_TYPES:
                raise ValueError(f"unknown event type {e.type!r}")
            if not isinstance(e.t, (int, float)) or not 0 <= e.t <= self.duration:
                raise ValueError(f"event {e.type}: t={e.t!r} outside [0, {self.duration}]")
            if e.t_end is not None:
                if isinstance(e.t_end, bool) or not isinstance(e.t_end, (int, float)) or not math.isfinite(e.t_end):
                    raise ValueError(f"event {e.type}: t_end must be a finite number, got {e.t_end!r}")
                if e.t_end < e.t:
                    raise ValueError(f"event {e.type}: t_end {e.t_end} before t {e.t}")
            if e.origin not in ORIGINS:
                raise ValueError(f"event {e.type}: invalid origin {e.origin!r}")
            for key, actor in (("who", e.who), ("from", e.from_)):
                if actor is not None and not (isinstance(actor, str) and _ACTOR_RE.fullmatch(actor)):
                    raise ValueError(f"event {e.type}: {key} must be cashier|client-<k>|other-<k>, got {actor!r}")
        for t in self.observable:
            if t not in EVENT_TYPES:
                raise ValueError(f"observable: unknown event type {t!r}")

    @classmethod
    def load(cls, path) -> "FactTimeline":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path) -> None:
        """Atomic write: temp file in the same directory, then os.replace."""
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def empty(cls, clip: str, duration: float) -> "FactTimeline":
        n = math.ceil(duration)
        states = {k: ([None] * n if k == "people_inside" else [UNKNOWN] * n) for k in STATE_VALUES}
        return cls(clip=clip, duration=duration, states=states)
