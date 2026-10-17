"""People tracks, roles and zones. Positions only, never faces or identity.

The extract stage builds the tracks frame by frame; this module re-links them, picks the cashier and
tells which zone of the room a person is in."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from kassa_pipeline import geometry as g
from kassa_pipeline.config import PeopleConfig
from kassa_pipeline.layout import LayoutTimeline
from kassa_pipeline.video import Video


@dataclass
class Obs:
    t: float
    box: g.Box
    score: float
    wrists: list[tuple[float, float]] = field(default_factory=list)  # confident wrist points


@dataclass
class Track:
    id: int
    obs: list[Obs]
    hist: np.ndarray | None = None  # clothing-colour signature for re-linking
    emb: np.ndarray | None = None  # unit SigLIP embedding of the person crops (PeopleConfig.relink_embed)
    role: str = "visitor"  # cashier | visitor

    @property
    def t0(self) -> float:
        return self.obs[0].t

    @property
    def t1(self) -> float:
        return self.obs[-1].t

    def at(self, t: float, tol: float = 0.3) -> Obs | None:
        best = min(self.obs, key=lambda o: abs(o.t - t))
        return best if abs(best.t - t) <= tol else None


@dataclass
class Seat:
    xy: tuple[float, float]  # foot point of the seated cashier (bottom centre of the box)
    height: float  # cashier's median box height: unit for "near" distances


@dataclass
class PeopleResult:
    tracks: list[Track]
    cashier_id: int | None
    seats: list[Seat | None]  # one per layout segment (the camera may move)
    body_height: float = 0.0

    def cashier(self) -> Track | None:
        return next((t for t in self.tracks if t.id == self.cashier_id), None)


def _foot(box: g.Box) -> tuple[float, float]:
    """Bottom centre of the box. Feet locate a person on the floor; box centres are distorted by
    perspective (the door looks 'close' to the seat in the image, its floor position is not)."""
    return ((box[0] + box[2]) / 2, box[3])


def _clothing_hist(rgb: np.ndarray, box: g.Box) -> np.ndarray | None:
    h = box[3] - box[1]
    torso = g.crop(rgb, (box[0], box[1] + 0.15 * h, box[2], box[1] + 0.65 * h), min_size=6)
    if torso is None:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten().astype(np.float32)


def _track_signature(video: Video, tr: Track) -> np.ndarray | None:
    frames = {round(f.t, 3): f for f in video.frames}
    hs = []
    for o in tr.obs[:: max(1, len(tr.obs) // 8)]:
        f = frames.get(round(o.t, 3))
        if f is None:
            continue
        h = _clothing_hist(f.rgb, o.box)
        if h is not None:
            hs.append(h)
    return np.mean(hs, axis=0) if hs else None


def _track_embedding(video: Video, tr: Track, encoder) -> np.ndarray | None:
    """Mean SigLIP embedding of up to 8 person crops, renormalized. Lighting changes move it less than colour."""
    frames = {round(f.t, 3): f for f in video.frames}
    crops = []
    for o in tr.obs[:: max(1, len(tr.obs) // 8)]:
        f = frames.get(round(o.t, 3))
        c = None if f is None else g.crop(f.norm, o.box, min_size=12)
        if c is not None:
            crops.append(c)
    if not crops:
        return None
    m = np.asarray(encoder.embed(crops), np.float32).mean(axis=0)
    return m / (np.linalg.norm(m) + 1e-6)


def _similar(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def _relink(tracks: list[Track], cfg: PeopleConfig) -> list[Track]:
    """Merge tracks of the same clothing that never overlap in time (left and came back)."""
    tracks = sorted(tracks, key=lambda t: t.t0)
    merged: list[Track] = []
    for tr in tracks:
        target, best = None, 0.0
        for m in merged:
            if tr.t0 < m.t1 or tr.t0 - m.t1 > cfg.relink_max_gap_sec:
                continue
            if cfg.relink_embed and m.emb is not None and tr.emb is not None:
                s, floor = float(m.emb @ tr.emb), cfg.relink_min_embed_similarity
            else:
                s, floor = _similar(m.hist, tr.hist), cfg.relink_min_similarity
            if s > floor and s - floor > best:
                target, best = m, s - floor
        if target is None:
            merged.append(tr)
        else:
            target.obs += tr.obs
            if target.hist is not None and tr.hist is not None:
                target.hist = (target.hist + tr.hist) / 2
            if target.emb is not None and tr.emb is not None:
                e = target.emb + tr.emb
                target.emb = e / (np.linalg.norm(e) + 1e-6)
    return merged


def _assign_roles(tracks: list[Track], layouts: LayoutTimeline) -> PeopleResult:
    def at_workstation(o: Obs) -> bool:
        ws = layouts.at(o.t).workstation
        return bool(ws) and g.inside_share(o.box, ws) > 0.3

    def anchor_dist(o: Obs) -> float:
        lay = layouts.at(o.t)
        anchors = [b for b in (lay.laptop, lay.counter, lay.telephone) if b]
        if not anchors:
            return 100.0
        ax = np.mean([g.center(b) for b in anchors], axis=0)
        return float(np.hypot(*(np.array(g.center(o.box)) - ax)))

    best, best_score = None, 0.0
    for tr in tracks:
        at_desk = [o for o in tr.obs if at_workstation(o)]
        if not at_desk:
            continue
        near = float(np.mean([1.0 / (1.0 + anchor_dist(o) / 100.0) for o in at_desk]))
        step = (tr.t1 - tr.t0) / max(len(tr.obs) - 1, 1)
        score = len(at_desk) * step * near  # seconds at the desk, weighted by closeness to the work equipment
        if score > best_score:
            best, best_score = tr, score
    if best is None:
        return PeopleResult(tracks, None, [None] * len(layouts.layouts))
    best.role = "cashier"
    seats: list[Seat | None] = []
    for k in range(len(layouts.layouts)):
        seg = [o for o in best.obs if layouts.index(o.t) == k and at_workstation(o)]
        if not seg:
            seats.append(None)
            continue
        xy = tuple(float(v) for v in np.median([_foot(o.box) for o in seg], axis=0))
        seats.append(Seat(xy, float(np.median([o.box[3] - o.box[1] for o in seg]))))
    height = next((s.height for s in seats if s), 0.0)
    return PeopleResult(tracks, best.id, seats, height)


STAND_RATIO = 1.35  # box taller than this x the seated height = standing (seated ~105 px, standing ~190 px)
NEAR_SEAT = 0.5  # feet within this many body heights of the cashier's seat = on the workplace


@dataclass
class Zone:
    name: str  # cashier_seat | standing_at_seat | at_desk | near_safe | doorway | elsewhere
    near_cashier_seat: bool
    seat_dist: float | None = None  # distance to the cashier's seat, in cashier body heights


def zone_of(t: float, box: g.Box, layouts: LayoutTimeline, res: PeopleResult) -> Zone:
    """Where a person is at time t. All thresholds are relative to the scene, not to pixels."""
    layout = layouts.at(t)
    seat = res.seats[layouts.index(t)] if res.seats else None
    near_seat, seat_dist = False, None
    if seat is not None and seat.height > 0:
        d = np.hypot(*(np.array(_foot(box)) - np.array(seat.xy)))
        seat_dist = float(d / seat.height)
        near_seat = seat_dist < NEAR_SEAT
    ws, safe = layout.workstation, layout.safe
    if near_seat:
        standing = seat.height > 0 and (box[3] - box[1]) > STAND_RATIO * seat.height
        return Zone("standing_at_seat" if standing else "cashier_seat", True, seat_dist)
    if safe and g.inside_share(box, g.expand(safe, 0.5)) > 0.25 and not (
        ws and g.inside_share(box, ws) > 0.5
    ):
        return Zone("near_safe", False, seat_dist)
    if ws and g.inside_share(box, ws) > 0.3:
        return Zone("at_desk", False, seat_dist)
    foot = ((box[0] + box[2]) / 2, box[3])
    if any(g.contains(g.expand(d, 0.05), foot) for d in layout.doors[:2]):
        return Zone("doorway", False, seat_dist)
    return Zone("elsewhere", False, seat_dist)
