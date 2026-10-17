"""Stage 2: people detection, tracking, roles and zones. Positions only, never faces or identity."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from kassa_pipeline import geometry as g
from kassa_pipeline.config import PeopleConfig
from kassa_pipeline.layout import Layout, LayoutTimeline
from kassa_pipeline.models import L_WRIST, R_WRIST, PersonDetector, PoseEstimator
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


def _similar(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def _relink(tracks: list[Track], cfg: PeopleConfig) -> list[Track]:
    """Merge tracks of the same clothing that never overlap in time (left and came back)."""
    tracks = sorted(tracks, key=lambda t: t.t0)
    merged: list[Track] = []
    for tr in tracks:
        target = None
        best = cfg.relink_min_similarity
        for m in merged:
            if tr.t0 < m.t1 or tr.t0 - m.t1 > cfg.relink_max_gap_sec:
                continue
            s = _similar(m.hist, tr.hist)
            if s > best:
                target, best = m, s
        if target is None:
            merged.append(tr)
        else:
            target.obs += tr.obs
            if target.hist is not None and tr.hist is not None:
                target.hist = (target.hist + tr.hist) / 2
    return merged


def track_people(
    video: Video,
    layouts: LayoutTimeline,
    detector: PersonDetector,
    pose: PoseEstimator | None,
    cfg: PeopleConfig,
    fps: float,
) -> PeopleResult:
    frames = video.every(fps)
    dets = detector([f.norm for f in frames], cfg.person_threshold)
    min_h = 0.10 * video.height  # smaller boxes are reflections or people in the next room

    tracker = ByteTrackTracker(
        lost_track_buffer=int(cfg.track_buffer_sec * fps),
        frame_rate=fps,
        track_activation_threshold=cfg.person_threshold,
        minimum_consecutive_frames=2,
        minimum_iou_threshold=0.1,
        high_conf_det_threshold=0.6,
    )
    raw: dict[int, list[Obs]] = {}
    for f, ds in zip(frames, dets):
        ds = [d for d in ds if d.box[3] - d.box[1] >= min_h]
        xyxy = np.array([d.box for d in ds], dtype=np.float32).reshape(-1, 4)
        sd = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([d.score for d in ds], dtype=np.float32),
            class_id=np.zeros(len(ds), dtype=int),
        )
        out = tracker.update(sd)
        for box, conf, tid in zip(out.xyxy, out.confidence, out.tracker_id):
            if int(tid) < 0:  # not yet confirmed by the tracker
                continue
            raw.setdefault(int(tid), []).append(Obs(f.t, tuple(float(v) for v in box), float(conf)))

    tracks = []
    for tid, obs in raw.items():
        if obs[-1].t - obs[0].t >= cfg.min_track_sec or len(obs) >= 3:
            tr = Track(tid, obs)
            tr.hist = _track_signature(video, tr)
            tracks.append(tr)
    tracks = _relink(tracks, cfg)
    for tr in tracks:
        tr.obs.sort(key=lambda o: o.t)

    if pose is not None:
        _add_wrists(video, tracks, pose, fps / 2)

    res = _assign_roles(tracks, layouts)
    res.tracks = _drop_background(res, layouts, cfg)
    return res


def _add_wrists(video: Video, tracks: list[Track], pose: PoseEstimator, fps: float):
    by_t: dict[float, list[tuple[Obs, Track]]] = {}
    for tr in tracks:
        for o in tr.obs:
            by_t.setdefault(round(o.t, 3), []).append((o, tr))
    for f in video.every(fps):
        items = by_t.get(round(f.t, 3), [])
        if not items:
            continue
        kpts, scores = pose(f.rgb, [o.box for o, _ in items])
        for (o, _), k, s in zip(items, kpts, scores):
            o.wrists = [tuple(map(float, k[i])) for i in (L_WRIST, R_WRIST) if s[i] > 0.4]


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


def _drop_background(res: PeopleResult, layouts: LayoutTimeline, cfg: PeopleConfig) -> list[Track]:
    """Remove reflections and people seen through the glass wall: long, nearly static tracks that
    never reach the desk. Real people at the door are moving."""
    unit = res.body_height or 0.2 * layouts.first.height
    keep = []
    for tr in res.tracks:
        if tr.id == res.cashier_id:
            keep.append(tr)
            continue
        zones = [zone_of(o.t, o.box, layouts, res).name for o in tr.obs]
        at_desk = sum(z in ("cashier_seat", "at_desk", "near_safe") for z in zones) / len(zones)
        if at_desk == 0:  # never reached the desk or the safe: next office, reflection, passer-by
            continue
        if tr.t1 - tr.t0 < cfg.background_min_sec:
            keep.append(tr)
            continue
        c = np.array([g.center(o.box) for o in tr.obs])
        spread = float(np.hypot(*(c.max(axis=0) - c.min(axis=0)))) / unit
        if at_desk < 0.2 and spread < cfg.background_max_spread:
            continue
        keep.append(tr)
    return keep


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
