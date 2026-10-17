"""Stage 1, Extract: run every model once per video and keep one record per sampled frame.

Frame by frame, so a caller can watch it run (the Modal service streams the records):
person detection -> ByteTrack -> pose (wrists) -> desk objects (Grounding DINO + SigLIP) -> object
tracks (ByteTrack) -> derived facts. Live records carry provisional roles and zones. After the last
frame the person tracks are re-linked and the roles assigned with the whole video in view, object
classes are voted per track, and every record is rebuilt (`_finalize`).

Every record is a plain dict, so the stream needs no package on the receiving side:
    {"i", "t", "people": [{track, role, box, score, zone, hands, holding}],
     "objects": [{track, cls, det_cls, box, score, on_desk, held_by}], "facts": {...}}
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from kassa_pipeline import geometry as g
from kassa_pipeline import layout as L
from kassa_pipeline import models as M
from kassa_pipeline import objects as O
from kassa_pipeline import people as P
from kassa_pipeline import video as V
from kassa_pipeline.config import PipelineConfig
from kassa_pipeline.detcache import CachedDetector
from kassa_pipeline.layout import LayoutTimeline

CLASS_PROMPTS = {**O.OBJECT_PROMPTS, "id_card": "a photo of an ID card, a plastic card or a passport"}
DROP = {"surface", "device", "hand"}  # classes that are not objects of interest
MONEY = ("cash", "cash_bag")
TRAINED = ("cash", "cash_bag", "id_card")  # the classes of the fine-tuned desk detector (det_data.CLASSES)
AT_WORKPLACE = ("cashier_seat", "at_desk")
HANDS = {"left": M.L_WRIST, "right": M.R_WRIST}  # the person's own left and right
FINGERS = {"left": M.L_HAND, "right": M.R_HAND}


def _hand_point(k: np.ndarray, s: np.ndarray, wrist: int, fingers: range | None) -> tuple[float, float] | None:
    """Centre of the confident finger keypoints (whole-body pose), else the wrist. An object in the
    hand sits between the fingers, a few centimetres past the wrist."""
    if fingers is not None:
        ok = [i for i in fingers if s[i] > 0.3]
        if len(ok) >= 5:
            return (float(np.mean(k[ok, 0])), float(np.mean(k[ok, 1])))
    if s[wrist] > 0.4:
        return (float(k[wrist][0]), float(k[wrist][1]))
    return None
ROLE_REFRESH_SEC = 1.0  # provisional roles are recomputed this often


@dataclass
class _Person:  # raw, before final ids
    obs: P.Obs  # shared with the final tracks, so a raw detection maps to its final track
    hands: dict


@dataclass
class _Object:
    track: int
    det_cls: str
    box: g.Box
    score: float


@dataclass
class _Raw:
    i: int
    t: float
    people: list[_Person]
    objects: list[_Object]
    safe: O.StateObs
    screen: O.StateObs
    safe_change: float | None = None


class _NoHeads:
    def has(self, task: str) -> bool:
        return False


class Models:
    """The large variants named in ExtractConfig (the analysis stages keep the small ones in ModelConfig)."""

    def __init__(self, cfg: PipelineConfig):
        e = cfg.extract
        dev = M.pick_device(cfg.models.device)
        self.person = M.person_detector(e.person_detector, dev)
        self.pose = None if e.pose_model == "none" else M.pose_estimator(e.pose_model, dev)
        self.ov = M.open_vocab_detector(e.open_vocab_detector, dev)
        if e.detection_cache_dir:
            self.ov = CachedDetector(self.ov, e.detection_cache_dir, e.open_vocab_detector)
        self.enc = M.crop_encoder(e.crop_encoder, dev)
        self.desk = M.desk_detector(e.desk_detector, dev) if e.desk_detector else None
        self.heads = _NoHeads()  # trained heads use the base encoder's embeddings; zero-shot here
        self.names = {"person": e.person_detector, "objects": e.open_vocab_detector, "classes": e.crop_encoder,
                      "pose": f"{e.pose_model} ({self.pose.device})" if self.pose else "none", "device": dev}
        if self.desk:
            self.names["desk_objects"] = e.desk_detector


# ---------------------------------------------------------------- per-frame models


def _people(f: V.Frame, mdl: Models, tracker: ByteTrackTracker, cfg: PipelineConfig, min_h: float, ms: dict):
    t = time.time()
    dets = [d for d in mdl.person([f.norm], cfg.people.person_threshold)[0] if d.box[3] - d.box[1] >= min_h]
    sd = sv.Detections(
        xyxy=np.array([d.box for d in dets], dtype=np.float32).reshape(-1, 4),
        confidence=np.array([d.score for d in dets], dtype=np.float32),
        class_id=np.zeros(len(dets), dtype=int),
    )
    out = tracker.update(sd)
    found = [
        (int(tid), tuple(float(v) for v in box), float(c))
        for box, c, tid in zip(out.xyxy, out.confidence, out.tracker_id)
        if int(tid) >= 0
    ]
    ms["people"] = (time.time() - t) * 1000

    t = time.time()
    hands = [{"left": None, "right": None} for _ in found]
    if mdl.pose is not None and found:
        kpts, scores = mdl.pose(f.rgb, [b for _, b, _ in found])
        for h, k, s in zip(hands, kpts, scores):
            for side, idx in HANDS.items():
                h[side] = _hand_point(k, s, idx, FINGERS[side] if len(k) >= 133 else None)
    ms["pose"] = (time.time() - t) * 1000
    return [
        (tid, _Person(P.Obs(f.t, box, c, [p for p in h.values() if p]), h))
        for (tid, box, c), h in zip(found, hands)
    ]


def _trained(f: V.Frame, region: g.Box, mdl: Models, ec) -> list[tuple[g.Box, float, str]]:
    """The fine-tuned desk detector on the raw RGB desk crop, upscaled as in training; boxes in video pixels."""
    crop = g.crop(f.rgb, region)
    if crop is None:
        return []
    s = ec.desk_detector_scale
    big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    x0, y0 = int(max(0, region[0])), int(max(0, region[1]))  # where g.crop cut
    return [((d.box[0] / s + x0, d.box[1] / s + y0, d.box[2] / s + x0, d.box[3] / s + y0), d.score, d.label)
            for d in mdl.desk([big], ec.desk_detector_threshold)[0] if d.label in TRAINED]


def _candidates(f: V.Frame, lay: L.Layout, mdl: Models, cfg: PipelineConfig, ms: dict | None = None):
    """Desk objects of one frame: (box, score, class). Grounding DINO proposes, SigLIP names; with a trained
    desk detector, money and ID cards come from it and DINO's boxes of those classes are dropped."""
    ms = {} if ms is None else ms
    t = time.time()
    ec = cfg.extract
    h, w = f.norm.shape[:2]
    trained = getattr(mdl, "desk", None)
    cands: list[tuple[g.Box, float]] = []
    kept: list[tuple[g.Box, float, str]] = []
    if lay.workstation is not None:
        region = g.expand(lay.workstation, cfg.objects.desk_crop_pad, w, h)
        if trained is not None:
            kept += _trained(f, region, mdl, ec)
        crop = g.crop(f.norm, region)
        if crop is not None:
            equipment = [b for b in (lay.laptop, lay.counter, lay.telephone) if b]
            for d in mdl.ov([crop], ec.prompt, ec.box_threshold, ec.text_threshold)[0]:
                box = (d.box[0] + region[0], d.box[1] + region[1], d.box[2] + region[0], d.box[3] + region[1])
                if any(g.iou(box, e) > 0.5 for e in equipment) or g.area(box) > 0.5 * g.area(lay.workstation):
                    continue  # the work equipment itself, or the whole desk
                cands.append((box, d.score))
    ms["detect"] = (time.time() - t) * 1000

    t = time.time()
    crops = [g.crop(f.norm, g.expand(b, 0.4, w, h), min_size=12) for b, _ in cands]
    valid = [(c, x) for c, x in zip(cands, crops) if x is not None]
    if valid:
        probs = mdl.enc.zero_shot(mdl.enc.embed([x for _, x in valid]), list(CLASS_PROMPTS.values()))
        names = list(CLASS_PROMPTS)
        for ((box, sc), _), p in zip(valid, probs):
            cls, prob = names[int(np.argmax(p))], float(np.max(p))
            if cls not in DROP and prob >= ec.min_class_prob and not (trained is not None and cls in TRAINED):
                kept.append((box, sc * prob, cls))
    ms["classify"] = (time.time() - t) * 1000
    return kept


def _objects(f: V.Frame, lay: L.Layout, mdl: Models, tracker: ByteTrackTracker, cfg: PipelineConfig, ms: dict):
    kept = _candidates(f, lay, mdl, cfg, ms)
    t = time.time()
    sd = sv.Detections(
        xyxy=np.array([k[0] for k in kept], dtype=np.float32).reshape(-1, 4),
        confidence=np.array([k[1] for k in kept], dtype=np.float32),
        class_id=np.zeros(len(kept), dtype=int),
    )
    out = tracker.update(sd)
    res = []
    for box, tid in zip(out.xyxy, out.tracker_id):
        if int(tid) < 0 or not kept:
            continue
        b = tuple(float(v) for v in box)
        src = max(kept, key=lambda k: g.iou(k[0], b))  # the tracker returns the input rows; find which
        res.append(_Object(int(tid), src[2], src[0], src[1]))
    ms["obj_track"] = (time.time() - t) * 1000
    return res


def _states(f: V.Frame, lay: L.Layout, mdl: Models, gain: float, ms: dict,
            safe_min_prob: float | None = None) -> tuple[O.StateObs, O.StateObs]:
    t = time.time()
    safe = O._state(f.norm, lay.safe, 0.5, mdl.enc, mdl.heads, "safe", O.SAFE_PROMPTS)
    if safe_min_prob is not None and safe.state != "unknown" and safe.prob < safe_min_prob:
        safe = O.StateObs("unknown", safe.prob)
    screen = O.classify_screen(O.screen_features(f.rgb, lay.laptop, gain)) if lay.laptop else O.StateObs("unknown", 0.0)
    ms["states"] = (time.time() - t) * 1000
    return safe, screen


def _safe_refs(frames: list[V.Frame], layouts: LayoutTimeline) -> list:
    """Median edge signature of the safe per camera segment (None where the layout has no safe)."""
    sigs: list[list] = [[] for _ in layouts.layouts]
    for f in frames:
        k = layouts.index(f.t)
        sig = O.safe_signature(f.norm, layouts.layouts[k].safe)
        if sig is not None:
            sigs[k].append(sig)
    return [O.safe_reference(s) for s in sigs]


# ---------------------------------------------------------------- records and facts


def _record(raw: _Raw, ids: dict[int, int], people: P.PeopleResult, layouts: LayoutTimeline,
            obj_cls: dict[int, str] | None, cfg: PipelineConfig) -> dict:
    """One frame record. `ids` maps id(Obs) -> person track id; missing = dropped (reflection, flicker)."""
    lay = layouts.at(raw.t)
    desk = lay.desk or lay.workstation
    persons = []
    for p in raw.people:
        tid = ids.get(id(p.obs))
        if tid is None:
            continue
        role = "cashier" if tid == people.cashier_id else "visitor"
        zone = P.zone_of(raw.t, p.obs.box, layouts, people).name
        persons.append({"track": tid, "role": role, "box": [round(v, 1) for v in p.obs.box],
                        "score": round(p.obs.score, 3), "zone": zone,
                        "hands": {k: (None if v is None else [round(v[0], 1), round(v[1], 1)]) for k, v in p.hands.items()},
                        "holding": {"left": None, "right": None}})

    objs = []
    for o in raw.objects:
        cls = obj_cls.get(o.track) if obj_cls is not None else o.det_cls
        if cls is None:
            continue  # flicker track, dropped at finalize
        on_desk = bool(desk) and g.contains(g.expand(desk, 0.05), g.center(o.box))
        objs.append({"track": o.track, "cls": cls, "det_cls": o.det_cls, "box": [round(v, 1) for v in o.box],
                     "score": round(o.score, 3), "on_desk": on_desk, "held_by": None})

    # which hand holds what: each object goes to the nearest wrist within the in-hand distance
    reach = cfg.objects.in_hand_px
    if cfg.objects.in_hand_body is not None and people.body_height > 0:
        reach = cfg.objects.in_hand_body * people.body_height
    for ob in objs:
        best, best_d = None, reach
        for pr in persons:
            for side, pt in pr["hands"].items():
                if pt is None:
                    continue
                d = g.gap(tuple(ob["box"]), (pt[0] - 1, pt[1] - 1, pt[0] + 1, pt[1] + 1))
                if d < best_d:
                    best, best_d = (pr, side), d
        if best:
            pr, side = best
            ob["held_by"] = [pr["track"], side]
            if pr["holding"][side] is None:
                pr["holding"][side] = ob["track"]

    cashier = next((p for p in persons if p["role"] == "cashier"), None)
    visitors = [p for p in persons if p["role"] != "cashier"]
    loose = [o for o in objs if o["on_desk"] and o["held_by"] is None]
    facts = {
        "cashier_zone": cashier["zone"] if cashier else "absent",
        "cashier_at_workplace": bool(cashier and cashier["zone"] in AT_WORKPLACE),
        "people_in_room": len(persons),
        "visitors_at_desk": sum(v["zone"] in ("at_desk", "cashier_seat") for v in visitors),
        "cash_on_desk": any(o["cls"] in MONEY for o in loose),
        "cash_in_hand": [[o["held_by"][0], o["held_by"][1]] for o in objs if o["cls"] in MONEY and o["held_by"]],
        "id_card_seen": any(o["cls"] == "id_card" for o in objs),
        "items_on_desk": sorted({o["cls"] for o in loose}),
        "safe": raw.safe.state, "safe_prob": round(raw.safe.prob, 2),
        "screen": raw.screen.state,
    }
    if raw.safe_change is not None:
        facts["safe_change"] = round(raw.safe_change, 3)
    return {"i": raw.i, "t": raw.t, "people": persons, "objects": objs, "facts": facts}


def _tracks(raws: list[_Raw], raw_ids: dict[int, int]) -> list[P.Track]:
    by: dict[int, list[P.Obs]] = defaultdict(list)
    for r in raws:
        for p in r.people:
            by[raw_ids[id(p.obs)]].append(p.obs)
    return [P.Track(tid, obs) for tid, obs in by.items()]


def _provisional(raws: list[_Raw], raw_ids: dict[int, int], layouts: LayoutTimeline) -> tuple[P.PeopleResult, dict[int, int]]:
    res = P._assign_roles(_tracks(raws, raw_ids), layouts)
    return res, dict(raw_ids)


def _drop_reflections(people: P.PeopleResult, layouts: LayoutTimeline, cfg: PipelineConfig) -> list[P.Track]:
    """Drop only long, nearly static tracks that never overlap the desk area: reflections in the glass
    wall, people in the next room. A client standing still in front of the desk is kept."""
    unit = people.body_height or 100.0
    keep = []
    for tr in people.tracks:
        if tr.id != people.cashier_id and tr.t1 - tr.t0 >= cfg.people.background_min_sec:
            c = np.array([g.center(o.box) for o in tr.obs])
            static = float(np.hypot(*(c.max(axis=0) - c.min(axis=0)))) / unit < cfg.people.background_max_spread
            ws = [layouts.at(o.t).workstation for o in tr.obs]
            near_desk = max((g.inside_share(o.box, w) for o, w in zip(tr.obs, ws) if w), default=0.0) >= 0.2
            if static and not near_desk:
                continue
        keep.append(tr)
    return keep


def _finalize(video: V.Video, layouts: LayoutTimeline, raws: list[_Raw], raw_ids: dict[int, int], cfg: PipelineConfig):
    """Whole-video pass: drop short tracks, re-link people who left and came back, assign roles,
    drop reflections, vote one class per object track."""
    tracks = []
    for tr in _tracks(raws, raw_ids):
        if tr.obs[-1].t - tr.obs[0].t >= cfg.people.min_track_sec or len(tr.obs) >= 3:
            tr.hist = P._track_signature(video, tr)
            tracks.append(tr)
    tracks = P._relink(tracks, cfg.people)
    for tr in tracks:
        tr.obs.sort(key=lambda o: o.t)
    people = P._assign_roles(tracks, layouts)
    people.tracks = _drop_reflections(people, layouts, cfg)
    ids = {id(o): tr.id for tr in people.tracks for o in tr.obs}

    votes: dict[int, Counter] = defaultdict(Counter)
    span: dict[int, list[float]] = {}
    for r in raws:
        for o in r.objects:
            votes[o.track][o.det_cls] += o.score
            s = span.setdefault(o.track, [r.t, r.t, 0])
            s[1], s[2] = r.t, s[2] + 1
    obj_cls = {tid: v.most_common(1)[0][0] for tid, v in votes.items() if span[tid][2] >= cfg.extract.min_object_frames}

    frames = [_record(r, ids, people, layouts, obj_cls, cfg) for r in raws]
    summary_people = [{"id": t.id, "role": t.role, "t0": t.t0, "t1": t.t1, "frames": len(t.obs)} for t in people.tracks]
    summary_objects = [
        {"id": tid, "cls": obj_cls[tid], "t0": span[tid][0], "t1": span[tid][1], "frames": span[tid][2],
         "votes": {k: round(v, 2) for k, v in votes[tid].most_common()}}
        for tid in sorted(obj_cls)
    ]
    return frames, summary_people, summary_objects, people


# ---------------------------------------------------------------- the stream


def stream(path: str | Path, cfg: PipelineConfig | None = None, mdl: Models | None = None) -> Iterator[dict]:
    """Yield events: stage (decode, layout), frame (one live record per sampled frame), final (everything)."""
    cfg = cfg or PipelineConfig()
    mdl = mdl or Models(cfg)
    ec = cfg.extract
    t_begin = time.time()

    video = V.load_video(path, ec.fps, cfg.sampling.scene_cut_threshold)
    timings = {"decode": time.time() - t_begin}
    meta = {"name": Path(path).name, "fps": video.fps, "duration": video.duration, "width": video.width,
            "height": video.height, "cuts": video.scene_cuts, "sample_fps": ec.fps, "frames": len(video.frames)}
    meta["models"] = mdl.names
    yield {"type": "stage", "stage": "decode", "sec": round(timings["decode"], 2), "meta": meta}

    t = time.time()
    layouts = L.detect_layout_timeline(video.frames, video.scene_cuts, mdl.ov, mdl.enc, cfg.layout, cfg.sampling.layout_frames)
    timings["layout"] = time.time() - t
    yield {"type": "stage", "stage": "layout", "sec": round(timings["layout"], 2), "layouts": layouts.to_dict()}

    fps = ec.fps
    ptracker = ByteTrackTracker(
        lost_track_buffer=int(cfg.people.track_buffer_sec * fps), frame_rate=fps,
        track_activation_threshold=cfg.people.person_threshold, minimum_consecutive_frames=2,
        minimum_iou_threshold=0.1, high_conf_det_threshold=0.6,
    )
    otracker = ByteTrackTracker(
        lost_track_buffer=int(ec.object_track_buffer_sec * fps), frame_rate=fps,
        track_activation_threshold=ec.object_activation, minimum_consecutive_frames=2,
        minimum_iou_threshold=0.1, high_conf_det_threshold=2 * ec.object_activation,
    )
    min_h = 0.10 * video.height  # smaller boxes are reflections or people in the next room
    base_luma = float(np.median([f.luma for f in video.frames])) or 1.0
    raws: list[_Raw] = []
    raw_ids: dict[int, int] = {}
    prov, prov_ids, prov_at = P.PeopleResult([], None, [None] * len(layouts.layouts)), {}, -1e9
    t_loop = time.time()
    model_ms: dict[str, float] = defaultdict(float)
    safe_refs = _safe_refs(video.frames, layouts) if ec.safe_reference else None

    for k, f in enumerate(video.frames):
        t_frame = time.time()
        ms: dict[str, float] = {}
        lay = layouts.at(f.t)
        found = _people(f, mdl, ptracker, cfg, min_h, ms)
        for tid, p in found:
            raw_ids[id(p.obs)] = tid
        objs = _objects(f, lay, mdl, otracker, cfg, ms)
        safe, screen = _states(f, lay, mdl, base_luma / max(f.luma, 1.0), ms, ec.safe_min_prob)
        change = O.safe_change(f.norm, lay.safe, safe_refs[layouts.index(f.t)]) if safe_refs else None
        raw = _Raw(f.index, f.t, [p for _, p in found], objs, safe, screen, change)
        raws.append(raw)

        if f.t - prov_at >= ROLE_REFRESH_SEC:
            prov, prov_ids = _provisional(raws, raw_ids, layouts)
            prov_at = f.t
        else:
            for _, p in found:
                prov_ids[id(p.obs)] = raw_ids[id(p.obs)]
        rec = _record(raw, prov_ids, prov, layouts, None, cfg)
        ms["total"] = (time.time() - t_frame) * 1000
        for key, v in ms.items():
            model_ms[key] += v
        yield {"type": "frame", "k": k, "n": len(video.frames), "rec": rec,
               "ms": {key: round(v, 1) for key, v in ms.items()}, "wall": round(time.time() - t_begin, 2)}
    timings["frames"] = time.time() - t_loop

    t = time.time()
    frames, ppl, objs_summary, people = _finalize(video, layouts, raws, raw_ids, cfg)
    timings["finalize"] = time.time() - t
    timings["total"] = time.time() - t_begin
    yield {
        "type": "final",
        "result": {
            "meta": meta, "layouts": layouts.to_dict(), "cashier_track": people.cashier_id,
            "people": ppl, "objects": objs_summary, "frames": frames,
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "model_ms_per_frame": {k: round(v / max(len(raws), 1), 1) for k, v in model_ms.items()},
        },
    }
