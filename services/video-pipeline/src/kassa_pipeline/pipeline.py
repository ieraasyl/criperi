"""Runs all stages that come before VLM verification and returns one result object."""

import hashlib
import json
import pickle
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from kassa_pipeline import events as E
from kassa_pipeline import layout as L
from kassa_pipeline import models, objects as O, people as P, report as R, timeline as T
from kassa_pipeline import video as V
from kassa_pipeline.config import PipelineConfig

Progress = Callable[[str, float, dict], None]  # (stage, seconds spent, extra info)


@dataclass
class Violation:
    candidate: E.Candidate
    type_label: str
    description: str
    rule: dict | None
    evidence: list[R.Evidence]

    def to_api(self) -> dict:
        return {
            "type": self.type_label,
            "timecode_sec": round(self.candidate.timecode, 2),
            "description": self.description,
            "evidence_frames": [e.to_api() for e in self.evidence],
        }

    def to_debug(self) -> dict:
        c = self.candidate
        return {
            **self.to_api(),
            "type_id": c.type,
            "score": round(c.score, 3),
            "t_range": [c.t_start, c.t_end],
            "facts": c.facts,
            "rule": self.rule,
        }


@dataclass
class AnalysisResult:
    video: V.Video
    layouts: L.LayoutTimeline
    people: P.PeopleResult
    objects: O.ObjectsResult
    timeline: T.Timeline
    candidates: list[E.Candidate]
    violations: list[Violation]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return "violation" if self.violations else "no_violation"

    def to_api(self) -> dict:
        return {"verdict": self.verdict, "violations": [v.to_api() for v in self.violations]}

    def to_debug(self) -> dict:
        return {
            "verdict": self.verdict,
            "duration": self.video.duration,
            "scene_cuts": self.video.scene_cuts,
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "layouts": self.layouts.to_dict(),
            "cashier_track": self.people.cashier_id,
            "tracks": [
                {"id": t.id, "role": t.role, "t0": t.t0, "t1": t.t1, "obs": len(t.obs)}
                for t in self.people.tracks
            ],
            "candidates": [
                {"type": c.type, "timecode": c.timecode, "range": [c.t_start, c.t_end], "score": round(c.score, 3), "facts": c.facts}
                for c in self.candidates
            ],
            "violations": [v.to_debug() for v in self.violations],
            "timeline": self.timeline.to_dict(),
        }


def _key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def _heads_fingerprint(heads_dir: str) -> list:
    d = Path(heads_dir)
    return sorted((f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in d.glob("*.joblib")) if d.is_dir() else []


def _cached(cache_dir: Path | None, clip_id: str, stage: str, key: str, fn):
    """Run fn() or load its pickled result. The key covers every input that changes the result,
    so editing a threshold in a later stage never invalidates an earlier expensive one."""
    if cache_dir is None:
        return fn(), False
    path = cache_dir / clip_id / f"{stage}-{key}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes()), True
        except Exception:  # corrupt or from an older code version: recompute
            pass
    val = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(val))
    for old in path.parent.glob(f"{stage}-*.pkl"):
        if old != path:
            old.unlink(missing_ok=True)
    return val, False


def analyze_video(
    path: str | Path,
    cfg: PipelineConfig | None = None,
    progress: Progress | None = None,
    with_evidence: bool = True,
    cache_dir: str | Path | None = None,
) -> AnalysisResult:
    cfg = cfg or PipelineConfig()
    dev = models.pick_device(cfg.models.device)
    cache = Path(cache_dir) if cache_dir else None
    path = Path(path)
    st = path.stat()
    clip_id = re.sub(r"\W+", "_", path.stem)
    timings: dict[str, float] = {}
    t_prev = time.time()

    def done(stage: str, **info):
        nonlocal t_prev
        now = time.time()
        timings[stage] = now - t_prev
        t_prev = now
        if progress:
            progress(stage, timings[stage], info)

    video = V.load_video(path, cfg.sampling.people_fps, cfg.sampling.scene_cut_threshold)
    done("decode", frames=len(video.frames), duration=video.duration, cuts=video.scene_cuts)

    m, sm = cfg.models, cfg.sampling
    k_layout = _key(
        st.st_size, int(st.st_mtime), asdict(cfg.layout), m.open_vocab_detector, m.crop_encoder,
        sm.layout_frames, sm.people_fps, sm.scene_cut_threshold,
    )

    def run_layout():
        ov = models.open_vocab_detector(m.open_vocab_detector, dev)
        enc = models.crop_encoder(m.crop_encoder, dev)
        return L.detect_layout_timeline(video.frames, video.scene_cuts, ov, enc, cfg.layout, sm.layout_frames)

    layouts, hit = _cached(cache, clip_id, "layout", k_layout, run_layout)
    done("layout", layouts=layouts, cached=hit)

    k_people = _key(k_layout, asdict(cfg.people), m.person_detector, m.pose_model)

    def run_people():
        pose = None if m.pose_model == "none" else models.pose_estimator(m.pose_model)
        pdet = models.person_detector(m.person_detector, dev)
        return P.track_people(video, layouts, pdet, pose, cfg.people, sm.people_fps)

    people, hit = _cached(cache, clip_id, "people", k_people, run_people)
    done("people", people=people, cached=hit)

    k_objects = _key(
        k_people, asdict(cfg.objects), sm.objects_fps, _heads_fingerprint(m.heads_dir),
        O.OBJECT_PROMPTS, O.SCREEN_PROMPTS, O.SAFE_PROMPTS,  # prompt edits must invalidate the cache
    )

    def run_objects():
        ov = models.open_vocab_detector(m.open_vocab_detector, dev)
        enc = models.crop_encoder(m.crop_encoder, dev)
        heads = models.trained_heads(m.heads_dir)
        return O.detect_objects(video, layouts, people, ov, enc, heads, cfg.objects, sm.objects_fps)

    objects, hit = _cached(cache, clip_id, "objects", k_objects, run_objects)
    done("objects", objects=objects, cached=hit)

    timeline = T.build_timeline(video, layouts, people, objects, cfg.timeline)
    done("timeline", timeline=timeline)

    candidates = E.find_candidates(timeline, cfg.events)
    done("events", candidates=candidates)

    violations: list[Violation] = []
    for c in candidates:
        if c.score < cfg.events.provisional_threshold:
            continue
        ev = R.evidence_frames(video, c, layouts, people, objects) if with_evidence else []
        violations.append(Violation(c, R.type_label(c.type), R.describe(c), R.rule_for(c.type), ev))
    done("report", violations=violations)

    return AnalysisResult(video, layouts, people, objects, timeline, candidates, violations, timings)
