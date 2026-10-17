"""Runs all stages that come before VLM verification and returns one result object."""

import time
from dataclasses import dataclass, field
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


def analyze_video(
    path: str | Path,
    cfg: PipelineConfig | None = None,
    progress: Progress | None = None,
    with_evidence: bool = True,
) -> AnalysisResult:
    cfg = cfg or PipelineConfig()
    dev = models.pick_device(cfg.models.device)
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

    ov = models.open_vocab_detector(cfg.models.open_vocab_detector, dev)
    enc = models.crop_encoder(cfg.models.crop_encoder, dev)
    layouts = L.detect_layout_timeline(
        video.frames, video.scene_cuts, ov, enc, cfg.layout, cfg.sampling.layout_frames
    )
    done("layout", layouts=layouts)

    pose = None if cfg.models.pose_model == "none" else models.pose_estimator(cfg.models.pose_model)
    pdet = models.person_detector(cfg.models.person_detector, dev)
    people = P.track_people(video, layouts, pdet, pose, cfg.people, cfg.sampling.people_fps)
    done("people", people=people)

    heads = models.trained_heads(cfg.models.heads_dir)
    objects = O.detect_objects(video, layouts, people, ov, enc, heads, cfg.objects, cfg.sampling.objects_fps)
    done("objects", objects=objects)

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
