"""Draft the per-second fact states: Stage-1 CV facts, with a local VLM answering closed zone questions."""

import math
from pathlib import Path

from .facts import FactTimeline
from .scenarios.stage1_facts import _by_second, _layout_at, facts_from_stage1
from .video import read_frame_at
from .zones import QUESTIONS, UNKNOWN, ask_zone_state, despeckle


def build_draft(video_path, stage1: dict, vlm, step: int = 1) -> FactTimeline:
    """CV facts from `stage1`; laptop_lid, screen (lid open), safe_door and desk_* come from the VLM
    wherever it answers yes/no. Skipped or unanswered seconds keep the CV value."""
    ft = facts_from_stage1(stage1, Path(video_path).name)
    duration = ft.duration
    n = math.ceil(duration)
    layouts = sorted(stage1["layouts"], key=lambda x: x["start"])
    secs = _by_second(stage1["frames"])
    raw = {k: [None] * n for k in QUESTIONS}
    reads = 0
    for s in range(0, n, step):
        try:
            rgb = read_frame_at(video_path, min(s + 0.5, duration - 0.05))
        except ValueError:
            continue
        reads += 1
        layout = _layout_at(layouts, s + 0.5)
        boxes = [p["box"] for f in secs.get(s, []) for p in f["people"]]
        for state in QUESTIONS:
            if state == "screen" and raw["laptop_lid"][s] != "open":
                continue
            raw[state][s] = ask_zone_state(video_path, duration, layout, boxes, vlm, state, s, rgb)
    if reads == 0:
        raise ValueError(f"no frames could be read from {video_path}")
    for state, answers in raw.items():
        smooth = despeckle([UNKNOWN if a is None else a for a in answers])
        for s, a in enumerate(answers):
            if a is not None:
                ft.states[state][s] = smooth[s]
                ft.state_origin[state][s] = "vlm"
    return ft
