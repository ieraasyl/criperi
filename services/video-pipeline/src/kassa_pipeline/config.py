"""All tunable parameters in one place. Values are starting points for calibration."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    device: str = "auto"  # auto | cuda | mps | cpu
    vlm_model: str = "Qwen/Qwen3.5-9B"  # runs on Modal (vlm_service/kassa_service.py)


@dataclass
class SamplingConfig:
    layout_frames: int = 10  # frames per camera segment used to find the desk, safe, laptop and door
    scene_cut_threshold: float = 0.5  # edge-structure correlation below this = camera cut/move


@dataclass
class LayoutConfig:
    prompt: tuple[str, ...] = (
        "desk",
        "safe",
        "metal cabinet",
        "laptop",
        "banknote counting machine",
        "door",
        "office chair",
        "telephone",
    )
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    counter_min_prob: float = 0.4  # SigLIP check that a "counter" candidate is a counting machine
    min_presence: float = 0.3  # object must be found in this share of layout frames
    desk_pad: float = 0.15  # workstation zone = desk box grown by this share


@dataclass
class PeopleConfig:
    person_threshold: float = 0.45
    track_buffer_sec: float = 3.0
    relink_max_gap_sec: float = 60.0
    relink_min_similarity: float = 0.55
    min_track_sec: float = 1.0
    background_min_sec: float = 8.0  # static non-desk tracks longer than this are reflections
    background_max_spread: float = 0.6  # ... if they move less than this many body heights


@dataclass
class ObjectsConfig:
    desk_crop_pad: float = 0.25  # desk objects are detected in the desk area grown by this share
    in_hand_px: float = 30.0  # hand point closer than this to a box = object is in hand


@dataclass
class ExtractConfig:
    # Large variants: the stage runs once per video on an L40S, so accuracy is worth the extra time.
    person_detector: str = "PekingU/rtdetr_v2_r101vd"
    # tiny, not base: on these low-resolution desk crops base found nothing with the full prompt and missed
    # the cash bag in X-17 even with short prompts, where tiny found it (0.38)
    open_vocab_detector: str = "IDEA-Research/grounding-dino-tiny"
    crop_encoder: str = "google/siglip-so400m-patch14-384"
    pose_model: str = "rtmw-x"  # whole body, 133 keypoints incl. 21 per hand; "none" disables pose
    fps: float = 4.0  # every model runs on every sampled frame (people, pose, desk objects)
    prompt: tuple[str, ...] = (
        "banknotes", "money", "cash bag", "handbag", "bag", "coat", "jacket", "cup", "mug", "bottle", "can",
        "food", "mobile phone", "id card", "plastic card", "passport", "papers", "tablet",
    )
    detection_cache_dir: str | None = None  # disk cache for raw open-vocab detections
    box_threshold: float = 0.25
    text_threshold: float = 0.2
    min_class_prob: float = 0.3  # SigLIP probability of the winning class
    object_track_buffer_sec: float = 2.0
    object_activation: float = 0.15  # detector score x class probability
    min_object_frames: int = 2  # shorter object tracks are flicker


@dataclass
class ScanConfig:
    window_sec: float = 8.0  # sliding windows: every moment is seen by two windows
    stride_sec: float = 4.0
    window_fps: float = 2.0  # full frames per second of window
    crop_fps: float = 1.0  # zoomed desk views per second of window
    frame_long_side: int = 512  # full frames are resized to this; the desk zoom keeps detail
    crop_width: int = 448
    session_fps: float = 1.0  # one window per client visit, for order rules (ID before cash)
    session_max_frames: int = 24
    session_pad_sec: float = 1.5
    max_new_tokens: int = 420
    batch: int = 4  # windows generated together on the GPU
    min_agree: int = 2  # a state needs this many overlapping windows saying "yes"
    merge_gap_sec: float = 10.0
    threshold: float = 0.5
    away_min_sec: float = 3.0  # laptop rule: the cashier is away at least this long
    door_fps: float = 1.0  # zoomed doorway views per second
    door_window_sec: float = 8.0  # views per VLM call = door_window_sec * door_fps
    door_open_sec: float = 4.0  # a client walking in opens the door for 1-3 s
    door_ignore_transits: bool = True  # V18 ignores the seconds around a client entering or leaving
    door_transit_sec: float = 3.0  # half-width of that window
    event_session_max_sec: float = 30.0  # longest session opened by an entry event with no exit logged
    sessions_from_vlm_events: bool = False  # a client entry the tracker missed still gets a session window
    scenario_rules: tuple[str, ...] = ()  # scenario rule ids (W01..W06, P01..P03) that also yield findings; off by default
    scenario_confidence: float = 0.45  # below threshold: alone it does not report, agreeing with a VLM finding it boosts it
    entrance_outsider: bool = False  # V18 from Stage 1: an outsider standing in the doorway while the cashier is in the room; off by default
    time_budget_sec: float | None = 270.0  # whole job, measured from analyze() start; None disables the guard
    finish_reserve_sec: float = 25.0  # kept for rules, evidence frames and JSON after the VLM stops
    entrance_sec: int = 3  # consecutive seconds of that before it counts


@dataclass
class PipelineConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    people: PeopleConfig = field(default_factory=PeopleConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
