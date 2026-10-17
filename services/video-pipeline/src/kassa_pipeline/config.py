"""All tunable parameters in one place. Values are starting points for calibration."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    device: str = "auto"  # auto | cuda | mps | cpu
    person_detector: str = "PekingU/rtdetr_v2_r18vd"
    open_vocab_detector: str = "IDEA-Research/grounding-dino-tiny"
    crop_encoder: str = "google/siglip-base-patch16-224"
    pose_model: str = "rtmpose-m"  # rtmlib body model; "none" disables pose
    heads_dir: str = "heads"  # trained crop-classifier heads (optional)
    detection_cache_dir: str | None = None  # disk cache for raw open-vocab detections


@dataclass
class SamplingConfig:
    people_fps: float = 4.0  # person detection + tracking
    objects_fps: float = 1.0  # desk objects, screen and safe state
    layout_frames: int = 10
    layout_refresh_sec: float = 10.0
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
    class_thresholds: dict[str, float] = field(default_factory=dict)  # canonical class -> box threshold
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
    prompt: tuple[str, ...] = (
        "banknotes",
        "money",
        "cash bag",
        "handbag",
        "coat",
        "jacket",
        "cup",
        "mug",
        "food",
        "mobile phone",
        "papers",
        "tablet",
    )
    box_threshold: float = 0.25
    text_threshold: float = 0.2
    category_thresholds: dict[str, float] = field(default_factory=dict)  # DINO category -> box threshold
    disagree_factor: float = 0.5  # score multiplier when DINO and SigLIP categories disagree
    desk_crop_pad: float = 0.25
    min_category_score: float = 0.35
    in_hand_px: float = 30.0  # wrist closer than this to a box = object is in hand
    instance_iou: float = 0.3  # same object across seconds


@dataclass
class TimelineConfig:
    min_on_sec: int = 2  # hysteresis: state turns on after N consistent seconds
    min_off_sec: int = 2
    lighting_jump: float = 25.0  # mean-luma change per second that counts as a lighting event


@dataclass
class EventsConfig:
    cash_away_sec: int = 2  # T1: cash on desk while cashier away for this long
    cash_no_client_sec: int = 5  # T1: cash on desk with no client and no handling
    cash_lag_sec: int = 4  # T1: cash is detected up to this long after the cashier got up
    personal_item_sec: int = 3  # T2
    food_sec: int = 2  # T3
    outsider_sec: float = 1.5  # T4
    outsider_seat_dist: float = 0.6  # T4: visitor closer than this many body heights to the cashier's seat
    unlocked_after_leave_sec: int = 3  # T5
    leave_during_service_sec: int = 3  # T6
    safe_open_away_sec: int = 3  # T7
    merge_gap_sec: float = 10.0  # same-type candidates closer than this are merged
    provisional_threshold: float = 0.5  # verdict before VLM verification


@dataclass
class PipelineConfig:
    models: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    people: PeopleConfig = field(default_factory=PeopleConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
