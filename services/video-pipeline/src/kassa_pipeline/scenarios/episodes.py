"""Client episodes: who was in the room and which cash/bag batches they handed over."""

from dataclasses import dataclass, field

from ..facts import Event, FactTimeline

_RECEIVED = {"cash_received": "cash", "bag_received": "bag"}
_STAGES = {"machine_count": "counted_t", "bag_opened": "opened_t", "cash_stored": "stored_t",
           "cash_returned": "returned_t"}


@dataclass
class Batch:
    id: str
    client: str
    received_t: float
    kind: str  # cash | bag
    counted_t: float | None = None
    opened_t: float | None = None
    stored_t: float | None = None
    returned_t: float | None = None


@dataclass
class Episode:
    client: str
    entry_t: float | None
    exit_t: float | None
    first_seen_t: float
    batches: list[Batch] = field(default_factory=list)


def _is_client(name: str | None) -> bool:
    return bool(name) and name.startswith("client-")


def _first_after(ft: FactTimeline, type: str, batch: str, t: float) -> float | None:
    times = [e.t for e in ft.events_of(type) if e.batch == batch and e.t >= t]
    return min(times, default=None)


def _client_events(ft: FactTimeline) -> dict[str, list[Event]]:
    by_client: dict[str, list[Event]] = {}
    for e in sorted(ft.events, key=lambda e: e.t):
        for name in {e.who, e.from_}:
            if _is_client(name):
                by_client.setdefault(name, []).append(e)
    return by_client


def build_episodes(ft: FactTimeline) -> list[Episode]:
    """One episode per client id seen in events, ordered by first event time."""
    episodes = []
    for client, events in _client_events(ft).items():
        entry = next((e.t for e in events if e.type == "room_entry" and e.who == client), None)
        exit_t = next((e.t for e in events if e.type == "room_exit" and e.who == client
                       and (entry is None or e.t > entry)), None)
        batches = []
        for e in events:
            if e.type in _RECEIVED and e.from_ == client:
                bid = e.batch or f"{client}-b{len(batches) + 1}"
                batches.append(Batch(
                    id=bid, client=client, received_t=e.t, kind=_RECEIVED[e.type],
                    counted_t=_first_after(ft, "machine_count", bid, e.t),
                    opened_t=_first_after(ft, "bag_opened", bid, e.t),
                    stored_t=_first_after(ft, "cash_stored", bid, e.t),
                    returned_t=_first_after(ft, "cash_returned", bid, e.t)))
        episodes.append(Episode(client, entry, exit_t, events[0].t, batches))
    _link_unlabelled_stages(ft, [b for ep in episodes for b in ep.batches])
    return sorted(episodes, key=lambda ep: ep.first_seen_t)


def _link_unlabelled_stages(ft: FactTimeline, batches: list[Batch]) -> None:
    """A stage event without a batch id goes to the latest batch received by then that lacks that stage."""
    for e in sorted(ft.events, key=lambda e: e.t):
        attr = _STAGES.get(e.type)
        if attr is None or e.batch:
            continue
        open_ = [b for b in batches if b.received_t <= e.t and getattr(b, attr) is None]
        if open_:
            setattr(max(open_, key=lambda b: b.received_t), attr, e.t)
