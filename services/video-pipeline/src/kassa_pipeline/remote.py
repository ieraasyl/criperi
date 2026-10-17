"""Thin client for the Modal service: uploads videos, gets results back. No ML runs locally."""

from pathlib import Path

APP, CLS = "kassa-service", "Service"


def _service():
    import modal

    return modal.Cls.from_name(APP, CLS)()


def extract_stream(path: str | Path, options: dict | None = None):
    """Stage 1 on Modal: yields stage and per-frame events while the GPU processes the video."""
    p = Path(path)
    yield from _service().extract.remote_gen(p.read_bytes(), p.name, options or {})


def analyze_many(paths: list[Path], options: dict | None = None):
    """Full analysis of several clips in parallel; yields results (or exceptions) as they finish."""
    svc = _service()
    args = [(p.read_bytes(), p.name, options or {}) for p in paths]
    yield from svc.analyze.starmap(args, order_outputs=False, return_exceptions=True)
