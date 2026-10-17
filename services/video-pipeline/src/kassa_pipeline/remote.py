"""Thin client for the Modal service: uploads a video, gets the analysis back. No ML runs locally."""

from pathlib import Path

APP, CLS = "kassa-service", "Service"


def _service():
    import modal

    return modal.Cls.from_name(APP, CLS)()


def analyze_remote(path: str | Path, options: dict | None = None) -> dict:
    p = Path(path)
    return _service().analyze.remote(p.read_bytes(), p.name, options or {})


def analyze_many(paths: list[Path], options: dict | None = None):
    """Yield results as they finish. Modal scales containers out, so clips run in parallel."""
    svc = _service()
    args = [(p.read_bytes(), p.name, options or {}) for p in paths]
    yield from svc.analyze.starmap(args, order_outputs=False, return_exceptions=True)
