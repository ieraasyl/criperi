"""The whole ML pipeline on Modal: CV models and Qwen3.5-9B in one GPU container.

    modal deploy vlm_service/kassa_service.py     # persistent; scales to zero when idle
    modal app stop kassa-service                  # remove

The container needs no outside AI service: all weights are open source and live in the Modal
volume. Locally there is only a thin client (src/kassa_pipeline/remote.py).
"""

import hashlib
import os
import pickle
import re
import tempfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src" / "kassa_pipeline"

app = modal.App("kassa-service")
hf_vol = modal.Volume.from_name("kassa-hf-cache", create_if_missing=True)  # model weights
work_vol = modal.Volume.from_name("kassa-work", create_if_missing=True)  # per-clip stage + VLM caches

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "transformers>=5.5", "accelerate", "pillow", "numpy",
        "opencv-python-headless", "supervision", "trackers", "rtmlib", "onnxruntime",
        "scikit-learn", "joblib", "certifi", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/cache", "PYTHONPATH": "/root"})
    .add_local_dir(str(PKG), remote_path="/root/kassa_pipeline", ignore=["__pycache__", "*.pyc"])
)


def _apply(cfg, overrides: dict):
    """Set dotted config fields, e.g. {"events.provisional_threshold": 0.0}."""
    for path, val in (overrides or {}).items():
        obj = cfg
        *parents, leaf = path.split(".")
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, val)


@app.cls(
    gpu="L40S", image=image, volumes={"/cache": hf_vol, "/work": work_vol},
    min_containers=1,  # keeps one L40S warm; it is billed while idle. Set 0 to scale to zero.
    scaledown_window=600, timeout=1800, max_containers=4,
)
class Service:
    @modal.enter()
    def load(self):
        from kassa_pipeline import models
        from kassa_pipeline.config import PipelineConfig
        from kassa_pipeline.vlm import QwenVLM

        t = time.time()
        cfg = PipelineConfig()
        self.vlm = QwenVLM(cfg.models.vlm_model)
        m = cfg.models
        dev = models.pick_device(m.device)
        models.open_vocab_detector(m.open_vocab_detector, dev)
        models.person_detector(m.person_detector, dev)
        models.crop_encoder(m.crop_encoder, dev)
        models.pose_estimator(m.pose_model)
        hf_vol.commit()
        print(f"service ready in {time.time() - t:.0f}s")

    @modal.method()
    def analyze(self, video_bytes: bytes, name: str, options: dict | None = None) -> dict:
        """options: with_evidence (bool), use_vlm (bool), only (list of detector ids), cfg (dotted overrides)."""
        from kassa_pipeline.config import PipelineConfig
        from kassa_pipeline.pipeline import analyze_video

        opt = options or {}
        cfg = PipelineConfig()
        _apply(cfg, opt.get("cfg", {}))
        digest = hashlib.sha1(video_bytes).hexdigest()[:10]
        clip = re.sub(r"\W+", "_", Path(name).stem)

        # stable path and mtime, so the stage cache key does not change between calls
        d = Path(tempfile.gettempdir()) / digest
        d.mkdir(exist_ok=True)
        path = d / name
        path.write_bytes(video_bytes)
        os.utime(path, (0, 0))

        use_vlm = opt.get("use_vlm", True)
        if use_vlm:
            self.vlm.use_cache(f"/work/vlm/{clip}-{digest}.json")
        t = time.time()
        res = analyze_video(
            path, cfg, with_evidence=opt.get("with_evidence", False), cache_dir="/work/cache",
            vlm=self.vlm if use_vlm else None, only_detectors=opt.get("only"),
        )
        if use_vlm:
            self.vlm.flush()
        work_vol.commit()
        if opt.get("api_only"):  # what the jury API needs; skips the large debug payload and replay blob
            return {"name": name, "wall_seconds": round(time.time() - t, 1), "api": res.to_api(),
                    "meta": {"timings": {k: round(v, 1) for k, v in res.timings.items()}, "detector_errors": res.detector_info.get("errors", {})}}
        debug = res.to_debug()
        debug["vlm"] = {"calls": self.vlm.calls, "cache_hits": self.vlm.hits, "gpu_seconds": round(self.vlm.gpu_seconds, 1)} if use_vlm else None
        return {
            "name": name, "wall_seconds": round(time.time() - t, 1),
            "api": res.to_api(), "debug": debug, "blob": res.to_blob(),
        }


# ---------------------------------------------------------------- public HTTP API (the case's jury interface)
#   POST /api/v1/analyze        multipart field "file" (.mov), header X-API-Key  ->  {"job_id", "status"}
#   GET  /api/v1/jobs/{job_id}  header X-API-Key  ->  {"job_id", "status": processing|done|failed, ...result}
# The key is the team's own (Modal secret "kassa-api-key", variable KASSA_API_KEY), not a Modal credential.

web_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi", "python-multipart")


@app.function(image=web_image, secrets=[modal.Secret.from_name("kassa-api-key")], timeout=120)
@modal.asgi_app()
def api():
    import secrets as pysecrets

    from fastapi import FastAPI, File, Header, HTTPException, UploadFile

    web = FastAPI(title="Cashier discipline video analysis", docs_url=None, redoc_url=None)
    MAX_BYTES = 200 * 1024 * 1024

    def check(key: str | None):
        expected = os.environ.get("KASSA_API_KEY", "")
        if not expected or not key or not pysecrets.compare_digest(key, expected):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @web.get("/health")
    def health():
        return {"ok": True}

    @web.post("/api/v1/analyze")
    async def analyze(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        if len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail="file too large")
        name = Path(file.filename or "video.mov").name
        call = Service().analyze.spawn(data, name, {"with_evidence": True, "api_only": True})
        return {"job_id": call.object_id, "status": "processing"}

    @web.get("/api/v1/jobs/{job_id}")
    def job(job_id: str, x_api_key: str | None = Header(default=None)):
        check(x_api_key)
        try:
            call = modal.FunctionCall.from_id(job_id)
        except Exception:
            raise HTTPException(status_code=404, detail="unknown job_id")
        try:
            out = call.get(timeout=0)
        except TimeoutError:
            return {"job_id": job_id, "status": "processing"}
        except Exception as e:  # the analysis itself failed
            return {"job_id": job_id, "status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
        return {"job_id": job_id, "status": "done", **out["api"]}

    return web
