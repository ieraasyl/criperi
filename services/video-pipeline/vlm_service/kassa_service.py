"""The whole ML pipeline on Modal: CV models and Qwen3.5-9B in one GPU container.

    modal deploy vlm_service/kassa_service.py     # persistent; scales to zero when idle
    modal app stop kassa-service                  # remove

Per video: Stage 1 extract (people, pose, desk objects, laptop screen, safe; src/kassa_pipeline/extract.py),
then the sliding-window VLM scan and the rules (vlm_scan.py), then the case's JSON (report.py).
The container needs no outside AI service: all weights are open source and live in the Modal volume.
"""

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / "src" / "kassa_pipeline"

app = modal.App("kassa-service")
hf_vol = modal.Volume.from_name("kassa-hf-cache", create_if_missing=True)  # model weights
work_vol = modal.Volume.from_name("kassa-work", create_if_missing=True)  # Stage 1 results and VLM answers per clip

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "transformers>=5.5", "accelerate", "pillow", "numpy",
        "opencv-python-headless", "supervision", "trackers", "rtmlib", "onnxruntime-gpu",
        "certifi", "sentencepiece", "protobuf",
    )
    # rtmlib depends on the CPU onnxruntime, which shadows the GPU build; keep only onnxruntime-gpu
    .run_commands("pip uninstall -y onnxruntime onnxruntime-gpu", "pip install --no-deps 'onnxruntime-gpu>=1.21'")
    .env({"HF_HOME": "/cache", "PYTHONPATH": "/root"})
    .add_local_dir(str(PKG), remote_path="/root/kassa_pipeline", ignore=["__pycache__", "*.pyc"])
)


def _apply(cfg, overrides: dict):
    """Set dotted config fields, e.g. {"scan.threshold": 0.6}."""
    for path, val in (overrides or {}).items():
        obj = cfg
        *parents, leaf = path.split(".")
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, val)


def _save(video_bytes: bytes, name: str) -> tuple[Path, str, str]:
    digest = hashlib.sha1(video_bytes).hexdigest()[:10]
    clip = re.sub(r"\W+", "_", Path(name).stem)
    d = Path(tempfile.gettempdir()) / digest
    d.mkdir(exist_ok=True)
    path = d / name
    path.write_bytes(video_bytes)
    return path, clip, digest


@app.cls(
    gpu="L40S", image=image, volumes={"/cache": hf_vol, "/work": work_vol},
    # keeps one L40S warm; it is billed while idle. KASSA_MIN_CONTAINERS=0 at deploy time scales to zero.
    min_containers=int(os.environ.get("KASSA_MIN_CONTAINERS") or "1"),
    scaledown_window=600, timeout=1800, max_containers=8,  # the jury may send several videos at once
)
class Service:
    @modal.enter()
    def load(self):
        from kassa_pipeline.config import PipelineConfig
        from kassa_pipeline.extract import Models
        from kassa_pipeline.vlm import QwenVLM

        t = time.time()
        cfg = PipelineConfig()
        self.vlm = QwenVLM(cfg.models.vlm_model)
        self.extract_models = Models(cfg)
        print("extract models:", self.extract_models.names)
        hf_vol.commit()
        print(f"service ready in {time.time() - t:.0f}s")

    @modal.method()
    def analyze(self, video_bytes: bytes, name: str, options: dict | None = None) -> dict:
        """Stage 1 (reused from the work volume when this exact video was extracted before), the
        sliding-window VLM scan, and the case's JSON in "api".
        options: with_evidence, api_only, cfg (dotted overrides), fresh_stage1, no_vlm_cache."""
        started = time.time()  # the job budget runs from here
        from kassa_pipeline import report
        from kassa_pipeline import video as V
        from kassa_pipeline.config import PipelineConfig
        from kassa_pipeline.extract import stream
        from kassa_pipeline.vlm_scan import scan

        opt = options or {}
        cfg = PipelineConfig()
        _apply(cfg, opt.get("cfg", {}))
        path, clip, digest = _save(video_bytes, name)

        t = time.time()
        work_vol.reload()  # see Stage 1 results other containers committed after this one started
        s1_path = Path("/work/extract") / f"{clip}-{digest}.json"
        cached = s1_path.exists() and not opt.get("fresh_stage1")
        if cached:
            stage1 = json.loads(s1_path.read_text())
        else:
            stage1 = next(ev["result"] for ev in stream(path, cfg, self.extract_models) if ev["type"] == "final")
            s1_path.parent.mkdir(parents=True, exist_ok=True)
            s1_path.write_text(json.dumps(stage1, ensure_ascii=False))
        t_s1 = time.time() - t

        video = V.load_video(path, cfg.extract.fps, cfg.sampling.scene_cut_threshold)
        self.vlm.use_cache(None if opt.get("no_vlm_cache") else f"/work/vlm/scan-{clip}-{digest}.json")
        res = scan(video, stage1, self.vlm, cfg, started=started)
        self.vlm.flush()
        work_vol.commit()
        api = report.to_api(res, stage1, path, cfg.scan.threshold, opt.get("with_evidence", True))
        res["timings"]["stage1"] = round(t_s1, 1)
        res["wall_seconds"] = round(time.time() - t, 1)
        if opt.get("api_only"):
            return {"name": name, "api": api, "timings": res["timings"]}
        res["name"] = name
        res["api"] = api
        res["stage1"] = {"seconds": round(t_s1, 1), "cached": cached, "people": stage1["people"], "objects": stage1["objects"]}
        return res

    @modal.method()
    def extract(self, video_bytes: bytes, name: str, options: dict | None = None):
        """Stage 1 only. A generator: yields stage events, one record per sampled frame while the models
        run, then the final timeline (also saved to the work volume as /work/extract/<clip>-<digest>.json)."""
        from kassa_pipeline.config import PipelineConfig
        from kassa_pipeline.extract import stream

        cfg = PipelineConfig()
        _apply(cfg, (options or {}).get("cfg", {}))
        path, clip, digest = _save(video_bytes, name)
        for ev in stream(path, cfg, self.extract_models):
            if ev["type"] == "final":
                out = Path("/work/extract") / f"{clip}-{digest}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(ev["result"], ensure_ascii=False))
                work_vol.commit()
                ev["saved_to"] = str(out)
            yield ev


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
