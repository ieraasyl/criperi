"""Probe service: Qwen3.5-9B on Modal (the full pipeline service is vlm_service/kassa_service.py)."""

import io
import time

import modal

MODEL_ID = "Qwen/Qwen3.5-9B"

app = modal.App("kassa-vlm")
cache = modal.Volume.from_name("kassa-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "transformers>=5.5", "accelerate", "pillow", "numpy", "huggingface_hub")
    .env({"HF_HOME": "/cache"})
)


@app.cls(gpu="L40S", image=image, volumes={"/cache": cache}, scaledown_window=300, timeout=3600)
class VLM:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t = time.time()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        cache.commit()
        print(f"loaded {MODEL_ID} in {time.time() - t:.0f}s; class={type(self.model).__name__}")

    @modal.method()
    def ask(self, jobs: list[dict]) -> list[dict]:
        import torch
        from PIL import Image

        out = []
        for j in jobs:
            t = time.time()
            imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in j["images"]]
            content = [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": j["prompt"]}]
            inputs = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
                enable_thinking=False,
            ).to("cuda")
            with torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=j.get("max_new_tokens", 220), do_sample=False)
            reply = self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            out.append({"id": j["id"], "text": reply.strip(), "sec": round(time.time() - t, 2)})
        return out
