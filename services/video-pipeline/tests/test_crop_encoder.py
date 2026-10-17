from types import SimpleNamespace

import torch

from kassa_pipeline.models import CropEncoder


class Batch(dict):
    def to(self, device):
        return self


class Proc:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kw):
        self.kwargs = kw
        return Batch(input_ids=torch.zeros((len(kw["text"]), 64), dtype=torch.long))


class Model:
    config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=64))

    def get_text_features(self, **kw):
        return torch.ones((kw["input_ids"].shape[0], 4))


def test_text_prompts_pad_to_the_trained_length():
    enc = object.__new__(CropEncoder)  # SigLIP 2 tokenizers report no usable model_max_length
    enc.device, enc.processor, enc.model, enc._text_cache = "cpu", Proc(), Model(), {}
    enc._text(("a", "b"))
    assert enc.processor.kwargs["padding"] == "max_length" and enc.processor.kwargs["max_length"] == 64
