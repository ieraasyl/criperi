import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_fact_layer_imports_in_fresh_interpreter():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = SRC
    for mod in ("kassa_pipeline.facts", "kassa_pipeline.facts_eval", "kassa_pipeline.review_server", "kassa_pipeline.scenarios"):
        r = subprocess.run([sys.executable, "-c", f"import {mod}"], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
