#!/usr/bin/env bash
# Usage: ./run.sh <video.mov> [--fresh-stage1]
#   Full analysis on Modal (Stage 1 extract -> sliding-window VLM scan -> rules -> case JSON).
#   Prints the verdict and violations; writes runs/run/<name>/result.json, debug.json and evidence JPEGs.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./run.sh <video.mov> [--fresh-stage1]"; exit 1; }
PYTHONPATH=src .venv/bin/python -m kassa_pipeline.cli run "$@" 2> >(grep -v -E "objc|Warning|warn" >&2)
