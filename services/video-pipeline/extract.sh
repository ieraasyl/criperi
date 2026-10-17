#!/usr/bin/env bash
# Usage: ./extract.sh <video.mov> [--no-window]
#   Stage 1 (extract) runs on the Modal GPU; every frame is drawn here as soon as its record arrives.
#   Writes runs/extract/<name>_extract.json (timeline) and runs/extract/<name>_extract_live.mp4 (recording).
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./extract.sh <video.mov> [--no-window]"; exit 1; }
PYTHONPATH=src .venv/bin/python -m kassa_pipeline.cli extract "$@" 2> >(grep -v -E "objc|Warning|warn" >&2)
