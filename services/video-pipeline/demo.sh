#!/usr/bin/env bash
# Usage: ./demo.sh <video.mov> [--window]
#   default: analyzes the video, writes runs/<name>_replay.mp4 and opens it
#   --window: plays the replay in a window at real-time speed (q or Esc closes it)
set -euo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./demo.sh <video.mov> [--window]"; exit 1; }
PYTHONPATH=src .venv/bin/python -m kassa_pipeline.cli demo "$@" 2> >(grep -v -E "objc|Loading weights|Warning|warn|detach|scale =|load /|NOTE: Redirects|transformers\]" >&2)
