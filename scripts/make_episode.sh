#!/bin/sh
# Make a full episode from a script, without the studio app.
#   scripts/make_episode.sh examples/the_lost_kite.yaml kite
# Art is cached in projects/<name>/art, so re-runs only redraw what's missing.
set -e
cd "$(dirname "$0")/.."
[ $# -eq 2 ] || { echo "usage: $0 <script.yaml> <project-name>" >&2; exit 1; }
.venv/bin/python scripts/make_art.py "$1" "$2"
.venv/bin/python scripts/render_episode.py "$1" "$2"
