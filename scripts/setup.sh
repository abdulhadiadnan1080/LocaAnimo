#!/bin/sh
# One-time setup for Hadi's LocaAnimo on an Apple Silicon Mac.
#   scripts/setup.sh           Python env + core models (~11 GB)
#   scripts/setup.sh --video   ...plus LTX-Video for AI action shots (~27 GB more)
set -e
cd "$(dirname "$0")/.."

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "LocaAnimo targets Apple Silicon Macs (M1 or newer)." >&2
  exit 1
fi
command -v ffmpeg >/dev/null || { echo "ffmpeg is required: install it from https://ffmpeg.org or with 'brew install ffmpeg'." >&2; exit 1; }
command -v uv >/dev/null || { echo "Installing uv (Python package manager)…"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

echo "→ Python 3.11 environment"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

echo "→ Models"
.venv/bin/python scripts/download_models.py "$@"

echo
echo "Done. Start the studio with:  ./studio.sh"
