"""Filesystem layout of the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"      # finished episodes (mp4 + thumbnail + metadata)
PROJECTS = ROOT / "projects"    # per-run working files (script, audio, frames)
CACHE = ROOT / "cache"          # voice previews etc.
WEB = ROOT / "web"
EXAMPLES = ROOT / "examples"
