"""Download the model weights Hadi's LocaAnimo uses into models/.

  python scripts/download_models.py            # core models (~11 GB)
  python scripts/download_models.py --video    # + LTX-Video for AI action shots (~27 GB)
  python scripts/download_models.py --list     # show what would be downloaded

Re-running is safe: finished files are skipped and interrupted ones resume.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"

# (what it's for, Hugging Face repo, files/patterns, local folder, approx size GB, license)
CORE = [
    ("Image generation (anime/cartoon art)", "OnomaAIResearch/Illustrious-XL-v2.0",
     ["Illustrious-XL-v2.0.safetensors"], "image/Illustrious-XL-v2.0", 6.9, "Fair AI Public License 1.0-SD"),
    ("Voices (28 English voices)", "hexgrad/Kokoro-82M",
     ["config.json", "kokoro-v1_0.pth", "voices/a*.pt", "voices/b*.pt"], "tts/Kokoro-82M", 0.35, "Apache-2.0"),
    ("Vision-language model: faces, free-text import, quality review", "mlx-community/Qwen3.5-4B-4bit",
     ["*"], "llm/Qwen3.5-4B-4bit", 2.9, "Apache-2.0"),
    ("Character cut-outs", "ZhengPeng7/BiRefNet", ["*.py", "*.json", "model.safetensors"],
     "vision/BiRefNet", 0.9, "MIT"),
    ("Depth for 2.5D camera moves", "depth-anything/Depth-Anything-V2-Small",
     ["depth_anything_v2_vits.pth"], "vision/Depth-Anything-V2-Small", 0.1, "Apache-2.0"),
]
VIDEO = [
    ("AI action shots: video model", "Lightricks/LTX-Video",
     ["ltxv-2b-0.9.8-distilled.safetensors", "scheduler/*", "tokenizer/*", "model_index.json"],
     "video/LTX-Video", 6.3, "LTX-Video license (see repo)"),
    ("AI action shots: T5 text encoder", "Lightricks/LTX-Video", ["text_encoder/*"], "video/LTX-Video", 19.0,
     "Apache-2.0"),
]
ANIMEFACE = ("Anime face detector", "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/"
             "lbpcascade_animeface.xml", "vision/animeface/lbpcascade_animeface.xml", 0.001, "MIT")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="store_true", help="also download LTX-Video (~27 GB) for AI action shots")
    ap.add_argument("--list", action="store_true", help="only list the models")
    args = ap.parse_args()

    plan = CORE + (VIDEO if args.video else [])
    total = sum(m[4] for m in plan) + ANIMEFACE[3]
    print(f"\n{'Model':<62} {'Size':>7}  License")
    for purpose, repo, *_, size, lic in plan:
        print(f"  {purpose:<60} {size:>5.1f} GB  {lic}")
    print(f"  {ANIMEFACE[0]:<60} {'<1 MB':>8}  {ANIMEFACE[4]}")
    print(f"\n  Total: ~{total:.1f} GB into {MODELS}\n")
    if args.list:
        return

    from huggingface_hub import snapshot_download
    for purpose, repo, patterns, folder, size, _ in plan:
        print(f"→ {purpose} ({repo})", flush=True)
        snapshot_download(repo, allow_patterns=patterns, local_dir=MODELS / folder)

    target = MODELS / ANIMEFACE[2]
    if not target.exists():
        print(f"→ {ANIMEFACE[0]}", flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ANIMEFACE[1], target)
    print("\nAll models ready.")


if __name__ == "__main__":
    sys.exit(main())
