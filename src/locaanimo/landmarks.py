"""Find a character's mouth and eyes with the local vision-language model (Qwen3.5-4B).

Pixel heuristics kept getting fooled by beards, shadows, scars and tilted heads; the
VLM points at them the way a person would. MLX can't share a process with PyTorch, so
this runs as its own process:

  python -m locaanimo.landmarks char_a.png char_b.png   → JSON {path: {"mouth": [...], ...}}

Boxes are pixel coordinates in the original image.
"""

from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image

from .paths import MODELS

LLM_DIR = MODELS / "llm" / "Qwen3.5-4B-4bit"
QUESTION = ("Locate this anime character's mouth, left eye and right eye. Reply with JSON only: "
            '{"mouth": [x1, y1, x2, y2], "left_eye": [x1, y1, x2, y2], "right_eye": [x1, y1, x2, y2]}')
HEAD_SIZE = 448  # the head crop is downscaled to this; fewer image tokens = much faster


def head_crop(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """A square around the head: the top of the figure, sized to the figure (full-body
    characters have much smaller heads than cowboy shots)."""
    import numpy as np
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    w, h = img.size
    corner = np.median(np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3)]), axis=0)
    ys, xs = np.where(np.abs(rgb - corner).sum(axis=2) > 60)
    if len(ys) < 1000:                                        # no clear backdrop: assume a cowboy shot
        top, fig_h, cx = 0, h, w / 2
    else:
        top, fig_h = ys.min(), ys.max() - ys.min()
        cx = xs[ys < top + 0.15 * fig_h].mean()
    side = int(min(w, max(256, 0.42 * fig_h)))
    x0 = int(min(max(0, cx - side / 2), w - side))
    y0 = int(max(0, top - 0.04 * side))
    box = (x0, y0, x0 + side, y0 + side)
    return img.crop(box).resize((HEAD_SIZE, HEAD_SIZE), Image.LANCZOS), box


def _parse(text: str) -> dict | None:
    """Accept both the requested {"mouth": [...]} shape and Qwen's native grounding
    format [{"bbox_2d": [...], "label": "mouth"}, ...]."""
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = {str(d.get("label", "")).lower().replace(" ", "_"): d.get("bbox_2d")
                for d in data if isinstance(d, dict)}
    return data if isinstance(data, dict) and "mouth" in data else None


def locate(paths: list[str]) -> dict:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    model, proc = load(str(LLM_DIR))
    prompt = apply_chat_template(proc, model.config, QUESTION, num_images=1, enable_thinking=False)
    results = {}
    tmp = "/tmp/locaanimo_head.png" if not os.environ.get("TMPDIR") else os.path.join(os.environ["TMPDIR"], "locaanimo_head.png")
    for path in paths:
        img = Image.open(path).convert("RGB")
        crop, (x0, y0, x1, _) = head_crop(img)
        crop.save(tmp)
        out = generate(model, proc, prompt, image=[tmp], max_tokens=120, verbose=False, temperature=0.0)
        raw = _parse(getattr(out, "text", out))
        if raw is None:
            results[path] = None
            continue
        scale = (x1 - x0) / 1000          # the model answers in 0-1000 relative coordinates
        boxes = {k: [x0 + v[0] * scale, y0 + v[1] * scale, x0 + v[2] * scale, y0 + v[3] * scale]
                 for k, v in raw.items() if k in ("mouth", "left_eye", "right_eye") and v and len(v) == 4}
        if "left_eye" in boxes and "right_eye" in boxes:
            boxes["mouth"] = _refine_mouth(model, proc, img, boxes, tmp) or boxes["mouth"]
        results[path] = boxes
    return results


def _refine_mouth(model, proc, img, boxes, tmp) -> list[float] | None:
    """Second look, zoomed in around the first guess: much more precise."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    (lx0, _, lx1, _), (rx0, _, rx1, _) = boxes["left_eye"], boxes["right_eye"]
    eye_dist = abs((lx0 + lx1) / 2 - (rx0 + rx1) / 2)
    mx0, my0, mx1, my1 = boxes["mouth"]
    cx, cy, half = (mx0 + mx1) / 2, (my0 + my1) / 2, max(40.0, 0.7 * eye_dist)
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    img.crop(box).resize((HEAD_SIZE, HEAD_SIZE), Image.LANCZOS).save(tmp)
    q = ('This is a close-up of an anime character\'s lower face. Locate the mouth (the lips line). '
         'Reply with JSON only: {"mouth": [x1, y1, x2, y2]}')
    prompt = apply_chat_template(proc, model.config, q, num_images=1, enable_thinking=False)
    raw = _parse(getattr(generate(model, proc, prompt, image=[tmp], max_tokens=60, verbose=False, temperature=0.0),
                         "text", ""))
    if not raw or not raw.get("mouth") or len(raw["mouth"]) != 4:
        return None
    s = (box[2] - box[0]) / 1000
    return [box[0] + raw["mouth"][0] * s, box[1] + raw["mouth"][1] * s,
            box[0] + raw["mouth"][2] * s, box[1] + raw["mouth"][3] * s]


if __name__ == "__main__":
    print(json.dumps(locate(sys.argv[1:])))
