"""Quality review by the local vision model (Qwen3.5-4B): looks at assets and preview
frames and reports obvious problems before the final cut.

Runs as its own process (MLX can't share a process with PyTorch):
  python -m locaanimo.review checks.json      → JSON list of {"id", "ok", "issue"}
checks.json: [{"id": ..., "image": path, "question": ...}, ...]
"""

from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from PIL import Image

from .landmarks import LLM_DIR

ANSWER = (' Answer with JSON only: {"ok": true} if it looks right, or {"ok": false, "issue": "<short description>"} '
          "if there is a clear, obvious problem. Small stylistic imperfections are fine.")

QUESTIONS = {
    "mouth": "This is an anime character's face with an open, talking mouth drawn on it. "
             "Is the open mouth placed naturally where a mouth belongs (below the nose, centred on the face)?",
    "blink": "This is an anime character's face that should have both eyes closed (blinking). "
             "Are both eyes clearly closed, with no smudges, leftover pupils or broken glasses?",
    "clip": "This is a frame from an AI-animated cartoon action shot. Do the characters look intact, "
            "without melted, merged or badly deformed faces and bodies?",
    "plate": "This should be an empty background for a cartoon. Is it free of people, characters and text?",
    "frame": "This is a frame from an animated cartoon episode. Is anything clearly broken: a mouth in the "
             "wrong place, a deformed face, a character cut off strangely or floating, or garbled visuals?",
}


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            return {"ok": bool(data.get("ok", True)), "issue": str(data.get("issue", "")).strip()}
        except json.JSONDecodeError:
            pass
    lowered = text.lower()
    return {"ok": not ("false" in lowered or "no," in lowered), "issue": text.strip()[:120]}


def review(checks: list[dict]) -> list[dict]:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    model, proc = load(str(LLM_DIR))
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "locaanimo_review.png")
    results = []
    for c in checks:
        img = Image.open(c["image"]).convert("RGB")
        img.thumbnail((640, 640))
        img.save(tmp)
        q = (c.get("question") or QUESTIONS[c["kind"]]) + ANSWER
        prompt = apply_chat_template(proc, model.config, q, num_images=1, enable_thinking=False)
        out = generate(model, proc, prompt, image=[tmp], max_tokens=60, verbose=False, temperature=0.0)
        results.append({"id": c["id"], **_parse(getattr(out, "text", out))})
    return results


if __name__ == "__main__":
    print(json.dumps(review(json.loads(open(sys.argv[1]).read()))))
