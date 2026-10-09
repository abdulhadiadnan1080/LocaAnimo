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
    # The mouth check is a "where is it?" question (see MOUTH_SPOT): a small model answers
    # that far more reliably than "is it placed naturally?", which it got wrong both ways.
    "blink": "This is an anime character's face that should have both eyes closed (blinking). "
             "Are both eyes clearly closed, with no smudges, leftover pupils or broken glasses?",
    "clip": "This is a frame from an AI-animated cartoon action shot. Do the characters look intact, "
            "without melted, merged or badly deformed faces and bodies?",
    "plate": "This should be an empty background for a cartoon. Is it free of people, characters and text?",
    "frame": "This is a frame from an animated cartoon episode. The style is intentional: flat 2D cut-out "
             "characters over painted backgrounds, glasses, beards and stylised anime faces are all fine. "
             "Is anything clearly broken: a deformed or melted face, a body part missing or duplicated, "
             "a character cut off strangely, or garbled visuals?",
}


MOUTH_SPOT = ("This picture shows a cartoon character with a small open, talking mouth (dark red or pink) "
              "drawn on it. Where on the character is that open mouth drawn? Pick one: "
              "\"below_nose\" (on the lower face, under the nose, possibly inside a beard or under a mustache), "
              "\"nose\", \"eye\", \"eyebrow\", \"forehead\", \"cheek\", \"chin_below_face\", \"neck\", "
              "\"clothes\", \"hand\", \"background\". "
              'Reply with JSON only: {"spot": "<one of the above>"}')
MOUTH_OK = {"below_nose", "chin_below_face"}   # "chin" is how it often describes a correct, low-drawn mouth


def _parse_spot(text: str) -> dict:
    match = re.search(r'"spot"\s*:\s*"([a-z_]+)"', text)
    spot = match.group(1) if match else "unclear"
    if spot in MOUTH_OK:
        return {"ok": True, "issue": ""}
    return {"ok": False, "issue": f"the mouth is drawn on the {spot.replace('_', ' ')}" if match
            else f"couldn't tell where the mouth is ({text.strip()[:80]})"}


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
        spot = c["kind"] == "mouth" and not c.get("question")
        q = MOUTH_SPOT if spot else (c.get("question") or QUESTIONS[c["kind"]]) + ANSWER
        prompt = apply_chat_template(proc, model.config, q, num_images=1, enable_thinking=False)
        out = generate(model, proc, prompt, image=[tmp], max_tokens=140, verbose=False, temperature=0.0)
        text = getattr(out, "text", out)
        results.append({"id": c["id"], **(_parse_spot(text) if spot else _parse(text))})
    return results


if __name__ == "__main__":
    print(json.dumps(review(json.loads(open(sys.argv[1]).read()))))
