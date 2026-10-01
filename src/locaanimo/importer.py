"""Turn free text (a story, a screenplay, notes) into a structured episode script.

Runs as its own process (`python -m locaanimo.importer`) because MLX and PyTorch
can't share a process. Protocol: JSON request on stdin; JSON lines on stdout:
  {"progress": tokens_so_far, "expected": n}   while the model writes
  {"script": {...}, "notes": [...], "via": "llm"|"structured"}  or  {"error": "..."}
"""

from __future__ import annotations

import json
import os
import re
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import yaml
from pydantic import ValidationError

from .paths import MODELS
from .schema import Camera, Emotion, Gesture, Script, Style

LLM_DIR = MODELS / "llm" / "Qwen3.5-4B-4bit"

AGES = ["child", "teen", "adult", "elder"]
GENDERS = ["male", "female", "neutral"]
TIMES = ["day", "sunset", "night"]
MOODS = ["calm", "tense", "happy", "sad", "epic"]

SYNONYMS = {
    # emotions
    "excited": "happy", "joyful": "happy", "cheerful": "happy", "laughing": "happy", "glad": "happy",
    "afraid": "scared", "fearful": "scared", "terrified": "scared", "frightened": "scared",
    "shocked": "surprised", "startled": "surprised", "nervous": "worried", "anxious": "worried",
    "concerned": "worried", "furious": "angry", "annoyed": "angry", "mad": "angry", "upset": "sad",
    "crying": "sad", "tearful": "sad", "proud": "confident", "smug": "confident", "calm": "neutral",
    "serious": "determined", "resolute": "determined", "awed": "amazed", "impressed": "amazed",
    "shy": "embarrassed", "flustered": "embarrassed",
    # cameras
    "closeup": "close_up", "close": "close_up", "medium_shot": "medium", "wide_shot": "wide",
    "establishing_shot": "establishing", "low": "low_angle", "high": "high_angle",
    # times & moods
    "morning": "day", "afternoon": "day", "noon": "day", "evening": "sunset", "dusk": "sunset",
    "dawn": "sunset", "midnight": "night", "peaceful": "calm", "dramatic": "tense", "scary": "tense",
    "action": "epic", "joyful_mood": "happy", "melancholy": "sad",
    # ages & genders
    "kid": "child", "boy": "male", "girl": "female", "man": "male", "woman": "female",
    "teenager": "teen", "young": "teen", "old": "elder", "elderly": "elder", "middle_aged": "adult",
}

SYSTEM = f"""You convert a story or screenplay written in plain text into a JSON episode script for an anime generator.

Output ONLY one JSON object (no markdown, no commentary) with this shape:
{{
  "episode": {{"title": "...", "style": "anime_tv"}},
  "characters": [{{"id": "kai", "name": "Kai", "look": "...", "age": "teen", "gender": "male"}}],
  "locations": [{{"id": "beach", "look": "..."}}],
  "scenes": [{{"id": "s1", "location": "beach", "time": "sunset", "mood": "calm", "beats": [ ... ]}}],
  "notes": ["short notes about anything you had to guess or invent"]
}}

Each beat is exactly ONE of:
  {{"line": {{"who": "<character id>", "text": "...", "emotion": "<emotion>", "gesture": "<gesture>"}}}}   (gesture is optional)
  {{"narration": "..."}}
  {{"action": "...", "characters": ["<character id>", ...]}}   (add "big": true only for spectacular moments)
  {{"camera": "<camera>"}}
  {{"pause": <seconds>}}

Allowed values:
  style: {", ".join(m.value for m in Style)}
  age: {", ".join(AGES)}
  gender: {", ".join(GENDERS)}
  time: {", ".join(TIMES)}
  mood: {", ".join(MOODS)}
  emotion: {", ".join(m.value for m in Emotion)}
  gesture: {", ".join(m.value for m in Gesture)}
  camera: {", ".join(m.value for m in Camera)}

Rules:
- Keep every line of dialogue exactly as written and in the original order. Never invent dialogue.
- Stage directions and descriptions become action beats. Narrator or voice-over text becomes narration.
- ids are short lowercase words (letters, digits, underscores), e.g. "kai", "old_pier". Every "who" and every "characters" entry must be a character id; every scene "location" must be a location id.
- "look" is comma-separated visual tags (hair, eyes, clothes, build, age) taken from the text. If the text gives none, invent a short fitting look and say so in notes.
- A location "look" is comma-separated visual tags for the place, including its lighting.
- Infer age, gender, time and mood from the text; if unsure use adult, neutral, day, calm.
- Use "big": true for at most one spectacular action moment per two minutes of story (fights, chases, transformations, huge stunts).
- Start each scene with {{"camera": "establishing"}} unless the text asks for a specific shot.
"""

EXAMPLE_IN = """Title: Lost Kitten
Characters: Hana, a cheerful little girl in a yellow raincoat. Mr. Oda, an old shopkeeper with a grey beard.

Scene 1 - A rainy street at night.
Hana searches under a bench.
HANA (worried): Mochi? Where are you?
Mr. Oda steps out of his shop holding an umbrella.
MR. ODA: Looking for something, little one?"""

EXAMPLE_OUT = {
    "episode": {"title": "Lost Kitten", "style": "anime_tv"},
    "characters": [
        {"id": "hana", "name": "Hana", "look": "little girl, short brown hair, yellow raincoat, red boots, cheerful face", "age": "child", "gender": "female"},
        {"id": "oda", "name": "Mr. Oda", "look": "old man, grey beard, round glasses, brown shopkeeper apron", "age": "elder", "gender": "male"},
    ],
    "locations": [{"id": "rainy_street", "look": "narrow city street at night, heavy rain, glowing shop signs, wet pavement, wooden bench"}],
    "scenes": [{"id": "s1", "location": "rainy_street", "time": "night", "mood": "sad", "beats": [
        {"camera": "establishing"},
        {"action": "Hana searches under a bench", "characters": ["hana"]},
        {"line": {"who": "hana", "text": "Mochi? Where are you?", "emotion": "worried"}},
        {"action": "Mr. Oda steps out of his shop holding an umbrella", "characters": ["oda"]},
        {"line": {"who": "oda", "text": "Looking for something, little one?", "emotion": "neutral"}},
    ]}],
    "notes": ["Hana's hair and boots were not described, so I invented them."],
}


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


# --- cleanup: make whatever the model (or a user) wrote fit the schema ---------------

def _slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_") or "item"


def _unique(base: str, taken: set[str]) -> str:
    out, n = base, 2
    while out in taken:
        out, n = f"{base}_{n}", n + 1
    taken.add(out)
    return out


def _pick(value, allowed: list[str], default: str | None) -> str | None:
    if value is None:
        return default
    v = _slug(value)
    v = SYNONYMS.get(v, v)
    return v if v in allowed else default


def sanitize(data: dict) -> tuple[dict, list[str]]:
    notes = [str(n) for n in (data.get("notes") or []) if str(n).strip()]
    ep_in = data.get("episode") or {}
    episode = {"title": str(ep_in.get("title") or "Untitled episode").strip()}
    episode["style"] = _pick(ep_in.get("style"), [m.value for m in Style], "anime_tv")
    for key in ("resolution", "music", "subtitles", "narrator_voice"):
        if key in ep_in:
            episode[key] = ep_in[key]

    # characters, with every way the text might refer to them
    taken: set[str] = set()
    characters, alias = [], {}
    for c in data.get("characters") or []:
        if not isinstance(c, dict) or not (c.get("name") or c.get("id")):
            continue
        name = str(c.get("name") or c.get("id")).strip()
        cid = _unique(_slug(c.get("id") or name), taken)
        characters.append({
            "id": cid, "name": name, "look": str(c.get("look") or "").strip(),
            "age": _pick(c.get("age"), AGES, "adult"), "gender": _pick(c.get("gender"), GENDERS, "neutral"),
            "voice": str(c.get("voice") or "auto"),
        })
        for key in {cid, _slug(name), _slug(name.split()[0]), _slug(name.split()[-1]), _slug(c.get("id") or "")}:
            alias.setdefault(key, cid)

    def who(ref) -> str:
        key = _slug(ref)
        if key in alias:
            return alias[key]
        for k, cid in alias.items():  # "mr_oda" vs "oda"
            if key and (key in k or k in key):
                return cid
        name = str(ref).strip().title() or "Someone"
        cid = _unique(key, taken)
        characters.append({"id": cid, "name": name, "look": "", "age": "adult", "gender": "neutral", "voice": "auto"})
        alias[key] = cid
        notes.append(f"Added {name} to the cast because they speak or act in the story.")
        return cid

    loc_taken: set[str] = set()
    locations, loc_alias = [], {}
    for l in data.get("locations") or []:
        if not isinstance(l, dict) or not (l.get("id") or l.get("look")):
            continue
        lid = _unique(_slug(l.get("id") or l.get("look"))[:30], loc_taken)
        locations.append({"id": lid, "look": str(l.get("look") or l.get("id")).strip()})
        loc_alias[_slug(l.get("id") or "")] = lid
        loc_alias[lid] = lid

    def where(ref) -> str:
        key = _slug(ref) if ref else ""
        if key in loc_alias:
            return loc_alias[key]
        if not key and locations:
            return locations[0]["id"]
        lid = _unique(key[:30] or "place", loc_taken)
        locations.append({"id": lid, "look": str(ref or "a simple background")})
        loc_alias[key] = lid
        return lid

    scenes = []
    for n, sc in enumerate(data.get("scenes") or [], 1):
        if not isinstance(sc, dict):
            continue
        beats = [b for b in (_beat(raw, who) for raw in sc.get("beats") or []) if b]
        if not beats:
            continue
        scenes.append({
            "id": f"s{len(scenes) + 1}", "location": where(sc.get("location")),
            "time": _pick(sc.get("time"), TIMES, "day"), "mood": _pick(sc.get("mood"), MOODS, "calm"),
            "beats": beats,
        })

    if not characters:
        characters.append({"id": "narrator", "name": "Narrator", "look": "", "age": "adult", "gender": "neutral", "voice": "auto"})
    if not locations:
        locations.append({"id": "place", "look": "a simple background"})
    return {"episode": episode, "characters": characters, "locations": locations, "scenes": scenes}, notes


def _beat(raw, who) -> dict | None:
    if not isinstance(raw, dict):
        return None
    camera = _pick(raw.get("camera"), [m.value for m in Camera], None)
    if isinstance(raw.get("line"), dict) or ("who" in raw and "text" in raw):
        ln = raw["line"] if isinstance(raw.get("line"), dict) else raw
        text = str(ln.get("text") or "").strip()
        if not text:
            return None
        line = {"who": who(ln.get("who") or "someone"), "text": text,
                "emotion": _pick(ln.get("emotion"), [m.value for m in Emotion], "neutral")}
        if g := _pick(ln.get("gesture"), [m.value for m in Gesture], None):
            line["gesture"] = g
        return {"line": line, **({"camera": camera} if camera else {})}
    if raw.get("narration"):
        return {"narration": str(raw["narration"]).strip(), **({"camera": camera} if camera else {})}
    if raw.get("action"):
        beat = {"action": str(raw["action"]).strip(),
                "characters": [who(c) for c in raw.get("characters") or []]}
        if raw.get("big"):
            beat["big"] = True
        if raw.get("sfx"):
            beat["sfx"] = [_slug(s) for s in raw["sfx"]] if isinstance(raw["sfx"], list) else [_slug(raw["sfx"])]
        if camera:
            beat["camera"] = camera
        return beat
    if "pause" in raw:
        try:
            return {"pause": min(10.0, max(0.1, float(raw["pause"])))}
        except (TypeError, ValueError):
            return None
    if camera:
        return {"camera": camera}
    return None


def _validated(data: dict) -> tuple[dict | None, list[str], str | None]:
    script, notes = sanitize(data)
    try:
        Script.model_validate(script)
    except ValidationError as e:
        return None, notes, "; ".join(err["msg"] for err in e.errors()[:6])
    return script, notes, None


# --- the model ---------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the model didn't return JSON")
    return json.loads(text[start:end + 1])


def _generate(model, tok, messages: list[dict], expected: int, max_tokens: int) -> str:
    from mlx_lm import stream_generate
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    out, n = [], 0
    for chunk in stream_generate(model, tok, prompt=prompt, max_tokens=max_tokens):
        out.append(chunk.text)
        n += 1
        if n % 16 == 0:
            emit({"progress": n, "expected": expected})
    return "".join(out)


def convert(text: str) -> None:
    # Already structured (YAML/JSON)? Then no model is needed.
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        data = None
    if isinstance(data, dict) and data.get("scenes"):
        script, notes, err = _validated(data)
        if script:
            return emit({"script": script, "notes": notes, "via": "structured"})

    if not LLM_DIR.exists():
        return emit({"error": "The story model (Qwen3.5-4B) isn't in models/llm/, so free text can't be read."})
    from mlx_lm import load
    model, tok = load(str(LLM_DIR))

    words = len(text.split())
    expected = int(words * 2.4 + 250)          # rough output length, for the progress bar
    max_tokens = min(24000, max(1500, expected * 2))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXAMPLE_IN},
        {"role": "assistant", "content": json.dumps(EXAMPLE_OUT)},
        {"role": "user", "content": text},
    ]

    problem = None
    for attempt in range(2):
        raw = _generate(model, tok, messages, expected, max_tokens)
        try:
            data = _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            problem = f"The reply wasn't valid JSON ({e})."
        else:
            script, notes, problem = _validated(data)
            if script:
                return emit({"script": script, "notes": notes, "via": "llm"})
        if attempt == 0:  # one retry, telling the model what went wrong
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": f"That had problems: {problem} Return the full corrected JSON only."}]
    emit({"error": f"Couldn't turn this text into a script: {problem}"})


def main() -> None:
    request = json.loads(sys.stdin.read() or "{}")
    try:
        convert(str(request.get("text") or ""))
    except Exception as e:  # report instead of crashing silently
        emit({"error": str(e)})


if __name__ == "__main__":
    main()
