"""Animate an episode from its script and the art drawn by make_art.py.

  python scripts/render_episode.py examples/ember_protocol.yaml ember

Progress lines go to projects/<name>/log.txt; the video lands in outputs/ (and the Library).
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PIL import Image  # noqa: E402

from locaanimo import animate, prep, sfx  # noqa: E402
from locaanimo.paths import OUTPUTS, PROJECTS  # noqa: E402
from locaanimo.schema import ActionBeat, CameraBeat, LineBeat, NarrationBeat, PauseBeat, load_script  # noqa: E402
from locaanimo.voices import SAMPLE_RATE, cast_voices, kokoro_cast, synth_lines  # noqa: E402

script = load_script(sys.argv[1])
project = PROJECTS / sys.argv[2]
art = project / "art"
log_file = open(project / "log.txt", "a")
t0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:5.0f}s] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


# 1. Real animation for every big action beat (LTX-Video), cached in clips/. Two
#     processes: text encoder, then video model (they don't fit in memory together).
from locaanimo import video  # noqa: E402

big_beats = [(s, b) for s in script.scenes for b in s.beats if isinstance(b, ActionBeat) and b.big]
clip_dir = project / "clips"
clip_files = [clip_dir / f"action_{n}.mp4" for n in range(1, len(big_beats) + 1)]
chars_by_id = {c.id: c for c in script.characters}
if big_beats and video.available():
    style_words = {"anime_tv": "anime", "chibi": "cute chibi anime", "cartoon": "cartoon"}[script.episode.style.value]
    jobs = []
    for n, ((s, b), out) in enumerate(zip(big_beats, clip_files), 1):
        if out.exists() or not (art / f"action_{n}.png").exists():
            continue
        who = "; ".join(f"{chars_by_id[c].name}: {chars_by_id[c].look}" for c in b.characters[:2])
        jobs.append({"prompt": f"{style_words} animation. {b.action}. {who}. Dynamic, smooth motion, "
                               f"consistent characters, {s.time} lighting.",
                     "image": str(art / f"action_{n}.png"), "embeds": str(clip_dir / f"action_{n}.pt"), "out": str(out)})
    if jobs:
        clip_dir.mkdir(exist_ok=True)
        (clip_dir / "jobs.json").write_text(json.dumps(jobs, indent=2))
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        for phase in ("encode", "animate"):
            log(f"render: AI action clips, {phase} ({len(jobs)} clip(s))")
            proc = subprocess.Popen([sys.executable, "-m", "locaanimo.video", phase, str(clip_dir / "jobs.json")],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
            for line in proc.stdout:
                if line.startswith("video:"):
                    log(f"render: {line.strip()}")
            if proc.wait() != 0:
                log(f"render: AI clips failed during {phase}; big moments will use stills")
                break
elif big_beats:
    log("render: LTX-Video not installed; big moments will use stills")

# 1b. Find every character's mouth and eyes with the vision model (own process: MLX),
#    cached so re-renders skip it.
lm_file = project / "landmarks.json"
landmarks = json.loads(lm_file.read_text()) if lm_file.exists() else {}
landmarks = {k: v for k, v in landmarks.items() if v}          # failed lookups get retried
missing = [str(art / f"char_{c.id}.png") for c in script.characters if str(art / f"char_{c.id}.png") not in landmarks]
if missing:
    log(f"render: locating mouths and eyes for {len(missing)} character(s)")
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-m", "locaanimo.landmarks", *missing],
                            capture_output=True, text=True, env=env, timeout=600)
    if result.returncode == 0 and result.stdout.strip():
        found = json.loads(result.stdout.strip().splitlines()[-1])
        landmarks.update({k: v for k, v in found.items() if v})
        for k, v in found.items():
            if not v:
                log(f"render: vision model couldn't find the face in {Path(k).name}; using pixel heuristics")
        lm_file.write_text(json.dumps(landmarks, indent=2))
    else:
        log("render: vision model failed, falling back to pixel heuristics")

# 1c. Closed-eye drawings for blinking, painted by the image model in the character's own
#     style (only the eyes are repainted). Cached in art/char_<id>_blink.png.
need_blink = [c for c in script.characters if not (art / f"char_{c.id}_blink.png").exists()
              and {"left_eye", "right_eye"} <= (landmarks.get(str(art / f"char_{c.id}.png")) or {}).keys()]
if need_blink:
    import torch  # noqa: E402
    from locaanimo.art import Artist, character_negative, character_prompt, seed_for  # noqa: E402
    log(f"render: painting closed eyes for {len(need_blink)} character(s)")
    artist = Artist()
    for c in need_blink:
        base = Image.open(art / f"char_{c.id}.png").convert("RGB")
        eye_mask = prep.eye_mask(base, landmarks[str(art / f"char_{c.id}.png")])
        blink = artist.inpaint(base, eye_mask, character_prompt(c.look, c.gender, script.episode.style.value,
                               "closed eyes, eyes closed, light smile", age=c.age), seed_for(c.id) + 1,
                               negative=character_negative(c.age, c.look) + ", open eyes")
        blink.save(art / f"char_{c.id}_blink.png")
        log(f"render: {c.name}: closed eyes painted")
    del artist
    torch.mps.empty_cache()

# 2. Characters → puppets with drawn mouths and blinks.
def run_review(checks: list[dict]) -> dict:
    """Ask the vision model about each image; {id: {"ok", "issue"}}. Empty if it can't run."""
    if not checks:
        return {}
    review_dir.mkdir(exist_ok=True)
    (review_dir / "checks.json").write_text(json.dumps(checks, indent=2))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    result = subprocess.run([sys.executable, "-m", "locaanimo.review", str(review_dir / "checks.json")],
                            capture_output=True, text=True, env=env, timeout=900)
    if result.returncode != 0 or not result.stdout.strip():
        log("render: review couldn't run; continuing without it")
        return {}
    return {r["id"]: r for r in json.loads(result.stdout.strip().splitlines()[-1])}


BEARD = re.compile(r"\b(beard|bearded|mustache|moustache|goatee|whiskers)\b", re.I)


def bearded(c) -> bool:
    """A beard or mustache hides the mouth: such characters talk by moving the beard."""
    return bool(BEARD.search(c.look))


def mouth_below_eyes(eyes, mouth) -> bool:
    """Geometry check, no AI needed: a mouth sits clearly below the eyes (0.4 to 1.7
    eye-distances lower) and between them sideways. Catches mouths on brows, noses, clothes."""
    if not eyes or len(eyes) != 2:
        return True
    (ax, ay), (bx, by) = [(x + w / 2, y + h / 2) for x, y, w, h in eyes]
    dist = max(1.0, abs(bx - ax))
    drop = (mouth["y"] - (ay + by) / 2) / dist
    return 0.4 <= drop <= 1.7 and abs(mouth["x"] - (ax + bx) / 2) <= 0.6 * dist


def build_character(c, base, alpha, mode: str):
    """mode: "vision" (Qwen's face points), "pixels" (heuristics) or "still" (no lip flap).
    Returns (face, the mode actually used): a misplaced mouth steps down to the next mode."""
    lm = landmarks.get(str(art / f"char_{c.id}.png"))
    if mode == "vision" and lm and {"mouth", "left_eye", "right_eye"} <= lm.keys():
        face, eyes, mouth = prep.parts_from_landmarks(base, lm)
    elif lm and {"left_eye", "right_eye"} <= lm.keys():
        # the vision model's face and eyes are still right when only its mouth was rejected:
        # search for the mouth line below those eyes, never anywhere else on the body
        face, eyes, _ = prep.parts_from_landmarks(base, lm)
        mouth = prep.find_mouth(base, face, eyes)
    else:
        face = prep.anime_face(base, alpha)
        if face is None:
            ys, xs = np.where(alpha > 0.5)
            top, h = ys.min(), ys.max() - ys.min()
            size = int(h * 0.3)
            face = (int(xs[ys < top + size].mean() - size / 2), int(top), size, size)
        eyes = prep.find_eyes(base, face)
        mouth = prep.find_mouth(base, face, eyes)
    if mode != "still" and not mouth_below_eyes(eyes, mouth):
        nxt = {"vision": "pixels", "pixels": "still"}[mode]
        report.append(f"{c.name}: mouth wasn't below the eyes in {mode} mode, switching to {nxt}")
        log(f"render: review: {report[-1]}")
        return build_character(c, base, alpha, nxt)
    if mode == "still":
        states = {"base": base, "half": base, "talk": base}
    elif bearded(c):
        if eyes and len(eyes) == 2:
            eye_dist = abs((eyes[0][0] + eyes[0][2] / 2) - (eyes[1][0] + eyes[1][2] / 2))
        else:
            eye_dist = 0.4 * face[2]
        mouths = prep.beard_states(base, mouth, eye_dist)
        states = {"base": mouths["closed"], "half": mouths["half"], "talk": mouths["open"]}
    else:
        mouths = prep.mouth_states(base, mouth)
        states = {"base": mouths["closed"], "half": mouths["half"], "talk": mouths["open"]}
    painted = art / f"char_{c.id}_blink.png"
    if painted.exists() and lm and c.id not in no_blink:      # model-painted closed eyes on every mouth shape
        mask = prep.eye_mask(base, lm)
        for name in list(states):
            states[f"{name}_blink" if name != "base" else "blink"] = Image.composite(
                Image.open(painted).convert("RGB"), states[name], mask)
    for name, img in states.items():
        img.save(project / f"sprite_{c.id}_{name}.png")
    sprites[c.id] = animate.make_sprite(states, alpha, (face[0] + face[2] / 2, face[1] + face[3] * 0.55))
    return face, mode


log("render: preparing characters")
review_dir = project / "review"
cutter = prep.Cutter()
sprites, faces, bases, mode, no_blink, report = {}, {}, {}, {}, set(), []
for c in script.characters:
    base = Image.open(art / f"char_{c.id}.png").convert("RGB")
    bases[c.id] = (base, cutter.alpha(base))
    mode[c.id] = "vision" if landmarks.get(str(art / f"char_{c.id}.png")) else "pixels"
    faces[c.id], mode[c.id] = build_character(c, *bases[c.id], mode[c.id])
    log(f"render: {c.name}: mouth placed by {mode[c.id]}")
del cutter


def face_crop(cid: str, state: str) -> str:
    fx, fy, fw, fh = faces[cid]
    pad = int(fw * 0.45)
    out = review_dir / f"{cid}_{state}.png"
    review_dir.mkdir(exist_ok=True)
    Image.open(project / f"sprite_{cid}_{state}.png").crop((fx - pad, fy - pad, fx + fw + pad, fy + fh + pad)).save(out)
    return str(out)


# 2b. Review: does every open mouth and every blink look right? Fix what doesn't.
log("render: review: checking mouths and blinks")
for attempt in range(2):
    checks = [{"id": f"mouth:{c.id}", "kind": "mouth", "image": face_crop(c.id, "talk")}
              for c in script.characters if mode[c.id] != "still" and not bearded(c)]   # beards: no drawn mouth
    if attempt == 0:
        checks += [{"id": f"blink:{c.id}", "kind": "blink", "image": face_crop(c.id, "blink")}
                   for c in script.characters if (project / f"sprite_{c.id}_blink.png").exists()
                   and "blink" in sprites[c.id].states]
    results = run_review(checks)
    redo = []
    for c in script.characters:
        r = results.get(f"blink:{c.id}")
        if r and not r["ok"]:
            no_blink.add(c.id)
            redo.append(c)
            report.append(f"{c.name}: blink rejected ({r['issue']}), blinking turned off")
            log(f"render: review: {report[-1]}")
        r = results.get(f"mouth:{c.id}")
        if r and not r["ok"]:
            nxt = {"vision": "pixels", "pixels": "still"}[mode[c.id]]
            report.append(f"{c.name}: mouth rejected ({r['issue']}), switching to {nxt}")
            log(f"render: review: {report[-1]}")
            mode[c.id] = nxt
            if c not in redo:
                redo.append(c)
    if not redo:
        break
    for c in redo:
        faces[c.id], mode[c.id] = build_character(c, *bases[c.id], mode[c.id])
log(f"render: review: characters OK ({', '.join(f'{c.name}={mode[c.id]}' for c in script.characters)})")

# 3. Backgrounds and action illustrations → depth-parallax plates.
log("render: depth maps")
depth = prep.Depth()
plate = lambda p: animate.load_plate(Image.open(p), depth.map(Image.open(p)))  # noqa: E731
plates = {p.stem.removeprefix("bg_"): plate(p) for p in sorted(art.glob("bg_*.png"))}
heroes = [plate(p) for p in sorted(art.glob("action_*.png"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))]
del depth
log(f"render: {len(plates)} plates, {len(heroes)} action shots")

# 4. Voices, sound effects and ambience.
log("render: recording voices")
cast, narrator = cast_voices(script.characters, script.episode.narrator_voice)
fast_cast, fast_narrator = kokoro_cast(script.characters, script.episode.narrator_voice)
spoken = []                                   # every line and narration, in episode order
for scene in script.scenes:
    for beat in scene.beats:
        if isinstance(beat, LineBeat):
            spoken.append({"text": beat.line.text, "voice": cast[beat.line.who],
                           "emotion": beat.line.emotion.value, "fallback": fast_cast[beat.line.who]})
        elif isinstance(beat, NarrationBeat):
            spoken.append({"text": beat.narration, "voice": narrator, "emotion": f"narration:{scene.mood}",
                           "fallback": fast_narrator})
t_voice = time.time()
recorded = iter(synth_lines(spoken, project, log))
log(f"render: {len(spoken)} lines recorded in {time.time() - t_voice:.0f}s")
locs = {l.id: l for l in script.locations}
clips, voice_track, events, scene_spans = [], [], [], []
big_seen = 0
t = 0.0
lead = int(0.15 * SAMPLE_RATE)
for scene in script.scenes:
    scene_start = t
    for beat in scene.beats:
        audio = None
        if isinstance(beat, LineBeat):
            audio = next(recorded)
            dur = len(audio) / SAMPLE_RATE + 0.45
        elif isinstance(beat, NarrationBeat):
            audio = next(recorded)
            dur = len(audio) / SAMPLE_RATE + 0.5
        elif isinstance(beat, ActionBeat):
            dur = 3.4 if beat.big else 2.6
            if beat.big:
                big_seen += 1
                if clip_files[big_seen - 1].exists():   # real animation: the shot lasts as long as its clip
                    dur = video.FRAMES / video.FPS + 0.4
            for name in beat.sfx:
                if name in sfx.EFFECTS:
                    events.append((t, sfx.EFFECTS[name]()))
            if beat.big:
                events.append((t, sfx.whoosh()))
                loud = any(w in beat.action.lower() for w in ("blast", "explo", "fire", "light"))
                events.append((t + 0.15, sfx.blast() if loud else sfx.impact()))
        elif isinstance(beat, CameraBeat):
            dur = 2.2
        else:
            dur = beat.pause
        clip = np.zeros(int(dur * SAMPLE_RATE), np.float32)
        if audio is not None:
            n = min(len(audio), len(clip) - lead)
            clip[lead:lead + n] = audio[:n]
        clips.append((clip if audio is not None else None, dur))
        voice_track.append(clip)
        t += dur
    scene_spans.append((scene_start, t, locs[scene.location].look))
total = t
mix = np.concatenate(voice_track)
for start, end, look in scene_spans:
    if amb := sfx.ambience_for(look):
        bed = amb(end - start + 0.5)
        i = int(start * SAMPLE_RATE)
        mix[i:i + len(bed)] += bed[: len(mix) - i]
for start, fx in events:
    i = int(start * SAMPLE_RATE)
    mix[i:i + len(fx)] += fx[: len(mix) - i]
mix = np.tanh(mix * 1.1) / np.tanh(1.1)
sf.write(project / "mix.wav", mix, SAMPLE_RATE)
log(f"render: audio ready, {total:.1f}s long")

# 5. Shots → frames → video.
clip_checks = []
for n, f in enumerate(clip_files, 1):
    if f.exists():
        for at in (1.2, 2.4):
            img = review_dir / f"clip{n}_{at}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", str(f), "-frames:v", "1", str(img)])
            clip_checks.append({"id": f"clip:{n}:{at}", "kind": "clip", "image": str(img)})
log(f"render: review: checking {len(clip_checks)} frames of AI clips")
clip_results = run_review(clip_checks)
bad_clips = {int(k.split(":")[1]) for k, r in clip_results.items() if not r["ok"]}
for n in sorted(bad_clips):
    issue = next(r["issue"] for k, r in clip_results.items() if k.startswith(f"clip:{n}:") and not r["ok"])
    report.append(f"AI clip {n} rejected ({issue}), using the still instead")
    log(f"render: review: {report[-1]}")
anim_clips = [video.read_video(f, (animate.W, animate.H)) if f.exists() and n not in bad_clips else None
              for n, f in enumerate(clip_files, 1)]
log(f"render: {sum(c is not None for c in anim_clips)} AI action clip(s) loaded")
assets = animate.Assets(plates, {}, heroes, sprites, anim_clips)
shots = animate.plan(script, assets, clips)
renderer = animate.Renderer(assets)

# 5a. Preview: one key frame per shot, reviewed before the final cut.
frame_checks = []
for i, shot in enumerate(shots):
    at = shot.start + shot.dur * (0.5 if shot.clip is None else 0.4)
    if shot.speaker and shot.audio is not None:     # pick a moment with the mouth open
        track = animate.mouth_track(shot.audio)
        at = shot.start + next((k / 12 for k, m in enumerate(track) if m == "talk"), shot.dur * 0.5)
    img = review_dir / f"shot_{i:02d}.png"
    Image.fromarray((renderer.frame(shot, at).clip(0, 1) * 255).astype(np.uint8)).save(img)
    frame_checks.append({"id": f"shot:{i}", "kind": "frame", "image": str(img)})
log(f"render: review: checking {len(frame_checks)} preview frames")
frame_results = run_review(frame_checks)
flagged = [(int(k.split(":")[1]), r["issue"]) for k, r in frame_results.items() if not r["ok"]]
# Whole-frame flags are reported, not acted on: in testing they were mostly false alarms,
# while the close-up mouth and blink checks above (which do fix things) proved reliable.
for i, issue in flagged:
    report.append(f"shot {i + 1} (for a human to check): {issue}")
    log(f"render: review: shot {i + 1} flagged for a look: {issue[:120]}")
if not flagged:
    log("render: review: all preview frames look right")
(review_dir / "report.json").write_text(json.dumps({"changes": report, "flagged_shots": flagged}, indent=2))
OUTPUTS.mkdir(exist_ok=True)
slug = re.sub(r"[^a-z0-9]+", "-", script.episode.title.lower()).strip("-")
out = OUTPUTS / f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
log(f"render: {len(shots)} shots, encoding {out.name}")
step = {"next": 0.1}


def progress(p: float) -> None:
    if p >= step["next"]:
        log(f"render: frames {p:.0%}")
        step["next"] += 0.1


animate.render(shots, renderer, project / "mix.wav", out, total, on_progress=progress)
big = next((s for s in shots if s.speedlines), shots[len(shots) // 2])
thumb = renderer.frame(big, big.start + big.dur * 0.6)
Image.fromarray((thumb.clip(0, 1) * 255).astype(np.uint8)).resize((640, 360)).save(out.with_suffix(".jpg"), quality=88)
out.with_suffix(".json").write_text(json.dumps({
    "title": script.episode.title, "style": script.episode.style.value, "duration": round(total, 2),
    "scenes": len(script.scenes), "created": time.time(), "kind": "animation",
    "review": report or ["Everything passed review."]}, indent=2))
log(f"render: ALL DONE → {out.name}")
