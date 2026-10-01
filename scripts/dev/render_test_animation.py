"""Render the 30-second 720p animation test from the example script and the art in projects/test/art/."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from PIL import Image  # noqa: E402

from locaanimo import animate, prep, sfx  # noqa: E402
from locaanimo.paths import EXAMPLES, OUTPUTS, PROJECTS  # noqa: E402
from locaanimo.schema import ActionBeat, CameraBeat, LineBeat, NarrationBeat, PauseBeat, load_script  # noqa: E402
from locaanimo.voices import EMOTION_SPEED, SAMPLE_RATE, cast_voices, engine  # noqa: E402

t0 = time.time()
script = load_script(EXAMPLES / "script_template.yaml")
art = PROJECTS / "test" / "art_v2"
work = PROJECTS / "test"
lap = lambda msg: print(f"[{time.time() - t0:5.1f}s] {msg}", flush=True)  # noqa: E731

# 1. Prepare art: cutouts, face points, expression variants, depth maps.
cutter = prep.Cutter()
sprites, faces = {}, {}
for c in script.characters:
    base = Image.open(art / f"{c.id}_base.png").convert("RGB")
    alpha = cutter.alpha(base)
    face = prep.anime_face(base, alpha)
    if face is None:  # fall back to the top of the silhouette
        ys, xs = np.where(alpha > 0.5)
        top, h = ys.min(), ys.max() - ys.min()
        size = int(h * 0.3)
        face = (int(xs[ys < top + size].mean() - size / 2), int(top), size, size)
        lap(f"{c.id}: no anime face detected, guessing from the silhouette")
    eyes = prep.find_eyes(base, face)
    mouth = prep.find_mouth(base, face, eyes)
    lap(f"{c.id}: face {face}, {len(eyes)} eye(s), mouth {'found' if mouth['found'] else 'guessed'}")
    mouths = prep.mouth_states(base, mouth)            # closed / half / open, drawn
    states = {"base": mouths["closed"], "half": mouths["half"], "talk": mouths["open"]}
    for name in list(states):                          # every mouth also gets a blink version
        blinked = prep.blink_state(states[name], eyes, face)
        if blinked is not None:
            states[f"{name}_blink" if name != "base" else "blink"] = blinked
    for name, img in states.items():
        img.save(work / f"sprite_{c.id}_{name}.png")
    head = (face[0] + face[2] / 2, face[1] + face[3] * 0.55)
    sprites[c.id] = animate.make_sprite(states, alpha, head)
del cutter
lap("cutouts + expressions ready")

depth = prep.Depth()
plates = {"beach": animate.load_plate(Image.open(art / "bg_beach.png"), depth.map(Image.open(art / "bg_beach.png")))}
vistas = {"beach": animate.load_plate(Image.open(art / "bg_ocean.png"), depth.map(Image.open(art / "bg_ocean.png")))}
heroes = [animate.load_plate(Image.open(art / "action_wave.png"), depth.map(Image.open(art / "action_wave.png")))]
del depth
lap("depth maps ready")

# 2. Record voices and lay out the audio timeline (voice + sound effects + ambience).
cast, narrator = cast_voices(script.characters, script.episode.narrator_voice)
clips, voice_track, sfx_events = [], [], []
t = 0.0
lead = int(0.15 * SAMPLE_RATE)
for scene in script.scenes:
    for beat in scene.beats:
        audio = None
        if isinstance(beat, LineBeat):
            audio = engine.synth(beat.line.text, cast[beat.line.who], EMOTION_SPEED[beat.line.emotion.value])
            dur = len(audio) / SAMPLE_RATE + 0.45
        elif isinstance(beat, NarrationBeat):
            audio = engine.synth(beat.narration, narrator)
            dur = len(audio) / SAMPLE_RATE + 0.5
        elif isinstance(beat, ActionBeat):
            dur = 3.4 if beat.big else 2.6
            for name in beat.sfx:
                if name in sfx.EFFECTS:
                    sfx_events.append((t, sfx.EFFECTS[name]()))
            if beat.big:
                sfx_events.append((t, sfx.whoosh()))
                sfx_events.append((t + 0.2, sfx.wave_roar(3.0, seed=9)))
        elif isinstance(beat, CameraBeat):
            dur = 2.2
        elif isinstance(beat, PauseBeat):
            dur = beat.pause
        clip = np.zeros(int(dur * SAMPLE_RATE), np.float32)
        if audio is not None:
            n = min(len(audio), len(clip) - lead)
            clip[lead:lead + n] = audio[:n]
        clips.append((clip if audio is not None else None, dur))
        voice_track.append(clip)
        t += dur
total = t
mix = np.concatenate(voice_track)
mix += sfx.ocean(total)[: len(mix)]
for start, fx in sfx_events:
    i = int(start * SAMPLE_RATE)
    mix[i:i + len(fx)] += fx[: len(mix) - i]
mix = np.tanh(mix * 1.1) / np.tanh(1.1)  # soft limiter
sf.write(work / "mix.wav", mix, SAMPLE_RATE)
lap(f"audio ready: {total:.1f}s")

# 3. Plan shots and render frames.
assets = animate.Assets(plates, vistas, heroes, sprites)
shots = animate.plan(script, assets, clips)
renderer = animate.Renderer(assets)
OUTPUTS.mkdir(exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
out = OUTPUTS / f"animation-test-{stamp}.mp4"
animate.render(shots, renderer, work / "mix.wav", out, total,
               on_progress=lambda p: print(f"  rendering {p:4.0%}", flush=True) if int(p * 100) % 20 < 4 else None)
lap(f"rendered {out.name}")

# Library entry: thumbnail from the middle of the action shot.
big = next((s for s in shots if s.speedlines), shots[len(shots) // 2])
thumb = renderer.frame(big, big.start + big.dur * 0.6)
Image.fromarray((thumb.clip(0, 1) * 255).astype(np.uint8)).resize((640, 360)).save(out.with_suffix(".jpg"), quality=88)
out.with_suffix(".json").write_text(json.dumps({
    "title": "The Last Wave · animation test", "style": script.episode.style.value, "duration": round(total, 2),
    "scenes": len(script.scenes), "created": time.time(), "kind": "animation"}, indent=2))
lap("done")
