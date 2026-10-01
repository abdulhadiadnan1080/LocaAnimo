"""Draw the art for the 30-second animation test (example script) into projects/test/art/."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locaanimo.art import Artist, action_prompt, character_negative, character_prompt, location_prompt, seed_for  # noqa: E402
from locaanimo.paths import EXAMPLES, PROJECTS  # noqa: E402
from locaanimo.schema import load_script  # noqa: E402

script = load_script(EXAMPLES / "script_template.yaml")
style = script.episode.style.value
out = PROJECTS / "test" / "art_v2"
out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
artist = Artist()
print(f"loaded in {time.time() - t0:.0f}s", flush=True)


def job(name, fn):
    path = out / f"{name}.png"
    if path.exists():
        print(f"skip {name}", flush=True)
        return
    t = time.time()
    fn().save(path)
    print(f"{name}: {time.time() - t:.0f}s", flush=True)


# Backgrounds: one plate per location, plus the scene-2 ocean angle.
beach = next(l for l in script.locations if l.id == "beach")
job("bg_beach", lambda: artist.draw(location_prompt(beach.look, "sunset", style), seed_for("beach"), 1344, 768,
                                    negative="1girl, 1boy, people, person"))
job("bg_ocean", lambda: artist.draw(location_prompt("open ocean, huge waves rolling in, sea spray, far horizon", "sunset", style),
                                    seed_for("ocean"), 1344, 768, negative="1girl, 1boy, people, person"))

# Characters: a base sprite, then open-mouth and closed-eyes variants of the same drawing.
for c in script.characters:
    seed = seed_for(c.id)
    job(f"{c.id}_base", lambda c=c, seed=seed: artist.draw(character_prompt(c.look, c.gender, style, age=c.age), seed, 832, 1216,
                                                                       negative=character_negative(c.age)))
    base = None
    for variant, expr in (("talk", "open mouth, talking"), ("blink", "closed eyes, closed mouth, light smile")):
        def make(c=c, seed=seed, expr=expr):
            from PIL import Image
            img = Image.open(out / f"{c.id}_base.png").convert("RGB")
            return artist.redraw(img, character_prompt(c.look, c.gender, style, expr, age=c.age), seed, strength=0.42,
                                 negative=character_negative(c.age))
        job(f"{c.id}_{variant}", make)

# The big action moment.
kai = next(c for c in script.characters if c.id == "kai")
job("action_wave", lambda: artist.draw(
    action_prompt("1boy, solo, teenager, surfing, riding a giant wave, surfboard, splash, from below", [kai.look], "sunset", style),
    seed_for("action_wave"), 1344, 768, negative="child, shota, chibi"))

print(f"all art done in {time.time() - t0:.0f}s", flush=True)
