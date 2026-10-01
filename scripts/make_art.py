"""Draw all art for an episode script.

  python scripts/make_art.py examples/ember_protocol.yaml ember

Writes projects/<name>/art/: bg_<location>@<time>.png per (location, time) used,
char_<id>.png per character, action_<n>.png per big action beat. Existing files are
kept (re-runs only draw what's missing). Progress lines go to projects/<name>/log.txt.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locaanimo.art import (Artist, action_prompt, character_negative, character_prompt,  # noqa: E402
                            location_prompt, seed_for)
from locaanimo.paths import PROJECTS  # noqa: E402
from locaanimo.schema import ActionBeat, load_script  # noqa: E402

script = load_script(sys.argv[1])
project = PROJECTS / sys.argv[2]
out = project / "art"
out.mkdir(parents=True, exist_ok=True)
log_file = open(project / "log.txt", "a")
t0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:5.0f}s] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


style = script.episode.style.value
chars = {c.id: c for c in script.characters}
locs = {l.id: l for l in script.locations}

jobs = []
for key in dict.fromkeys(f"{s.location}@{s.time}" for s in script.scenes):
    loc, tod = key.split("@")
    jobs.append((f"bg_{key}", lambda loc=loc, tod=tod: artist.draw(
        location_prompt(locs[loc].look, tod, style), seed_for(loc), 1344, 768, negative="1girl, 1boy, people, person")))
for c in script.characters:
    jobs.append((f"char_{c.id}", lambda c=c: artist.draw(
        character_prompt(c.look, c.gender, style, age=c.age), seed_for(c.id), 832, 1216,
        negative=character_negative(c.age))))
n = 0
for s in script.scenes:
    for b in s.beats:
        if isinstance(b, ActionBeat) and b.big:
            n += 1
            looks = [f"{chars[c].look}" for c in b.characters[:2]]
            jobs.append((f"action_{n}", lambda b=b, s=s, looks=looks, n=n: artist.draw(
                action_prompt(b.action, looks, s.time, style), seed_for(f"action_{n}"), 1344, 768,
                negative="child, shota, loli, chibi")))

todo = [(name, fn) for name, fn in jobs if not (out / f"{name}.png").exists()]
log(f"art: {len(jobs)} images planned, {len(todo)} to draw")
if todo:
    artist = Artist()
    log("art: model loaded")
for i, (name, fn) in enumerate(todo, 1):
    t = time.time()
    log(f"art: drawing {name} ({i}/{len(todo)})")
    fn().save(out / f"{name}.png")
    log(f"art: done {name} in {time.time() - t:.0f}s")
log("art: ALL DONE")
