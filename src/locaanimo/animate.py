"""Shot planner + compositor: script beats and prepared art → animated frames.

Camera moves over depth-parallaxed background plates; characters are puppets that
breathe, sway, walk in, bob while talking, lip-flap from their audio and blink.
Character animation is held on twos (12 drawings/s) like TV anime; the camera runs on ones.

The camera works like a TV-anime layout artist: most dialogue shots are held, moves are
slow and constant-speed (no stop-and-go easing inside a shot), shot sizes and move types
vary from cut to cut, and consecutive shots of the same framing continue the previous move
instead of snapping back and starting over.
"""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schema import ActionBeat, CameraBeat, LineBeat, NarrationBeat, PauseBeat, Script, strip_tags

FPS = 24
W, H = 1280, 720
WORLD_SCALE = 2.0          # background plates are upscaled 2x into "world" space
VIEW_FRACTION = 0.9        # at zoom 1 the camera sees 90% of the plate width (room to move)
CHAR_HEIGHT = 0.82         # cowboy-shot sprite height as a fraction of world height
ZOOM_RATE = 0.018          # gentle push/pull: relative zoom change per second
PAN_RATE = 0.022           # gentle pan: fraction of the plate width per second
MAX_ZOOM_MOVE = 0.14       # a move never zooms more than this in one shot
MAX_PAN_MOVE = 0.12        # ... or pans further than this fraction of the plate
INTENSE = ("angry", "scared", "surprised", "determined", "amazed")
GRADE = {"day": (1.0, 1.0, 1.0), "sunset": (1.05, 0.95, 0.86), "night": (0.72, 0.78, 0.98)}


# --- assets ------------------------------------------------------------------------

@dataclass
class Plate:
    rgb: np.ndarray            # float32 HxWx3 in [0, 1], world resolution
    depth: np.ndarray          # float32 HxW in [0, 1], 1 = near

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]


@dataclass
class Sprite:
    states: dict[str, np.ndarray]   # base / talk / blink / talk_blink → float32 RGBA, trimmed
    head: tuple[float, float]       # face center in sprite pixels


@dataclass
class Assets:
    plates: dict[str, Plate]                 # location id → plate
    vistas: dict[str, Plate]                 # location id → wide plate for shots without characters
    heroes: list[Plate]                      # illustrations for big action beats, in order
    sprites: dict[str, Sprite]
    clips: list = field(default_factory=list)  # LTX clips per big beat (frames [T,H,W,3]) or None


def load_plate(img: Image.Image, depth: np.ndarray) -> Plate:
    w, h = int(img.width * WORLD_SCALE), int(img.height * WORLD_SCALE)
    rgb = cv2.resize(np.asarray(img.convert("RGB"), np.float32) / 255, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return Plate(rgb.clip(0, 1), cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR))


def make_sprite(states: dict[str, Image.Image], alpha: np.ndarray, head: tuple[float, float]) -> Sprite:
    ys, xs = np.where(alpha > 0.05)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    out = {}
    for name, img in states.items():
        rgb = np.asarray(img.convert("RGB"), np.float32) / 255
        out[name] = np.dstack([rgb, alpha])[y0:y1, x0:x1].copy()
    return Sprite(out, (head[0] - x0, head[1] - y0))


# --- planning ----------------------------------------------------------------------

@dataclass
class OnStage:
    id: str
    x0: float                  # world x at shot start (feet center)
    x1: float                  # world x at shot end
    gait: str | None = None    # "walk" | "run" while moving


@dataclass
class Shot:
    start: float
    dur: float
    plate: Plate
    time: str
    cam0: tuple[float, float, float]           # (cx, cy, zoom) at start
    cam1: tuple[float, float, float]
    cast: list[OnStage] = field(default_factory=list)
    speaker: str | None = None
    audio: np.ndarray | None = None            # this shot's voice clip, for lip flap
    shake: float = 0.0
    shake_hold: bool = False                   # keep shaking the whole shot (else a jolt that settles)
    speedlines: bool = False
    flash: bool = False
    pop: bool = False                          # quick zoom pop at the start (surprise)
    subtitle: str | None = None
    subtitle_name: str | None = None
    parallax: float = 0.0                      # world px of near-vs-far drift over the shot
    clip: np.ndarray | None = None             # real animation for this shot (AI action clip)
    move: str = "hold"                         # hold | push | pull | pan | tilt | continue | dramatic
    framing: str = ""                          # what the shot frames, to continue matching shots


def _ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def _clamp_cam(plate: Plate, cx: float, cy: float, z: float) -> tuple[float, float, float]:
    pw, ph = plate.size
    vw = pw * VIEW_FRACTION / z
    vh = vw * H / W
    cx = min(max(cx, vw / 2 + pw * 0.02), pw - vw / 2 - pw * 0.02)
    cy = min(max(cy, vh / 2 + ph * 0.02), ph - vh / 2 - ph * 0.02)
    return cx, cy, z


def _ease_out(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


def _shake(t: float, local: float, amp: float, hold: bool) -> tuple[float, float]:
    """Handheld-style shake: a few low sine waves instead of per-frame random jumps, so it
    reads as camera weight, not video noise. Unless held, it is a jolt that settles."""
    env = 1.0 if hold else math.exp(-local / 0.7)
    x = math.sin(t * 31.0) * 0.6 + math.sin(t * 53.0 + 1.3) * 0.4
    y = math.sin(t * 37.0 + 0.7) * 0.6 + math.sin(t * 47.0 + 2.1) * 0.4
    return x * amp * env, y * amp * env


def _move(plate: Plate, cam: tuple[float, float, float], kind: str, dur: float, sign: int = 1,
          rate: float = 1.0) -> tuple[tuple, tuple]:
    """(start, end) camera for a constant-speed move of `kind` that keeps `cam` as the
    framing in the middle of the shot."""
    cx, cy, z = cam
    pw, ph = plate.size
    if kind == "push" or kind == "pull":
        dz = min(MAX_ZOOM_MOVE, ZOOM_RATE * rate * dur) / 2
        a, b = (z * (1 - dz), z * (1 + dz))
        if kind == "pull":
            a, b = b, a
        return _clamp_cam(plate, cx, cy, a), _clamp_cam(plate, cx, cy, b)
    if kind == "pan":
        d = min(MAX_PAN_MOVE, PAN_RATE * rate * dur) * pw / 2 * sign
        return _clamp_cam(plate, cx - d, cy, z), _clamp_cam(plate, cx + d, cy, z)
    if kind == "tilt":  # top to bottom: sky first, then the ground
        d = min(MAX_PAN_MOVE, PAN_RATE * rate * dur) * ph / 2 * sign
        return _clamp_cam(plate, cx, cy - d, z), _clamp_cam(plate, cx, cy + d, z)
    c = _clamp_cam(plate, cx, cy, z)
    return c, c


def _continue(prev: "Shot", plate: Plate, dur: float) -> tuple[tuple, tuple]:
    """Pick up exactly where the previous shot's camera ended, at the same speed, so a run of
    matching shots plays as one unbroken move instead of a jump back and a restart."""
    v = [(b - a) / max(1e-3, prev.dur) for a, b in zip(prev.cam0, prev.cam1)]
    c0 = prev.cam1
    end = [a + vi * dur for a, vi in zip(c0, v)]
    # keep the whole run gentle: stop adding zoom once it has pushed in a lot
    end[2] = min(end[2], c0[2] * (1 + MAX_ZOOM_MOVE / 2))
    return c0, _clamp_cam(plate, *end)


def _slots(n: int) -> list[float]:
    return {1: [0.5], 2: [0.34, 0.66], 3: [0.24, 0.5, 0.76]}.get(n, [0.15 + 0.7 * i / max(1, n - 1) for i in range(n)])


def plan(script: Script, assets: Assets, clips: list[tuple[np.ndarray | None, float]]) -> list[Shot]:
    """One shot per beat (pauses extend the previous shot). `clips` gives each beat's
    (audio, duration) in script order."""
    shots: list[Shot] = []
    t = 0.0
    rng = random.Random(script.episode.title)       # same script → same camera choices
    audio_iter = iter(clips)
    hero_iter = iter(assets.heroes)
    anim_iter = iter(assets.clips)
    names = {c.id: c.name for c in script.characters}

    for scene in script.scenes:
        plate = assets.plates.get(f"{scene.location}@{scene.time}") or assets.plates[scene.location]
        vista = assets.vistas.get(scene.location, plate)
        pw, ph = plate.size
        order: list[str] = []                       # who appears in this scene, first come first served
        for b in scene.beats:
            ids = [b.line.who] if isinstance(b, LineBeat) else (b.characters if isinstance(b, ActionBeat) else [])
            order += [i for i in ids if i not in order]
        slot = {cid: x * pw for cid, x in zip(order, _slots(max(1, len(order))))}
        present: list[str] = []

        def wide_cam(p, z=1.06):
            return _clamp_cam(p, p.size[0] * 0.5, p.size[1] * 0.5, z)

        def stage(entering=(), gait=None):
            cast = []
            for cid in present:
                if cid in entering:
                    side = -0.25 * pw if slot[cid] < pw / 2 else 1.25 * pw
                    cast.append(OnStage(cid, side, slot[cid], gait))
                else:
                    cast.append(OnStage(cid, slot[cid], slot[cid]))
            return cast

        def head(cid):
            sp = assets.sprites[cid]
            char_h = ph * CHAR_HEIGHT
            k = char_h / sp.states["base"].shape[0]
            return (slot[cid] + (sp.head[0] - sp.states["base"].shape[1] / 2) * k,
                    ph * 1.04 - char_h + sp.head[1] * k)

        top = ph * 1.04 - ph * CHAR_HEIGHT          # world y of the top of every character's hair

        def face_cam(hx, hy, z):
            """Frame a face with the whole head in view: the eyes sit about a third of the way
            down and the top of the hair always has headroom, even after a push-in (zoom out
            if needed)."""
            z = min(z, _max_zoom_for_headroom(hy))
            vh = pw * VIEW_FRACTION / z * H / W
            end_vh = vh / (1 + MAX_ZOOM_MOVE / 2)                        # the view after a push-in
            cy = min(hy + (0.5 - 0.36) * vh, top - 0.05 * end_vh + end_vh / 2)
            return _clamp_cam(plate, hx, cy, z)

        def _max_zoom_for_headroom(hy):
            """The tightest zoom that still fits from the top of the hair to just below the
            face, with room to push in: big heads and tall hair get a wider shot."""
            need = (hy - top) * 2.1 + 0.08 * ph               # hair top → a bit below the chin, plus margin
            return max(1.0, pw * VIEW_FRACTION * H / W / max(1.0, need) / (1 + MAX_ZOOM_MOVE / 2))

        first = True                                # establishing shot of the scene
        said: set[str] = set()                      # who has spoken in this scene
        for beat in scene.beats:
            audio, dur = next(audio_iter)
            if isinstance(beat, PauseBeat):
                if shots:
                    shots[-1].dur += dur
                    if shots[-1].move not in ("hold", "dramatic"):  # stretch the move, same speed
                        c0, c1 = shots[-1].cam0, shots[-1].cam1
                        k = shots[-1].dur / max(1e-3, shots[-1].dur - dur)
                        shots[-1].cam1 = _clamp_cam(shots[-1].plate, *(a + (b - a) * k for a, b in zip(c0, c1)))
                t += dur
                continue
            prev = shots[-1] if shots else None
            last_move = prev.move if prev else ""

            def pick(options):
                """A random move, but not the same non-hold move twice in a row."""
                opts = [o for o in options if o == "hold" or o != last_move] or options
                return rng.choice(opts)

            if isinstance(beat, LineBeat):
                who = beat.line.who
                if who not in present:
                    present.append(who)
                emo = beat.line.emotion.value
                hx, hy = head(who)
                others = [c for c in present if c != who]
                if prev and prev.speaker == who and prev.plate is plate and prev.clip is None:
                    # same speaker keeps talking: no cut-and-restart, the shot just carries on
                    c0, c1 = _continue(prev, plate, dur)
                    move, framing, in_shot = "continue", prev.framing, [a.id for a in prev.cast]
                else:
                    if emo in INTENSE or beat.line.gesture:
                        size, move = "close", "push"
                    elif others and who in said and rng.random() < 0.3:
                        size, move = "two", pick(["hold", "hold", "pan"])
                    else:
                        size = "close" if who in said else "medium"
                        move = pick(["hold", "hold", "push", "pan"])
                    if size == "two":
                        partner = min(others, key=lambda c: abs(slot[c] - slot[who]))
                        ox, oy = head(partner)
                        z = min(1.6, max(1.0, pw * VIEW_FRACTION / (abs(ox - hx) * 2.2 + 1e-3)))
                        cam = face_cam((hx + ox) / 2, (hy + oy) / 2, z)
                        in_shot = [who, partner]
                    else:
                        cam = face_cam(hx, hy, 1.6 if size == "close" else 1.3)
                        in_shot = [who]
                    sign = rng.choice([1, -1])
                    c0, c1 = _move(plate, cam, move, dur, sign, rate=1.8 if emo in INTENSE else 1.0)
                    framing = f"{size}:{who}"
                said.add(who)
                # A single shows only the speaker and a two-shot only those two, like an anime
                # layout: nobody else ever appears half-cut at the edge of the frame.
                shot = Shot(t, dur, plate, scene.time, c0, c1, [a for a in stage() if a.id in in_shot],
                            speaker=who, audio=audio,
                            subtitle=strip_tags(beat.line.text), subtitle_name=names[who],
                            pop=emo in ("surprised", "amazed", "scared") and move != "continue",
                            shake=2.0 if emo == "angry" else 0.0, move=move, framing=framing)
            elif isinstance(beat, NarrationBeat):
                p = plate if present else vista
                if prev and prev.framing == "wide" and prev.plate is p and prev.clip is None:
                    (c0, c1), move = _continue(prev, p, dur), "continue"
                elif first:
                    move = rng.choice(["pan", "tilt"])
                    c0, c1 = _move(p, wide_cam(p, 1.12), move, dur, 1 if len(shots) % 2 == 0 else -1)
                else:
                    move = pick(["hold", "pan", "pull"])
                    c0, c1 = _move(p, wide_cam(p, 1.08), move, dur, rng.choice([1, -1]))
                shot = Shot(t, dur, p, scene.time, c0, c1, stage(), subtitle=beat.narration,
                            subtitle_name=None, move=move, framing="wide")
            elif isinstance(beat, ActionBeat) and beat.big:
                hero = next(hero_iter, None)
                anim = next(anim_iter, None)
                p = hero or plate
                c0, c1 = wide_cam(p, 1.0), wide_cam(p, 1.32)       # a deliberate dramatic punch-in
                shot = Shot(t, dur, p, scene.time, c0, c1, [] if hero else stage(), shake=7.0,
                            speedlines=anim is None, flash=True, parallax=120, clip=anim,
                            move="dramatic", framing="hero")
            elif isinstance(beat, ActionBeat):
                entering = [c for c in beat.characters if c not in present]
                present += entering
                gait = "run" if any(w in beat.action.lower() for w in ("run", "rush", "dash", "sprint")) else "walk"
                if present:
                    if entering:  # follow the newcomer in with a slow pan from their side
                        sign = -1 if slot[entering[0]] < pw / 2 else 1
                        move = "pan"
                        c0, c1 = _move(plate, wide_cam(plate), move, dur, sign)
                    elif prev and prev.framing == "wide" and prev.plate is plate and prev.clip is None:
                        (c0, c1), move = _continue(prev, plate, dur), "continue"
                    else:
                        move = pick(["hold", "push"])
                        c0, c1 = _move(plate, wide_cam(plate), move, dur)
                    shot = Shot(t, dur, plate, scene.time, c0, c1, stage(entering, gait), move=move, framing="wide")
                else:  # pure scenery action, e.g. "a giant wave rises": a fast, deliberate push
                    c0, c1 = wide_cam(vista, 1.0), wide_cam(vista, 1.22)
                    shot = Shot(t, dur, vista, scene.time, c0, c1, shake=3.0, parallax=90,
                                move="dramatic", framing="scenery")
                cam = beat.camera.value if beat.camera else None
                if cam == "low_angle":
                    p = shot.plate
                    shot.cam0 = _clamp_cam(p, p.size[0] / 2, p.size[1] * 0.7, 1.15)
                    shot.cam1 = _clamp_cam(p, p.size[0] / 2, p.size[1] * 0.45, 1.3)
                    shot.move = "tilt"
            else:  # CameraBeat: the script asked for this move, so it's allowed to be bigger
                cam = beat.camera.value
                p = plate if present else vista
                c = wide_cam(p, 1.04)
                if cam in ("zoom_in", "close_up"):
                    c0, c1, move = c, _clamp_cam(p, c[0], c[1], 1.5), "dramatic"
                elif cam in ("pan_left", "pan_right"):
                    d = 0.08 * p.size[0] * (-1 if cam == "pan_left" else 1)
                    c0, c1, move = _clamp_cam(p, c[0] - d, c[1], c[2]), _clamp_cam(p, c[0] + d, c[1], c[2]), "pan"
                elif cam == "shake":
                    c0, c1, move = c, c, "hold"
                else:  # establishing, wide, high angle…: a calm look around the place
                    move = pick(["pan", "tilt"]) if cam == "establishing" else pick(["hold", "pan", "pull"])
                    c0, c1 = _move(p, c, move, dur, rng.choice([1, -1]))
                shot = Shot(t, dur, p, scene.time, c0, c1, stage(), shake=4.0 if cam == "shake" else 0.0,
                            shake_hold=cam == "shake", move=move, framing=f"camera:{cam}")

            if shot.move in ("pan", "continue") and shot.clip is None and not shot.parallax:
                # depth parallax only follows real sideways camera travel: near things slide
                # past faster than far ones. Held shots and zooms don't make the background swim.
                shot.parallax = 0.35 * (shot.cam1[0] - shot.cam0[0]) / WORLD_SCALE
            first = False
            shots.append(shot)
            t += dur
    return shots


# --- rendering ---------------------------------------------------------------------

class Renderer:
    def __init__(self, assets: Assets, seed: int = 7) -> None:
        self.assets = assets
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        self.gx, self.gy = gx - W / 2, gy - H / 2
        r = np.sqrt((self.gx / (W / 2)) ** 2 + (self.gy / (H / 2)) ** 2)
        self.vignette = (1 - 0.28 * np.clip(r - 0.55, 0, 1) ** 1.4)[..., None].astype(np.float32)
        rng = random.Random(seed)
        self.blinks = {cid: self._blink_times(rng) for cid in assets.sprites}
        self.font = _font(34, bold=True)
        self.name_font = _font(24, bold=True)
        self._subs: dict[tuple, np.ndarray] = {}
        self._mouths: dict[int, list[str]] = {}
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _blink_times(rng, total: float = 600.0) -> list[float]:
        out, t = [], rng.uniform(0.8, 2.5)
        while t < total:
            out.append(t)
            t += rng.uniform(2.2, 5.0)
        return out

    def frame(self, shot: Shot, t_abs: float) -> np.ndarray:
        local = t_abs - shot.start
        if shot.clip is not None:
            return self._clip_frame(shot, t_abs, local)
        # constant speed: the camera is already moving when we cut in and still moving when
        # we cut out, like a real camera move (only the script's dramatic beats ease)
        lin = min(1.0, max(0.0, local / max(1e-3, shot.dur)))
        u = _ease(lin) if shot.move == "dramatic" else lin
        cx, cy, z = (a + (b - a) * u for a, b in zip(shot.cam0, shot.cam1))
        if shot.pop:  # punch-in on surprise: start wider, snap in fast and settle softly
            z *= 1 - 0.1 * (1 - _ease_out(local / 0.35))
        if shot.shake:
            jx, jy = _shake(t_abs, local, shot.shake * WORLD_SCALE, shot.shake_hold)
            cx, cy = cx + jx, cy + jy

        p = shot.plate
        pw, ph = p.size
        scale = pw * VIEW_FRACTION / z / W            # world px per output px
        sx = (cx + self.gx * scale).astype(np.float32)   # remap needs float32 maps
        sy = (cy + self.gy * scale).astype(np.float32)
        d = cv2.remap(p.depth, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        drift = np.float32((u - 0.5) * shot.parallax * WORLD_SCALE)
        img = cv2.remap(p.rgb, sx + (d - 0.5) * drift, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        t2 = math.floor(t_abs * 12) / 12             # puppets animate on twos
        for actor in shot.cast:
            self._draw_actor(img, shot, actor, local, t2, cx, cy, scale)

        if shot.speedlines:
            self._speedlines(img, int(t_abs * 12))
        img *= self.vignette
        if shot.flash and local < 0.18:
            img += (1 - local / 0.18) * 0.85
        if shot.subtitle:
            self._subtitle(img, shot.subtitle, shot.subtitle_name)
        return img

    def _clip_frame(self, shot: Shot, t_abs: float, local: float) -> np.ndarray:
        """Real animation: play the clip (held on its last frame if the shot runs longer)
        with a slow push-in, a light shake and the impact flash."""
        idx = min(int(local * FPS), len(shot.clip) - 1)
        z = 1.0 + 0.06 * min(1.0, local / max(1e-3, shot.dur))
        jitter = _shake(t_abs, local, 2.0, False)
        M = np.array([[z, 0, (1 - z) * W / 2 + jitter[0]], [0, z, (1 - z) * H / 2 + jitter[1]]], np.float32)
        img = cv2.warpAffine(shot.clip[idx], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        img *= self.vignette
        if shot.flash and local < 0.18:
            img += (1 - local / 0.18) * 0.85
        if shot.subtitle:
            self._subtitle(img, shot.subtitle, shot.subtitle_name)
        return img

    def _draw_actor(self, img, shot: Shot, actor: OnStage, local: float, t2: float, cx, cy, scale) -> None:
        sp = self.assets.sprites[actor.id]
        ph = shot.plate.size[1]
        base = sp.states["base"]
        sh, sw = base.shape[:2]
        k = ph * CHAR_HEIGHT / sh                    # sprite px → world px

        # where the feet are: walk/run in over the first part of the shot
        enter_dur = 1.0 if actor.gait == "run" else 1.6
        m = _ease(min(1.0, local / enter_dur)) if actor.x0 != actor.x1 else 1.0
        fx = actor.x0 + (actor.x1 - actor.x0) * m
        fy = ph * 1.04
        moving = actor.x0 != actor.x1 and local < enter_dur
        hz = 2.6 if actor.gait == "run" else 1.8
        bob = abs(math.sin(math.pi * hz * t2)) * (22 if actor.gait == "run" else 12) * WORLD_SCALE if moving else 0.0

        # mouth: flap track of this shot's voice clip, on twos
        mouth = "base"
        if shot.speaker == actor.id and shot.audio is not None:
            if id(shot) not in self._mouths:
                self._mouths[id(shot)] = mouth_track(shot.audio)
            track = self._mouths[id(shot)]
            i = int(local * 12)
            mouth = track[i] if i < len(track) else "base"
        level = {"talk": 1.0, "half": 0.5, "base": 0.0}[mouth]
        blink = any(0 <= t2 - b < 0.13 for b in self.blinks[actor.id])
        name = (f"{mouth}_blink" if mouth != "base" else "blink") if blink else mouth
        sprite = sp.states[name] if name in sp.states else sp.states.get(mouth, base)

        breathe = 1 + 0.009 * math.sin(2 * math.pi * t2 / 3.4 + hash(actor.id) % 7)
        sway = math.radians(0.7 * math.sin(2 * math.pi * t2 / 5.2 + hash(actor.id) % 5))
        lean = math.radians(-3.0 if moving else 0.0) * (1 if actor.x1 > actor.x0 else -1)
        talk_bob = -level * 5 * WORLD_SCALE

        # sprite px → output px, anchored at the feet (bottom center of the sprite)
        s_out = k / scale
        ang = sway + lean
        a, b = math.cos(ang) * s_out, math.sin(ang) * s_out
        ox = (fx - cx) / scale + W / 2
        oy = (fy - bob + talk_bob - cy) / scale + H / 2
        M = np.array([[a, -b * breathe, 0], [b, a * breathe, 0]], np.float32)
        anchor = np.array([sw / 2, sh], np.float32)
        M[:, 2] = np.array([ox, oy], np.float32) - M[:, :2] @ anchor
        warped = cv2.warpAffine(sprite, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        alpha = warped[..., 3:4]
        rgb = warped[..., :3] * np.array(GRADE.get(shot.time, (1, 1, 1)), np.float32)
        img *= 1 - alpha
        img += rgb * alpha

    def _speedlines(self, img, step: int) -> None:
        rng = np.random.default_rng(step)
        layer = np.zeros((H, W), np.float32)
        for _ in range(70):
            ang = rng.uniform(0, 2 * math.pi)
            r0 = rng.uniform(0.42, 0.62) * W
            x0, y0 = W / 2 + math.cos(ang) * r0, H / 2 + math.sin(ang) * r0
            x1, y1 = W / 2 + math.cos(ang) * W, H / 2 + math.sin(ang) * W
            cv2.line(layer, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, int(rng.integers(1, 4)), cv2.LINE_AA)
        img += layer[..., None] * 0.55
        np.clip(img, 0, 1, out=img)

    def _subtitle(self, img, text: str, name: str | None) -> None:
        key = (text, name)
        if key not in self._subs:
            layer = Image.new("RGBA", (W, 150), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            lines = _wrap(draw, text, self.font, W - 260)[-2:]
            y = 150 - 26 - len(lines) * 44
            if name:
                draw.text((W / 2, y - 30), name.upper(), font=self.name_font, fill=(255, 214, 120, 255),
                          anchor="mt", stroke_width=3, stroke_fill=(0, 0, 0, 255))
            for line in lines:
                draw.text((W / 2, y), line, font=self.font, fill=(255, 255, 255, 255), anchor="mt",
                          stroke_width=4, stroke_fill=(0, 0, 0, 255))
                y += 44
            self._subs[key] = np.asarray(layer, np.float32) / 255
        sub = self._subs[key]
        region = img[H - 150:]
        a = sub[..., 3:4]
        region *= 1 - a
        region += sub[..., :3] * a


def mouth_track(audio: np.ndarray, sr: int = 24000) -> list[str]:
    """Mouth shape per 1/12 s (on twos): 'talk' (open), 'half' or 'base' (closed).
    Loudness picks the shape; a held open mouth drops to half unless the voice gets
    louder again, so the mouth flaps with the syllables instead of hanging open."""
    step = sr // 12
    rms = np.array([np.sqrt(np.mean(audio[i:i + step] ** 2)) for i in range(0, len(audio), step)])
    voiced = rms[rms > 0.005]
    ref = float(np.percentile(voiced, 90)) if len(voiced) else 1.0
    track, prev = [], 0.0
    for level in rms / ref:
        if level < 0.18:
            track.append("base")
        elif level > 0.6 and (not track or track[-1] != "talk" or level > prev * 1.15):
            track.append("talk")
        else:
            track.append("half")
        prev = level
    return track


def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default(size)


def _wrap(draw, text, font, max_w) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur] if cur else lines


def render(shots: list[Shot], renderer: Renderer, audio_path, out_path, total: float, on_progress=None) -> None:
    """Stream frames straight into ffmpeg (hardware H.264 encoder)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(audio_path), "-c:v", "h264_videotoolbox", "-b:v", "10M",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(math.ceil(total * FPS))
    si = 0
    for f in range(n):
        t = f / FPS
        while si + 1 < len(shots) and t >= shots[si + 1].start:
            si += 1
        img = renderer.frame(shots[si], t)
        fade = min(1.0, t / 0.5, (total - t) / 0.6)   # fade in from / out to black
        img *= max(0.0, fade)
        proc.stdin.write((img.clip(0, 1) * 255).astype(np.uint8).tobytes())
        if on_progress and f % 24 == 0:
            on_progress(f / n)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed while encoding the animation")
