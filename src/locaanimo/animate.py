"""Shot planner + compositor: script beats and prepared art → animated frames.

Camera moves over depth-parallaxed background plates; characters are puppets that
breathe, sway, walk in, bob while talking, lip-flap from their audio and blink.
Character animation is held on twos (12 drawings/s) like TV anime; the camera runs on ones.
"""

from __future__ import annotations

import math
import random
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schema import ActionBeat, CameraBeat, LineBeat, NarrationBeat, PauseBeat, Script

FPS = 24
W, H = 1280, 720
WORLD_SCALE = 2.0          # background plates are upscaled 2x into "world" space
VIEW_FRACTION = 0.9        # at zoom 1 the camera sees 90% of the plate width (room to move)
CHAR_HEIGHT = 0.82         # cowboy-shot sprite height as a fraction of world height
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
    speedlines: bool = False
    flash: bool = False
    pop: bool = False                          # quick zoom pop at the start (surprise)
    subtitle: str | None = None
    subtitle_name: str | None = None
    parallax: float = 60.0                     # world px of near-vs-far drift over the shot
    clip: np.ndarray | None = None             # real animation for this shot (AI action clip)


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


def _slots(n: int) -> list[float]:
    return {1: [0.5], 2: [0.34, 0.66], 3: [0.24, 0.5, 0.76]}.get(n, [0.15 + 0.7 * i / max(1, n - 1) for i in range(n)])


def plan(script: Script, assets: Assets, clips: list[tuple[np.ndarray | None, float]]) -> list[Shot]:
    """One shot per beat (pauses extend the previous shot). `clips` gives each beat's
    (audio, duration) in script order."""
    shots: list[Shot] = []
    t = 0.0
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

        def wide(z0=1.0, z1=1.07, drift=0.04, p=plate):
            w_ = p.size[0]
            return (_clamp_cam(p, w_ * (0.5 - drift), p.size[1] * 0.5, z0),
                    _clamp_cam(p, w_ * (0.5 + drift), p.size[1] * 0.5, z1))

        def stage(entering=(), gait=None):
            cast = []
            for cid in present:
                if cid in entering:
                    side = -0.25 * pw if slot[cid] < pw / 2 else 1.25 * pw
                    cast.append(OnStage(cid, side, slot[cid], gait))
                else:
                    cast.append(OnStage(cid, slot[cid], slot[cid]))
            return cast

        for beat in scene.beats:
            audio, dur = next(audio_iter)
            if isinstance(beat, PauseBeat):
                if shots:
                    shots[-1].dur += dur
                t += dur
                continue

            if isinstance(beat, LineBeat):
                who = beat.line.who
                if who not in present:
                    present.append(who)
                sp = assets.sprites[who]
                char_h = ph * CHAR_HEIGHT
                k = char_h / sp.states["base"].shape[0]
                head_x = slot[who] + (sp.head[0] - sp.states["base"].shape[1] / 2) * k
                head_y = ph * 1.04 - char_h + sp.head[1] * k
                z = 1.85
                vh = pw * VIEW_FRACTION / z * H / W
                cy = head_y + (0.5 - 0.36) * vh
                emo = beat.line.emotion.value
                push = 0.1 if emo in ("confident", "determined", "angry") or beat.line.gesture else 0.04
                shot = Shot(t, dur, plate, scene.time, _clamp_cam(plate, head_x, cy, z),
                            _clamp_cam(plate, head_x, cy, z + push), stage(), speaker=who, audio=audio,
                            subtitle=beat.line.text, subtitle_name=names[who], parallax=25,
                            pop=emo in ("surprised", "amazed", "scared"), shake=2.0 if emo == "angry" else 0.0)
            elif isinstance(beat, NarrationBeat):
                c0, c1 = wide(1.04, 1.12)
                shot = Shot(t, dur, plate if present else vista, scene.time, c0, c1, stage(),
                            audio=None, subtitle=beat.narration, subtitle_name=None)
                if not present:
                    shot.cam0, shot.cam1 = wide(1.04, 1.12, p=vista)
            elif isinstance(beat, ActionBeat) and beat.big:
                hero = next(hero_iter, None)
                anim = next(anim_iter, None)
                p = hero or plate
                c0, c1 = wide(1.0, 1.32, 0.0, p)
                shot = Shot(t, dur, p, scene.time, c0, c1, [] if hero else stage(), shake=7.0,
                            speedlines=anim is None, flash=True, parallax=120, clip=anim)
            elif isinstance(beat, ActionBeat):
                entering = [c for c in beat.characters if c not in present]
                present += entering
                gait = "run" if any(w in beat.action.lower() for w in ("run", "rush", "dash", "sprint")) else "walk"
                if present:
                    c0, c1 = wide(1.02, 1.1)
                    shot = Shot(t, dur, plate, scene.time, c0, c1, stage(entering, gait))
                else:  # pure scenery action, e.g. "a giant wave rises"
                    c0, c1 = wide(1.0, 1.22, 0.0, vista)
                    shot = Shot(t, dur, vista, scene.time, c0, c1, shake=3.0, parallax=90)
                cam = beat.camera.value if beat.camera else None
                if cam == "low_angle":
                    p = shot.plate
                    shot.cam0 = _clamp_cam(p, p.size[0] / 2, p.size[1] * 0.7, 1.15)
                    shot.cam1 = _clamp_cam(p, p.size[0] / 2, p.size[1] * 0.45, 1.3)
            else:  # CameraBeat
                cam = beat.camera.value
                p = plate if present else vista
                c0, c1 = wide(1.0, 1.08, 0.06, p)
                if cam in ("zoom_in", "close_up"):
                    c1 = _clamp_cam(p, c1[0], c1[1], 1.5)
                elif cam == "pan_left":
                    c0, c1 = c1, c0
                shot = Shot(t, dur, p, scene.time, c0, c1, stage(), shake=4.0 if cam == "shake" else 0.0)

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
        u = _ease(local / max(1e-3, shot.dur))
        cx, cy, z = (a + (b - a) * u for a, b in zip(shot.cam0, shot.cam1))
        if shot.pop:  # punch-in on surprise: start wider, snap in over 0.25 s
            z *= 1 - 0.12 * max(0.0, 1 - local / 0.25)
        if shot.shake:
            step = int(t_abs * FPS)
            jitter = np.random.default_rng(step).normal(0, shot.shake, 2) * WORLD_SCALE
            cx, cy = cx + jitter[0], cy + jitter[1]

        p = shot.plate
        pw, ph = p.size
        scale = pw * VIEW_FRACTION / z / W            # world px per output px
        sx = (cx + self.gx * scale).astype(np.float32)   # remap needs float32 maps
        sy = (cy + self.gy * scale).astype(np.float32)
        d = cv2.remap(p.depth, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        drift = (u - 0.5) * shot.parallax * WORLD_SCALE
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
        z = 1.0 + 0.06 * _ease(local / max(1e-3, shot.dur))
        jitter = np.random.default_rng(int(t_abs * FPS)).normal(0, 2.0, 2) if local < 1.2 else (0, 0)
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
