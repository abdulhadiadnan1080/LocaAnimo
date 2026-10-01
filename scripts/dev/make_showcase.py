"""Build the 40-second GitHub showcase video (docs/media/showcase.mp4) from real episode
footage, pipeline assets and a synthesized 128 BPM track. Everything is cut on the beat.

  python scripts/dev/make_showcase.py
"""

import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "media" / "showcase.mp4"
W, H, FPS, SR = 1920, 1080, 30, 48000
BPM = 128
BEAT = 60 / BPM
BAR = 4 * BEAT
TOTAL = 40.0

EP = {
    "ember": ROOT / "outputs" / "ember-protocol-20261001-141513.mp4",
    "kite": ROOT / "outputs" / "the-lost-kite-20261001-152505.mp4",
    "beach": ROOT / "outputs" / "animation-test-20261001-133556.mp4",
}
KITE, EMBER = ROOT / "projects" / "kite", ROOT / "projects" / "ember"
ACCENT = (47, 91, 234)
FONT = "/System/Library/Fonts/Avenir Next.ttc"   # index 8 = Heavy, 2 = Demi Bold, 0 = Bold


def font(size, weight="heavy"):
    return ImageFont.truetype(FONT, size, index={"heavy": 8, "demi": 2, "bold": 0, "medium": 5}[weight])


def ease(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


# --- footage ---------------------------------------------------------------------------

class Footage:
    """Sequential frame reader for a slice of a video, scaled/cropped to 1920x1080."""

    def __init__(self, path, start, dur, fit="cover"):
        vf = ("scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080" if fit == "cover"
              else "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2")
        self.cmd = ["ffmpeg", "-loglevel", "error", "-ss", str(start), "-t", str(dur + 0.2), "-i", str(path),
                    "-vf", f"{vf},fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.proc, self.last, self.n = None, None, 0

    def frame(self, i):
        if self.proc is None:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
        while self.n <= i:
            raw = self.proc.stdout.read(W * H * 3)
            if len(raw) < W * H * 3:
                break
            self.last = np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(np.float32) / 255
            self.n += 1
        return self.last.copy() if self.last is not None else np.zeros((H, W, 3), np.float32)


def still(path, box=None, fit="cover", size=(W, H)):
    img = Image.open(path).convert("RGB")
    if box:
        img = img.crop(box)
    tw, th = size
    if fit == "cover":
        s = max(tw / img.width, th / img.height)
    else:
        s = min(tw / img.width, th / img.height)
    img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", size, (12, 14, 20))
    canvas.paste(img, ((tw - img.width) // 2, (th - img.height) // 2))
    return np.asarray(canvas, np.float32) / 255


def zoom(img, z, cx=0.5, cy=0.5):
    M = np.array([[z, 0, (1 - z) * W * cx], [0, z, (1 - z) * H * cy]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# --- text ------------------------------------------------------------------------------

_text_cache = {}


def text_layer(lines, size, weight="heavy", color=(255, 255, 255), track=0, shadow=True, align="center", y=None):
    """RGBA float layer with centered text, its block centred at height `y` (default: middle); cached."""
    lines = [l for l in lines if l]
    key = (tuple(lines), size, weight, color, track, shadow, align, y)
    if key in _text_cache:
        return _text_cache[key]
    f = font(size, weight)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    heights = [f.getbbox(l)[3] - f.getbbox(l)[1] for l in lines]
    gap = int(size * 0.25)
    y = (y if y is not None else H / 2) - (sum(heights) + gap * (len(lines) - 1)) / 2
    for l, h in zip(lines, heights):
        w = d.textlength(l, font=f) + track * (len(l) - 1)
        x = (W - w) / 2 if align == "center" else 140
        if track:
            for ch in l:
                d.text((x, y - f.getbbox(l)[1]), ch, font=f, fill=color + (255,))
                x += d.textlength(ch, font=f) + track
        else:
            d.text((x, y - f.getbbox(l)[1]), l, font=f, fill=color + (255,))
        y += h + gap
    arr = np.asarray(layer, np.float32) / 255
    if shadow:
        sh = np.asarray(layer.filter(ImageFilter.GaussianBlur(18)), np.float32)[..., 3:4] / 255
        arr = np.concatenate([arr[..., :3], np.maximum(arr[..., 3:4], sh * 0.55)], axis=2)
        arr[..., :3] = np.where(arr[..., 3:4] > 0, arr[..., :3] * (np.asarray(layer, np.float32)[..., 3:4] / 255 /
                                np.maximum(arr[..., 3:4], 1e-6)), 0)
    _text_cache[key] = arr
    return arr


def over(img, layer, alpha=1.0, dx=0, dy=0, scale=1.0):
    if scale != 1.0 or dx or dy:
        M = np.array([[scale, 0, (1 - scale) * W / 2 + dx], [0, scale, (1 - scale) * H / 2 + dy]], np.float32)
        layer = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = layer[..., 3:4] * alpha
    return img * (1 - a) + layer[..., :3] * a


def chip(img, number, title, sub, t):
    """Label for the pipeline section: number badge + title + model name, slides in."""
    key = ("chip", number, title, sub)
    if key not in _text_cache:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        f1, f2, f3 = font(46, "heavy"), font(38, "heavy"), font(28, "demi")
        x, y = 110, H - 210
        d.rounded_rectangle((x - 30, y - 28, x + 30 + max(d.textlength(title, font=f2), d.textlength(sub, font=f3)) + 110,
                             y + 112), radius=26, fill=(10, 12, 18, 215))
        d.rounded_rectangle((x, y, x + 72, y + 72), radius=18, fill=ACCENT + (255,))
        d.text((x + 36, y + 36), str(number), font=f1, fill=(255, 255, 255, 255), anchor="mm")
        d.text((x + 100, y - 2), title, font=f2, fill=(255, 255, 255, 255))
        d.text((x + 100, y + 46), sub, font=f3, fill=(165, 182, 255, 255))
        _text_cache[key] = np.asarray(layer, np.float32) / 255
    return over(img, _text_cache[key], ease(t / 0.18), dx=-60 * (1 - ease(t / 0.18)))


def flash(img, t, length=0.12, strength=0.9):
    return img + max(0.0, 1 - t / length) * strength if t < length else img


def vignette():
    gx, gy = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    return (1 - 0.35 * np.clip(np.sqrt(gx ** 2 + gy ** 2) - 0.45, 0, 1) ** 1.3)[..., None].astype(np.float32)


VIG = vignette()


# --- scenes ----------------------------------------------------------------------------

def build_timeline():
    tl = []  # (start, dur, render(t_local) -> img)

    # A. hook: words slam in on the beat
    words = [("ONE SCRIPT.", 0.0), ("ONE MACBOOK.", 2 * BEAT), ("ZERO CLOUD.", 4 * BEAT)]
    bg = np.zeros((H, W, 3), np.float32)

    def hook(t):
        img = bg.copy() + 0.02
        for i, (w, s) in enumerate(words):
            e = s + 2 * BEAT if i < 2 else 2 * BAR - 0.35
            if s <= t < e:
                lt = t - s
                img = over(img, text_layer([w], 190), 1.0, scale=1.25 - 0.25 * ease(lt / 0.12))
                img = flash(img, lt, 0.08, 0.5)
        if t > 2 * BAR - 0.35:   # riser: white bloom into the drop
            img += ease((t - (2 * BAR - 0.35)) / 0.35) ** 2
        return img
    tl.append((0.0, 2 * BAR, hook))

    # B. drop: 8 hero cuts, then the title
    cuts = [("ember", 23.3), (KITE / "clips" / "action_2.mp4", 0.6), ("ember", 17.2), ("beach", 22.6),
            ("ember", 11.6), (KITE / "clips" / "action_1.mp4", 0.9), ("ember", 56.6), ("ember", 72.6)]
    t0 = 2 * BAR
    for i, (src, at) in enumerate(cuts):
        path = EP[src] if isinstance(src, str) else src
        clip = Footage(path, at, BEAT + 0.1)
        is_title = i >= 4

        def cut(t, clip=clip, i=i, is_title=is_title):
            img = zoom(clip.frame(int(t * FPS)), 1.12 - 0.08 * ease(t / BEAT))
            img *= VIG
            if is_title:
                img *= 0.55
                img = over(img, text_layer(["HADI'S LOCAANIMO"], 150, track=6, y=H / 2 - 50), 1.0,
                           scale=1.0 + 0.04 * (1 - ease((t + (i - 4) * BEAT) / (4 * BEAT))))
                img = over(img, text_layer(["write a story  ·  get an animated episode"], 50, "demi",
                                           (205, 215, 255), y=H / 2 + 85), ease((i - 4 + t / BEAT) / 1.2))
            return flash(img, t, 0.07, 0.6)
        tl.append((t0 + i * BEAT, BEAT, cut))

    # C. how it works: 8 steps, two beats each
    t0 = 4 * BAR
    step = 2 * BEAT
    story = ["Title: The Lost Kite", "", "Leo is a little boy with curly brown hair,", "a yellow t-shirt and red sneakers.",
             "", "Scene 1: A sunny city park. Happy.", "LEO (happy): Look, Grandma! It's flying!",
             "BIG MOMENT: A gust snaps the string and", "the kite shoots up into the sky."]

    def s_script(t):
        img = np.full((H, W, 3), 0.97, np.float32)
        n = int(len(" ".join(story)) * ease(t / (step * 0.9)))
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.rounded_rectangle((300, 130, W - 300, H - 230), radius=28, fill=(255, 255, 255, 255), outline=(225, 228, 235, 255), width=3)
        f = font(40, "demi")
        y, used = 190, 0
        for line in story:
            shown = line[:max(0, n - used)]
            used += len(line) + 1
            d.text((370, y), shown, font=f, fill=(25, 28, 38, 255) if not line.startswith(("LEO", "BIG")) else ACCENT + (255,))
            y += 62
        return chip(over(img, np.asarray(lay, np.float32) / 255), 1, "Write the story", "plain text or YAML · any format", t)

    ui = still(ROOT / "docs" / "media" / "ui-story.png", box=(0, 0, 2880, 1620))

    def s_ui(t):
        return chip(zoom(ui, 1.0 + 0.1 * ease(t / step), 0.55, 0.3), 2, "Hadi's LocaAnimo studio", "local web app · voice previews · one click", t)

    plate = still(KITE / "art" / "bg_hill@sunset.png")

    rose = still(KITE / "art" / "char_rose.png", fit="contain")[:, W // 2 - 380:W // 2 + 380]

    def s_art(t):
        img = plate.copy()
        img[:, int(W * ease(t / (step * 0.5))):] = 0.05             # background paints in left to right
        a = ease((t - step * 0.35) / 0.25) if t > step * 0.35 else 0.0
        img[:, W - 840:W - 80] = img[:, W - 840:W - 80] * (1 - a) + rose * a   # then a character card
        return chip(img * VIG, 3, "Art", "Illustrious XL · backgrounds, characters, action shots", t)

    orig = still(KITE / "art" / "char_leo.png", fit="contain")
    
    def s_cut(t):
        a = ease((t - 0.25) / 0.45)   # the raw drawing dissolves into Leo cut out and placed in the park
        return chip(orig * (1 - a) + KITE_FRAME * a, 4, "Cut-out", "BiRefNet · the character is lifted off the drawing", t)

    rin = Image.open(EMBER / "art" / "char_rin.png").convert("RGB").crop((240, 120, 640, 470))
    boxes = [((370, 246, 419, 283), (70, 200, 120)), ((462, 246, 510, 283), (70, 200, 120)), ((395, 322, 449, 337), (230, 70, 80))]

    def s_face(t):
        img = rin.copy()
        d = ImageDraw.Draw(img)
        for k, (b, c) in enumerate(boxes):
            p = ease((t - 0.15 * k) / 0.25)
            if p > 0:
                x0, y0, x1, y1 = [v - o for v, o in zip(b, (240, 120, 240, 120))]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                hw, hh = (x1 - x0) / 2 * (2 - p), (y1 - y0) / 2 * (2 - p)
                d.rectangle((cx - hw, cy - hh, cx + hw, cy + hh), outline=c, width=3)
        arr = still_from_pil(img)
        return chip(arr, 5, "Faces", "Qwen3.5 vision model finds the eyes and mouth", t)

    kaito_states = [Image.open(EMBER / f"sprite_kaito_{s}.png").convert("RGB").crop((330, 200, 740, 560)) for s in ("base", "half", "talk", "half")]
    kaito_arr = [still_from_pil(k) for k in kaito_states]

    def s_lips(t):
        return chip(kaito_arr[int(t * 12) % 4], 6, "Lip sync", "drawn mouths flap with the Kokoro voice, on twos", t)

    night = still(EMBER / "art" / "bg_rooftop@night.png")
    depth = depth_preview(EMBER / "art" / "bg_rooftop@night.png")

    def s_depth(t):
        x = int(W * (0.15 + 0.7 * ease(t / step)))
        img = night.copy()
        img[:, :x] = depth[:, :x]
        img[:, max(0, x - 3):x + 3] = 1.0
        return chip(img, 7, "Depth", "Depth Anything V2 · 2.5D parallax camera", t)

    ai_still = still(KITE / "art" / "action_2.png")
    ai_clip = Footage(KITE / "clips" / "action_2.mp4", 0.0, step + 0.2)

    def s_ai(t):
        if t < 0.3:
            img = ai_still.copy()
            return chip(over(img, text_layer(["STILL"], 90, track=8, y=H / 2 - 60), 0.9), 8, "AI motion", "LTX-Video · the still comes alive", t)
        img = zoom(ai_clip.frame(int((t - 0.3) * FPS * 1.4)), 1.02)
        img = flash(img, t - 0.3, 0.08, 0.7)
        return chip(img, 8, "AI motion", "LTX-Video · the still comes alive", t)

    for k, fn in enumerate([s_script, s_ui, s_art, s_cut, s_face, s_lips, s_depth, s_ai]):
        tl.append((t0 + k * step, step, fn))

    # D. showcase with their own voices
    t0 = 8 * BAR
    show = [("beach", 9.0, 3.0, "The Last Wave"), ("ember", 19.85, 3.05, "Ember Protocol"),
            ("kite", 47.35, 3.1, "The Lost Kite · AI motion"), ("ember", 69.05, 15 * BAR / 4 - 9.15 + 0.0, "Ember Protocol")]
    remaining = 7 * BAR
    durs = [3.0, 3.05, 3.1]
    durs.append(remaining - sum(durs))
    s = t0
    for (src, at, _, label), d in zip(show, durs):
        clip = Footage(EP[src], at, d)

        def sc(t, clip=clip, label=label, d=d):
            img = zoom(clip.frame(int(t * FPS)), 1.0 + 0.04 * t / d)
            img = over(img, label_layer(label), ease(t / 0.3))
            return flash(img, t, 0.06, 0.35)
        tl.append((s, d, sc))
        AUDIO_SLICES.append((s, EP[src], at, d))
        s += d

    # E. stats
    t0 = 15 * BAR
    stats = [("100%", "LOCAL"), ("0", "CLOUD APIs"), ("1", "MACBOOK · M3 PRO · 18 GB"), ("6", "OPEN MODELS")]
    bgclip = Footage(KITE / "clips" / "action_2.mp4", 0.0, 3 * BAR)

    def s_stats(t):
        frame = bgclip.frame(int(t * FPS * 0.6) % 60)
        img = cv2.GaussianBlur(frame, (0, 0), 18) * 0.28
        k = min(len(stats) - 1, int(t / (2 * BEAT)))
        lt = t - k * 2 * BEAT
        big, small = stats[k]
        img = over(img, text_layer([big], 280, color=(255, 255, 255), y=H / 2 - 110), 1.0, scale=1.18 - 0.18 * ease(lt / 0.15))
        img = over(img, text_layer([small], 66, "heavy", (165, 182, 255), y=H / 2 + 110), ease(lt / 0.2), dy=20 * (1 - ease(lt / 0.2)))
        if t > 6 * BEAT:
            img = over(img, text_layer(["Illustrious XL  ·  LTX-Video  ·  Qwen3.5  ·  Kokoro  ·  BiRefNet  ·  Depth Anything"],
                                       36, "demi", (215, 220, 235), y=H - 150), ease((t - 6 * BEAT) / 0.4))
        return flash(img, lt, 0.07, 0.4)
    tl.append((t0, TOTAL - 6.0 - t0, s_stats))

    # F. end card
    def s_end(t):
        img = np.full((H, W, 3), 0.985, np.float32)
        a = ease(t / 0.5)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        s = 150
        d.rounded_rectangle((W / 2 - s / 2, 240, W / 2 + s / 2, 240 + s), radius=38, fill=(17, 24, 39, 255))
        d.text((W / 2, 240 + s / 2), "LA", font=font(70, "heavy"), fill=(255, 255, 255, 255), anchor="mm")
        d.text((W / 2, 500), "Hadi's LocaAnimo", font=font(120, "heavy"), fill=(17, 24, 39, 255), anchor="mm")
        d.text((W / 2, 610), "Write the story. Your Mac makes the episode.", font=font(50, "demi"), fill=(75, 85, 99, 255), anchor="mm")
        d.rounded_rectangle((W / 2 - 330, 700, W / 2 + 330, 790), radius=45, fill=ACCENT + (255,))
        d.text((W / 2, 745), "Open source  ·  100% local", font=font(40, "heavy"), fill=(255, 255, 255, 255), anchor="mm")
        img = over(img, np.asarray(lay, np.float32) / 255, a, dy=40 * (1 - a), scale=0.96 + 0.04 * a)
        if t > 5.4:
            img *= max(0.0, 1 - (t - 5.4) / 0.6)
        return flash(img, t, 0.1, 0.5)
    tl.append((TOTAL - 6.0, 6.0, s_end))
    return tl


def still_from_pil(img):
    s = min(W / img.width, H / img.height) * 0.92
    img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (245, 246, 248))
    canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
    return np.asarray(canvas, np.float32) / 255


def label_layer(label):
    key = ("label", label)
    if key not in _text_cache:
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        f = font(34, "heavy")
        w = d.textlength(label.upper(), font=f)
        d.rounded_rectangle((70, 60, 70 + w + 56, 124), radius=32, fill=(10, 12, 18, 200))
        d.ellipse((92, 84, 108, 100), fill=(239, 68, 68, 255))
        d.text((122, 92), label.upper(), font=f, fill=(255, 255, 255, 255), anchor="lm")
        _text_cache[key] = np.asarray(lay, np.float32) / 255
    return _text_cache[key]


def depth_preview(path):
    sys.path.insert(0, str(ROOT / "src"))
    from locaanimo.prep import Depth
    d = Depth().map(Image.open(path))
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    return still_from_pil(Image.fromarray(colored)) if False else \
        np.asarray(Image.fromarray(colored).resize((W, H), Image.LANCZOS), np.float32) / 255


AUDIO_SLICES = []
KITE_FRAME = None


# --- music -------------------------------------------------------------------------------

def music():
    n = int(TOTAL * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng(3)

    def at(sec):
        return int(sec * SR)

    def add(sig, start, gain=1.0):
        i = at(start)
        if i >= n:
            return
        out[i:i + len(sig)] += sig[: n - i] * gain

    def kick():
        k = np.arange(int(0.42 * SR)) / SR
        f = 45 + 110 * np.exp(-k * 28)
        return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-k * 7.5)).astype(np.float32)

    def clap():
        k = np.arange(int(0.25 * SR)) / SR
        noise = sosfilt(butter(2, [900, 6000], "band", fs=SR, output="sos"), rng.normal(0, 1, len(k)))
        return (noise * np.exp(-k * 18)).astype(np.float32)

    def hat():
        k = np.arange(int(0.06 * SR)) / SR
        return (sosfilt(butter(2, 7000, "high", fs=SR, output="sos"), rng.normal(0, 1, len(k))) * np.exp(-k * 70)).astype(np.float32)

    def boom(length=2.5):
        k = np.arange(int(length * SR)) / SR
        sub = np.sin(2 * np.pi * np.cumsum(30 + 70 * np.exp(-k * 6)) / SR) * np.exp(-k * 1.6)
        crack = sosfilt(butter(2, 3000, "low", fs=SR, output="sos"), rng.normal(0, 1, len(k))) * np.exp(-k * 9)
        return (sub + 0.5 * crack).astype(np.float32)

    def riser(length):
        k = np.arange(int(length * SR)) / SR
        noise = rng.normal(0, 1, len(k))
        sweep = np.zeros_like(noise)
        for j in range(0, len(k), 2400):          # stepped band sweep upward
            lo = 300 + 6000 * (j / len(k)) ** 2
            seg = noise[j:j + 2400]
            sweep[j:j + 2400] = sosfilt(butter(2, [lo, lo * 2], "band", fs=SR, output="sos"), seg)
        return (sweep * (k / length) ** 2).astype(np.float32)

    def saw(freq, length, detune=(-0.12, -0.05, 0, 0.05, 0.12)):
        k = np.arange(int(length * SR)) / SR
        sig = sum(2 * ((freq * 2 ** (dt / 12) * k + rng.random()) % 1) - 1 for dt in detune) / len(detune)
        return sig.astype(np.float32)

    chords = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [261.63, 329.63, 392.0], [196.0, 246.94, 293.66]]
    roots = [55.0, 43.65, 65.41, 49.0]

    # sections
    drop_start, how_start, show_start, stats_start, end_start = 2 * BAR, 4 * BAR, 8 * BAR, 15 * BAR, TOTAL - 6.0

    # intro heartbeat + drone + riser
    for b in range(8):
        if b % 2 == 0:
            add(kick(), b * BEAT, 0.55)
    add(saw(55, 2 * BAR) * 0.15 * np.linspace(0, 1, int(2 * BAR * SR)), 0)
    add(riser(1.6), drop_start - 1.6, 0.35)
    for sec in (drop_start, show_start, stats_start, end_start):
        add(boom(), sec, 0.9)

    # beat, bass and chords from the drop to the end card
    sidechain = np.ones(n, np.float32)
    beats = int((end_start - drop_start) / BEAT)
    for b in range(beats):
        s = drop_start + b * BEAT
        energy = 0.6 if show_start <= s < stats_start else 1.0      # pull back under dialogue
        add(kick(), s, 0.95 * energy)
        i = at(s)
        env = np.minimum(1, np.arange(int(0.22 * SR)) / (0.22 * SR)) ** 0.6
        sidechain[i:i + len(env)] = np.minimum(sidechain[i:i + len(env)], 0.25 + 0.75 * env)
        if b % 4 in (1, 3):
            add(clap(), s, 0.45 * energy)
        for h in range(2):
            add(hat(), s + BEAT / 2 * h + BEAT / 4, 0.22 * energy)
    for bar in range(int((end_start - drop_start) / BAR) + 1):
        s = drop_start + bar * BAR
        ci = bar % 4
        for e in range(8):                                           # 8th-note bass
            note = saw(roots[ci] * (2 if e % 2 else 1), BEAT / 2 * 0.9, (0,))
            note *= np.exp(-np.arange(len(note)) / SR * 6)
            add(sosfilt(butter(2, 400, "low", fs=SR, output="sos"), note).astype(np.float32), s + e * BEAT / 2, 0.5)
        pad = sum(saw(f, BAR) for f in chords[ci])
        pad = sosfilt(butter(2, 2600, "low", fs=SR, output="sos"), pad).astype(np.float32)
        i = at(s)
        seg = pad[: max(0, n - i)] * sidechain[i:i + len(pad)][: max(0, n - i)]
        out[i:i + len(seg)] += seg * 0.16
    # stats build: snare roll riser into the end card
    for k in range(16):
        add(clap(), stats_start + 6 * BEAT + k * (end_start - stats_start - 6 * BEAT) / 16, 0.15 + 0.4 * k / 16)
    add(riser(end_start - stats_start - 0.2), stats_start + 0.1, 0.25)
    # end chord swell
    end_pad = sum(saw(f, 6.0) for f in chords[0]) * np.linspace(1, 0, int(6 * SR)) ** 1.5
    add(sosfilt(butter(2, 1800, "low", fs=SR, output="sos"), end_pad).astype(np.float32), end_start, 0.22)
    return out


def dialogue(total_len):
    """The episodes' own audio for the showcase slices; the music ducks under it."""
    out = np.zeros(total_len, np.float32)
    duck = np.ones(total_len, np.float32)
    for start, path, at, d in AUDIO_SLICES:
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(at), "-t", str(d), "-i", str(path),
                              "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True).stdout
        a = np.frombuffer(raw, np.float32)
        i = int(start * SR)
        fade = np.ones(len(a), np.float32)
        f = int(0.05 * SR)
        fade[:f] = np.linspace(0, 1, f)
        fade[-f:] = np.linspace(1, 0, f)
        out[i:i + len(a)] += (a * fade * 1.3)[: total_len - i]
        duck[i:i + len(a)] = 0.45
    duck = np.convolve(duck, np.ones(2400) / 2400, mode="same")
    return out, duck


def main():
    global KITE_FRAME
    print("preparing assets…", flush=True)
    KITE_FRAME = Footage(EP["kite"], 8.2, 0.2).frame(0)     # Leo standing in the park, cut out
    tl = build_timeline()
    print("music…", flush=True)
    m = music()
    d, duck = dialogue(len(m))
    mix = m * duck + d
    mix = np.tanh(mix * 1.15) / np.tanh(1.15) * 0.95
    wav = ROOT / "projects" / "showcase_audio.wav"   # working file (git-ignored)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(wav, mix, SR)

    print("rendering frames…", flush=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(OUT)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = int(TOTAL * FPS)
    for f in range(n):
        t = f / FPS
        seg = next((s for s in reversed(tl) if s[0] <= t), tl[0])
        img = seg[2](t - seg[0])
        proc.stdin.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
        if f % (FPS * 5) == 0:
            print(f"  {t:4.1f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
