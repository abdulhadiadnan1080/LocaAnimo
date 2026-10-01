"""Turn raw art into animation-ready assets: cutouts (BiRefNet), face points (DWPose),
expression variants merged onto the base drawing, and depth maps (Depth Anything V2 Small)."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from .paths import MODELS, ROOT

VISION = MODELS / "vision"


# --- cutout ------------------------------------------------------------------------

class Cutter:
    def __init__(self) -> None:
        from transformers import AutoModelForImageSegmentation
        self.model = AutoModelForImageSegmentation.from_pretrained(
            str(VISION / "BiRefNet"), trust_remote_code=True).to("mps").eval().half()

    @torch.no_grad()
    def alpha(self, img: Image.Image) -> np.ndarray:
        """Float mask in [0, 1] at the image's size."""
        x = np.asarray(img.convert("RGB").resize((1024, 1024), Image.BILINEAR), np.float32) / 255
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to("mps", torch.float16)
        pred = self.model(t)[-1].sigmoid()[0, 0].float().cpu().numpy()
        return _main_subject(cv2.resize(pred, img.size, interpolation=cv2.INTER_LINEAR))


def _main_subject(alpha: np.ndarray) -> np.ndarray:
    """Keep only the character: the biggest connected shape that touches the image's
    horizontal centre. Stray background decorations the model drew get dropped."""
    solid = (alpha > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, 8)
    if n <= 2:
        return alpha
    centre = labels[:, labels.shape[1] // 2]
    candidates = [i for i in set(centre.tolist()) if i] or list(range(1, n))
    keep = max(candidates, key=lambda i: stats[i][4])
    mask = cv2.dilate((labels == keep).astype(np.uint8), np.ones((9, 9), np.uint8))  # keep soft edges
    return alpha * cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0)


# --- face points -------------------------------------------------------------------

def face_points(img: Image.Image) -> dict | None:
    """Eyes and mouth centers from DWPose's 68 face landmarks; None if no confident face."""
    from rtmlib import Wholebody
    pose = Wholebody(det=str(VISION / "DWPose" / "yolox_l.onnx"),
                     pose=str(VISION / "DWPose" / "dw-ll_ucoco_384.onnx"), backend="onnxruntime", device="cpu")
    kps, scores = pose(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR))
    if len(kps) == 0:
        return None
    best = int(np.argmax(scores[:, 23:91].mean(axis=1)))
    face, conf = kps[best, 23:91], scores[best, 23:91]
    if conf.mean() < 0.3:
        return None
    left_eye, right_eye, mouth = face[36:42].mean(0), face[42:48].mean(0), face[48:68].mean(0)
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    return {"left_eye": left_eye.tolist(), "right_eye": right_eye.tolist(), "mouth": mouth.tolist(),
            "eye_dist": eye_dist, "nose": face[30].tolist(), "confidence": float(conf.mean())}


def _ellipse_mask(shape, center, axes, feather) -> np.ndarray:
    m = np.zeros(shape[:2], np.float32)
    cv2.ellipse(m, (int(center[0]), int(center[1])), (max(1, int(axes[0])), max(1, int(axes[1]))), 0, 0, 360, 1.0, -1)
    k = max(3, int(feather) | 1)
    return cv2.GaussianBlur(m, (k, k), 0)


def merge_variant(base: Image.Image, variant: Image.Image, face: dict, part: str) -> Image.Image:
    """Paste only the mouth (or eyes) of the variant onto the base, so nothing else flickers."""
    b = np.asarray(base.convert("RGB"), np.float32)
    v = np.asarray(variant.convert("RGB").resize(base.size), np.float32)
    d = face["eye_dist"]
    if part == "mouth":
        mask = _ellipse_mask(b.shape, face["mouth"], (0.55 * d, 0.38 * d), 0.25 * d)
    else:
        mask = np.maximum(_ellipse_mask(b.shape, face["left_eye"], (0.42 * d, 0.3 * d), 0.2 * d),
                          _ellipse_mask(b.shape, face["right_eye"], (0.42 * d, 0.3 * d), 0.2 * d))
    out = b * (1 - mask[..., None]) + v * mask[..., None]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


# --- anime face, mouth & eyes (drawn, not generated) -----------------------------------
# TV anime animates mouths with a few simple drawn shapes; we do the same. Generated
# "open mouth" variants and real-face landmarks both proved unreliable on anime faces.

ANIMEFACE = VISION / "animeface" / "lbpcascade_animeface.xml"


def anime_face(img: Image.Image, alpha: np.ndarray | None = None) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) of the face: nagadomi's lbpcascade_animeface (strict, then relaxed),
    falling back to the biggest skin-coloured area in the top of the silhouette."""
    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    cascade = cv2.CascadeClassifier(str(ANIMEFACE))
    h_img = rgb.shape[0]
    for scale, neighbours in ((1.05, 4), (1.03, 2)):
        faces = [f for f in cascade.detectMultiScale(gray, scale, neighbours, minSize=(64, 64))
                 if f[1] < 0.5 * h_img]                     # faces are in the top half of a sprite
        if faces:
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    return _skin_face(rgb, alpha) if alpha is not None else None


def _skin_face(rgb: np.ndarray, alpha: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, _ = np.where(alpha > 0.5)
    if not len(ys):
        return None
    top, height = ys.min(), ys.max() - ys.min()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    skin = ((hsv[..., 0] < 25) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 15) & (hsv[..., 1] < 170) & (hsv[..., 2] > 110)
    skin &= alpha > 0.5
    skin[int(top + 0.45 * height):] = False                  # head and neck only
    n, _, stats, _ = cv2.connectedComponentsWithStats(skin.astype(np.uint8), 8)
    if n < 2:
        return None
    x, y, w, h, _ = stats[1 + int(np.argmax(stats[1:, 4]))]
    size = int(max(w, h) * 1.15)                             # hair usually covers the forehead
    return (int(x + w / 2 - size / 2), int(y + h - size), size, size)


def find_mouth(img: Image.Image, face, eyes=None) -> dict:
    """The drawn mouth line: darkest small stroke where anime faces put the mouth.
    With both eyes found, that's a fixed distance below them; otherwise the lower face."""
    fx, fy, fw, fh = face
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    if eyes and len(eyes) == 2:
        (ax, ay), (bx, by) = [(x + w / 2, y + h / 2) for x, y, w, h in eyes]
        dist, mx, my = abs(bx - ax), (ax + bx) / 2, (ay + by) / 2
        x0, x1 = int(mx - 0.4 * dist), int(mx + 0.4 * dist)
        y0, y1 = int(my + 0.35 * dist), int(my + 1.2 * dist)
        target = (mx - x0, my + 0.72 * dist - y0)
        max_w = 0.7 * dist
    else:
        x0, x1, y0, y1 = fx + int(0.3 * fw), fx + int(0.7 * fw), fy + int(0.62 * fh), fy + int(0.98 * fh)
        target = ((x1 - x0) / 2, (y1 - y0) / 2)
        max_w = 0.3 * fw
    region = rgb[max(0, y0):y1, max(0, x0):x1]
    lum = region.mean(axis=2)
    skin = np.median(lum)
    best = None
    for contrast in (45, 28):                 # strong lines first, then faint ones (light smiles)
        dark = (lum < skin - contrast).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
        best_score = 1e9
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < 0.04 * fw or w > max_w or area < 6 or h > 0.2 * fh:   # too small, or hair/collar lines
                continue
            score = abs(cents[i][0] - target[0]) + 0.6 * abs(cents[i][1] - target[1])
            if score < best_score:
                best, best_score = i, score
        if best is not None:
            break
    x0, y0 = max(0, x0), max(0, y0)
    if best is None:
        return {"x": fx + fw / 2, "y": fy + 0.72 * fh, "w": 0.12 * fw, "found": False,
                "mask": None, "face": face}
    x, y, w, h, _ = stats[best]
    mask = np.zeros(rgb.shape[:2], np.uint8)
    mask[y0:y1, x0:x1][labels == best] = 255
    return {"x": x0 + x + w / 2, "y": y0 + y + h / 2, "w": float(w), "found": True, "mask": mask, "face": face}


def _erase(img: np.ndarray, mask: np.ndarray, grow: int) -> np.ndarray:
    m = cv2.dilate(mask, np.ones((grow, grow), np.uint8))
    return cv2.inpaint(img, m, 5, cv2.INPAINT_TELEA)


def mouth_states(img: Image.Image, mouth: dict) -> dict[str, Image.Image]:
    """closed (original drawing), half and open mouths drawn in TV-anime style."""
    rgb = np.asarray(img.convert("RGB")).copy()
    fw = mouth["face"][2]
    base = _erase(rgb, mouth["mask"], max(3, int(fw * 0.02))) if mouth["mask"] is not None else rgb
    w = max(mouth["w"] * 0.9, 0.11 * fw)
    out = {"closed": img.convert("RGB")}
    for name, openness in (("half", 0.32), ("open", 0.62)):
        out[name] = _draw_mouth(base, mouth["x"], mouth["y"], w, w * openness)
    return out


def _draw_mouth(rgb: np.ndarray, cx: float, cy: float, w: float, h: float) -> Image.Image:
    """A rounded 'D' mouth: dark inside, pink tongue, thin outline. Drawn 4x and downsampled."""
    S = 4
    pad = int(w * 1.5)
    x0, y0 = int(cx - pad), int(cy - pad)
    patch = Image.fromarray(rgb[y0:y0 + 2 * pad, x0:x0 + 2 * pad]).resize((2 * pad * S, 2 * pad * S), Image.BICUBIC)
    shape = Image.new("L", patch.size, 0)
    d = ImageDraw.Draw(shape)
    c = pad * S
    top = c - h * S * 0.35
    box = (c - w * S / 2, top - h * S * 0.3, c + w * S / 2, top + h * S)
    d.chord(box, 0, 180, fill=255)                                    # flat-ish top, round bottom
    inside = Image.new("RGB", patch.size, (92, 28, 38))
    tongue = Image.new("L", patch.size, 0)
    ImageDraw.Draw(tongue).ellipse((c - w * S * 0.3, top + h * S * 0.45, c + w * S * 0.3, top + h * S * 1.15), fill=255)
    inside.paste((222, 112, 122), mask=Image.fromarray(np.minimum(np.asarray(tongue), np.asarray(shape))))
    outline = shape.filter(ImageFilter.MaxFilter(2 * S + 1))
    patch.paste((48, 18, 24), mask=outline)
    patch.paste(inside, mask=shape)
    patch = patch.resize((2 * pad, 2 * pad), Image.LANCZOS)
    out = Image.fromarray(rgb)
    out.paste(patch, (x0, y0))
    return out


def find_eyes(img: Image.Image, face) -> list[tuple[int, int, int, int]]:
    """Up to two eye boxes, found by the saturated iris colour in the face's eye band."""
    fx, fy, fw, fh = face
    hsv = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2HSV)
    x0, x1, y0, y1 = fx + int(0.05 * fw), fx + int(0.95 * fw), fy + int(0.3 * fh), fy + int(0.75 * fh)
    band = hsv[y0:y1, x0:x1]
    iris = ((band[..., 1] > 90) & (band[..., 2] > 50)).astype(np.uint8)
    iris = cv2.morphologyEx(iris, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(iris, 8)
    def iris_like(i) -> bool:  # small and roundish; long saturated strands are hair
        x, y, w, h, area = stats[i]
        return (area > 0.0015 * fw * fh and 0.04 * fw <= w <= 0.24 * fw and 0.04 * fh <= h <= 0.24 * fh
                and 0.4 <= w / h <= 2.5 and area > 0.35 * w * h)
    blobs = sorted((i for i in range(1, n) if iris_like(i)), key=lambda i: -stats[i][4])[:8]
    hue = {i: float(np.median(band[..., 0][labels == i])) for i in blobs}

    # Eyes come in pairs: side by side, at about the same height, size and colour.
    best, best_score = None, 0.0
    for a in blobs:
        for b in blobs:
            if a >= b:
                continue
            (ax, ay, aw, ah, aa), (bx, by, bw, bh, ba) = stats[a], stats[b]
            dy = abs((ay + ah / 2) - (by + bh / 2))
            dx = abs((ax + aw / 2) - (bx + bw / 2))
            dh = min(abs(hue[a] - hue[b]), 180 - abs(hue[a] - hue[b]))
            if dy > 0.15 * fh or dx < 0.2 * fw or dh > 18 or max(aa, ba) > 4 * min(aa, ba):
                continue
            score = aa + ba
            if score > best_score:
                best, best_score = (a, b), score
    picks = best or blobs[:1]
    return [(x0 + stats[i][0], y0 + stats[i][1], stats[i][2], stats[i][3]) for i in picks]


def blink_state(img: Image.Image, eyes, face) -> Image.Image | None:
    """Closed eyes: paint over each eye with nearby skin and draw a lash arc."""
    if len(eyes) != 2:  # one closed eye looks like a wink; skip blinking instead
        return None
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    arcs = []
    for x, y, w, h in eyes:
        cx, cy = x + w / 2, y + h / 2
        ew, eh = w * 1.9, h * 1.6                      # iris → whole eye incl. whites and lashes
        # skin colour from the cheek just below the eye
        cheek = rgb[int(cy + eh * 0.75):int(cy + eh * 1.2), int(cx - w * 0.4):int(cx + w * 0.4)].reshape(-1, 3)
        skin = np.median(cheek, axis=0) if len(cheek) else np.array([240, 215, 200], np.float32)
        lid = _ellipse_mask(rgb.shape, (cx, cy), (ew / 2, eh / 2), max(3, ew * 0.12))
        # keep strands of hair that hang over the eye (hair colour = just above the eye)
        above = rgb[max(0, int(cy - eh * 1.1)):max(1, int(cy - eh * 0.6)), int(cx - w * 0.5):int(cx + w * 0.5)].reshape(-1, 3)
        if len(above):
            hair = np.median(above, axis=0)
            if np.linalg.norm(hair - skin) > 60:   # only when hair clearly differs from skin
                lid = lid * (np.linalg.norm(rgb - hair, axis=2) > 45)
        lid = lid[..., None]
        rgb = rgb * (1 - lid) + skin * lid
        arcs.append((cx, cy + eh * 0.12, ew))
    out = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    for cx, cy, ew in arcs:
        d.arc((cx - ew / 2, cy - ew * 0.28, cx + ew / 2, cy + ew * 0.28), 15, 165,
              fill=(38, 28, 40), width=max(3, int(ew * 0.07)))
    return out


def eye_mask(img: Image.Image, lm: dict) -> Image.Image:
    """Soft mask over both eyes (the vision model's boxes, slightly enlarged)."""
    m = np.zeros((img.height, img.width), np.float32)
    for x0, y0, x1, y1 in (lm["left_eye"], lm["right_eye"]):
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * 0.68, (y1 - y0) * 0.72
        m = np.maximum(m, _ellipse_mask(m.shape, (cx, cy), (rx, ry), max(5, rx * 0.35)))
    return Image.fromarray((m * 255).clip(0, 255).astype(np.uint8), "L")


def parts_from_landmarks(img: Image.Image, lm: dict):
    """(face, eyes, mouth) from the vision model's boxes (see landmarks.py), snapping the
    mouth to the actual drawn mouth line near the model's box when there is one."""
    (lx0, ly0, lx1, ly1), (rx0, ry0, rx1, ry1) = lm["left_eye"], lm["right_eye"]
    ecx = ((lx0 + lx1) / 2, (rx0 + rx1) / 2)
    ecy = ((ly0 + ly1) / 2, (ry0 + ry1) / 2)
    eye_dist = max(20.0, abs(ecx[1] - ecx[0]))
    mid_x, mid_y = sum(ecx) / 2, sum(ecy) / 2
    face = (int(mid_x - 1.25 * eye_dist), int(mid_y - 0.9 * eye_dist), int(2.5 * eye_dist), int(2.5 * eye_dist))

    # Eyes, as iris-sized boxes (blink_state grows them back to the whole eye). Skip
    # blinking when the "eyes" are glasses: bright, flat lenses.
    rgb = np.asarray(img.convert("RGB"))
    eyes = []
    for x0, y0, x1, y1 in (lm["left_eye"], lm["right_eye"]):
        patch = rgb[int(y0):int(y1), int(x0):int(x1)].mean(axis=2)
        if patch.size and patch.mean() < 200 and patch.std() > 25:
            w, h = (x1 - x0) * 0.8 / 1.9, (y1 - y0) * 0.8 / 1.6  # blink covers ~80% of the box
            eyes.append((int((x0 + x1) / 2 - w / 2), int((y0 + y1) / 2 - h / 2), int(w), int(h)))

    mx0, my0, mx1, my1 = lm["mouth"]
    bw, bh = mx1 - mx0, max(my1 - my0, 0.12 * eye_dist)
    cx, cy = (mx0 + mx1) / 2, (my0 + my1) / 2
    mouth = {"x": cx, "y": cy, "w": float(np.clip(bw * 0.8, 0.3 * eye_dist, 0.6 * eye_dist)),
             "found": True, "mask": None, "face": face}
    # Erase the original closed-mouth line inside the (refined) box before drawing over it.
    region = rgb[int(my0):int(my1) + 1, int(mx0):int(mx1) + 1].astype(np.int16)
    if region.size:
        lum = region.mean(axis=2)
        dark = (lum < np.median(lum) - 30).astype(np.uint8) * 255
        if dark.any():
            mask = np.zeros(rgb.shape[:2], np.uint8)
            mask[int(my0):int(my1) + 1, int(mx0):int(mx1) + 1] = dark
            mouth["mask"] = mask
    return face, eyes, mouth


# --- depth -------------------------------------------------------------------------

class Depth:
    def __init__(self) -> None:
        from .third_party.depth_anything_v2.dpt import DepthAnythingV2   # vendored, Apache-2.0
        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(VISION / "Depth-Anything-V2-Small" / "depth_anything_v2_vits.pth",
                                              map_location="cpu"))
        self.model = self.model.to("mps").eval()

    @torch.no_grad()
    def map(self, img: Image.Image) -> np.ndarray:
        """Relative nearness in [0, 1] (1 = closest), at the image's size."""
        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        d = self.model.infer_image(bgr, 518).astype(np.float32)
        d = (d - d.min()) / max(1e-6, d.max() - d.min())
        return cv2.GaussianBlur(d, (0, 0), 3)
