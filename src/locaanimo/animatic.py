"""Storyboard cards for the animatic preview.

Until the art and animation stages land, every beat renders as a styled card so the
whole pipeline runs end to end with real voices and timing.
"""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
MARGIN = 90

# Speaker name colors, assigned by cast order.
PALETTE = ["#FF8A5B", "#5BC0FF", "#C792EA", "#7EE787", "#FFD166", "#FF6B9A"]

TIME_TINT = {"day": (0.0, 1.0), "sunset": (0.06, 0.9), "night": (0.62, 0.55)}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path, index in (("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
                        ("/System/Library/Fonts/SFNS.ttf", 0)):
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _background(location_id: str, time: str) -> Image.Image:
    """Vertical gradient whose hue comes from the location and is tinted by time of day."""
    hue = int(hashlib.md5(location_id.encode()).hexdigest()[:4], 16) / 0xFFFF
    tint_hue, brightness = TIME_TINT.get(time, (0.0, 1.0))
    if time != "day":
        hue = (tint_hue + (hue - 0.5) * 0.08) % 1.0  # time of day dominates; location only nudges it
    top = np.array(colorsys.hsv_to_rgb(hue, 0.55, 0.42 * brightness)) * 255
    bottom = np.array(colorsys.hsv_to_rgb(hue, 0.35, 0.08)) * 255
    t = np.linspace(0, 1, H)[:, None]
    rows = top * (1 - t) + bottom * t
    img = np.repeat(rows[:, None, :], W, axis=1).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _text_block(draw, text, font, y, fill, center=False, line_gap=1.25) -> int:
    for line in _wrap(draw, text, font, W - 2 * MARGIN):
        x = (W - draw.textlength(line, font=font)) / 2 if center else MARGIN
        draw.text((x, y), line, font=font, fill=fill)
        y += int(font.size * line_gap)
    return y


def _badge(draw, text, x, y, color) -> None:
    font = _font(20, bold=True)
    w = draw.textlength(text, font=font)
    draw.rounded_rectangle((x, y, x + w + 28, y + 36), radius=18, fill=color)
    draw.text((x + 14, y + 7), text, font=font, fill="#111111")


def render_card(kind: str, *, scene_no: int, location_id: str, location_look: str, time: str,
                text: str = "", speaker: str = "", speaker_color: str = "#FFFFFF",
                detail: str = "", big: bool = False) -> Image.Image:
    img = _background(location_id, time)
    draw = ImageDraw.Draw(img)
    header = f"SCENE {scene_no}  ·  {location_id.upper()}  ·  {time.upper()}"
    draw.text((MARGIN, 60), header, font=_font(22, bold=True), fill=(255, 255, 255, 150))
    draw.text((W - MARGIN - 210, H - 50), "LocaAnimo · animatic", font=_font(18), fill="#8a8a8a")

    if kind == "scene":
        _text_block(draw, f"Scene {scene_no}", _font(84, bold=True), 250, "#FFFFFF", center=True)
        _text_block(draw, location_look, _font(28), 370, "#D0D0D0", center=True)
    elif kind == "camera":
        _text_block(draw, text.replace("_", " ").upper() + " SHOT", _font(64, bold=True), 270,
                    "#FFFFFF", center=True)
        _text_block(draw, location_look, _font(28), 370, "#D0D0D0", center=True)
    elif kind == "action":
        _badge(draw, "AI ACTION SHOT" if big else "ACTION", MARGIN, 200, "#FFD166" if big else "#DDDDDD")
        y = _text_block(draw, text, _font(50, bold=True), 270, "#FFFFFF")
        if detail:
            draw.text((MARGIN, y + 20), detail, font=_font(26), fill="#BBBBBB")
    elif kind == "narration":
        draw.text((MARGIN, 220), "NARRATOR", font=_font(26, bold=True), fill="#AAAAAA")
        _text_block(draw, f"“{text}”", _font(46), 280, "#EEEEEE")
    elif kind == "line":
        name_font = _font(34, bold=True)
        draw.text((MARGIN, 330), speaker, font=name_font, fill=speaker_color)
        if detail:
            x = MARGIN + draw.textlength(speaker, font=name_font) + 18
            draw.text((x, 338), detail, font=_font(26), fill="#AAAAAA")
        _text_block(draw, text, _font(52, bold=True), 390, "#FFFFFF")
    return img
