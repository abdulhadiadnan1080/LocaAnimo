"""Build docs/media/cast.png: a gallery of characters LocaAnimo has drawn, for the README.

  python scripts/dev/make_cast_sheet.py

Each character is cut out with the same BiRefNet model the renderer uses and shown on a
soft card with its name and the one-line description from its script.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from locaanimo import prep  # noqa: E402

# (project, character id, card colour): picked to show range: ages, styles, beards, glasses.
CAST = [
    ("ember", "rin", (255, 231, 226)),
    ("ember", "kaito", (226, 232, 245)),
    ("lantern-keeper", "tomo", (246, 240, 220)),
    ("lantern-keeper", "hana", (226, 240, 236)),
    ("clockmaker", "ivy", (240, 236, 222)),
    ("star-cartographers", "zara", (242, 228, 245)),
    ("star-cartographers", "hugo", (232, 228, 248)),
    ("fox-and-ferryman", "osei", (226, 238, 246)),
    ("kite", "rose", (232, 242, 228)),
    ("star-cartographers", "momo", (236, 244, 226)),
]
COLS, CARD_W, CARD_H, GAP = 5, 300, 420, 18
FONT = "/System/Library/Fonts/Helvetica.ttc"


def describe(project: str, cid: str) -> tuple[str, str]:
    path = ROOT / "projects" / project / "script.yaml"
    if not path.exists():   # rendered straight from an example script
        path = ROOT / "examples" / {"ember": "ember_protocol.yaml", "kite": "the_lost_kite.yaml"}[project]
    script = yaml.safe_load(path.read_text())
    c = next(x for x in script["characters"] if x["id"] == cid)
    look = c["look"].split(",")
    return c["name"], ",".join(look[:2]).strip()


def card(img: Image.Image, alpha: np.ndarray, colour, name: str, line: str) -> Image.Image:
    out = Image.new("RGB", (CARD_W, CARD_H), colour)
    ys, xs = np.where(alpha > 0.3)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    fig = rgba.crop(box)
    area_h = CARD_H - 86
    k = min((CARD_W - 24) / fig.width, area_h / fig.height)
    fig = fig.resize((int(fig.width * k), int(fig.height * k)), Image.LANCZOS)
    shadow = Image.new("RGBA", fig.size, (0, 0, 0, 0))
    shadow.putalpha(fig.getchannel("A").point(lambda a: a * 0.18))
    x, y = (CARD_W - fig.width) // 2, area_h - fig.height + 8
    out.paste(shadow.filter(ImageFilter.GaussianBlur(8)), (x + 6, y + 8), shadow.filter(ImageFilter.GaussianBlur(8)))
    out.paste(fig, (x, y), fig)
    d = ImageDraw.Draw(out)
    d.rectangle((0, CARD_H - 78, CARD_W, CARD_H), fill=(255, 255, 255))
    d.text((16, CARD_H - 66), name, font=ImageFont.truetype(FONT, 22, index=1), fill=(17, 24, 39))
    if len(line) > 34:                      # cut at a word, not mid-word
        line = line[:34].rsplit(" ", 1)[0].rstrip(",") + "…"
    d.text((16, CARD_H - 36), line, font=ImageFont.truetype(FONT, 14),
           fill=(107, 114, 128))
    mask = Image.new("L", out.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, CARD_W - 1, CARD_H - 1), 18, fill=255)
    rounded = Image.new("RGBA", out.size, (0, 0, 0, 0))
    rounded.paste(out, (0, 0), mask)
    return rounded


def main() -> None:
    cutter = prep.Cutter()
    rows = (len(CAST) + COLS - 1) // COLS
    sheet = Image.new("RGBA", (COLS * CARD_W + (COLS + 1) * GAP, rows * CARD_H + (rows + 1) * GAP), (255, 255, 255, 255))
    for i, (project, cid, colour) in enumerate(CAST):
        img = Image.open(ROOT / "projects" / project / "art" / f"char_{cid}.png").convert("RGB")
        name, line = describe(project, cid)
        c = card(img, cutter.alpha(img), colour, name, line)
        sheet.paste(c, (GAP + (i % COLS) * (CARD_W + GAP), GAP + (i // COLS) * (CARD_H + GAP)), c)
    out = ROOT / "docs" / "media" / "cast.png"
    sheet.convert("RGB").save(out, optimize=True)
    print(out)


if __name__ == "__main__":
    main()
