"""Illustrious XL image generation (diffusers on Apple MPS): prompts, drawing, redrawing."""

from __future__ import annotations

import hashlib

import torch
from PIL import Image

from .paths import MODELS

ILLUSTRIOUS = MODELS / "image" / "Illustrious-XL-v2.0" / "Illustrious-XL-v2.0.safetensors"

QUALITY = "masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest"
STYLE = {
    "anime_tv": "anime screencap, anime coloring, clean lineart",
    "chibi": "chibi, cute, super deformed, anime coloring",
    "cartoon": "cartoon, flat colors, bold outlines",
}
NEGATIVE = ("worst quality, low quality, lowres, bad anatomy, bad hands, extra fingers, missing fingers, "
            "sketch, monochrome, jpeg artifacts, text, watermark, signature, logo, blurry")
TIME_LIGHT = {"day": "daytime, bright sunlight", "sunset": "sunset, golden hour, orange sky, warm lighting",
              "night": "night, moonlight, dark blue sky, city lights"}
COUNT = {"male": "1boy", "female": "1girl", "neutral": "1other"}


def seed_for(key: str) -> int:
    """Stable seed per character/location, so re-runs draw the same art."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


AGE_TAGS = {"child": "child", "teen": "teenager, 17 years old", "adult": "adult, 25 years old", "elder": "old, elderly, wrinkles"}
LOWER_GARMENTS = ("pants", "jeans", "shorts", "skirt", "dress", "trousers", "leggings", "kimono", "robe", "overalls")
CHARACTER_NEGATIVE = "scenery, room, wall, door, indoors, outdoors, furniture, multiple views, bottomless, no pants"


def character_prompt(look: str, gender: str, style: str, expression: str = "closed mouth, light smile",
                     age: str = "adult") -> str:
    # Sprites need a plain backdrop (for the cutout) and a full outfit (the model often drops trousers).
    # CLIP reads only ~77 tokens: what must not be lost goes first, quality tags last.
    clothes = "" if any(g in look.lower() for g in LOWER_GARMENTS) else ", long pants"
    style_tags = STYLE[style].replace("anime screencap, ", "")  # "screencap" pulls in backgrounds
    return (f"{COUNT[gender]}, solo, white background, simple background, {AGE_TAGS[age]}, {look}{clothes}, "
            f"standing, cowboy shot, facing viewer, arms at sides, {expression}, {style_tags}, {QUALITY}")


def character_negative(age: str) -> str:
    young = "" if age == "child" else ", child, loli, shota, chibi, kid"
    return CHARACTER_NEGATIVE + young


def location_prompt(look: str, time: str, style: str) -> str:
    return f"scenery, no humans, {look}, {TIME_LIGHT[time]}, wide shot, detailed background, {STYLE[style]}, {QUALITY}"


def action_prompt(text: str, looks: list[str], time: str, style: str) -> str:
    who = ", ".join(looks)
    return (f"{text}, {who}, dynamic pose, dynamic angle, action shot, motion lines, {TIME_LIGHT[time]}, "
            f"cinematic composition, {STYLE[style]}, {QUALITY}")


class Artist:
    """Loads Illustrious once; draws new images and redraws variants of existing ones."""

    def __init__(self, steps: int = 20, cfg: float = 5.5) -> None:
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
        self.pipe = StableDiffusionXLPipeline.from_single_file(str(ILLUSTRIOUS), torch_dtype=torch.float16).to("mps")
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        self.img2img = StableDiffusionXLImg2ImgPipeline(**self.pipe.components)
        self.steps, self.cfg = steps, cfg

    def draw(self, prompt: str, seed: int, width: int, height: int, negative: str = "") -> Image.Image:
        return self.pipe(prompt, negative_prompt=f"{NEGATIVE}, {negative}".strip(", "),
                         width=width, height=height, num_inference_steps=self.steps, guidance_scale=self.cfg,
                         generator=torch.Generator("cpu").manual_seed(seed)).images[0]

    def inpaint(self, image: Image.Image, mask: Image.Image, prompt: str, seed: int,
                strength: float = 0.9, negative: str = "") -> Image.Image:
        """Repaint only the white area of `mask` (e.g. the eyes), matching the drawing's style.
        Only the masked area is kept, so nothing else in the drawing shifts."""
        from diffusers import StableDiffusionXLInpaintPipeline
        if not hasattr(self, "_inpaint"):
            self._inpaint = StableDiffusionXLInpaintPipeline(**self.pipe.components)
        out = self._inpaint(prompt, image=image, mask_image=mask, negative_prompt=f"{NEGATIVE}, {negative}".strip(", "),
                            strength=strength, num_inference_steps=self.steps, guidance_scale=self.cfg,
                            width=image.width, height=image.height,
                            generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        return Image.composite(out.resize(image.size), image, mask.convert("L"))

    def redraw(self, image: Image.Image, prompt: str, seed: int, strength: float = 0.45,
               negative: str = "") -> Image.Image:
        """Same picture, small change (e.g. open mouth): low strength keeps the composition."""
        return self.img2img(prompt, image=image, negative_prompt=f"{NEGATIVE}, {negative}".strip(", "), strength=strength,
                            num_inference_steps=self.steps, guidance_scale=self.cfg,
                            generator=torch.Generator("cpu").manual_seed(seed)).images[0]
