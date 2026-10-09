"""Episode script schema: validates a user-written YAML script before any rendering starts."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Nonverbal sounds a line can contain, performed by the voice engine and hidden from subtitles.
NONVERBAL_TAGS = ("laugh", "chuckle", "sigh", "gasp", "cough", "groan", "sniff", "clear throat", "shush")
_TAG = re.compile(r"\[([^\]]+)\]")


# Adult content is allowed for adult characters. Anything sexual is never allowed in a script
# that has a child or teen character: the script is rejected before any art is drawn.
MINOR_AGES = ("child", "teen")
EXPLICIT = re.compile(
    r"\b(nsfw|nude|nudity|naked|topless|bottomless|undress\w*|strip(s|ped|ping)?|lingerie|"
    r"erotic\w*|explicit|sex|sexy|sexual\w*|porn\w*|hentai|nipples?|genitals?|breasts?|"
    r"orgasm\w*|aroused|seduc\w*|fetish\w*|intercourse|make love|making love|lewd)\b", re.I)


def is_explicit(text: str) -> bool:
    return bool(EXPLICIT.search(text or ""))


def strip_tags(text: str) -> str:
    """Text without [laugh]-style tags, for subtitles and voices that can't perform them."""
    return re.sub(r"\s{2,}", " ", _TAG.sub("", text)).strip()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Style(str, Enum):
    anime_tv = "anime_tv"
    chibi = "chibi"
    cartoon = "cartoon"


class Emotion(str, Enum):
    neutral = "neutral"
    happy = "happy"
    sad = "sad"
    angry = "angry"
    surprised = "surprised"
    worried = "worried"
    determined = "determined"
    confident = "confident"
    amazed = "amazed"
    scared = "scared"
    embarrassed = "embarrassed"


class Gesture(str, Enum):
    points = "points"
    waves = "waves"
    nods = "nods"
    shakes_head = "shakes_head"
    shrugs = "shrugs"
    crosses_arms = "crosses_arms"
    fist_pump = "fist_pump"
    facepalm = "facepalm"


class Camera(str, Enum):
    establishing = "establishing"
    wide = "wide"
    medium = "medium"
    close_up = "close_up"
    over_shoulder = "over_shoulder"
    low_angle = "low_angle"
    high_angle = "high_angle"
    pan_left = "pan_left"
    pan_right = "pan_right"
    zoom_in = "zoom_in"
    shake = "shake"


class Episode(Strict):
    title: str
    style: Style = Style.anime_tv
    resolution: Literal["720p", "1080p"] = "1080p"
    music: bool = True
    subtitles: bool = True
    narrator_voice: str = "auto"   # a voice description (see Character.voice)


class Character(Strict):
    id: str
    name: str
    look: str
    age: Literal["child", "teen", "adult", "elder"] = "adult"
    gender: Literal["male", "female", "neutral"] = "neutral"
    # A description of the voice, e.g. "teen girl, bright, a bit sarcastic". "auto" builds one
    # from age and gender; an exact Kokoro id (e.g. "af_heart") uses the fast Kokoro voice.
    voice: str = "auto"


class Location(Strict):
    id: str
    look: str


# --- beats: each beat is exactly one kind -------------------------------------

class Line(Strict):
    who: str
    text: str
    emotion: Emotion = Emotion.neutral
    gesture: Gesture | None = None

    @field_validator("text")
    @classmethod
    def known_tags(cls, text: str) -> str:
        unknown = [t for t in _TAG.findall(text) if t.strip().lower() not in NONVERBAL_TAGS]
        if unknown:
            raise ValueError(f"unknown sound tag [{unknown[0]}]; use one of "
                             + ", ".join(f"[{t}]" for t in NONVERBAL_TAGS))
        return text


class LineBeat(Strict):
    line: Line
    camera: Camera | None = None


class ActionBeat(Strict):
    action: str
    characters: list[str] = []
    big: bool = False
    sfx: list[str] = []
    camera: Camera | None = None


class NarrationBeat(Strict):
    narration: str
    camera: Camera | None = None


class CameraBeat(Strict):
    camera: Camera


class PauseBeat(Strict):
    pause: Annotated[float, Field(gt=0, le=10)]


Beat = Union[LineBeat, ActionBeat, NarrationBeat, PauseBeat, CameraBeat]


class Scene(Strict):
    id: str
    location: str
    time: Literal["day", "sunset", "night"] = "day"
    mood: Literal["calm", "tense", "happy", "sad", "epic"] = "calm"
    beats: list[Beat] = Field(min_length=1)


class Script(Strict):
    episode: Episode
    characters: list[Character] = Field(min_length=1)
    locations: list[Location] = Field(min_length=1)
    scenes: list[Scene] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references(self) -> Script:
        """Every id used in a scene must be defined, so typos fail fast."""
        char_ids = {c.id for c in self.characters}
        loc_ids = {l.id for l in self.locations}
        errors = []
        for kind, ids in (("character", [c.id for c in self.characters]),
                          ("location", [l.id for l in self.locations]),
                          ("scene", [s.id for s in self.scenes])):
            dupes = {i for i in ids if ids.count(i) > 1}
            if dupes:
                errors.append(f"duplicate {kind} id(s): {sorted(dupes)}")
        for scene in self.scenes:
            if scene.location not in loc_ids:
                errors.append(f"scene {scene.id}: unknown location '{scene.location}'")
            for n, beat in enumerate(scene.beats, 1):
                used = []
                if isinstance(beat, LineBeat):
                    used = [beat.line.who]
                elif isinstance(beat, ActionBeat):
                    used = beat.characters
                for who in used:
                    if who not in char_ids:
                        errors.append(f"scene {scene.id}, beat {n}: unknown character '{who}'")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @model_validator(mode="after")
    def protect_minors(self) -> Script:
        """Explicit content is for adult characters only. If any character is a child or a
        teen, nothing in the script may be sexual or explicit."""
        minors = [c.name for c in self.characters if c.age in MINOR_AGES]
        if not minors:
            return self
        texts = [self.episode.title] + [c.look for c in self.characters] + [l.look for l in self.locations]
        for scene in self.scenes:
            for b in scene.beats:
                texts += [getattr(b, "action", ""), getattr(b, "narration", ""),
                          b.line.text if isinstance(b, LineBeat) else ""]
        hit = next((m.group(0) for t in texts if (m := EXPLICIT.search(t or ""))), None)
        if hit:
            raise ValueError(f"explicit content (\"{hit}\") isn't allowed in a script with a child or teen "
                             f"character ({', '.join(minors)}). Explicit stories may only have adult characters.")
        return self

    @property
    def big_action_count(self) -> int:
        return sum(isinstance(b, ActionBeat) and b.big for s in self.scenes for b in s.beats)


def load_script(path: str | Path) -> Script:
    with open(path, encoding="utf-8") as f:
        return Script.model_validate(yaml.safe_load(f))
