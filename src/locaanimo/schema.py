"""Episode script schema: validates a user-written YAML script before any rendering starts."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    narrator_voice: str = "auto"


class Character(Strict):
    id: str
    name: str
    look: str
    age: Literal["child", "teen", "adult", "elder"] = "adult"
    gender: Literal["male", "female", "neutral"] = "neutral"
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

    @property
    def big_action_count(self) -> int:
        return sum(isinstance(b, ActionBeat) and b.big for s in self.scenes for b in s.beats)


def load_script(path: str | Path) -> Script:
    with open(path, encoding="utf-8") as f:
        return Script.model_validate(yaml.safe_load(f))
