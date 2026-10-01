"""Kokoro voice engine: local-only synthesis, voice catalog, auto-casting and cached previews."""

from __future__ import annotations

import hashlib
import io
import os
import threading

# Never let any library fetch files on its own; downloads are always explicit.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import soundfile as sf

from .paths import CACHE, MODELS
from .schema import Character

KOKORO_DIR = MODELS / "tts" / "Kokoro-82M"
KOKORO_REPO = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000

ACCENTS = {"a": "American", "b": "British"}
GENDERS = {"f": "female", "m": "male"}

# Highest-rated English voices in Kokoro's VOICES.md; listed first in the picker.
FEATURED = {"af_heart", "af_bella", "af_nicole", "bf_emma", "am_fenrir", "am_michael", "am_puck"}

# Auto-casting pools per (gender, age), best match first. Kokoro has no true child
# voices, so children get the lightest-sounding ones.
CASTING = {
    ("female", "child"): ["af_sky", "af_nova", "af_jessica"],
    ("female", "teen"): ["af_heart", "af_sky", "af_jessica", "af_nova"],
    ("female", "adult"): ["af_bella", "af_nicole", "bf_emma", "af_sarah", "af_aoede", "af_kore"],
    ("female", "elder"): ["bf_isabella", "bf_alice", "af_river"],
    ("male", "child"): ["am_puck", "am_echo"],
    ("male", "teen"): ["am_puck", "am_echo", "am_liam"],
    ("male", "adult"): ["am_michael", "am_fenrir", "am_adam", "am_eric", "bm_daniel", "am_onyx"],
    ("male", "elder"): ["bm_george", "bm_lewis", "am_onyx"],
}
NEUTRAL = ["af_river", "af_alloy", "am_echo"]
NARRATORS = ["bm_fable", "bm_george", "af_river", "am_michael"]

# Kokoro has no emotion control; delivery speed is the one lever we have.
EMOTION_SPEED = {
    "neutral": 1.0, "happy": 1.05, "sad": 0.88, "angry": 1.1, "surprised": 1.08,
    "worried": 0.97, "determined": 1.0, "confident": 0.98, "amazed": 1.05,
    "scared": 1.12, "embarrassed": 0.95,
}


class VoiceUnavailable(RuntimeError):
    pass


def list_voices() -> list[dict]:
    voices = []
    for path in sorted((KOKORO_DIR / "voices").glob("*.pt")):
        vid = path.stem
        if len(vid) < 4 or vid[0] not in ACCENTS or vid[1] not in GENDERS:
            continue  # non-English voice
        voices.append({
            "id": vid,
            "name": vid[3:].capitalize(),
            "accent": ACCENTS[vid[0]],
            "gender": GENDERS[vid[1]],
            "featured": vid in FEATURED,
        })
    voices.sort(key=lambda v: (v["accent"], v["gender"], not v["featured"], v["name"]))
    return voices


def voice_ids() -> set[str]:
    return {v["id"] for v in list_voices()}


def cast_voices(characters: list[Character], narrator: str = "auto") -> tuple[dict[str, str], str]:
    """Give every character a distinct voice; explicit choices are kept as-is."""
    available = voice_ids()
    used = {c.voice for c in characters if c.voice != "auto"}
    cast = {}
    for c in characters:
        if c.voice != "auto":
            cast[c.id] = c.voice
            continue
        pool = [v for v in CASTING.get((c.gender, c.age), NEUTRAL) if v in available]
        pick = next((v for v in pool if v not in used), pool[0] if pool else "af_heart")
        cast[c.id] = pick
        used.add(pick)
    if narrator == "auto":
        narrator = next((v for v in NARRATORS if v not in used and v in available), NARRATORS[0])
    return cast, narrator


class VoiceEngine:
    """One shared Kokoro model; synthesis is serialized because the model isn't thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._pipelines: dict[str, object] = {}
        self._ready = False

    def missing(self) -> str | None:
        """Why voices can't run yet, or None if everything is in place."""
        if self._ready:
            return None
        if not (KOKORO_DIR / "kokoro-v1_0.pth").exists():
            return "Kokoro model files are missing from models/tts/Kokoro-82M."
        import spacy
        if not spacy.util.is_package("en_core_web_sm"):
            return ("The English pronunciation dictionary (spaCy en_core_web_sm, ~13 MB) "
                    "isn't installed yet.")
        self._ready = True
        return None

    def _pipeline(self, lang: str):
        from kokoro import KModel, KPipeline
        if self._model is None:
            self._model = KModel(
                repo_id=KOKORO_REPO,
                config=str(KOKORO_DIR / "config.json"),
                model=str(KOKORO_DIR / "kokoro-v1_0.pth"),
            ).eval()
        if lang not in self._pipelines:
            self._pipelines[lang] = KPipeline(lang_code=lang, repo_id=KOKORO_REPO, model=self._model)
        return self._pipelines[lang]

    def synth(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        problem = self.missing()
        if problem:
            raise VoiceUnavailable(problem)
        voice_file = KOKORO_DIR / "voices" / f"{voice}.pt"
        if not voice_file.exists():
            raise VoiceUnavailable(f"Unknown voice '{voice}'.")
        with self._lock:
            pipe = self._pipeline(voice[0])
            chunks = [r.audio.numpy() for r in pipe(text, voice=str(voice_file), speed=speed)
                      if r.audio is not None]
        return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)

    def preview(self, text: str, voice: str, emotion: str = "neutral") -> bytes:
        """WAV bytes for a short sample, cached on disk so repeat previews are instant."""
        speed = EMOTION_SPEED.get(emotion, 1.0)
        key = hashlib.sha1(f"{voice}|{speed}|{text}".encode()).hexdigest()[:16]
        path = CACHE / "voice_previews" / f"{voice}_{key}.wav"
        if path.exists():
            return path.read_bytes()
        audio = self.synth(text, voice, speed)
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.getvalue())
        return buf.getvalue()


engine = VoiceEngine()
