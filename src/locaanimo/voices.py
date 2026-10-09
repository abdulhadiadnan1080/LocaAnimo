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


NARRATOR_VOICE = "warm, rich mature storyteller voice, clear and expressive"
AGE_GENDER = {
    ("child", "male"): "young boy", ("child", "female"): "young girl", ("child", "neutral"): "young child",
    ("teen", "male"): "teenage boy", ("teen", "female"): "teenage girl", ("teen", "neutral"): "teenager",
    ("adult", "male"): "adult man", ("adult", "female"): "adult woman", ("adult", "neutral"): "adult",
    ("elder", "male"): "old man", ("elder", "female"): "old woman", ("elder", "neutral"): "elderly person",
}
# One distinguishing trait per auto-described character, so voices in a cast differ.
TRAITS = ["warm and friendly", "bright and lively", "husky and relaxed", "crisp and clear",
          "gentle and soft-spoken", "deep and steady", "quick and energetic", "smooth and calm"]


def is_kokoro_id(voice: str) -> bool:
    return voice in voice_ids()


def describe_voice(c: Character, index: int = 0) -> str:
    """The voice description a character uses: theirs, or one built from age and gender."""
    if c.voice and c.voice != "auto":
        return c.voice
    return f"{AGE_GENDER[(c.age, c.gender)]}, {TRAITS[index % len(TRAITS)]}"


def cast_voices(characters: list[Character], narrator: str = "auto") -> tuple[dict[str, str], str]:
    """Every character's voice spec: a description for the expressive engine, or a Kokoro
    voice id when the script names one. Used for rendering and shown in the studio."""
    cast = {c.id: describe_voice(c, i) for i, c in enumerate(characters)}
    return cast, (narrator if narrator and narrator != "auto" else NARRATOR_VOICE)


def kokoro_cast(characters: list[Character], narrator: str = "auto") -> tuple[dict[str, str], str]:
    """Fallback casting with Kokoro's preset voices: distinct per character."""
    available = voice_ids()
    used = {c.voice for c in characters if c.voice in available}
    cast = {}
    for c in characters:
        if c.voice in available:
            cast[c.id] = c.voice
            continue
        pool = [v for v in CASTING.get((c.gender, c.age), NEUTRAL) if v in available]
        pick = next((v for v in pool if v not in used), pool[0] if pool else "af_heart")
        cast[c.id] = pick
        used.add(pick)
    if narrator not in available:
        narrator = next((v for v in NARRATORS if v not in used and v in available), NARRATORS[0])
    return cast, narrator


def synth_lines(lines: list[dict], work_dir, log=print) -> list[np.ndarray]:
    """Speak every line of an episode, in order. Each line: {"text", "voice", "emotion",
    "fallback"} where voice is a description (expressive engine) or a Kokoro id, and
    fallback is the Kokoro voice to use if the expressive engine can't. Returns 24 kHz audio."""
    import json
    import subprocess
    import sys
    from . import speech
    from .paths import ROOT
    from .schema import strip_tags

    out: list[np.ndarray | None] = [None] * len(lines)
    expressive = [i for i, l in enumerate(lines) if not is_kokoro_id(l["voice"])]
    if expressive and speech.available():
        work = work_dir / "voices"
        work.mkdir(parents=True, exist_ok=True)
        jobs = [{"id": str(i), "text": lines[i]["text"], "voice": lines[i]["voice"],
                 "emotion": lines[i]["emotion"], "out": str(work / f"line_{i:03d}.wav")} for i in expressive]
        (work / "jobs.json").write_text(json.dumps({"lines": jobs}, indent=2))
        proc = subprocess.Popen([sys.executable, "-m", "locaanimo.speech", "batch", str(work / "jobs.json")],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        for msg in proc.stdout:
            if msg.startswith("speech:") and "ALL DONE" not in msg:
                log(f"render: {msg.strip()}")
        proc.wait()
        results_file = work / "results.json"
        results = json.loads(results_file.read_text()) if results_file.exists() else {}
        for job in jobs:
            if results.get(job["id"], {}).get("ok"):
                audio, sr = sf.read(job["out"], dtype="float32")
                out[int(job["id"])] = audio if audio.ndim == 1 else audio.mean(axis=1)
        if any(out[i] is None for i in expressive):
            log("render: some lines couldn't use expressive voices; those use the fast voices")
    elif expressive:
        log("render: expressive voice models not downloaded; using the fast voices (Kokoro)")
    for i, l in enumerate(lines):   # Kokoro for named Kokoro voices and anything that failed
        if out[i] is None:
            voice = l["voice"] if is_kokoro_id(l["voice"]) else l["fallback"]
            out[i] = engine.synth(strip_tags(l["text"]), voice, EMOTION_SPEED.get(l["emotion"], 1.0))
    return out


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
