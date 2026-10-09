"""Expressive voices with Qwen3-TTS VoiceDesign: every line is spoken from a description.

Each character has a voice description ("teen girl, bright, a bit sarcastic"). Every line
is generated from that description plus the line's emotion ("speaking angrily, sharp and
forceful"). The seed is fixed per character, so the same character keeps sounding like the
same person. [laugh]-style tags become part of the delivery instruction and are never spoken
as words. In listening tests this sounded better than designing a reference clip and cloning
it with Chatterbox, both in voice fit and smoothness.

MLX can't share a process with PyTorch, so this runs as its own process:
  python -m locaanimo.speech batch jobs.json   one-shot, for renders (prints "speech: ..." lines)
  python -m locaanimo.speech serve             long-lived, for studio previews (JSON lines in/out)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from .paths import MODELS
from .schema import _TAG, strip_tags

HF_CACHE = MODELS / "hf-cache"

import numpy as np
import soundfile as sf

DESIGNER = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
SAMPLE_RATE = 24000

# Delivery per emotion, appended to the character's voice description.
EMOTION_GROUP = {
    "neutral": "neutral", "happy": "happy", "sad": "sad", "angry": "angry",
    "surprised": "surprised", "amazed": "surprised", "scared": "scared", "worried": "worried",
    "determined": "confident", "confident": "confident", "embarrassed": "embarrassed",
}
DELIVERY = {
    "neutral": "calm, natural, conversational delivery",
    "happy": "speaking happily, warm and upbeat, with a smile in the voice",
    "sad": "speaking sadly, soft, slow and heavy, close to tears",
    "angry": "speaking angrily, sharp, tense and forceful",
    "scared": "speaking fearfully, breathless, shaky and urgent",
    "worried": "speaking with worry, hesitant and concerned",
    "surprised": "speaking in surprise, bright and amazed, rising pitch",
    "confident": "speaking confidently, firm, steady and assured",
    "embarrassed": "speaking shyly, embarrassed, a little awkward and quiet",
}
# Narration is told, not chatted: an audiobook read whose color follows the scene's mood.
# (A flat "calm, measured" direction is what made narration sound robotic.)
NARRATION = ("telling a story like an audiobook narrator, expressive and engaging, natural rises "
             "and falls in pitch, a brief pause between sentences")
NARRATION_MOOD = {
    "calm": "gentle and inviting", "happy": "bright and warm, smiling", "sad": "soft and wistful",
    "tense": "hushed and suspenseful, building tension", "epic": "grand and stirring, with rising intensity",
}

# How a sound tag is asked for, since Qwen performs it from the instruction.
TAG_DELIVERY = {
    "laugh": "with a short laugh", "chuckle": "with a soft chuckle", "sigh": "with a tired sigh",
    "gasp": "with a sharp gasp", "cough": "with a small cough", "groan": "with a groan",
    "sniff": "sniffling", "clear throat": "clearing the throat first", "shush": "hushed, almost whispering",
}


def available() -> bool:
    """The voice model is fully downloaded."""
    root = HF_CACHE / f"models--{DESIGNER.replace('/', '--')}"
    weights = list((root / "snapshots").glob("*/*.safetensors"))
    unfinished = list((root / "blobs").glob("*.incomplete"))
    return bool(weights) and all(w.exists() for w in weights) and not unfinished


def voice_key(description: str) -> str:
    return hashlib.sha1(f"v1|{description.strip().lower()}".encode()).hexdigest()[:12]


def instruction(description: str, emotion: str, text: str) -> str:
    """The VoiceDesign instruction for one line: who is speaking, and how."""
    sounds = [TAG_DELIVERY[t.strip().lower()] for t in _TAG.findall(text) if t.strip().lower() in TAG_DELIVERY]
    if emotion.startswith("narration"):           # "narration" or "narration:<scene mood>"
        mood = NARRATION_MOOD.get(emotion.partition(":")[2])
        how = NARRATION + (f", {mood}" if mood else "")
    else:
        how = DELIVERY[EMOTION_GROUP.get(emotion, "neutral")]
    how += ", " + ", ".join(sounds) if sounds else ""
    return f"{description.strip().rstrip('.')}. {how}."


def log(msg: str) -> None:
    print(f"speech: {msg}", flush=True)


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly
    g = np.gcd(sr, SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)


def _fade(audio: np.ndarray, ms: int = 15) -> np.ndarray:
    n = min(len(audio) // 4, int(SAMPLE_RATE * ms / 1000))
    if n:
        audio[:n] *= np.linspace(0, 1, n)
        audio[-n:] *= np.linspace(1, 0, n)
    return audio


class Voices:
    def __init__(self) -> None:
        self.model = None

    def speak(self, text: str, description: str, emotion: str = "neutral") -> np.ndarray:
        import mlx.core as mx
        if self.model is None:
            from mlx_audio.tts.utils import load_model
            self.model = load_model(DESIGNER)
        mx.random.seed(int(voice_key(description), 16) % (2 ** 31))   # same character, same voice
        parts = list(self.model.generate_voice_design(text=strip_tags(text), language="English",
                                                      instruct=instruction(description, emotion, text)))
        audio = np.concatenate([np.asarray(p.audio, np.float32).reshape(-1) for p in parts])
        return _fade(_resample(audio, parts[0].sample_rate))


def batch(jobs_file: str) -> None:
    """jobs.json: {"lines": [{"id", "text", "voice", "emotion", "out"}, ...]} → results.json."""
    lines = json.loads(Path(jobs_file).read_text())["lines"]
    voices = Voices()
    results = {}
    for i, l in enumerate(lines, 1):
        t = time.time()
        try:
            audio = voices.speak(l["text"], l["voice"], l.get("emotion", "neutral"))
            sf.write(l["out"], audio, SAMPLE_RATE)
            results[l["id"]] = {"ok": True, "seconds": round(len(audio) / SAMPLE_RATE, 3)}
            log(f"line ({i}/{len(lines)}) {len(audio) / SAMPLE_RATE:.1f}s of speech in {time.time() - t:.1f}s")
        except Exception as e:   # one bad line falls back to Kokoro instead of failing the episode
            results[l["id"]] = {"ok": False, "error": str(e)}
            log(f"line ({i}/{len(lines)}) failed: {e}")
    Path(jobs_file).with_name("results.json").write_text(json.dumps(results, indent=2))
    log("ALL DONE")


def serve() -> None:
    """Studio previews: one JSON request per line on stdin, one JSON reply per line on stdout."""
    voices = Voices()
    print(json.dumps({"ready": True}), flush=True)
    for raw in sys.stdin:
        try:
            req = json.loads(raw)
            sf.write(req["out"], voices.speak(req["text"], req["voice"], req.get("emotion", "neutral")), SAMPLE_RATE)
            reply = {"ok": True, "out": req["out"]}
        except Exception as e:
            reply = {"ok": False, "error": str(e)}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    # Only in the worker process: models live in models/hf-cache and are never fetched implicitly.
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"
    {"batch": lambda: batch(sys.argv[2]), "serve": serve}[sys.argv[1]]()
