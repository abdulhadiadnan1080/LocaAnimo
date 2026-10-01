"""Procedural sound effects and ambience (filtered noise): a stopgap until the CC0 library."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SR = 24000


def _noise(seconds: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 1, int(seconds * SR)).astype(np.float32)


def _lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    return sosfilt(butter(2, hz, "low", fs=SR, output="sos"), x).astype(np.float32)


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x).astype(np.float32)


def _norm(x: np.ndarray, peak: float) -> np.ndarray:
    return x / (np.abs(x).max() + 1e-6) * peak


def ocean(seconds: float, seed: int = 1) -> np.ndarray:
    """Waves washing in and out every ~6 s."""
    t = np.arange(int(seconds * SR)) / SR
    swell = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 6.5)) ** 2
    return _norm(_lowpass(_noise(seconds, seed), 900) * swell, 0.08)


def wave_roar(seconds: float = 2.6, seed: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = np.minimum(1, t / (seconds * 0.7)) ** 2 * np.minimum(1, (seconds - t) / 0.5)
    return _norm(_lowpass(_noise(seconds, seed), 500) * env, 0.32)


def footsteps_sand(seconds: float = 2.0, rate: float = 2.2, seed: int = 3) -> np.ndarray:
    out = np.zeros(int(seconds * SR), np.float32)
    step = _bandpass(_noise(0.12, seed), 300, 2500) * np.exp(-np.linspace(0, 6, int(0.12 * SR)))
    for i in range(int(seconds * rate)):
        s = int(i / rate * SR)
        out[s:s + len(step)] += step[: len(out) - s]
    return _norm(out, 0.18)


def whoosh(seconds: float = 0.9, seed: int = 4) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = np.sin(np.pi * t / seconds) ** 2
    lo = _bandpass(_noise(seconds, seed), 200, 1200)
    hi = _bandpass(_noise(seconds, seed + 1), 1200, 5000)
    mix = lo * (1 - t / seconds) + hi * (t / seconds)  # rising sweep
    return _norm(mix * env, 0.3)


def rain(seconds: float, seed: int = 5) -> np.ndarray:
    """Steady rain hiss with occasional heavier gusts."""
    t = np.arange(int(seconds * SR)) / SR
    gust = 0.7 + 0.3 * np.sin(2 * np.pi * t / 7.3) ** 2
    return _norm(_bandpass(_noise(seconds, seed), 1500, 8000) * gust, 0.06)


def hum(seconds: float, seed: int = 6) -> np.ndarray:
    """Indoor machine hum: mains buzz plus a little low noise."""
    t = np.arange(int(seconds * SR)) / SR
    buzz = 0.6 * np.sin(2 * np.pi * 60 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
    return _norm(buzz + 0.5 * _lowpass(_noise(seconds, seed), 300), 0.04)


def siren(seconds: float = 2.5) -> np.ndarray:
    """Distant two-tone siren."""
    t = np.arange(int(seconds * SR)) / SR
    freq = np.where((t * 1.6) % 1 < 0.5, 700, 950)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / SR)
    env = np.minimum(1, t / 0.4) * np.minimum(1, (seconds - t) / 0.6)
    return _norm(_lowpass(tone * env, 2500), 0.12)


def impact(seconds: float = 1.4, seed: int = 7) -> np.ndarray:
    """A heavy crash: a low boom with a noisy attack."""
    t = np.arange(int(seconds * SR)) / SR
    boom = np.sin(2 * np.pi * (55 + 40 * np.exp(-t * 8)) * t) * np.exp(-t * 3.5)
    crack = _bandpass(_noise(seconds, seed), 400, 6000) * np.exp(-t * 14)
    return _norm(boom + 0.6 * crack, 0.55)


def blast(seconds: float = 2.2, seed: int = 8) -> np.ndarray:
    """Energy blast: rising whine into a boom."""
    t = np.arange(int(seconds * SR)) / SR
    whine = np.sin(2 * np.pi * np.cumsum(300 + 1500 * np.minimum(1, t / 0.5)) / SR) * np.minimum(1, t / 0.5) * (t < 0.55)
    tail = np.zeros_like(t)
    hit = impact(seconds - 0.5, seed)
    tail[int(0.5 * SR):int(0.5 * SR) + len(hit)] = hit[: len(tail) - int(0.5 * SR)]
    return _norm(0.3 * whine + tail, 0.6)


EFFECTS = {"wave_roar": wave_roar, "footsteps_sand": footsteps_sand, "footsteps": footsteps_sand, "whoosh": whoosh,
           "siren": siren, "impact": impact, "crash": impact, "door_bang": impact, "blast": blast, "explosion": blast}

# Background ambience picked from words in a location's look.
AMBIENCE = [(("rain", "storm"), rain), (("beach", "ocean", "sea", "wave"), ocean),
            (("station", "hideout", "lab", "factory", "server", "indoor", "room"), hum)]


def ambience_for(look: str):
    look = look.lower()
    return next((fn for words, fn in AMBIENCE if any(w in look for w in words)), None)
