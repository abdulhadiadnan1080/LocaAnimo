"""Episode runner behind the studio app.

Two modes:
  full        the real thing: scripts/make_art.py then scripts/render_episode.py, run as
              subprocesses; their log lines drive the stage list the app shows
  storyboard  a quick animatic (cards + real voices), handy for checking timing

Jobs run one at a time on a single worker thread: on an 18 GB Mac only one heavy model
fits in memory, so episodes queue instead of competing for it.
"""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np
import soundfile as sf
import yaml

from . import animatic
from .paths import OUTPUTS, PROJECTS, ROOT
from .schema import ActionBeat, CameraBeat, LineBeat, NarrationBeat, PauseBeat, Script
from .voices import EMOTION_SPEED, SAMPLE_RATE, cast_voices, engine

FPS = 24

# (key, label, implemented) for the full pipeline; each matches log lines from the scripts.
FULL_STAGES = [
    ("check", "Check script", True),
    ("art", "Draw backgrounds and characters", True),
    ("clips", "AI action shots", True),
    ("faces", "Faces, lip sync and blinks", True),  # (render_episode runs clips first, then faces)
    ("voices", "Voices and sound", True),
    ("review", "Quality review", True),
    ("render", "Render final cut", True),
]
STAGES = [  # storyboard mode
    ("check", "Check script", True),
    ("cast", "Cast voices", True),
    ("record", "Record dialogue", True),
    ("characters", "Draw characters", False),
    ("locations", "Paint locations", False),
    ("animate", "Animate scenes", False),
    ("action", "AI action shots", False),
    ("music", "Score music", False),
    ("assemble", "Assemble episode", True),
]


class Cancelled(Exception):
    pass


@dataclass
class Stage:
    key: str
    label: str
    implemented: bool
    state: str = "pending"      # pending | running | done | skipped | failed
    detail: str = ""


@dataclass
class Segment:
    """One stretch of the episode timeline: a card on screen plus optional audio."""
    card: dict
    duration: float
    audio: np.ndarray | None = None
    reuse_previous_card: bool = False


@dataclass
class Job:
    title: str
    script: Script
    raw: dict
    mode: str = "full"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "queued"      # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = "Waiting to start"
    stages: list[Stage] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    video: str | None = None
    error: str | None = None
    cancel_requested: bool = False

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = [Stage(*s) for s in (FULL_STAGES if self.mode == "full" else STAGES)]
        self.process: subprocess.Popen | None = None

    @property
    def slug(self) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-") or "episode"
        return f"{base[:40]}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.created))}"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "status": self.status,
            "progress": round(self.progress, 4), "message": self.message,
            "stages": [asdict(s) for s in self.stages],
            "created": self.created, "started": self.started, "finished": self.finished,
            "video": self.video, "error": self.error, "mode": self.mode,
        }


class Runner:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, script: Script, raw: dict, mode: str = "full") -> Job:
        job = Job(title=script.episode.title, script=script, raw=raw, mode=mode)
        self.jobs[job.id] = job
        self._queue.put(job)
        return job

    def cancel(self, job_id: str) -> None:
        if job := self.jobs.get(job_id):
            job.cancel_requested = True
            if job.process and job.process.poll() is None:
                job.process.terminate()

    def latest(self) -> Job | None:
        return max(self.jobs.values(), key=lambda j: j.created, default=None)

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job.cancel_requested:
                job.status, job.message = "cancelled", "Cancelled"
                continue
            job.status, job.started = "running", time.time()
            try:
                (_run_full if job.mode == "full" else _run)(job)
                job.status, job.progress, job.message = "done", 1.0, "Episode ready"
            except Cancelled:
                job.status, job.message = "cancelled", "Cancelled"
            except Exception as e:  # surface any failure in the UI instead of dying silently
                job.status, job.error, job.message = "failed", str(e), "Something went wrong"
                traceback.print_exc()
                for s in job.stages:
                    if s.state == "running":
                        s.state, s.detail = "failed", str(e)
            finally:
                job.finished = time.time()


# --- the run itself ----------------------------------------------------------------

def _stage(job: Job, key: str) -> Stage:
    return next(s for s in job.stages if s.key == key)


def _begin(job: Job, key: str, message: str) -> Stage:
    if job.cancel_requested:
        raise Cancelled()
    stage = _stage(job, key)
    stage.state, job.message = "running", message
    return stage


# Log line → (stage key, progress floor). Checked in order; the first match wins.
LOG_STAGES = [
    (re.compile(r"art: drawing (\S+) \((\d+)/(\d+)\)"), "art"),
    (re.compile(r"art: (\d+) images planned"), "art"),
    (re.compile(r"render: (AI action clips|video: )"), "clips"),
    (re.compile(r"render: (locating mouths|preparing characters|painting closed eyes|.*: mouth placed)"), "faces"),
    (re.compile(r"render: review: (checking mouths|characters OK)"), "faces"),
    (re.compile(r"render: (depth maps|\d+ plates|recording voices|audio ready)"), "voices"),
    (re.compile(r"render: review: (checking \d+ (frames of AI clips|preview frames)|all preview|shot)"), "review"),
    (re.compile(r"render: (\d+ shots, encoding|frames \d+%)"), "render"),
]
PROGRESS = {"check": 0.0, "art": 0.02, "clips": 0.55, "faces": 0.68, "voices": 0.74, "review": 0.78, "render": 0.85}


def _run_full(job: Job) -> None:
    work = PROJECTS / job.slug
    work.mkdir(parents=True, exist_ok=True)
    script_file = work / "script.yaml"
    script_file.write_text(yaml.safe_dump(job.raw, sort_keys=False, allow_unicode=True))
    s = _begin(job, "check", "Checking script")
    s.state, s.detail = "done", f"{len(job.script.scenes)} scenes · {sum(len(x.beats) for x in job.script.scenes)} beats"
    if not job.script.big_action_count:
        st = _stage(job, "clips")
        st.state, st.detail = "skipped", "No big moments in this script"

    py = str(ROOT / ".venv" / "bin" / "python")
    for step in ("make_art.py", "render_episode.py"):
        if job.cancel_requested:
            raise Cancelled()
        job.process = subprocess.Popen([py, str(ROOT / "scripts" / step), str(script_file), job.slug],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        tail: list[str] = []
        for line in job.process.stdout:
            line = line.strip()
            if not line.startswith("["):
                tail = (tail + [line])[-6:]
                continue
            _follow(job, line.split("] ", 1)[-1])
        if job.process.wait() != 0:
            if job.cancel_requested:
                raise Cancelled()
            raise RuntimeError(next((t for t in reversed(tail) if "Error" in t), "the pipeline stopped unexpectedly"))
    for st in job.stages:
        if st.state in ("pending", "running"):
            st.state = "done"


def _follow(job: Job, msg: str) -> None:
    """Advance the stage list from one pipeline log message."""
    if m := re.search(r"ALL DONE → (\S+\.mp4)", msg):
        job.video = m.group(1)
        return
    for pattern, key in LOG_STAGES:
        if not pattern.search(msg):
            continue
        order = [k for k, *_ in FULL_STAGES]
        for st in job.stages:              # everything before this stage is finished
            if order.index(st.key) < order.index(key) and st.state in ("pending", "running"):
                st.state = "done"
        st = _stage(job, key)
        if st.state != "skipped":
            st.state = "running"
        st.detail = msg.split(": ", 1)[-1][:90]
        job.message = st.label
        floor = PROGRESS[key]
        if m := re.search(r"\((\d+)/(\d+)\)", msg):
            floor += (PROGRESS["clips"] - PROGRESS["art"]) * int(m.group(1)) / int(m.group(2)) * 0.95
        if m := re.search(r"frames (\d+)%", msg):
            floor += (0.99 - PROGRESS["render"]) * int(m.group(1)) / 100
        job.progress = max(job.progress, floor)
        return


def _run(job: Job) -> None:
    script = job.script
    work = PROJECTS / job.slug
    work.mkdir(parents=True, exist_ok=True)
    (work / "script.yaml").write_text(yaml.safe_dump(job.raw, sort_keys=False, allow_unicode=True))

    stage = _begin(job, "check", "Checking script")
    beats = sum(len(s.beats) for s in script.scenes)
    stage.state, stage.detail = "done", f"{len(script.scenes)} scenes · {beats} beats"
    job.progress = 0.02

    stage = _begin(job, "cast", "Casting voices")
    cast, narrator = cast_voices(script.characters, script.episode.narrator_voice)
    names = {c.id: c.name for c in script.characters}
    stage.state = "done"
    stage.detail = ", ".join(f"{names[cid]} → {v[3:].capitalize()}" for cid, v in cast.items())
    job.progress = 0.05

    stage = _begin(job, "record", "Recording dialogue")
    segments = _build_timeline(job, script, cast, narrator, stage)
    stage.state = "done"
    stage.detail = f"{sum(s.audio is not None for s in segments)} lines recorded"

    for key in ("characters", "locations", "animate", "action", "music"):
        s = _stage(job, key)
        s.state, s.detail = "skipped", "Coming soon. Using storyboard cards for now"

    stage = _begin(job, "assemble", "Assembling episode")
    video = _assemble(job, script, segments, work)
    stage.state = "done"
    stage.detail = f"{sum(s.duration for s in segments) / 60:.1f} min · {video.name}"
    job.video = video.name


def _build_timeline(job: Job, script: Script, cast: dict, narrator: str, stage: Stage) -> list[Segment]:
    locations = {l.id: l for l in script.locations}
    chars = {c.id: c for c in script.characters}
    colors = {c.id: animatic.PALETTE[i % len(animatic.PALETTE)] for i, c in enumerate(script.characters)}
    total = sum(len(s.beats) for s in script.scenes)
    done = 0
    segments: list[Segment] = []

    for n, scene in enumerate(script.scenes, 1):
        loc = locations[scene.location]
        base = dict(scene_no=n, location_id=loc.id, location_look=loc.look, time=scene.time)
        segments.append(Segment(card={"kind": "scene", **base}, duration=1.6))

        for beat in scene.beats:
            if job.cancel_requested:
                raise Cancelled()
            if isinstance(beat, LineBeat):
                ln = beat.line
                audio = engine.synth(ln.text, cast[ln.who], EMOTION_SPEED[ln.emotion.value])
                detail = ln.emotion.value + (f" · {ln.gesture.value.replace('_', ' ')}" if ln.gesture else "")
                card = {"kind": "line", **base, "text": ln.text, "speaker": chars[ln.who].name,
                        "speaker_color": colors[ln.who], "detail": detail}
                segments.append(Segment(card, len(audio) / SAMPLE_RATE + 0.45, audio))
            elif isinstance(beat, NarrationBeat):
                audio = engine.synth(beat.narration, narrator)
                segments.append(Segment({"kind": "narration", **base, "text": beat.narration},
                                        len(audio) / SAMPLE_RATE + 0.5, audio))
            elif isinstance(beat, ActionBeat):
                who = ", ".join(chars[c].name for c in beat.characters)
                sfx = ", ".join(beat.sfx)
                detail = " · ".join(x for x in (who, f"sfx: {sfx}" if sfx else "") if x)
                segments.append(Segment({"kind": "action", **base, "text": beat.action,
                                         "detail": detail, "big": beat.big},
                                        3.2 if beat.big else 2.4))
            elif isinstance(beat, CameraBeat):
                segments.append(Segment({"kind": "camera", **base, "text": beat.camera.value}, 2.0))
            elif isinstance(beat, PauseBeat):
                segments.append(Segment({}, beat.pause, reuse_previous_card=True))
            done += 1
            job.progress = 0.05 + 0.6 * done / total
            stage.detail = f"{done}/{total} beats"
    return segments


def _assemble(job: Job, script: Script, segments: list[Segment], work) -> "Path":
    frames = work / "frames"
    frames.mkdir(exist_ok=True)
    concat, audio_parts, last_card = [], [], None
    lead = int(0.15 * SAMPLE_RATE)  # small breath before each line

    for i, seg in enumerate(segments):
        if job.cancel_requested:
            raise Cancelled()
        if not seg.reuse_previous_card or last_card is None:
            last_card = frames / f"card_{i:04d}.png"
            animatic.render_card(**seg.card).save(last_card)
        concat += [f"file '{last_card.name}'", f"duration {seg.duration:.3f}"]

        clip = np.zeros(int(seg.duration * SAMPLE_RATE), np.float32)
        if seg.audio is not None:
            n = min(len(seg.audio), len(clip) - lead)
            clip[lead:lead + n] = seg.audio[:n]
        audio_parts.append(clip)
        job.progress = 0.65 + 0.2 * (i + 1) / len(segments)

    concat.append(f"file '{last_card.name}'")  # concat demuxer needs the last frame repeated
    (frames / "list.txt").write_text("\n".join(concat) + "\n")
    sf.write(work / "dialogue.wav", np.concatenate(audio_parts), SAMPLE_RATE)

    OUTPUTS.mkdir(exist_ok=True)
    out = OUTPUTS / f"{job.slug}.mp4"
    width, height = (1920, 1080) if script.episode.resolution == "1080p" else (1280, 720)
    job.message = "Encoding video"
    _encode(frames / "list.txt", work / "dialogue.wav", out, width, height)

    first_line = next((f for f in sorted(frames.glob("card_*.png"))), None)
    if first_line:
        from PIL import Image
        Image.open(first_line).convert("RGB").resize((640, 360)).save(out.with_suffix(".jpg"), quality=85)
    meta = {
        "title": script.episode.title, "style": script.episode.style.value,
        "duration": round(sum(s.duration for s in segments), 2),
        "scenes": len(script.scenes), "created": job.created, "kind": "animatic",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    job.progress = 0.99
    return out


def _encode(concat_list, audio, out, width, height) -> None:
    base = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-i", str(audio), "-vf", f"scale={width}:{height},fps={FPS},format=yuv420p"]
    tail = ["-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    # Hardware encoder first; fall back to libx264 if VideoToolbox is unavailable.
    for codec in (["-c:v", "h264_videotoolbox", "-b:v", "8M"], ["-c:v", "libx264", "-crf", "20"]):
        result = subprocess.run(base + codec + tail, capture_output=True, text=True)
        if result.returncode == 0:
            return
    raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-400:]}")


runner = Runner()
