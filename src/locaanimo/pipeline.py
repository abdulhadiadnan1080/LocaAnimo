"""Episode runner behind the studio app, and the live progress of any episode being made.

A studio job runs scripts/make_art.py then scripts/render_episode.py as subprocesses; their
log lines drive the stage list, the "what's happening now" text, the preview image and the
time estimate the app shows. Episodes started from the terminal (scripts/make_episode.sh)
are picked up too, by reading the same log, so the app always shows what's being made.

Jobs run one at a time on a single worker thread: on an 18 GB Mac only one heavy model
fits in memory, so episodes queue instead of competing for it.
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from .paths import PROJECTS, ROOT
from .schema import ActionBeat, CameraBeat, LineBeat, NarrationBeat, PauseBeat, Script

# (key, label) in the order the pipeline runs them; each matches log lines from the scripts.
FULL_STAGES = [
    ("check", "Check script"),
    ("art", "Draw backgrounds and characters"),
    ("clips", "AI action shots"),
    ("faces", "Faces, lip sync and blinks"),
    ("voices", "Voices and sound"),
    ("review", "Quality review"),
    ("render", "Render final cut"),
]
ORDER = [k for k, _ in FULL_STAGES]


class Cancelled(Exception):
    pass


@dataclass
class Stage:
    key: str
    label: str
    state: str = "pending"      # pending | running | done | skipped | failed
    detail: str = ""
    started: float | None = None
    finished: float | None = None


@dataclass
class Job:
    title: str
    script: Script
    project: Path
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "queued"      # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = "Waiting to start"
    now: str = ""               # what's being made right now, in plain words
    stages: list[Stage] = field(default_factory=lambda: [Stage(k, label) for k, label in FULL_STAGES])
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    video: str | None = None
    error: str | None = None
    external: bool = False      # started from the terminal, not from the app
    cancel_requested: bool = False

    def __post_init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.budget = stage_budget(self.script, self.project)
        self._counter: dict[str, float] = {}       # fraction done inside a stage, from (i/n) and %

    def to_dict(self) -> dict:
        eta, total = self.estimate()
        return {
            "id": self.id, "title": self.title, "status": self.status,
            "progress": round(self.progress, 4), "message": self.message, "now": self.now,
            "stages": [asdict(s) for s in self.stages],
            "created": self.created, "started": self.started, "finished": self.finished,
            "video": self.video, "error": self.error, "external": self.external,
            "eta": eta, "total": total, "preview": latest_preview(self.project, self.started or self.created),
        }

    # --- progress and time left ---------------------------------------------------------

    def _done_fraction(self, st: Stage, now: float) -> float:
        if st.state in ("done", "skipped"):
            return 1.0
        if st.state != "running":
            return 0.0
        if st.key in self._counter:
            return min(0.97, self._counter[st.key])
        spent = now - (st.started or now)                # no counter: assume it runs to budget
        return min(0.9, spent / max(1.0, self.budget[st.key]))

    def update_progress(self, now: float | None = None) -> None:
        now = now or time.time()
        total = sum(self.budget.values()) or 1.0
        done = sum(self.budget[s.key] * self._done_fraction(s, now) for s in self.stages)
        self.progress = max(self.progress, min(0.99, done / total))

    def estimate(self) -> tuple[float | None, float | None]:
        """(seconds left, expected total seconds). The budget is scaled by how fast this
        Mac has actually been on the finished stages, so it corrects itself as it goes."""
        if self.status not in ("queued", "running") or not self.started:
            return None, None
        now = time.time()
        spent = [(s.finished - s.started, self.budget[s.key]) for s in self.stages
                 if s.state == "done" and s.started and s.finished and self.budget[s.key] > 5]
        speed = 1.0
        if spent:
            speed = min(2.0, max(0.5, sum(a for a, _ in spent) / sum(b for _, b in spent)))
        left = sum(self.budget[s.key] * (1 - self._done_fraction(s, now)) for s in self.stages) * speed
        return round(left), round(now - self.started + left)


def stage_budget(script: Script, project: Path) -> dict[str, float]:
    """Expected seconds per stage for this script on an M3 Pro (measured), before any
    speed correction. Art already drawn for this project is cached, so it costs nothing."""
    art = project / "art"
    images = ([f"bg_{s.location}@{s.time}" for s in script.scenes] + [f"char_{c.id}" for c in script.characters]
              + [f"action_{i}" for i in range(1, script.big_action_count + 1)])
    to_draw = sum(not (art / f"{name}.png").exists() for name in dict.fromkeys(images))
    big = script.big_action_count
    lines = sum(isinstance(b, (LineBeat, NarrationBeat)) for s in script.scenes for b in s.beats)
    episode = episode_seconds(script)
    return {
        "check": 1.0,
        "art": (15 + 100 * to_draw) if to_draw else 2.0,
        "clips": (130 + 50 * big) if big else 0.0,
        "faces": 40 + 70 * len(script.characters),
        "voices": 25 + 3.0 * lines,
        "review": 40 + 12 * big,
        "render": 15 + 0.7 * episode,
    }


def episode_seconds(script: Script) -> float:
    seconds = 0.0
    for scene in script.scenes:
        seconds += 1.6
        for b in scene.beats:
            if isinstance(b, LineBeat):
                seconds += max(1.2, len(b.line.text.split()) / 2.7) + 0.45
            elif isinstance(b, NarrationBeat):
                seconds += max(1.2, len(b.narration.split()) / 2.6) + 0.5
            elif isinstance(b, ActionBeat):
                seconds += 3.2 if b.big else 2.4
            elif isinstance(b, CameraBeat):
                seconds += 2.0
            elif isinstance(b, PauseBeat):
                seconds += b.pause
    return seconds


def latest_preview(project: Path, since: float) -> str | None:
    """The newest image this run has made (a drawing, a blink, a preview frame), as a path
    relative to projects/, so the app can show what's being created."""
    pics = [p for pattern in ("art/*.png", "review/shot_*.png") for p in project.glob(pattern)
            if p.stat().st_mtime >= since - 2]
    if not pics:
        return None
    pic = max(pics, key=lambda p: p.stat().st_mtime)
    return f"{pic.relative_to(PROJECTS)}?v={int(pic.stat().st_mtime)}"


class Runner:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, script: Script, raw: dict) -> Job:
        title = script.episode.title
        base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "episode"
        project = PROJECTS / f"{base}-{time.strftime('%Y%m%d-%H%M%S')}"
        project.mkdir(parents=True, exist_ok=True)
        (project / "script.yaml").write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
        job = Job(title=title, script=script, project=project)
        self.jobs[job.id] = job
        self._queue.put(job)
        return job

    def cancel(self, job_id: str) -> None:
        if job := self.jobs.get(job_id):
            job.cancel_requested = True
            if job.process and job.process.poll() is None:
                job.process.terminate()
        elif job_id.startswith("ext"):
            terminal.cancel(job_id)

    def active(self) -> Job | None:
        return next((j for j in self.jobs.values() if j.status in ("queued", "running")), None)

    def latest(self) -> Job | None:
        """What the app should show: a job in progress (the app's own, or one started from
        the terminal), else the most recent one."""
        if job := self.active():
            return job
        if ext := terminal.current():
            return ext
        return max(self.jobs.values(), key=lambda j: j.created, default=None)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id) or (terminal.get(job_id) if job_id.startswith("ext") else None)

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job.cancel_requested:
                job.status, job.message = "cancelled", "Cancelled"
                continue
            job.status, job.started = "running", time.time()
            try:
                _run_full(job)
                job.status, job.progress, job.message, job.now = "done", 1.0, "Episode ready", ""
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


def _start(job: Job, at: float) -> None:
    """Mark the script checked and skip what this script doesn't need."""
    s = _stage(job, "check")
    s.state, s.started, s.finished = "done", at, at
    s.detail = f"{len(job.script.scenes)} scenes · {sum(len(x.beats) for x in job.script.scenes)} beats"
    if not job.script.big_action_count:
        st = _stage(job, "clips")
        st.state, st.detail = "skipped", "No big moments in this script"


def _run_full(job: Job) -> None:
    _start(job, job.started)
    job.message = "Starting"
    script_file = job.project / "script.yaml"
    py = str(ROOT / ".venv" / "bin" / "python")
    for step in ("make_art.py", "render_episode.py"):
        if job.cancel_requested:
            raise Cancelled()
        job.process = subprocess.Popen([py, str(ROOT / "scripts" / step), str(script_file), job.project.name],
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
    _finish_stages(job, time.time())


def _finish_stages(job: Job, at: float) -> None:
    for st in job.stages:
        if st.state in ("pending", "running"):
            st.state, st.finished = "done", st.finished or at


# Log line → stage. Checked in order; the first match wins.
LOG_STAGES = [
    (re.compile(r"art: drawing (\S+) \((\d+)/(\d+)\)"), "art"),
    (re.compile(r"art: (\d+) images planned"), "art"),
    (re.compile(r"render: (AI action clips|video: )"), "clips"),
    (re.compile(r"render: (locating mouths|preparing characters|painting closed eyes|.*: (mouth placed|closed eyes))"), "faces"),
    (re.compile(r"render: review: (checking mouths|characters OK|.*: (mouth|blink) rejected)"), "faces"),
    (re.compile(r"render: (depth maps|\d+ plates|recording voices|speech: |\d+ lines recorded|audio ready)"), "voices"),
    (re.compile(r"render: (review: (checking \d+ (frames of AI clips|preview frames)|all preview|shot|AI clip)|\d+ AI action clip)"), "review"),
    (re.compile(r"render: (\d+ shots, encoding|frames \d+%)"), "render"),
]


def _now_text(job: Job, msg: str) -> str:
    """A log message in plain words: what is being made right now."""
    names = {c.id: c.name for c in job.script.characters}
    if m := re.search(r"art: drawing (\S+) \((\d+)/(\d+)\)", msg):
        what, i, n = m.groups()
        if what.startswith("char_"):
            thing = f"Drawing {names.get(what[5:], what[5:])}"
        elif what.startswith("bg_"):
            place, _, tod = what[3:].partition("@")
            thing = f"Painting the {place.replace('_', ' ')} ({tod})"
        else:
            thing = f"Drawing big moment {what.rsplit('_', 1)[-1]}"
        return f"{thing} · image {i} of {n}"
    rules = [
        (r"video: loading text encoder", "Reading the action scenes for the video model"),
        (r"video: encoded .*action_(\d+)", r"Prepared action shot \1"),
        (r"video: loading video model", "Loading the video model"),
        (r"video: clip (\d+)/(\d+) done", r"Animated action shot \1 of \2"),
        (r"locating mouths", "Finding mouths and eyes"),
        (r"painting closed eyes", "Painting blinks"),
        (r"(.+): closed eyes painted", r"Painted \1's blink"),
        (r"(.+): mouth placed", r"Placed \1's mouth"),
        (r"review: checking mouths", "Checking mouths and blinks"),
        (r"review: checking (\d+) frames of AI clips", r"Checking \1 frames of the action shots"),
        (r"review: checking (\d+) preview frames", r"Checking \1 preview frames before the final cut"),
        (r"speech: line \((\d+)/(\d+)\)", r"Recording line \1 of \2"),
        (r"recording voices", "Recording voices"),
        (r"(\d+) lines recorded", r"\1 lines recorded"),
        (r"depth maps|(\d+) plates", "Building 3D depth for the backgrounds"),
        (r"(\d+) shots, encoding", r"Rendering \1 shots"),
        (r"frames (\d+)%", r"Rendering frames · \1%"),
    ]
    for pattern, text in rules:
        if m := re.search(pattern, msg):
            return m.expand(text)
    return msg.split(": ", 1)[-1][:90]


def _follow(job: Job, msg: str, at: float | None = None) -> None:
    """Advance the stages, the progress and the "now" text from one pipeline log message."""
    at = at or time.time()
    if m := re.search(r"ALL DONE → (\S+\.mp4)", msg):
        job.video = m.group(1)
        return
    for pattern, key in LOG_STAGES:
        if not pattern.search(msg):
            continue
        for st in job.stages:              # everything before this stage is finished
            if ORDER.index(st.key) < ORDER.index(key) and st.state in ("pending", "running"):
                st.state = "done"
                st.finished = st.finished or at
        st = _stage(job, key)
        if st.state not in ("skipped", "running"):
            st.state, st.started = "running", at
        st.detail = msg.split(": ", 1)[-1][:90]
        job.message = st.label
        job.now = _now_text(job, msg)
        if m := re.search(r"\((\d+)/(\d+)\)", msg):
            job._counter[key] = (int(m.group(1)) - 0.5) / int(m.group(2))
        elif m := re.search(r"clip (\d+)/(\d+) done", msg):
            job._counter[key] = 0.4 + 0.6 * int(m.group(1)) / int(m.group(2))
        elif m := re.search(r"frames (\d+)%", msg):
            job._counter[key] = int(m.group(1)) / 100
        job.update_progress(at)
        return


# --- episodes started from the terminal --------------------------------------------

class Terminal:
    """Finds an episode being made by scripts/make_episode.sh (or the two scripts run by
    hand) and replays its log into a Job, so the app shows it like one of its own."""

    SCRIPTS = ("scripts/make_episode.sh", "scripts/make_art.py", "scripts/render_episode.py")

    def __init__(self) -> None:
        self.last: Job | None = None
        self._pids: list[int] = []
        self._cache = (0.0, None)

    def _processes(self) -> list[tuple[int, int, str, str]]:
        """(pid, seconds running, script path, project name) for each pipeline process. Only
        the processes themselves count ("python scripts/x.py a b" or "sh scripts/x.sh a b"),
        not shell wrappers that merely mention them."""
        out = subprocess.run(["ps", "-axo", "pid=,etime=,command="], capture_output=True, text=True).stdout
        procs = []
        for row in out.splitlines():
            pid, etime, *args = row.split()
            if len(args) >= 4 and args[1].endswith(self.SCRIPTS) and Path(args[0]).name.startswith(("python", "sh", "bash", "zsh")):
                procs.append((int(pid), _seconds(etime), args[2], args[3]))
        return procs

    def current(self) -> Job | None:
        """The job in progress right now, if any (checked at most once a second)."""
        t, job = self._cache
        if time.time() - t < 1.0:
            return job
        job = self._scan()
        self._cache = (time.time(), job)
        return job

    def get(self, job_id: str) -> Job | None:
        if (job := self.current()) and job.id == job_id:
            return job
        return self.last if self.last and self.last.id == job_id else None

    def cancel(self, job_id: str) -> None:
        if self.last and self.last.id == job_id:
            for pid in self._pids:
                subprocess.run(["kill", str(pid)], capture_output=True)

    def _scan(self) -> Job | None:
        procs = [p for p in self._processes() if not runner_owns(p[3])]
        if not procs:
            if self.last and self.last.status == "running":      # it just ended: settle the outcome
                self._settle(self.last)
            return None
        pid, age, script_path, name = max(procs, key=lambda p: p[1])  # the outer process started first
        started = time.time() - age
        self._pids = [p[0] for p in procs if p[3] == name]
        job_id = f"ext{pid}"
        if not (self.last and self.last.id == job_id):
            path = Path(script_path) if Path(script_path).is_absolute() else ROOT / script_path
            try:
                script = Script.model_validate(yaml.safe_load(path.read_text()))
            except Exception:
                return None
            self.last = Job(title=script.episode.title, script=script, project=PROJECTS / name,
                            id=job_id, external=True, created=started, started=started, status="running")
            self.last.message = "Starting"
        self._replay(self.last)
        return self.last

    def _replay(self, job: Job) -> None:
        """Rebuild the job's state from the run's log lines (the log keeps earlier runs too)."""
        log = job.project / "log.txt"
        fresh = Job(title=job.title, script=job.script, project=job.project, id=job.id, external=True,
                    created=job.created, started=job.started, status="running")
        _start(fresh, job.started)
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                m = re.match(r"\[(\d\d):(\d\d):(\d\d) \+\s*\d+s\] (.*)", line)
                if not m:
                    continue
                at = _clock_to_epoch(*map(int, m.groups()[:3]), near=job.started)
                if at >= job.started - 1:
                    _follow(fresh, m.group(4), at)
        fresh.update_progress()
        for attr in ("stages", "progress", "message", "now", "video", "_counter"):
            setattr(job, attr, getattr(fresh, attr))

    def _settle(self, job: Job) -> None:
        self._replay(job)
        job.finished = time.time()
        if job.video:
            job.status, job.progress, job.message, job.now = "done", 1.0, "Episode ready", ""
            _finish_stages(job, job.finished)
        else:
            job.status, job.message = "failed", "Stopped before the video was finished"
            job.error = "The run stopped. See the project's log.txt for details."


def runner_owns(project_name: str) -> bool:
    return any(j.project.name == project_name and j.status in ("queued", "running") for j in runner.jobs.values())


def _seconds(etime: str) -> int:
    """ps elapsed time "[[dd-]hh:]mm:ss" → seconds."""
    days, _, rest = etime.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _clock_to_epoch(h: int, m: int, s: int, near: float) -> float:
    """A log's HH:MM:SS as a timestamp, on whichever day puts it closest to `near`."""
    base = datetime.fromtimestamp(near).replace(hour=h, minute=m, second=s, microsecond=0)
    best = min((base + timedelta(days=d) for d in (-1, 0, 1)), key=lambda d: abs(d.timestamp() - near - 1800))
    return best.timestamp()


runner = Runner()
terminal = Terminal()
