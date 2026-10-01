"""Hadi's LocaAnimo, the local studio app: `python -m locaanimo.server` → http://127.0.0.1:4747

Standard library only (no web framework), bound to localhost.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import yaml
from pydantic import ValidationError

from . import schema
from .paths import EXAMPLES, OUTPUTS, ROOT, WEB
from .pipeline import runner
from .schema import ActionBeat, LineBeat, NarrationBeat, CameraBeat, PauseBeat, Script
from .voices import VoiceUnavailable, cast_voices, engine, list_voices, voice_ids

HOST, PORT = "127.0.0.1", 4747


# --- helpers ---------------------------------------------------------------------

def _enum(e) -> list[str]:
    return [m.value for m in e]


def options() -> dict:
    return {
        "styles": _enum(schema.Style), "emotions": _enum(schema.Emotion),
        "gestures": _enum(schema.Gesture), "cameras": _enum(schema.Camera),
        "ages": ["child", "teen", "adult", "elder"], "genders": ["male", "female", "neutral"],
        "times": ["day", "sunset", "night"], "moods": ["calm", "tense", "happy", "sad", "epic"],
        "resolutions": ["720p", "1080p"],
    }


def _errors(exc: ValidationError) -> list[dict]:
    out = []
    for err in exc.errors():
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "value_error" and "; " in msg:  # cross-reference errors arrive joined
            out += [{"where": "", "message": m} for m in msg.split("; ")]
        else:
            where = ".".join(str(p) for p in err["loc"] if not str(p)[0].isupper())
            out.append({"where": where, "message": msg})
    # Unions report one error per beat type; keep the first per location to stay readable.
    seen, unique = set(), []
    for e in out:
        if e["where"] and re.search(r"beats\.\d+", e["where"]):
            key = re.search(r".*beats\.\d+", e["where"]).group(0)
            if key in seen:
                continue
            seen.add(key)
            e = {"where": key, "message": "This beat is incomplete or has an unknown field."}
        unique.append(e)
    return unique


def estimate_minutes(script: Script) -> float:
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
    return round(seconds / 60, 2)


def validate(raw: dict) -> tuple[Script | None, dict]:
    try:
        script = Script.model_validate(raw)
    except ValidationError as e:
        return None, {"ok": False, "errors": _errors(e)}
    known = voice_ids()
    bad = [f"{c.name}: unknown voice '{c.voice}'" for c in script.characters
           if c.voice != "auto" and c.voice not in known]
    if script.episode.narrator_voice not in ("auto", *known):
        bad.append(f"Narrator: unknown voice '{script.episode.narrator_voice}'")
    if bad:
        return None, {"ok": False, "errors": [{"where": "", "message": m} for m in bad]}
    summary = {
        "characters": len(script.characters), "locations": len(script.locations),
        "scenes": len(script.scenes), "beats": sum(len(s.beats) for s in script.scenes),
        "big": script.big_action_count, "minutes": estimate_minutes(script),
    }
    summary["cast"], summary["narrator"] = cast_voices(script.characters, script.episode.narrator_voice)
    return script, {"ok": True, "errors": [], "summary": summary}


class ImportTask:
    """Free text → script, done by the story model in its own process (MLX can't share
    a process with PyTorch)."""

    def __init__(self, text: str) -> None:
        self.id = uuid.uuid4().hex[:8]
        self.status, self.progress, self.expected = "running", 0, 1
        self.result: dict | None = None
        self.error: str | None = None
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        self.proc = subprocess.Popen([sys.executable, "-m", "locaanimo.importer"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
        self.proc.stdin.write(json.dumps({"text": text}))
        self.proc.stdin.close()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "progress" in msg:
                self.progress, self.expected = msg["progress"], max(1, msg["expected"])
            elif "script" in msg:
                self.result, self.status = msg, "done"
            elif "error" in msg:
                self.error, self.status = msg["error"], "failed"
        self.proc.wait()
        if self.status == "running":
            self.status = "cancelled" if self.proc.returncode < 0 else "failed"
            self.error = self.error or "The story model stopped unexpectedly."

    def cancel(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()

    def to_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "progress": self.progress, "expected": self.expected,
                "error": self.error, **(self.result or {})}


imports: dict[str, ImportTask] = {}


def library() -> list[dict]:
    items = []
    for meta_file in sorted(OUTPUTS.glob("*.json"), reverse=True):
        video = meta_file.with_suffix(".mp4")
        if not video.exists():
            continue
        meta = json.loads(meta_file.read_text())
        thumb = meta_file.with_suffix(".jpg")
        items.append({**meta, "id": meta_file.stem, "video": video.name,
                      "thumb": thumb.name if thumb.exists() else None,
                      "size_mb": round(video.stat().st_size / 1e6, 1)})
    return items


# --- request handler ---------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "LocaAnimo"

    def log_message(self, fmt, *args) -> None:  # keep the terminal quiet except for errors
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s\n" % (fmt % args))

    # responses
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    def _file(self, path, ranged: bool = False) -> None:
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        size = path.stat().st_size
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "")) if ranged else None
        if not match:
            return self._send(200, path.read_bytes(), ctype, {"Accept-Ranges": "bytes"})
        # Safari only plays <video> when the server honors byte ranges.
        start = int(match.group(1)) if match.group(1) else max(0, size - int(match.group(2)))
        end = int(match.group(2)) if match.group(1) and match.group(2) else size - 1
        end = min(end, size - 1)
        with open(path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start + 1)
        self._send(206, chunk, ctype, {"Accept-Ranges": "bytes",
                                       "Content-Range": f"bytes {start}-{end}/{size}"})

    # routing
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self._file(WEB / "index.html")
        if path.startswith("/static/"):
            return self._file(_safe(WEB, path.removeprefix("/static/")))
        if path.startswith("/media/"):
            return self._file(_safe(OUTPUTS, path.removeprefix("/media/")), ranged=True)
        if path == "/api/options":
            return self._json(options())
        if path == "/api/voices":
            return self._json({"voices": list_voices(), "missing": engine.missing()})
        if path == "/api/template":
            return self._json({"script": yaml.safe_load((EXAMPLES / "script_template.yaml").read_text())})
        if path == "/api/library":
            return self._json({"items": library()})
        if path == "/api/jobs/latest":
            job = runner.latest()
            return self._json({"job": job.to_dict() if job else None})
        if m := re.fullmatch(r"/api/jobs/(\w+)", path):
            job = runner.jobs.get(m.group(1))
            return self._json({"job": job.to_dict()}) if job else self._json({"error": "no such job"}, 404)
        if m := re.fullmatch(r"/api/import/(\w+)", path):
            task = imports.get(m.group(1))
            return self._json(task.to_dict()) if task else self._json({"error": "no such import"}, 404)
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON"}, 400)

        if path == "/api/validate":
            return self._json(validate(body.get("script") or {})[1])
        if path == "/api/yaml/parse":
            try:
                data = yaml.safe_load(body.get("yaml") or "") or {}
                if not isinstance(data, dict):
                    raise yaml.YAMLError("The script must start with episode:, characters:, ...")
                return self._json({"script": data})
            except yaml.YAMLError as e:
                return self._json({"error": str(e).split("\n")[0]}, 400)
        if path == "/api/yaml/dump":
            text = yaml.safe_dump(body.get("script") or {}, sort_keys=False, allow_unicode=True, width=100)
            return self._json({"yaml": text})
        if path == "/api/voices/preview":
            try:
                wav = engine.preview(body.get("text") or "Hello there.", body.get("voice") or "af_heart",
                                     body.get("emotion") or "neutral")
            except VoiceUnavailable as e:
                return self._json({"error": str(e)}, 503)
            return self._send(200, wav, "audio/wav")
        if path == "/api/jobs":
            script, result = validate(body.get("script") or {})
            if not script:
                return self._json(result, 400)
            if problem := engine.missing():
                return self._json({"ok": False, "errors": [{"where": "", "message": problem}]}, 503)
            mode = "storyboard" if body.get("mode") == "storyboard" else "full"
            return self._json({"job": runner.submit(script, body["script"], mode).to_dict()})
        if path == "/api/import":
            text = (body.get("text") or "").strip()
            if not text:
                return self._json({"error": "Paste some text first."}, 400)
            for task in imports.values():  # one at a time: the model needs the memory
                task.cancel()
            task = ImportTask(text)
            imports[task.id] = task
            return self._json(task.to_dict())
        if m := re.fullmatch(r"/api/import/(\w+)/cancel", path):
            if task := imports.get(m.group(1)):
                task.cancel()
            return self._json({"ok": True})
        if m := re.fullmatch(r"/api/jobs/(\w+)/cancel", path):
            runner.cancel(m.group(1))
            return self._json({"ok": True})
        if m := re.fullmatch(r"/api/library/([\w.-]+)/reveal", path):
            video = _safe(OUTPUTS, m.group(1) + ".mp4")
            if video.exists():
                subprocess.run(["open", "-R", str(video)])
            return self._json({"ok": video.exists()})
        self._json({"error": "not found"}, 404)


def _safe(base, rel: str):
    """Resolve a path under base, refusing anything that escapes it."""
    target = (base / rel).resolve()
    return target if target.is_relative_to(base.resolve()) else base / "__denied__"


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"Hadi's LocaAnimo running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
