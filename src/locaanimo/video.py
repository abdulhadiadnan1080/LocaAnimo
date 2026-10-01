"""AI action clips with LTX-Video 2B distilled (diffusers, Apple MPS).

The 19 GB T5 text encoder and the video model don't fit in 18 GB together, so each
phase is its own process:
  python -m locaanimo.video encode  jobs.json   (prompts → embeddings file per job)
  python -m locaanimo.video animate jobs.json   (start image + embeddings → mp4 per job)
jobs.json: [{"prompt": ..., "image": ..., "embeds": ..., "out": ...}, ...]
Progress goes to stdout as "video: ..." lines.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import torch

from .paths import MODELS

LTX = MODELS / "video" / "LTX-Video"
CKPT = LTX / "ltxv-2b-0.9.8-distilled.safetensors"
WIDTH, HEIGHT, FRAMES, FPS = 768, 448, 65, 24          # ~2.7 s per clip
STEPS = [1000, 993, 987, 981, 975, 909, 725, 0.03]      # the distilled model's 8-step schedule


def available() -> bool:
    return CKPT.exists() and (LTX / "text_encoder" / "model-00004-of-00004.safetensors").exists()


def log(msg: str) -> None:
    print(f"video: {msg}", flush=True)


def encode(jobs: list[dict]) -> None:
    from transformers import T5EncoderModel, T5TokenizerFast
    log("loading text encoder")
    tok = T5TokenizerFast.from_pretrained(LTX / "tokenizer")
    enc = T5EncoderModel.from_pretrained(LTX / "text_encoder", torch_dtype=torch.bfloat16,
                                         low_cpu_mem_usage=True).to("mps")
    for j in jobs:
        ids = tok(j["prompt"], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        with torch.no_grad():
            emb = enc(ids.input_ids.to("mps"), attention_mask=ids.attention_mask.to("mps"))[0]
        torch.save({"embeds": emb.cpu(), "mask": ids.attention_mask}, j["embeds"])
        log(f"encoded {j['out']}")


def animate(jobs: list[dict]) -> None:
    from diffusers import (AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline,
                           LTXVideoTransformer3DModel)
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    from PIL import Image
    log("loading video model")
    pipe = LTXConditionPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(LTX / "scheduler", use_dynamic_shifting=False),
        vae=AutoencoderKLLTXVideo.from_single_file(str(CKPT), torch_dtype=torch.bfloat16),
        transformer=LTXVideoTransformer3DModel.from_single_file(str(CKPT), torch_dtype=torch.bfloat16),
        text_encoder=None, tokenizer=None).to("mps")
    pipe.vae.enable_tiling()
    for i, j in enumerate(jobs, 1):
        t = time.time()
        e = torch.load(j["embeds"])
        img = Image.open(j["image"]).convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS)
        frames = pipe(conditions=[LTXVideoCondition(image=img, frame_index=0)],
                      prompt_embeds=e["embeds"].to("mps", torch.bfloat16), prompt_attention_mask=e["mask"].to("mps"),
                      width=WIDTH, height=HEIGHT, num_frames=FRAMES, timesteps=STEPS, guidance_scale=1.0,
                      decode_timestep=0.05, decode_noise_scale=0.025, output_type="pil",
                      generator=torch.Generator().manual_seed(7 + i)).frames[0]
        write_video(frames, j["out"])
        log(f"clip {i}/{len(jobs)} done in {time.time() - t:.0f}s")


def write_video(frames, out: str) -> None:
    """Pipe frames into the system ffmpeg (no extra Python video packages needed)."""
    w, h = frames[0].size
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "h264_videotoolbox", "-b:v", "8M",
                             "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.asarray(f.convert("RGB"), np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


def read_video(path, size: tuple[int, int]) -> np.ndarray:
    """All frames of a clip as float32 [T, H, W, 3] in [0, 1], resized to `size` (w, h)."""
    w, h = size
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", f"scale={w}:{h}:flags=lanczos",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(np.float32) / 255


if __name__ == "__main__":
    jobs = json.loads(open(sys.argv[2]).read())
    {"encode": encode, "animate": animate}[sys.argv[1]](jobs)
