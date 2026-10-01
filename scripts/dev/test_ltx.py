"""Smoke test: animate one action still with LTX-Video 2B distilled on Apple MPS.

Two phases in separate processes so the 19 GB text encoder and the video model are
never in memory together (18 GB Mac):
  python scripts/test_ltx.py encode "<prompt>" out.pt
  python scripts/test_ltx.py animate image.png out.pt out.mp4
"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LTX = ROOT / "models" / "video" / "LTX-Video"
CKPT = LTX / "ltxv-2b-0.9.8-distilled.safetensors"
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - t0:5.0f}s] {msg}", flush=True)


def encode(prompt: str, out: str) -> None:
    from transformers import T5EncoderModel, T5TokenizerFast
    log("ltx: loading T5 text encoder (bf16)")
    tok = T5TokenizerFast.from_pretrained(LTX / "tokenizer")
    enc = T5EncoderModel.from_pretrained(LTX / "text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps")
    log("ltx: encoding prompt")
    ids = tok(prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    with torch.no_grad():
        emb = enc(ids.input_ids.to("mps"), attention_mask=ids.attention_mask.to("mps"))[0]
    torch.save({"embeds": emb.cpu(), "mask": ids.attention_mask}, out)
    log("ltx: ENCODE DONE")


def animate(image: str, emb_path: str, out: str, frames: int = 65, width: int = 768, height: int = 448) -> None:
    from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXConditionPipeline, LTXVideoTransformer3DModel
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    from PIL import Image

    log("ltx: loading video transformer + VAE (bf16)")
    transformer = LTXVideoTransformer3DModel.from_single_file(str(CKPT), torch_dtype=torch.bfloat16)
    vae = AutoencoderKLLTXVideo.from_single_file(str(CKPT), torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(LTX / "scheduler", use_dynamic_shifting=False)  # distilled model uses fixed timesteps
    pipe = LTXConditionPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
                                transformer=transformer).to("mps")
    pipe.vae.enable_tiling()
    e = torch.load(emb_path)
    img = Image.open(image).convert("RGB").resize((width, height), Image.LANCZOS)
    log(f"ltx: generating {frames} frames at {width}x{height}")
    t = time.time()
    video = pipe(
        conditions=[LTXVideoCondition(image=img, frame_index=0)],
        prompt_embeds=e["embeds"].to("mps", torch.bfloat16), prompt_attention_mask=e["mask"].to("mps"),
        width=width, height=height, num_frames=frames,
        timesteps=[1000, 993, 987, 981, 975, 909, 725, 0.03],   # the distilled model's 8-step schedule
        guidance_scale=1.0, decode_timestep=0.05, decode_noise_scale=0.025,
        generator=torch.Generator().manual_seed(7), output_type="pil",
    ).frames[0]
    log(f"ltx: generated in {time.time() - t:.0f}s")
    _write_video(video, out)
    log(f"ltx: ANIMATE DONE → {out}")


def _write_video(frames, out: str, fps: int = 24) -> None:
    """Pipe frames into the system ffmpeg (no extra Python video packages needed)."""
    import subprocess
    import numpy as np
    w, h = frames[0].size
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "h264_videotoolbox",
                             "-b:v", "8M", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.asarray(f.convert("RGB"), np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    if sys.argv[1] == "encode":
        encode(sys.argv[2], sys.argv[3])
    else:
        animate(sys.argv[2], sys.argv[3], sys.argv[4])
