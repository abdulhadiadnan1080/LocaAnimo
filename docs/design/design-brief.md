# AnimoLocal: Master Project Brief (v3, 2026-09-30)

> Paste this into your AI code editor as project context.
> Earlier versions: `context.v1.md` (original idea), `context.v2.md` (first research pass).
> The cloned source code of every tool lives in `gitrepos/` (§9).

---

## 1. The pitch

**"Type one prompt → get a fully voiced, animated anime episode. 100% local. Runs on a MacBook."**

This is an open-source repo built to trend on GitHub. It takes a prompt and produces a complete episode: story, consistent characters, real animation, voices, lip sync, music, sound effects and subtitles, delivered as one `.mp4`.

**Non-negotiables**
- **No APIs and no cloud.** Nothing leaves the machine once the models are downloaded.
- **Runs on a MacBook** (the reference machine is below). Support for NVIDIA GPUs comes later.
- **Real animation.** Characters move their bodies. This is not a slideshow or a motion comic. The big scenes get AI-generated video.
- **Optional user input at every stage.** A prompt alone is enough, but the user can paste their own story, dialogue, scenes or shot list.
- **Permissive licenses by default.** Non-commercial models are opt-in only (§8).

---

## 2. Target spec (decided)

| Item | Target |
|---|---|
| Reference machine | **MacBook, Apple M3 Pro**: 11-core CPU (5 performance + 6 efficiency), 14-core GPU, **18 GB unified RAM**, macOS 26.3 |
| Tools already installed | ffmpeg 8.1, Ollama, uv, git. System Python is 3.9, so use `uv` to get Python 3.11 |
| Language | **English** |
| v1 episode | **10 minutes**, processed in **≤ 20 minutes** on a warm run (models already downloaded) |
| v0.1 launch demo | **60 seconds**, as good-looking as possible (§6) |

**The 18 GB of RAM is the hard constraint.** Only one GPU model can be loaded at a time. The pipeline runs in batches: load a model, run it on every item that needs it, unload it, and move to the next model. CPU jobs (voices, compositing) run alongside.

---

## 3. How the animation works (hybrid)

No computer, local or cloud, can generate 10 minutes of AI video in 20 minutes today. An M3 Pro takes several minutes for each few-second clip. So the episode mixes three techniques, the way real anime studios budget their animation:

| Share of episode | Technique | What the viewer sees | Cost |
|---|---|---|---|
| **~70%** dialogue and everyday scenes | **Rigged 2D "cut-out" animation** | Characters really move: gestures, head turns, walking, blinking, breathing, lip sync | Seconds, on the CPU |
| **~25%** establishing and emotional beats | **Camera animation** over illustrated shots: 2.5D depth parallax, pans, zooms, shakes, speed lines | Cinematic "anime PV" shots | Seconds |
| **~5%** (4–6 clips, about 15–25 s) **big action scenes** | **AI image-to-video** (LTX-Video 2B distilled), started from the consistent keyframes | Full hand-drawn-style motion: fights, running, transformations | ~1 min per clip |

Everything is timed **on 2s/3s** (each drawing held for 2–3 frames at 24 fps). That is authentic anime timing, and it halves the rendering work.

### 3.1 The rig pipeline (the core technical bet)

This is the hardest and most important module. It is what makes the animation "true", and it has to be built well.

1. **Character model sheet.** Illustrious SDXL (plus ControlNet OpenPose to force a neutral A-pose) generates each character full-body on a plain background. It also produces **3 head views** (front, 3/4, side), **eye sprites** (open, half, closed) and **3 mouth sprites** (closed, half, open) by inpainting.
2. **Cutout + skeleton.**
   - BiRefNet removes the background.
   - DWPose finds the body keypoints.
   - SAM2 splits the character into parts (head/hair, torso, upper and lower arms, legs) so limbs can overlap each other.
   - LaMa fills in the hidden areas under the parts.
3. **Deformation.** An as-rigid-as-possible (ARAP) mesh warp, like the one in **AnimatedDrawings**, follows the skeleton, so limbs bend smoothly.
4. **Motion library.**
   - Procedural motions: idle breathing, sway, talking gestures, nods, head turns (swapping between the 3 head views), walk and run cycles, pointing and surprise poses.
   - Optional mocap: **CMU mocap BVH** retargeted onto the skeleton. (Mixamo files can't be redistributed.)
5. **Direction.** The screenplay tags each line with an action and an emotion, such as `(gestures angrily)`. A rule table maps tags to motions and expressions.
6. **Lip flap.** The mouth sprite follows how loud the voice is (closed / half / open), the way TV anime does it. Rhubarb phoneme-accurate mouths come in Quality mode.

**Known risk:** anime clothing and hair can deform like rubber under mesh warps. Mitigations are rigging at medium-shot framing, using a separate secondary-motion layer for hair, and cutting to camera-animated shots for wide framings. The v0.1 demo should favour the angles where the rig looks good.

### 3.2 AI action clips

- **LTX-Video 2B distilled** (`gitrepos/LTX-Video`) is the realistic choice for 18 GB.
  - Its T5 text encoder (~9.5 GB) runs first to encode all action prompts, then unloads before the 2B video model loads.
  - Each clip is conditioned on a **start keyframe** (and an end keyframe where possible) made from the character sheet, so identity stays consistent.
  - Generate at about 512×320 and 12 fps, hold each frame for 2 frames to reach 24 fps, then upscale with Real-ESRGAN anime.
- **Candidate upgrade:** LTX-2 via MLX (`gitrepos/ltx-2-mlx`). It is better but needs 16 GB or more for itself, so the benchmark (§7, step 0) decides whether it fits in 18 GB. It stays optional for Quality mode.

---

## 4. Input modes

> **Decision (2026-10-01): no LLM for now.** The user writes the whole storyline scene by scene in a structured script file (a YAML/JSON template to be defined). The pipeline turns it into shots and image prompts with rule-based code. Qwen3.5-4B stays downloaded (`models/llm/`) but isn't used by default. It's kept for the optional "prompt only" mode below and for helpers such as turning free prose into the script format.

```
prompt ─► story.md ─► screenplay.json ─► shots.json ─► assets ─► shots ─► episode.mp4
            ▲              ▲                 ▲            ▲
       paste story    paste dialogue/   paste shot    supply character art /
                      scenes            list          voice sample
```

- `animolocal make "a shy robot learns to surf" --length 10m`
- `animolocal make --story story.md`
- `animolocal make --screenplay screenplay.json`: dialogue comes from the user; shots and art are generated.
- `animolocal regen <project> --shot 12`: every stage is cached and seeded, so any single piece can be redone.
- Flags: `--budget 20m` (trades image count and resolution to fit), `--mode fast|quality`.

---

## 5. Tech stack (M3 Pro)

| Stage | Tool (repo in `gitrepos/`) | Model | Size | License |
|---|---|---|---|---|
| Story → screenplay | `mlx-lm` (or Ollama, already installed) | **Qwen3.5-4B** 4-bit, JSON forced to the schema, thinking off | ~2.5 GB | Apache-2.0 |
| Shot breakdown | **Our own rule-based code** (camera-grammar rules) | none | — | — |
| Character sheets, sprites, backgrounds, keyframes | `stable-diffusion.cpp` (Metal) or `mflux` (MLX); the benchmark picks the faster | **Illustrious XL** + Lightning 4-step LoRA + OpenPose ControlNet | ~7 GB | FAIPL (commercial OK) / openrail++ |
| Quality mode: per-shot consistency edits | `mflux` / `stable-diffusion.cpp` | FLUX.2 klein 4B | ~4–8 GB | Apache-2.0 |
| Cutout / pose / parts / inpaint / depth | `BiRefNet`, `DWPose`, `sam2`, `lama`, `Depth-Anything-V2` | small variants | ~1.8 GB | MIT / Apache ⚠️ (Depth Anything: only the **Small** model is Apache) |
| Rig and deformation | `AnimatedDrawings` (reference for the ARAP method) + our rig module | none | — | MIT |
| Action clips | `LTX-Video` | LTX-Video 2B distilled + T5 | ~15 GB | ⚠️ LTX license (check the terms for your model version) |
| Voices (default) | `kokoro` / `mlx-audio` | **Kokoro-82M**, 50+ English voices, auto-assigned by character age and gender | ~0.3 GB | Apache-2.0 |
| Voices (quality) | `OmniVoice` (the engine behind the trending **VoiceStudio**) | OmniVoice 0.6B: voice design by description, then cloning for consistency | ~1.2 GB | Apache-2.0 |
| Voice fallback | `chatterbox` | Chatterbox Turbo / Multilingual | ~1 GB | MIT |
| Lip sync | our volume-driven lip flap; `rhubarb-lip-sync` in Quality mode | none | — | MIT |
| Music | `ACE-Step-1.5` | 2B turbo (MLX); 2–3 cues per episode, looped per scene | ~5 GB | MIT |
| Sound effects | a bundled CC0 SFX library, plus `stable-audio-tools` for missing sounds | Stable Audio Open Small | ~1.3 GB | ⚠️ Stability Community License (free under $1M revenue; gated on Hugging Face) |
| Upscaling | `Real-ESRGAN` | realesr-animevideov3 | < 0.1 GB | BSD-3 |
| Subtitles | our ASS generator using TTS line timings, burned in with ffmpeg; `pycaps` optional for fancy styles | none | — | MIT |
| Assembly | ffmpeg: `h264_videotoolbox` hardware encoder, stem mix (dialogue / SFX / music, with ducking) | none | — | LGPL/GPL (external binary) |

**Core model downloads come to about 33 GB.** They aren't downloaded yet. The disk has 258 GB free.

---

## 6. Milestones: demo first

### v0.1: the launch demo (target: weeks, not months)
- **A 60-second episode** from one prompt, with no processing-time limit. Quality mode is allowed here.
- Must contain:
  - 2 consistent characters with distinct voices
  - rigged dialogue with lip flap
  - **one jaw-dropping AI action shot**
  - music and SFX
  - animated subtitles
- One-command install on macOS: `curl -fsSL …/install.sh | sh`. It uses `uv`, detects the chip, and fetches only the needed models with a progress bar.
- The README hero is the video, with the headline **"This ran on a MacBook. No cloud. No API."**, and next to it the prompt, the character sheet and the timing breakdown.

### v0.5
- Paste-your-own-script modes (§4), `regen --shot`, and budget/mode flags.
- 3–5 minute episodes.

### v1.0
- **10-minute episodes in ≤ 20 minutes** on the M3 Pro in Fast mode (budget below).
- Character reuse across episodes (a series bible), so a second episode skips creating the characters.

### Later
- NVIDIA tiers (Wan 2.2 5B, Index-AniSora), a local timeline UI, character LoRA training, and Rhubarb plus better rigs.

### v1.0 time budget: 10-minute episode on the M3 Pro (estimates; the benchmark replaces them)

| Stage | Runs on | Estimate |
|---|---|---|
| Story + screenplay (Qwen3.5-4B, ~4–5k tokens) | GPU | 2–3 min |
| Voices (Kokoro, ~7 min of audio) | CPU, **in parallel with images** | < 1 min |
| ~50 images (sheets, sprites, plates, keyframes; Illustrious + Lightning) | GPU | 6–8 min |
| Cutouts, parts, pose, depth, inpainting | GPU/CPU | ~1 min |
| 4–6 LTX-Video action clips (T5 encoding, then 2B generation) | GPU | 4–6 min |
| Music (2–3 ACE-Step cues) | GPU | 1–2 min |
| Rig + camera compositing, lip flap, subtitles, encode (11 cores + VideoToolbox) | CPU | ~2 min |
| Model load/unload | — | ~1 min |
| **Total** | | **~18–22 min** (tight; the `--budget` flag trims image count and clip count to fit) |

---

## 7. Build order

0. **Benchmark (`bench.py`) before anything else.** Time every model on this M3 Pro and write `bench.json`:
   - LLM tokens per second.
   - Seconds per image for sd.cpp vs mflux.
   - Seconds per clip for LTX-Video, and whether LTX-2 MLX fits in 18 GB.
   - Kokoro real-time factor.
   - ACE-Step seconds per cue.
   - Compositing frames per second.

   The budget planner uses these real numbers.
1. `schema.py` (Pydantic: Project, Character, Location, Scene, Shot, DialogueLine, Motion/Emotion tags), `pipeline.py` (batching by stage, caching, seeds, resume) and `hardware.py`.
2. `stages/story.py` (LLM writes the screenplay as JSON matching the schema) and `stages/shots.py` (rule-based shot breakdown).
3. `stages/voice.py` (Kokoro, automatic voice casting, line timings).
4. `stages/characters.py` (model sheet, head views, eye and mouth sprites) and `stages/backgrounds.py`.
5. **`rig/`**: cutout, pose, parts, ARAP deformation and the procedural motion library. Needs the most time.
6. `stages/camera.py` (parallax, pans, zooms, shakes, speed lines).
7. `stages/action.py` (LTX-Video clips started from keyframes).
8. `stages/audio.py` (music cues, SFX library, stem mix) and `stages/subtitles.py` (ASS).
9. `assemble.py`. This completes the **v0.1 demo**. Then the install script and the README video.

### Repo layout

```
animolocal/
├── src/animolocal/{cli,schema,pipeline,hardware,budget}.py
├── src/animolocal/stages/   story, shots, voice, characters, backgrounds, camera, action, audio, subtitles
├── src/animolocal/rig/      cutout, skeleton, parts, arap, motions/, retarget
├── src/animolocal/backends/ llm, image, video, tts, music  (adapters, each tagged with its license)
├── assets/sfx/              CC0 library
├── projects/<name>/         cached artifacts per episode
├── scripts/install.sh, bench.py
└── gitrepos/                reference clones (not shipped; install uses pip packages where available)
```

---

## 8. License policy

- **The repo itself:** Apache-2.0 or MIT.
- **Default install:** Apache, MIT and BSD, plus FAIPL for Illustrious. The revenue-capped tools (LTX and Stable Audio Open) are flagged in `animolocal doctor`.
- **Opt-in only** (`--allow-noncommercial`):
  - NoobAI XL, Anima, DMD2, SANA, FLUX.1 Kontext-dev, insightface.
  - Depth Anything V2 Base/Large, MMAudio, MusicGen, Fish Speech, F5-TTS, Wav2Lip, Sonic.
  - IndexTTS-2.5 (custom license).
- **Never bundled** (region-restricted): HunyuanVideo, FramePack weights, MiniMax H3.
- **Used only as external processes, never imported:**
  - VoiceStudio: AGPL-3.0. Use its engine, OmniVoice (Apache), directly.
  - ComfyUI (GPL), APISR (GPL), TheAnimeScripter (AGPL).

---

## 9. Cloned repos (`gitrepos/`, shallow clones, 2026-09-30)

**Core:**
- `stable-diffusion.cpp`, `mflux`: image generation
- `mlx-lm`: story LLM
- `OmniVoice`, `kokoro`, `mlx-audio`: voices
- `rhubarb-lip-sync`: lip sync (Quality mode)
- `ACE-Step-1.5`: music
- `LTX-Video`: action clips
- `AnimatedDrawings`, `sam2`, `DWPose`, `Depth-Anything-V2`, `BiRefNet`, `lama`: rig and depth
- `stable-audio-tools`: SFX
- `Real-ESRGAN`: upscaling

**Optional:** `LTX-2`, `ltx-2-mlx`, `chatterbox`, `VoiceStudio`, `pycaps`

**Reference only:**
- `Toonflow-app`: closest competitor, a short-drama app that supports a local LLM and ComfyUI.
- `ViMax`: agent design for idea/novel/script → video.
- `AIComicBuilder`: Drizzle + SQLite data model for shots, characters and dialogue.

Note: `stable-diffusion.cpp` needs `git submodule update --init` before it can be built.

---

## 10. Launch plan (how it trends)

- **Proof of demand:** VoiceStudio hit #1 on GitHub Trending (about 50k stars) with "local ElevenLabs alternative". MoneyPrinterTurbo has 100k+ stars and Toonflow about 16k. Nobody offers a fully local prompt-to-episode tool for laptops yet.
- **What decides it:**
  1. The demo video's quality.
  2. An install that works the first time.
  3. A same-day launch on X, r/LocalLLaMA, r/StableDiffusion and Hacker News ("Show HN").
- The README opens with the video, then one install command, one example prompt, and a hardware/time table.
- Consider a punchier name before launch.
- Ship narrow, then let the community build out the NVIDIA tiers, the UI and extra languages.

---

## 11. Corrections to v1 (the original brief)

| v1 claim | Reality |
|---|---|
| `tofuSu/PuLID` | Doesn't exist. PuLID is weak on anime and depends on non-commercial insightface, so it was replaced by character sheets and reference edits. |
| FLUX.1-schnell + PuLID-Flux | PuLID-Flux was trained on FLUX.1-dev (non-commercial) and needs CUDA. |
| ComfyUI `--headless` | No such flag. The repo moved to Comfy-Org/ComfyUI. |
| RIFE/TheAnimeScripter for action scenes | Frame interpolation can't create motion from a still. Action comes from LTX-Video. |
| MultiPassDedup gives on-twos timing | It does the reverse (it smooths video that's already on twos). On-twos timing comes from holding frames. |
| edge-tts is local | It calls Microsoft's cloud service. Replaced by Kokoro and OmniVoice. |
| rhasspy/piper | Archived; moved to OHF-Voice/piper1-gpl and is now GPL. |
| LivePortrait for dialogue | Loses track of anime faces. Replaced by rigged lip flap. |
| AIComicBuilder as the base UI | It's cloud-only, so it's a reference for its data model. |
| RTX 3060/4090 target | The target is now an M3 Pro with 18 GB. |

---

## 12. Open questions

- Is commercial use of the generated episodes a goal? This affects the LTX and Stable Audio license caps.
- Final project name.
- Art style default: modern TV anime vs. chibi/cartoon. Chibi rigs look much better with cut-out animation, so this is worth considering for the demo.
