# AnimoLocal — Master Project Brief (v2, researched 2026-09-30)

> Paste this into your AI code editor as project context.
> v1 of this brief is kept in `context.v1.md`. Section 10 lists what v1 got wrong.

---

## 1. Goal

Build an open-source repo, meant to trend on GitHub, that takes **a single prompt** and produces **a complete animated cartoon/anime episode**: story, consistent characters, animation, voices, lip sync, sound effects, music and subtitles. The output is one `.mp4`.

**Non-negotiables**
- **No APIs.** Everything runs locally, and there are no network calls after the models are downloaded.
- **Runs on weak hardware.** It must work on CPU-only machines and Apple Silicon Macs. It scales up automatically when a GPU is present (see §3).
- **Every stage can be overridden by the user.** The default is prompt → everything, and the user can paste their own story, dialogue, scenes or shot list at any stage (see §2).
- **Permissive licenses by default.** Non-commercial or region-restricted models are opt-in only (see §7).

---

## 1b. Target spec (decided 2026-09-30)

| Item | Target |
|---|---|
| Language | **English only** |
| Reference machine | **MacBook with Apple M3** (Metal/MLX) |
| Episode length | **10 minutes** |
| Processing budget | **≤ 15 minutes wall-clock** for a warm run (models already downloaded; the first-run download doesn't count) |

### The budget forces these design choices ("Fast mode", the default)

A 10-minute episode is about **120–150 shots**, **~1,300 words of dialogue** (~7 min of speech) and **14,400 frames** at 24 fps. At roughly 5–15 s per SDXL image on an M3, generating an image for every shot alone would take 15–35 min. So Fast mode works the way limited-budget TV anime and visual novels do:

1. **Generate assets, not shots.** Make a small reusable set of images, and build every shot from them:
   - Per character: 1 turnaround sheet, ~6 pose/expression sprites and 3 mouth shapes.
   - Per location: 1–2 background plates.
   - About 6–10 illustrated "hero" moments.
   - Total: **~40–60 images**, not 150.
2. **Shots are composites.** Each shot is a background plate plus character sprite layers, with a camera move (pan, zoom, parallax), puppet motion (blink, breathe, sway) and cuts. Consistency is guaranteed because every shot reuses the same sprites.
3. **Crop reuse.** A wide shot at 1536 px gives medium and close-up framings by cropping, so no new image is needed. Real anime reuses shots heavily.
4. **Lip flap by volume, not phonemes.** TV anime uses 3 mouth shapes (closed / half / open) driven by loudness. It is essentially instant, and it's the correct look. Rhubarb (phoneme-accurate) is available in Quality mode.
5. **The LLM writes less.** The LLM writes the screenplay with scene-level visual tags. The **shot breakdown is rule-based code** (dialogue line → shot, with camera-grammar rules), so there's no ~7k-token shot-list JSON to generate. Thinking/reasoning mode is off.
6. **Fast English TTS by default.** **Kokoro-82M** (Apache-2.0, 50+ English voices, many times faster than real time on Apple Silicon via mlx-audio) is the default. **OmniVoice** (voice design and cloning) is the Quality-mode option.
7. **Audio needs no transcription.** TTS output gives line timings directly, so no Whisper pass is needed; subtitles are generated from the script as ASS.
8. **SFX come from a bundled CC0 library** (whooshes, impacts, ambience) chosen by tag. Stable Audio Open Small is used only for sounds the library lacks.
9. **Music: 2–3 ACE-Step 1.5 turbo cues** (~60–90 s each), looped and crossfaded per scene.
10. **Render in parallel.** Shots are composited in numpy across all CPU cores at 1280×720 → encoded with the `h264_videotoolbox` hardware encoder → upscaled to 1080p in the final encode.
11. **No generative video in Fast mode.** LTX-2/Wan on an M3 take minutes *per clip*, so they only exist in Quality mode (no time cap).

### Time budget (estimates, to be confirmed by the benchmark in §9 step 0)

| Stage | Runs on | Est. time on M3 |
|---|---|---|
| Story + screenplay (Qwen3.5-4B, MLX 4-bit, ~4–5k tokens) | GPU | 2–3 min |
| Voices (Kokoro, ~7 min of audio) | CPU/ANE — **runs in parallel with the image stage** | < 1 min |
| Character sheets, sprites, backgrounds and hero shots (~50 images; Illustrious SDXL + Lightning 4-step) | GPU | 5–7 min |
| Background removal, depth maps and inpainting (BiRefNet / Depth Anything V2-S / LaMa) | GPU/CPU | ~1 min |
| Music (2–3 ACE-Step turbo cues) | GPU | 1–2 min |
| Shot compositing + lip flap + subtitles + encode | CPU (all cores) + VideoToolbox | 1–2 min |
| Model load/unload overhead | — | ~1 min |
| **Total** | | **~12–16 min** |

**Risk:** this is tight on a base M3. It fits comfortably on an M3 Pro/Max. **RAM matters.** With 16 GB or more, one model runs at a time alongside the CPU jobs. With 8 GB, run a 2B LLM and fewer images, or accept ~20 min. The tool shows a live ETA and has a `--budget 15m` flag that automatically trades image count and resolution to fit.

**Two modes:**
- `--mode fast`: the default, budgeted, motion-comic anime.
- `--mode quality`: no time cap. It adds OmniVoice, Rhubarb lip sync, per-shot FLUX.2 klein keyframes, and LTX-2 (MLX) clips for key action shots.

---

## 2. Input modes (all optional, beyond the prompt)

The pipeline is a chain of stages that each produce a JSON artifact. If the user supplies an artifact, that stage is skipped.

```
prompt ─► story.md ─► screenplay.json ─► shotlist.json ─► assets ─► shots ─► episode.mp4
            ▲              ▲                  ▲              ▲
       user may paste  user may paste    user may paste   user may supply
       a story         dialogue/scenes   a shot list      character refs / voice samples
```

- `animolocal make "a shy robot learns to surf"`: runs the full pipeline.
- `animolocal make --story story.md`: skips story writing.
- `animolocal make --screenplay screenplay.json`: dialogue and scenes come from the user; the LLM only breaks them into shots.
- `animolocal make --shotlist shots.json --character hero=ref.png --voice hero=hero.wav`
- `--resume <project>`: every stage caches its results and can be re-run one by one (for example, regenerate shot 12 only).

---

## 3. Hardware tiers (auto-detected, and the user can override)

| Tier | Hardware | Motion approach | Honest quality ceiling |
|---|---|---|---|
| **0: CPU / Mac** | Any CPU, 16 GB RAM; Apple Silicon | Keyframe stills plus **2.5D depth parallax**, Ken Burns camera moves, **layered puppet animation** (blink, breathe, head sway), and **Rhubarb mouth-sprite lip sync** | A polished "motion comic" or anime-PV look. Characters don't walk or fight. |
| **0+: Mac with 16–32 GB unified memory** | M-series | Tier 0, plus **LTX-2.x via MLX (int4)** for a few hero shots | A few seconds of real motion per key shot, a few minutes each |
| **1: 6–8 GB NVIDIA** | RTX 3060/4060 class | **Wan 2.2 TI2V-5B** (4-step LightX2V distill) or Wan 2.1 14B GGUF Q4 with anime LoRAs | Short TV-anime-like shots with occasional warping; an episode is an overnight batch |
| **2: 12 GB+** | RTX 3080/4070 and up | **Index-AniSora V3.2** (built for anime, keyframe control), ToonCrafter in-betweens, InfiniteTalk for talking close-ups | Close to fan-made anime-short quality at 480–720p, then upscaled |

Design rule: **limited animation is a feature, not a fallback.** Anime genuinely animates on 2s/3s (8–12 fps holds). Generate fewer frames, hold them, and spend the compute on keyframe quality.

---

## 4. Tech stack by stage

Legend: ✅ default · ⚙️ optional or higher tier · ⚠️ license caution

### 4.1 Story → screenplay → shot list (local LLM)
- ✅ **Ollama** (MIT) or **llama.cpp** (MIT): ggml-org/llama.cpp. Output is forced to valid JSON with the JSON-schema `format=` parameter (Ollama) or GBNF grammars (llama.cpp), then validated with Pydantic and retried on failure.
- ✅ Model: **Qwen3.5-4B** (Apache-2.0, about 3 GB at Q4). Use **Qwen3.5-9B** when 8 GB or more is available, and **Gemma 4 E4B** (Apache-2.0) as the alternative.
- ⚙️ Mac: MLX-LM + Outlines.
- Design references: **HKUDS/ViMax** (MIT; its Idea2Video, Novel2Video and Script2Video agents and character/environment tracking, though it is cloud-only) and **HBAI-Ltd/Toonflow-app** (MIT; short-drama app that supports local ComfyUI and a local LLM).

### 4.2 Images and character consistency
- ✅ Engine: **leejet/stable-diffusion.cpp** (MIT). One binary covers CPU, CUDA, Metal, Vulkan and OpenCL, with GGUF, LoRA, ControlNet, IP-Adapter and VAE tiling. It supports SDXL, FLUX.2, Z-Image, Qwen-Image(-Edit), Wan and LTX.
- ⚙️ **ComfyUI** (Comfy-Org/ComfyUI, GPL-3.0) is an optional backend for GPU tiers, run as a separate process over its HTTP/WebSocket API (`/prompt`, `/ws`, `/history`, `/view`, `/upload/image`, workflows in API format). There is no `--headless` flag; the server simply runs.
- ✅ Anime base model: **Illustrious XL** (SDXL; Fair AI Public License, commercial use OK) with a **Lightning/Hyper** 4–8-step LoRA, which runs on CPU at 512–768px.
- ✅ Quality preset: **Z-Image-Turbo GGUF** (Apache-2.0, 4–6 GB).
- ✅ **Consistency: FLUX.2 klein 4B GGUF** (Apache-2.0).
  - It is an edit model that accepts multiple reference images.
  - Step 1: generate a **character turnaround sheet** once.
  - Step 2: edit each shot from those references.
  - No training is needed.
- ⚙️ Strongest lock: a one-click **character LoRA** (kohya sd-scripts Apache-2.0 / ai-toolkit MIT), 15–30 images, fits 8 GB.
- ✅ Poses and layout: SDXL **ControlNet** (OpenPose, lineart, depth).
- ✅ Sets: generate one **background plate** per location, cut characters out with **BiRefNet/rembg**, and composite them as layers. Layering also enables the puppet animation and parallax in §4.3.
- ❌ PuLID / InstantID / IP-Adapter-FaceID: built on insightface, which is trained on real faces. They work poorly on anime faces, and insightface's models are non-commercial.

### 4.3 Motion
**Tier 0 (CPU/Mac) toolkit, which is always available:**
- **Depth Anything V2 Small** (Apache-2.0; ⚠️ the Base and Large sizes are non-commercial) plus **LaMa** inpainting (Apache-2.0) → layered 2.5D parallax camera moves.
- **Puppet layers.** Segment the character (SAM or layer-diffusion) into head, eyes, mouth and body, then animate blinks, breathing and sway with affine/mesh warps in numpy.
- Camera language in FFmpeg/numpy: Ken Burns pans, zooms, shakes, speed lines, and impact frames.
- Frame holds on 2s/3s at 24 fps, which gives the authentic anime cadence.

**GPU tiers:**
- ✅ Tier 1: **Wan 2.2 TI2V-5B** (Wan-Video/Wan2.2, Apache-2.0) with the LightX2V 4-step distill. **Wan 2.1 14B I2V GGUF Q4** (city96/ComfyUI-GGUF) plus anime LoRAs.
  - Launcher reference: **deepbeepmeep/Wan2GP**, the "GPU-poor" launcher that runs on 6 GB NVIDIA; no Mac.
- ✅ Tier 2: **bilibili/index-anisora** V3.2 (Apache-2.0, built for anime, 8 steps, keyframe and arbitrary-frame guidance; a 12 GB build exists).
- ⚙️ **Doubiiu/ToonCrafter** (Apache-2.0): cartoon in-betweens between two keyframes, 10–12 GB.
- ⚙️ Mac: **LTX-2.x via MLX** (dgrauet/ltx-2-mlx). ⚠️ LTX Community License, free under $10M revenue.
- ❌ Excluded:
  - Wan 2.5–2.7: API-only.
  - HunyuanVideo / FramePack weights: not licensed in the EU, UK or South Korea.
  - MiniMax H3: excludes the US, EU, UK and South Korea.
  - CogVideoX: outdated.

### 4.4 Voices ⭐ (the trending repo)
- ⭐ **debpalash/VoiceStudio**: #1 on GitHub Trending (Sept 29–30, 2026), about 50k stars. ⚠️ AGPL-3.0.
  - Pitched as the "fully-local ElevenLabs alternative".
  - Features: voice cloning, voice design, dubbing, transcription, audiobooks, 646 languages.
  - It's a studio (Electron + Python) that exposes a local API on `127.0.0.1:3900` and an MCP server.
  - **Use:** call it as an *optional external service* through its local API, or better, use its default engine directly (below), which avoids linking AGPL code.
- ✅ **k2-fsa/OmniVoice** (Apache-2.0, 0.6B, runs on MPS/CPU/GGUF). This is VoiceStudio's default engine.
  - Zero-shot cloning in 600+ languages.
  - **Voice design by attributes** (`instruct="female, child, high pitch"`), which gives every generated character a unique voice with no sample needed.
  - Non-verbal tags: `[laughter]`, `[sigh]`, `[pause 500ms]`.
  - Recommended minimum is 6 GB VRAM; the GGUF build works on CPU.
  ```python
  from omnivoice import OmniVoice
  m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="mps")
  a = m.generate(text="...", instruct="male, teen, energetic")      # design a voice
  a = m.generate(text="...", ref_audio="hero.wav", ref_text="...")  # then clone it, so the voice stays the same
  ```
- ✅ Fallback cloner: **resemble-ai/chatterbox** (MIT; Multilingual V3 0.5B, Turbo 350M).
- ✅ Weak CPUs: **Kokoro-82M** (Apache-2.0, preset voices), **kyutai-labs/pocket-tts** (MIT code, 100M, cloning), and **MOSS-TTS-Nano** (Apache-2.0).
- ⚙️ Emotional scenes:
  - **CosyVoice 3** instruct (Apache-2.0).
  - **IndexTTS-2.5**: best emotion and duration control. ⚠️ Custom bilibili license.
- ⚙️ Anime timbre polish: **RVC** voice conversion (MIT, about 4 GB) as a post-pass.
- ⚙️ Multi-speaker dialogue done natively: **MOSS-TTSD** (Apache-2.0), **Dia** (Apache-2.0, stale).
  - Default approach: generate each line separately and stitch the lines with timed pauses.
- ❌ Not local or not allowed:
  - edge-tts: calls Microsoft's cloud service.
  - Fish Speech/S2, F5-TTS weights: non-commercial.
  - VibeVoice 1.5B: disabled by Microsoft.

### 4.5 Sound effects and music
- ✅ SFX: **Stable Audio Open Small** (341M, runs on CPU). ⚠️ Stability Community License, free under $1M revenue.
- ⚙️ SFX: **MOSS-SoundEffect v2** (OpenMOSS) on GPU.
- ✅ Music: **ace-step/ACE-Step-1.5** (MIT).
  - The 2B turbo model runs in 6 GB or less with INT8 and offload.
  - Has an MLX launcher for Mac.
  - The LLM writes a music cue per scene.
- ❌ MusicGen, MMAudio: non-commercial weights. YuE: needs 24 GB or more.

### 4.6 Lip sync
- ✅ **DanielSWolf/rhubarb-lip-sync** (MIT, CPU, seconds). It turns audio into timed mouth-shape cues (A–H/X).
  - Generate each character's **mouth sprites** once, by inpainting on the character sheet, then swap them per frame.
  - This is how 2D anime is actually animated, and it works for every tier.
- ⚙️ Tier 2 close-ups: **MeiGen-AI/InfiniteTalk** (Apache-2.0; Wan-based; v1.5 generalizes to anime; about 8–12 GB with GGUF, slow).
- ❌ LivePortrait: loses track of anime faces, and its insightface detector is non-commercial.
- ❌ Wav2Lip and Sonic: non-commercial. SadTalker and MuseTalk: realistic faces only.

### 4.7 Subtitles, post and assembly
- ✅ Timing: TTS word timestamps, or **SYSTRAN/faster-whisper** (MIT) forced alignment against the known script. On a Mac it runs on CPU only, so use small or turbo.
- ✅ Subtitles: generate **ASS** files (karaoke `\k` and `\t` animation) and burn them in with FFmpeg's `ass=` filter. No browser is needed.
  - ⚙️ Fancy style: **francozanardi/pycaps** (MIT, alpha, headless Chromium; keep its OpenAI tagging disabled).
- ✅ Upscale: **Real-ESRGAN animevideov3** (BSD-3, ncnn-vulkan; CoreML port on Mac). ⚠️ APISR is GPL.
- ⚙️ Interpolation: **Practical-RIFE** (MIT). Use it **only** on generated motion, never on the held frames of limited animation, because it ruins the anime look.
- ✅ Assembly: **FFmpeg**, fed by piping numpy frames. Normalize every shot (codec, fps, resolution, timebase) before concatenating.
  - Mix the audio in stems: dialogue, SFX and music, with ducking.

---

## 5. Architecture

```
animolocal/
├── src/animolocal/
│   ├── cli.py                 # `animolocal make|resume|regen|doctor`
│   ├── hardware.py            # detect CPU/MPS/CUDA + VRAM → pick tier & model presets
│   ├── schema.py              # Pydantic: Project, Character, Location, Scene, Shot, DialogueLine, AudioCue
│   ├── pipeline.py            # stage DAG, caching, resume, per-shot regeneration
│   ├── models/                # model registry: URLs, hashes, licenses, tiers; lazy download
│   ├── stages/
│   │   ├── story.py           # prompt → story → screenplay → shotlist (LLM + JSON schema)
│   │   ├── characters.py      # turnaround sheet, mouth sprites, voice design → character bible
│   │   ├── backgrounds.py     # location plates
│   │   ├── keyframes.py       # per-shot stills (reference edit + ControlNet)
│   │   ├── motion.py          # tier-dispatched: parallax/puppet | Wan | AniSora | LTX-MLX
│   │   ├── voice.py           # per-line TTS, durations drive shot length
│   │   ├── lipsync.py         # Rhubarb → sprite compositing | InfiniteTalk
│   │   ├── sfx_music.py       # Stable Audio Open Small, ACE-Step
│   │   ├── subtitles.py       # alignment → ASS
│   │   └── assemble.py        # FFmpeg: normalize, composite, mix, concat
│   └── backends/              # adapters: llm (ollama/llamacpp/mlx), image (sdcpp/comfy),
│                              #   video (wan/anisora/ltx-mlx), tts (omnivoice/chatterbox/kokoro/voicestudio-api)
├── projects/<name>/           # cached artifacts: story.md, screenplay.json, shots/, audio/, out/
├── scripts/setup.sh           # one-click install: uv venv, detect hardware, fetch the tier's models
└── examples/                  # sample prompts + pasted-screenplay examples
```

**Key design rules**
1. **Audio first.** Voice lines are generated before motion, and their durations set each shot's length.
2. **Load one model at a time.** Load a model, run it over every shot, then unload it (batching by stage). This keeps peak memory at the size of a single model.
3. **Every backend is an adapter** with a common interface and a license tag. The tier preset picks the adapters, and the user can swap them in `animolocal.toml`.
4. **Every stage is deterministic and cached** (seeds are stored). "Regenerate shot 12" is a first-class command.
5. **GPL/AGPL tools run as separate processes** (ComfyUI, VoiceStudio, TheAnimeScripter), so the repo itself can stay under a permissive license (Apache-2.0 or MIT).

**Core schema (sketch)**
- **Character:** id, name, description, visual tags, reference images, LoRA path, voice (`instruct` or `ref_audio`), mouth sprite set.
- **Scene:** id, location id, time of day, mood, music cue.
- **Shot:** id, scene id, shot type (wide / medium / close-up), camera move, action description, motion kind (`static | parallax | puppet | generated`), characters in the shot, dialogue lines, SFX cues, duration (derived from the audio).
- **DialogueLine:** character id, text, emotion, audio path, word timings.

---

## 6. Reference projects (study them, don't depend on them)

| Repo | What to borrow |
|---|---|
| HBAI-Ltd/Toonflow-app (MIT, about 16k stars) | Short-drama workflow and UX; it already supports local ComfyUI and a local LLM, so it's the closest competitor |
| HKUDS/ViMax (MIT, about 12.5k stars) | Agent design for idea → novel → script → video, and character/environment tracking |
| LingyiChen-AI/AIComicBuilder (Apache-2.0) | Drizzle + SQLite data model (projects, episodes, characters, scenes, shots, dialogues, shot assets); cloud-only |
| debpalash/VoiceStudio (AGPL-3.0, about 50k stars) | Local voice studio and API; shows the demand for a local-first "ElevenLabs killer" |
| harry0703/MoneyPrinterTurbo | The viral README and one-click install pattern |

**The gap:** none of these is a fully local pipeline from prompt to a *complete anime episode* that runs on low-end hardware. That is the pitch.

---

## 7. License policy

- **Default install includes only:** Apache-2.0, MIT, BSD, and FAIPL for Illustrious.
- **Opt-in, behind `--allow-noncommercial` or with a warning shown:**
  - Non-commercial: NoobAI XL, Anima, DMD2, SANA, FLUX.1 Kontext-dev, insightface, Depth Anything V2 Base/Large, MMAudio, MusicGen, Fish S2, F5-TTS, Wav2Lip, Sonic.
  - Revenue caps: LTX (under $10M revenue), Stable Audio Open (under $1M).
  - Custom license: IndexTTS-2.5.
- **Never bundled** (region-restricted): HunyuanVideo, FramePack weights, MiniMax H3.
- **External processes only** (GPL/AGPL): ComfyUI, VoiceStudio, TheAnimeScripter, APISR, Seed-VC.
- `animolocal doctor` prints each active model with its license.

---

## 8. Virality plan

- **One-command install:** `curl … | sh` or `uvx animolocal`, with hardware auto-detection. The first run downloads only the models for that tier.
- **The "runs on my MacBook Air" demo:** a Tier 0 episode rendered on a laptop, shown next to a Tier 2 render of the same shot.
- **The README hero shows** a single prompt, the character sheet, and a 30-second clip with voices and lip sync.
- Share **pasted-screenplay examples** so writers can bring their own scripts.
- **A web UI later:** a local Gradio or Next.js timeline editor for regenerating single shots. The CLI comes first.

---

## 9. Build order (MVP = Fast mode on the M3)

0. **Benchmark first (half a day).** Build `bench.py`, which times each model on this M3 and writes `bench.json`:
   - Qwen3.5-4B MLX tokens/s.
   - Illustrious + Lightning s/image at 1024 and 768 (MLX/Draw Things engine vs stable-diffusion.cpp Metal; pick the faster).
   - Kokoro real-time factor.
   - ACE-Step s/cue.
   - Compositing fps.

   This replaces the estimates in §1b with real numbers, and the budget planner uses them.
1. `schema.py` + `hardware.py` + `pipeline.py`: caching, resume and a **budget planner** (it picks image count and resolution to fit `--budget`).
2. `stages/story.py`: MLX-LM (or Ollama) with Qwen3.5-4B, JSON forced to the schema; the screenplay only, plus the paste-in overrides. Then `shots.py` does the **rule-based shot breakdown**.
3. `stages/voice.py`: Kokoro with a voice auto-assigned per character (by age/gender tags); gives line timings.
4. `stages/characters.py` + `backgrounds.py`:
   - Character: sheet → pose/expression sprites → 3 mouth shapes (inpaint).
   - Background: plates.
   - Then BiRefNet cutouts and Depth Anything V2-S depth maps.
5. `stages/compose.py`: layered compositor with camera moves, parallax, puppet (blink/breathe/sway), volume-driven lip flap and frame holds on 2s/3s. Uses multiprocessing.
6. `assemble.py` + ASS subtitles + audio mix (dialogue / SFX / music stems, with ducking). This gives the **first 10-minute episode in ≤15 min**.
7. CC0 SFX library + ACE-Step music cues.
8. Quality mode: OmniVoice, Rhubarb, per-shot FLUX.2 klein keyframes, LTX-2 MLX clips. Later: NVIDIA tiers (Wan 2.2 / AniSora) and the timeline UI.

---

## 10. Corrections to v1 of this brief

| v1 claim | Reality |
|---|---|
| `tofuSu/PuLID` | Doesn't exist (404). The real repo is ToTheBeginning/PuLID, and it's weak on anime and depends on non-commercial insightface anyway. Replaced by FLUX.2 klein and LoRAs. |
| FLUX.1-schnell + PuLID-Flux | PuLID-Flux was trained on FLUX.1-dev, which is non-commercial, and the nodes require CUDA. |
| ComfyUI `--headless` | That flag doesn't exist, and the repo moved to Comfy-Org/ComfyUI. |
| "Action scenes: run RIFE/TheAnimeScripter" | Interpolation can't create motion from a still. Real motion needs an image-to-video model. |
| "MultiPassDedup gives on-twos timing" | It does the opposite: it smooths video that is already on twos up to 60 fps. On-twos timing comes from holding frames. |
| "edge-tts, 100% local" | edge-tts calls Microsoft's cloud service. |
| rhasspy/piper | Archived. It moved to OHF-Voice/piper1-gpl and is now GPL-3.0. |
| LivePortrait for dialogue | Loses track of anime faces, and its insightface detector is non-commercial. Rhubarb sprites replace it. |
| AIComicBuilder as the base UI | It's a Next.js app that only uses cloud services; it's a reference, not a foundation. |
| "RTX 3060/4090 target" | The new target is CPU/Mac first, scaling up with a GPU. |

---

## 11. Open questions

- ~~Language~~: **English**, decided.
- ~~Length and hardware~~: **10 min episodes, M3 Mac, ≤15 min processing**, decided.
- Which M3 exactly: base, Pro or Max, and how much RAM (8 / 16 / 24 GB+)? This decides whether the 15-min budget is comfortable or tight.
- Is commercial use of generated episodes a goal? This decides whether the opt-in model list matters.
- Interface: CLI-only MVP, or a timeline UI in v1?
