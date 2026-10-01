This is the "Master Project Brief" formatted specifically for Cursor (or any LLM-based code editor). You can paste this directly into your project’s README.md or a prompt_instructions.txt file to give Cursor the full context of what you are building.
Project Master Plan: "AnimoLocal" - The Open-Source Anime Studio
1. The Core Goal
Create a GitHub-trending repository that allows users to transform a structured script and a character reference image into a consistent, high-quality 5-minute anime episode.
Key Value Proposition: 100% Local (no subscription APIs), Perfect Character Consistency (PuLID), and Professional Anime Timing (Limited Animation).
2. Technical Stack & Dependencies
Cursor, use these specific repositories and libraries as the foundation:
A. The Dashboard & Management
Base UI: LingyiChen-AI/AIComicBuilder (Modified for video timelines).
Database: Drizzle ORM + SQLite (To store shots, character seeds, and audio paths).
B. Image & Character Engine
Backend: comfyanonymous/ComfyUI (Running in --headless API mode).
Identity Lock: tofuSu/PuLID (PuLID-Flux node) – This is the critical component for facial consistency.
Model: FLUX.1-schnell (8-step) or SDXL (For speed and high-quality 2D anime aesthetics).
C. Motion & Frame Engine
Orchestrator: NevermindNilas/TheAnimeScripter (CLI for RIFE/Dedup).
Interpolation: hzwer/Practical-RIFE (Generating 24fps motion from keyframes).
Timing: routineLife1/MultiPassDedup (Ensuring "Animation on Twos/Threes" timing so it doesn't look like "AI jelly").
D. Audio & Subtitles
TTS: rany2/edge-tts (High-speed voice gen) or rhasspy/piper (100% Offline).
Transcription: SYSTRAN/faster-whisper (For millisecond-accurate timing).
Subtitles: francozanardi/pycaps (Dynamic, "kinetic" bouncing subtitles).
E. Video Assembly
Final Render: FFmpeg (For panning/zooming effects, audio muxing, and concatenation).
3. System Architecture (The "Orchestrator")
Cursor, your job is to build the Python Glue that connects these modules. The logic flow is:
Input Parsing: Take a script.json (Scene descriptions, Dialogue, Camera instructions) and a character_ref.jpg.
Asset Generation (The Loop):
Image Gen: Send visual prompts to ComfyUI API using Flux + PuLID.
Audio Gen: Generate TTS for the dialogue; measure the duration.
Motion Gen: If the scene is action, run RIFE/TheAnimeScripter. If the scene is dialogue, use LivePortrait or simple FFmpeg Panning (Ken Burns) to match audio length.
Post-Processing:
Run faster-whisper on the audio to create timing for pycaps subtitles.
Assembly:
Use FFmpeg to merge video, audio, and subtitles into a single .mp4 shot.
Concatenate all shots into the final 5-minute master file.
4. Cursor Instructions for Initial Setup
"Cursor, please help me initialize the project structure with the following requirements:"
Directory Layout: Create a structure with /src, /scripts, /assets, and /output.
ComfyUI Connection: Write a Python module in src/comfy_client.py that sends a JSON prompt to a local ComfyUI instance (Port 8188) via WebSockets.
JSON Schema: Define a Scene class using Pydantic that includes: scene_id, character_ref, visual_description, dialogue, and camera_movement.
FFmpeg Wrapper: Create a utility that takes a list of video clips and audio files and generates an FFmpeg command to stitch them perfectly.
5. Viral Hook Strategy (The GitHub Trending Plan)
To ensure this repository trends, the code must prioritize:
The "One-Click" Install: A setup.sh that handles the environment.
Minimal VRAM Footprint: Implement "Model Offloading" (loading/unloading Flux/RIFE sequentially).
The Visual Demo: The README.md must display a side-by-side of the "Character Reference" vs. the "Final Rendered Shot" to prove consistency.
Output Goal:
A high-performance, modular Python pipeline that turns a text-based script into a broadcast-ready 5-minute anime episode, running locally on a single consumer GPU (RTX 3060/4090).
Ready to start? Let's begin by coding the ComfyUI API Client to handle the PuLID face injection.