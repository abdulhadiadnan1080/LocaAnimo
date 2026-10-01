# Contributing to Hadi's LocaAnimo

Thanks for wanting to help! LocaAnimo turns a written story into a voiced, animated episode, entirely on a Mac. Contributions of all sizes are welcome: bug reports, ideas, docs, example scripts and code.

## Ways to contribute

- **Report a bug.** Open an issue with the *Bug report* template. Include your Mac model, RAM and the `projects/<name>/log.txt` lines around the error.
- **Suggest a feature.** Open an issue with the *Feature request* template. The [roadmap](README.md#roadmap) lists what's planned.
- **Share an episode script.** Good example scripts in `examples/` help everyone.
- **Fix or build something.** Look for issues labelled **`good first issue`** or **`help wanted`**.

## Getting set up

You need an Apple Silicon Mac (16 GB+ RAM), [ffmpeg](https://ffmpeg.org) and about 15 GB of free disk space.

```sh
git clone https://github.com/<your-username>/LocaAnimo.git
cd LocaAnimo
scripts/setup.sh            # Python 3.11 env + core models (~11 GB)
./studio.sh                 # the studio app at http://127.0.0.1:4747
```

AI motion needs LTX-Video (`scripts/setup.sh --video`, ~27 GB more). Everything else works without it, and big moments fall back to stills.

## Making a change

1. **Fork** the repo and create a branch: `git checkout -b fix/blink-on-glasses`.
2. Keep changes focused: one fix or feature per pull request.
3. **Test it for real.** Render a short episode with your change:
   ```sh
   scripts/make_episode.sh examples/the_lost_kite.yaml kite
   ```
   Art is cached in `projects/kite/art/`, so after the first run, re-renders only take a few minutes.
4. Open a **pull request** against `main` and fill in the template. Screenshots or a short clip of the result help a lot.

## Code guidelines

- **Match the surrounding code.** Plain Python 3.11, type hints, small functions, a docstring that says *why* a module or function exists.
- **Everything stays local.** No cloud APIs and no telemetry. Models must not download anything without the user asking: keep `HF_HUB_OFFLINE=1` for runtime code.
- **Memory matters.** The target is an 18 GB Mac, so heavy models run one at a time, and MLX models run in their own process (MLX and PyTorch can't share one).
- **Licenses.** New models or vendored code must have a license that allows redistribution and use. List them in the README's model table.
- **The web app has no build step.** `web/` is plain HTML, CSS and JavaScript. Please keep it that way.

## Where things live

| Area | Files |
|---|---|
| Script format | `src/locaanimo/schema.py` |
| Art prompts and drawing | `src/locaanimo/art.py`, `scripts/make_art.py` |
| Faces, mouths, blinks, cut-outs, depth | `src/locaanimo/prep.py`, `src/locaanimo/landmarks.py` |
| Shots, camera and compositing | `src/locaanimo/animate.py` |
| AI motion | `src/locaanimo/video.py` |
| Quality review | `src/locaanimo/review.py` |
| Voices | `src/locaanimo/voices.py` |
| The full render | `scripts/render_episode.py` |
| Studio app | `src/locaanimo/server.py`, `src/locaanimo/pipeline.py`, `web/` |

## Community

Please be kind and constructive. This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
