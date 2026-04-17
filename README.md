# vla

A vision-language-action pipeline for a UR5 arm in CoppeliaSim. You give it a
natural-language task, a local LLaVA model picks the right tool, and the arm
plans + executes the motion using IK and OMPL.

Two pipelines live here:

- **`vla.py`** — original blocks demo. Procedurally builds a scene with three
  colored blocks (red/green/blue) on a pedestal and picks one.
- **`vla_task.py`** — toolbox/grate/screws demo. Loads a `.ttt` scene with a
  toolbox (red=screwdriver, green=grabber, blue=riveter), a grate with two
  screws, and a yellow panel. Can pick and place the grate, then drive its
  screws.

```
red   = screwdriver / hammer
green = grabber     / screwdriver (vla.py)
blue  = riveter     / wrench      (vla.py)
```

## What you need installed

**1. CoppeliaSim** — https://www.coppeliarobotics.com/. Open it before running
anything; the Python scripts connect to a running instance via the ZMQ remote
API, they do not launch it.

**2. Ollama + LLaVA** (for vision-language reasoning)
```bash
# install ollama from https://ollama.com/
ollama pull llava
```
Ollama needs to be running in the background. Skip this only if you always
use `--color` to bypass the VLM.

**3. Python packages**
```bash
pip install coppeliasim-zmqremoteapi-client opencv-python numpy ollama imageio imageio-ffmpeg
```

## Running `vla.py` (blocks demo)

Open CoppeliaSim (empty scene is fine — `vla.py` builds its own).

```bash
# LLaVA picks the block. Default task: "I need to tighten a screw."
python vla.py

# Custom task:
python vla.py --task "I need to drive a nail."       # → red
python vla.py --task "I need to tighten a screw."    # → green
python vla.py --task "I need to loosen a bolt."      # → blue

# Bypass LLaVA:
python vla.py --color red
```

Outputs: `robot_view.jpg`, `vision_debug.jpg`, `final_equipped_view.jpg`,
`trajectory.json`.

## Running `vla_task.py` (toolbox + grate + screws demo)

Open CoppeliaSim. The script loads the `.ttt` scene itself.

```bash
# Full demo — no LLaVA: places grate, then drives screws.
python vla_task.py

# LLaVA picks ONE tool and ONLY that routine runs.
python vla_task.py --task "pick and place the grate over the panel"   # → green
python vla_task.py --task "screw the screws"                           # → red

# Force a tool directly (LLaVA bypassed), ONLY that routine runs.
python vla_task.py --color green
python vla_task.py --color red
```

### Chaining tasks with `--no-reload`

`--task` and `--color` leave the simulation running so you can continue from
the current scene state. The second run skips scene loading and reconnects:

```bash
# Run 1: place the grate on the panel. Sim stays running.
python vla_task.py --task "pick and place the grate over the panel"

# Run 2: reuse the running sim (grate is already placed), drive the screws.
python vla_task.py --task "screw the screws" --no-reload
```

### Side-view recording

By default, `vla_task.py` records an MP4 from a side-view vision sensor:

```bash
python vla_task.py --side-video my_run.mp4   # custom path
python vla_task.py --no-side-video           # disable
```

Run artifacts (`side_view.mp4`, `task_trajectory.json`, `task_metrics.json`)
are regenerated on each run and are git-ignored.

## VLM evaluation sweep

**`run_trials_vlm.py`** drives `query_vlm` repeatedly against a grid of
phrasings × tools to measure accuracy, latency, and memory.

```bash
python run_trials_vlm.py --n 30                      # full sweep (default)
python run_trials_vlm.py --n 5 --out dryrun.json     # quick smoke test
```
Outputs a JSON (`trial_results_vlm.json` by default) with per-call data and
a summary (overall/per-tool/per-prompt accuracy, confusion matrix, latency,
RSS, cold-start time).

## Other scripts

- **`diagnose.py`** — no VLM, no motion. Spawns the blocks scene, takes a
  top-down photo, runs deprojection on known ground-truth positions, and
  prints pixel → world error. Output in `diag.log` / `diag_*.jpg`.
  ```bash
  python diagnose.py
  ```
- **`replay_trajectory.py`** — replays a recorded `trajectory.json` on any
  UR5 in an open CoppeliaSim scene. No vision, no planning.
  ```bash
  python replay_trajectory.py                 # uses trajectory.json
  python replay_trajectory.py my_traj.json    # custom file
  ```

## Troubleshooting

- **"Connection refused" / ZMQ errors** → CoppeliaSim isn't running. Open it
  and leave the default scene up; the ZMQ remote API is on by default in
  recent versions.
- **"Ollama error" / LLaVA not responding** → run `ollama serve` in another
  terminal, or confirm `ollama pull llava` finished. Or use `--color` to
  skip the VLM.
- **Arm picks the wrong block/tool** → run with `--color <name>` to confirm
  motion works, then look at the printed `LLaVA: '...'` reply. Tweak the
  task string if LLaVA is mis-classifying.
- **Wrong pixel detection (vla.py)** → check `vision_debug.jpg`. The yellow
  circle marks what OpenCV thought was the target. If it's off, adjust the
  HSV mask in `get_object_pixel()`.
- **`--no-reload` can't find objects** → the previous run's simulation was
  stopped or the CoppeliaSim window was closed. Start over without
  `--no-reload` to reload the scene.
