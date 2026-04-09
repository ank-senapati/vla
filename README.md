# vla

A tiny vision-language-action pipeline for a UR5 arm in CoppeliaSim. You give
it a natural-language task (or force a specific color), a local LLaVA model
figures out which colored block is the right "tool" to grab, OpenCV finds the
block in the camera image, depth-based deprojection turns the pixel into a
world coordinate, and then OMPL + IK drive the arm to pick it up.

```
red   block = Hammer
green block = Screwdriver
blue  block = Wrench
```

## What you need installed

**1. CoppeliaSim** (the simulator itself)
Download from https://www.coppeliarobotics.com/. Open it before running
anything — the Python scripts all connect to a running CoppeliaSim via its
ZMQ remote API, they do not launch it for you.

**2. Ollama + the LLaVA model** (for the vision-language reasoning)
```bash
# install ollama from https://ollama.com/
ollama pull llava
```
Ollama needs to be running in the background (it usually starts automatically
after install). You can skip this if you always use `--color` to bypass the
VLM (see below).

**3. Python packages**
```bash
pip install coppeliasim-zmqremoteapi-client opencv-python numpy ollama
```

## How to run it

**Step 1.** Open CoppeliaSim. You don't need to load any scene — `vla.py`
builds its own scene (floor, pedestal, three colored blocks, UR5, vision
sensor) on startup.

**Step 2.** In a terminal, from this directory:

```bash
# Let LLaVA pick the right block based on a natural-language task.
# Default task is "I need to tighten a screw." (picks the green block).
python vla.py

# Or give it your own task:
python vla.py --task "I need to drive a nail."       # → red   (hammer)
python vla.py --task "I need to tighten a screw."    # → green (screwdriver)
python vla.py --task "I need to loosen a bolt."      # → blue  (wrench)

# Or skip LLaVA entirely and force a specific block.
# Use this if you just want to test the arm/motion and not the VLM.
python vla.py --color red
python vla.py --color green
python vla.py --color blue
```

You'll see the scene build in CoppeliaSim, the camera snap a picture
(`robot_view.jpg`), LLaVA print its answer, OpenCV locate the block,
deprojection compute a world position, and then the arm plan a path and
pick the block up. At the end it takes a final photo
(`final_equipped_view.jpg`) and writes the joint trajectory to
`trajectory.json`.

## The other two scripts

**`diagnose.py`** — no VLM, no arm motion. Just spawns the scene, takes a
top-down photo, runs deprojection on all three blocks using their known
ground-truth positions, and prints the pixel → world error for each. Useful
to confirm the camera + deprojection math is working before debugging
anything else.
```bash
python diagnose.py
```
Output ends up in `diag.log` and `diag_*.jpg`.

**`replay_trajectory.py`** — plays back a recorded `trajectory.json` on any
UR5 in an open CoppeliaSim scene. No vision, no planning, no LLaVA — just
writes joint angles in the same timing as the original run. Good for showing
a motion without re-running the whole pipeline.
```bash
python replay_trajectory.py                 # uses trajectory.json
python replay_trajectory.py my_traj.json    # uses a custom file
```

## Troubleshooting

- **"Connection refused" / ZMQ errors** → CoppeliaSim isn't running, or it's
  running but the remote API isn't enabled. Just open CoppeliaSim and leave
  it on the default empty scene; the ZMQ remote API is on by default in
  recent versions.
- **"Ollama error" / LLaVA not responding** → run `ollama serve` in another
  terminal, or check that `ollama pull llava` actually finished. Or just use
  `--color` to skip LLaVA.
- **Arm picks the wrong block** → run with `--color <name>` to confirm the
  motion works, then compare against what LLaVA is replying (the reply is
  printed as `LLaVA: '...'`). If LLaVA is picking the wrong color, tweak the
  task string to be more explicit.
- **Wrong pixel detection** → look at `vision_debug.jpg` after a run. A
  yellow circle marks whatever OpenCV thought was the target block. If it's
  on the robot instead of a block, the HSV mask in `get_object_pixel()` needs
  adjusting for that color.
