"""
vla_task.py — Task runner for the toolbox / grate / screws scene.

Loads the .ttt scene, asks LLaVA which tool to use given a natural-language
task, attaches that tool (even if LLaVA is wrong), and executes the
corresponding physical sub-routine. The task is driven BY THE VLM's
selection, so if LLaVA picks the wrong color the downstream step fails
and gets logged.

Routines (dispatched by the color LLaVA chooses):
  green  / grabber     → pick up grate, place on yellow base
  red    / screwdriver → drive screw[0] then screw[1]
  blue   / riveter     → attach + lift (placeholder, no rivet sub-action yet)

Outputs:
  - task_metrics.json     (VLM accuracy, position errors, phase timings, success)
  - task_trajectory.json  (full joint trajectory for every smooth-motion step)
  - vlm_view_*.jpg        (the image sent to LLaVA at each selection step)

Usage:
  python3 vla_task.py --task "I need to drive screws into the grate."
  python3 vla_task.py --task "I need to place a grate on the base."
  python3 vla_task.py --color red       # bypass LLaVA and force the screwdriver
  python3 vla_task.py                   # default: runs both grate + screws demo

Pre-flight:
  1. Start CoppeliaSim with an empty scene (the script will load the .ttt)
  2. Make sure Ollama is running with `llava` pulled
"""
import os
import sys
import time
import math
import json
import argparse

import cv2
import numpy as np
import ollama
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Reuse utilities from vla.py
from vla import (
    _normalize_angle_close,
    _normalize_config_close,
    _traj_record,
    _traj_set_phase,
    move_joints_smooth,
    TRAJECTORY,
)

# ─────────────────────────── Scene config ─────────────────────
SCENE_PATH = os.path.abspath("3-AD6ECD9793D23BD9 1.ttt")

# ─────────────────────────── Task tuning ──────────────────────
APPROACH_HEIGHT = 0.15      # meters above a target before descending
GRATE_APPROACH_HEIGHT = 0.25
SMOOTH_STEPS_TRANSIT = 180  # cosine-eased joint steps for transit moves
SMOOTH_STEPS_DESCEND = 80
SMOOTH_DELAY = 0.012
GRATE_PLACE_Z_OFFSET = 0.05  # drop the grate this high above panel origin
GRATE_PICK_Z_OFFSET = 0.11   # stop this high ABOVE the grate's origin Z when picking
SCREW_DRIVE_Z_OFFSET = 0.10  # stop this high ABOVE the screw's origin Z when driving
# The grate mesh extends asymmetrically from its origin (more toward +X/+Y
# where the screws are). After placing the origin at the panel center, the
# grate body is visually shifted. This nudge compensates so the grate's
# VISUAL center lands on the panel's center. Tune these if placement looks off.
GRATE_PLACE_XY_NUDGE = (0.01, -0.02)  # meters, applied to the tip target
SCREW_PUSH_DEPTH = 0.01      # meters to press each screw down
SCREW_DRIVE_XY_NUDGE = (0.0, 0.01)  # (dX, dY) meters — nudge applied to tip target for screw driving
IK_SETTLE_ITERS = 20
IK_SOLVE_ITERS = 100

# ─── Tip orientation for picking up and placing the grate ───
# Absolute world-frame Euler angles (α, β, γ) that the user verified in
# the CoppeliaSim object-properties dialog as the correct orientation
# for the grabber to sit on the grate. Used for BOTH pick_grate and
# place_grate so the grate keeps the same world orientation end-to-end.
GRATE_TIP_ORI_DEG = (172.05, 0.889, 94.295)
GRATE_TIP_ORI_RAD = [math.radians(a) for a in GRATE_TIP_ORI_DEG]

# ─────────────────────────── VLM config ───────────────────────
VLM_MODEL = "llava"
VLM_PROMPT_TEMPLATE = (
    "Look at the toolbox in the image. It contains three tools:\n"
    "- RED tool = electric screwdriver\n"
    "- GREEN tool = grabber / claw\n"
    "- BLUE tool = riveter\n"
    "Task: \"{task}\"\n"
    "Which tool color is needed to complete this task? "
    "Reply with exactly one word: red, green, or blue."
)

TASK_A_DESC = "I need to pick up a grate and place it on top of a yellow base."
TASK_A_EXPECTED = "green"  # grabber

TASK_B_DESC = "I need to drive screws down into a grate."
TASK_B_EXPECTED = "red"    # screwdriver

COLOR_TO_TOOL_KEY = {
    "red": "screwdriver",
    "green": "grabber",
    "blue": "riveter",
}

# ─────────────────────────── Metrics ──────────────────────────
METRICS = {
    "vlm_tool_selections": [],
    "position_errors_mm": [],
    "orientation_errors_deg": [],
    "pose_errors": [],   # combined position+orientation per alignment phase
    "phase_timings_sec": {},
    "task_success": False,
    "failure_phase": None,
    "failure_reason": None,
}
_phase_start = {}


def _begin_phase(name):
    _phase_start[name] = time.time()
    _traj_set_phase(name)
    print(f"\n=== PHASE: {name} ===")


def _end_phase(name):
    if name in _phase_start:
        METRICS["phase_timings_sec"][name] = round(time.time() - _phase_start[name], 3)


def _wrap_angle(a):
    """Wrap to [-π, π]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _log_position_error(phase, target, actual):
    err = math.sqrt(sum((target[i] - actual[i]) ** 2 for i in range(3)))
    METRICS["position_errors_mm"].append({
        "phase": phase,
        "target_xyz": [float(v) for v in target],
        "actual_xyz": [float(v) for v in actual],
        "error_mm": round(err * 1000, 2),
    })
    print(f"   [{phase}] position error: {err * 1000:.1f} mm")
    return err * 1000


def _log_orientation_error(phase, target_ori, actual_ori):
    """Log per-axis and max Euler-angle error in degrees. Target/actual
    are CoppeliaSim world-frame orientations (alpha, beta, gamma)."""
    diffs = [abs(_wrap_angle(actual_ori[i] - target_ori[i])) for i in range(3)]
    max_deg = math.degrees(max(diffs))
    per_axis_deg = [round(math.degrees(d), 2) for d in diffs]
    METRICS["orientation_errors_deg"].append({
        "phase": phase,
        "target_abg_rad": [float(v) for v in target_ori],
        "actual_abg_rad": [float(v) for v in actual_ori],
        "per_axis_deg": per_axis_deg,
        "max_deg": round(max_deg, 2),
    })
    print(f"   [{phase}] orientation error: max={max_deg:.2f}°  per-axis={per_axis_deg}°")
    return max_deg


def _log_pose_error(phase, target_pos, target_ori, actual_pos, actual_ori):
    """Log position + orientation error in one go."""
    pos_mm = _log_position_error(phase, target_pos, actual_pos)
    ori_deg = _log_orientation_error(phase, target_ori, actual_ori)
    METRICS["pose_errors"].append({
        "phase": phase,
        "pos_error_mm": round(pos_mm, 2),
        "ori_error_deg": round(ori_deg, 2),
    })


def _fail(phase, reason):
    METRICS["failure_phase"] = phase
    METRICS["failure_reason"] = reason
    print(f"   !! FAIL at '{phase}': {reason}")


# ─────────────────────────── State container ─────────────────
class SimState:
    pass


# ─────────────────────────── Scene loading ────────────────────
def find_ur5_joints(sim, ur5):
    """Return the 6 kinematic joints under /UR5, excluding /UR5/Revolute_joint,
    ordered by initial Z (base = lowest).
    The end-effector Revolute_joint is skipped because it's not in the main chain.
    """
    tree = sim.getObjectsInTree(ur5, sim.handle_all, 0)
    joints = []
    for h in tree:
        if sim.getObjectType(h) != sim.object_joint_type:
            continue
        alias = sim.getObjectAlias(h, 1)
        if "Revolute_joint" in alias:
            continue
        joints.append(h)
    joints.sort(key=lambda j: sim.getObjectPosition(j, sim.handle_world)[2])
    if len(joints) != 6:
        print(f"   WARNING: expected 6 kinematic joints, got {len(joints)}")
    return joints


def setup_scene(sim, simIK, no_reload=False):
    """Load the scene and build SimState with all object handles.

    If ``no_reload`` is True, assumes the scene is ALREADY loaded and the
    simulation is ALREADY running (from a previous ``vla_task.py`` run).
    Skips ``loadScene``, ``stopSimulation``, and UR5 stabilization — just
    finds the existing object handles and sets up a fresh IK environment.
    This lets you chain tasks across multiple script invocations without
    losing the scene state (e.g. a placed grate).
    """
    if not no_reload:
        sim.stopSimulation()
        time.sleep(0.3)
        print(f"Loading scene: {SCENE_PATH}")
        sim.loadScene(SCENE_PATH)
        time.sleep(0.6)
    else:
        print("--no-reload: reusing existing scene (simulation must be running)")

    st = SimState()
    st.sim = sim
    st.simIK = simIK

    st.ur5 = sim.getObject("/UR5")
    st.tip_dummy = sim.getObject("/UR5/dummy")
    st.joints = find_ur5_joints(sim, st.ur5)
    print(f"   Found {len(st.joints)} UR5 kinematic joints")

    # ─── Stabilize the robot: remove ALL scripts in the UR5 subtree, set
    # every shape static, every joint kinematic, and mark the whole model
    # non-dynamic. Without this the robot links fall under gravity as
    # soon as startSimulation() is called. Mirrors vla.py setup_vla_environment.
    # SKIPPED on --no-reload because this was already done on the first run.
    if not no_reload:
        ur5_tree = sim.getObjectsInTree(st.ur5, sim.handle_all, 0)
        n_scripts_removed = 0
        n_shapes_static = 0
        n_joints_kinematic = 0
        for obj in ur5_tree:
            try:
                t = sim.getObjectType(obj)
                if t == sim.object_script_type:
                    sim.removeObjects([obj])
                    n_scripts_removed += 1
                elif t == sim.object_shape_type:
                    sim.setObjectInt32Param(obj, sim.shapeintparam_static, 1)
                    try:
                        sim.setObjectInt32Param(obj, sim.shapeintparam_respondable, 0)
                    except Exception:
                        pass
                    n_shapes_static += 1
                elif t == sim.object_joint_type:
                    try:
                        sim.setJointMode(obj, sim.jointmode_kinematic, 0)
                    except TypeError:
                        sim.setJointMode(obj, sim.jointmode_kinematic)
                    n_joints_kinematic += 1
            except Exception as e:
                pass
        try:
            sim.setObjectInt32Param(st.ur5, sim.modelproperty_not_dynamic, 1)
        except Exception as e:
            print(f"   (modelproperty_not_dynamic: {e})")
        print(f"   Stabilized UR5: {n_scripts_removed} scripts removed, "
              f"{n_shapes_static} shapes→static, {n_joints_kinematic} joints→kinematic")

    # Work-piece handles
    st.grate = sim.getObject("/Grate_Assembly")
    st.panel = sim.getObject("/Bottom_Panel")
    st.screws = [
        sim.getObject("/91400A242_Mil__Spec__Phillips_Rounded_Head_Screws[0]"),
        sim.getObject("/91400A242_Mil__Spec__Phillips_Rounded_Head_Screws[1]"),
    ]
    st.toolbox = sim.getObject("/Preliminary_Toolbox_decimated")

    # Make the work pieces and toolbox static (skipped on --no-reload)
    if not no_reload:
        for handle in [st.grate, st.panel, st.toolbox] + list(st.screws):
            try:
                sim.setObjectInt32Param(handle, sim.shapeintparam_static, 1)
            except Exception as e:
                print(f"   (static set on {handle}: {e})")

    # Tools and their attach dummies
    claw = sim.getObject("/Claw_Assembly")
    rivet = sim.getObject("/Riveter_Assembly")
    screwdriver = sim.getObject("/Electric_Screwdriver_Top")
    claw_dummy = sim.getObject("/claw_dummy")
    rivet_dummy = sim.getObject("/rivet_dummy")
    screw_dummy = sim.getObject("/screw_dummy")

    # Defensive: an earlier (now-removed) visibility-hide approach could
    # have left tool / arm shapes stuck on visibility layer 0 across
    # reload-less re-runs. Force every shape in the UR5 + tools subtrees
    # back to visibility layer 1 (the standard "always visible" layer)
    # so a freshly loaded scene starts cleanly visible no matter what.
    for root in [st.ur5, claw, rivet, screwdriver]:
        try:
            for o in sim.getObjectsInTree(root, sim.handle_all, 0):
                try:
                    if sim.getObjectType(o) == sim.object_shape_type:
                        sim.setObjectInt32Param(
                            o, sim.objintparam_visibility_layer, 1
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # Tools: also make them static so they stay in their toolbox slots
    for tool_root in [claw, rivet, screwdriver]:
        try:
            sim.setObjectInt32Param(tool_root, sim.shapeintparam_static, 1)
        except Exception as e:
            print(f"   (static set on tool {tool_root}: {e})")

    def snap(name, root, attach_dummy, color):
        # Capture BOTH the dummy's orientation AND the tool root's
        # orientation. The tip position target = dummy position (marks the
        # attach point on the tool). The tip ORIENTATION target = the tool
        # root's orientation — so when the tool is reparented with
        # keepInPlace=True, its pose RELATIVE to the tip is identity and
        # the tool hangs as an extension of the tip, not perpendicular.
        return {
            "name": name,
            "root": root,
            "attach_dummy": attach_dummy,
            "attach_pos": list(sim.getObjectPosition(attach_dummy, sim.handle_world)),
            "dummy_ori": list(sim.getObjectOrientation(attach_dummy, sim.handle_world)),
            "attach_ori": list(sim.getObjectOrientation(root, sim.handle_world)),
            "original_parent": sim.getObjectParent(root),
            "original_pos": list(sim.getObjectPosition(root, sim.handle_world)),
            "original_ori": list(sim.getObjectOrientation(root, sim.handle_world)),
            "color": color,
        }

    st.tools = {
        "grabber":     snap("grabber",     claw,       claw_dummy,  "green"),
        "screwdriver": snap("screwdriver", screwdriver, screw_dummy, "red"),
        "riveter":     snap("riveter",     rivet,      rivet_dummy, "blue"),
    }
    st.current_tool = None

    # ─── UNIVERSAL ATTACH ORIENTATION ───────────────────────────
    # The red screwdriver attached visually-correctly using
    # screw_dummy's world orientation as the tip target. The grabber
    # was hanging perpendicular because claw_dummy's orientation
    # differs from screw_dummy's by ~90° around Z. To make all three
    # tools behave the same way, we override every tool's attach_ori
    # to use the RED screwdriver's dummy orientation. This puts the
    # tip in the exact same pose for every attach/detach, regardless
    # of which tool is being handled.
    universal_ori = list(sim.getObjectOrientation(screw_dummy, sim.handle_world))
    for tool_key in ("grabber", "screwdriver", "riveter"):
        st.tools[tool_key]["attach_ori"] = list(universal_ori)
    st.universal_attach_ori = list(universal_ori)

    # ─── Print captured poses so you can verify the override.
    print("\n   Captured tool poses (all three toolheads):")
    uni_deg = [round(math.degrees(v), 2) for v in universal_ori]
    print(f"     UNIVERSAL tip target ori (from screw_dummy): "
          f"[{uni_deg[0]:+.2f}, {uni_deg[1]:+.2f}, {uni_deg[2]:+.2f}]°")
    for key in ("grabber", "screwdriver", "riveter"):
        t = st.tools[key]
        pos = t["attach_pos"]
        dum_ori_deg = [round(math.degrees(v), 2) for v in t["dummy_ori"]]
        root_ori_deg = [round(math.degrees(v), 2) for v in t["original_ori"]]
        print(f"     {key:12s} ({t['color']:5s})  "
              f"pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")
        print(f"                    own dummy_ori_deg=[{dum_ori_deg[0]:+.2f}, "
              f"{dum_ori_deg[1]:+.2f}, {dum_ori_deg[2]:+.2f}]  "
              f"own root_ori_deg=[{root_ori_deg[0]:+.2f}, "
              f"{root_ori_deg[1]:+.2f}, {root_ori_deg[2]:+.2f}]")

    # Create a fixed overhead camera above the toolbox for VLM queries
    tb_pos = sim.getObjectPosition(st.toolbox, sim.handle_world)
    vlm_cam = sim.createVisionSensor(
        0,
        [512, 512, 0, 0],
        [0.01, 10.0, 60.0 * math.pi / 180.0, 0.1, 0, 0, 0, 0, 0, 0, 0],
    )
    sim.setObjectPosition(
        vlm_cam, sim.handle_world,
        [tb_pos[0], tb_pos[1], tb_pos[2] + 0.8],
    )
    sim.setObjectOrientation(vlm_cam, sim.handle_world, [math.pi, 0, 0])
    sim.setObjectInt32Param(vlm_cam, sim.objintparam_visibility_layer, 0)
    st.vlm_camera = vlm_cam

    # IK target dummy positioned at the tip so IK starts matched
    st.ik_target = sim.createDummy(0.02)
    tip_pos0 = sim.getObjectPosition(st.tip_dummy, sim.handle_world)
    tip_ori0 = sim.getObjectOrientation(st.tip_dummy, sim.handle_world)
    sim.setObjectPosition(st.ik_target, sim.handle_world, tip_pos0)
    sim.setObjectOrientation(st.ik_target, sim.handle_world, tip_ori0)

    st.ik_env = simIK.createEnvironment()
    st.ik_group = simIK.createGroup(st.ik_env)
    simIK.setGroupCalculation(
        st.ik_env, st.ik_group, simIK.method_damped_least_squares, 0.1, 99
    )
    simIK.addElementFromScene(
        st.ik_env, st.ik_group, st.ur5,
        st.tip_dummy, st.ik_target,
        simIK.constraint_position + simIK.constraint_orientation,
    )

    # Target orientation to use for ALL IK solves — preserves the tip's
    # natural upright orientation. Captured before we touch anything.
    st.target_ori = list(tip_ori0)

    return st


# ─────────────────────────── IK helpers ───────────────────────
def resync_ik_target_local(st):
    """Move the IK target dummy to the tip's current world pose so future
    IK solves don't get a step discontinuity. Preserves tip orientation.
    """
    sim = st.sim
    tip_pos = sim.getObjectPosition(st.tip_dummy, sim.handle_world)
    tip_ori = sim.getObjectOrientation(st.tip_dummy, sim.handle_world)
    sim.setObjectPosition(st.ik_target, sim.handle_world, tip_pos)
    sim.setObjectOrientation(st.ik_target, sim.handle_world, tip_ori)
    for _ in range(10):
        st.simIK.handleGroup(st.ik_env, st.ik_group, {"syncWorlds": True})
        time.sleep(0.005)


def compute_ik_no_seed(st, target_pos, target_ori=None, iters=IK_SOLVE_ITERS):
    """Solve IK for (target_pos, target_ori) without leaving scene state
    modified. The iterations do touch scene joints briefly (via
    syncWorlds=True), but we save joint positions + IK target pose
    before and restore them in the finally block, so after the function
    returns the sim is back to its original state.

    Returns the converged joint config (normalized to the shortest
    angular path from the current state).
    """
    sim = st.sim
    simIK = st.simIK
    joints = st.joints
    ik_target = st.ik_target
    ik_env = st.ik_env
    ik_group = st.ik_group

    if target_ori is None:
        target_ori = st.target_ori

    saved_q = [sim.getJointPosition(j) for j in joints]
    saved_tp = list(sim.getObjectPosition(ik_target, sim.handle_world))
    saved_to = list(sim.getObjectOrientation(ik_target, sim.handle_world))

    try:
        sim.setObjectPosition(ik_target, sim.handle_world,
                              [float(x) for x in target_pos])
        sim.setObjectOrientation(ik_target, sim.handle_world,
                                  [float(x) for x in target_ori])
        for _ in range(iters):
            simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})
        goal = [float(sim.getJointPosition(j)) for j in joints]
    finally:
        for j_idx, j in enumerate(joints):
            sim.setJointPosition(j, saved_q[j_idx])
        sim.setObjectPosition(ik_target, sim.handle_world, saved_tp)
        sim.setObjectOrientation(ik_target, sim.handle_world, saved_to)
        simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})

    return _normalize_config_close(goal, saved_q)


def compute_ik_staged(st, target_pos, target_ori=None,
                       stages=20, iters_per_stage=60):
    """
    Solve IK in staged substeps to avoid large-move non-convergence.

    Damped least-squares IK gets stuck in local minima when the target is
    far from the current pose, especially when it also requires a big
    wrist rotation. We fix this by moving the IK TARGET incrementally
    from the current tip position toward the final target across `stages`
    steps, running `iters_per_stage` IK iterations between each. Each
    sub-move is small enough for DLS to converge locally.

    Joint state is saved before and restored after, so the sim is left
    unchanged — the caller still smoothly interpolates joints via
    move_joints_smooth. Returns the converged joint config (normalized to
    the shortest angular path from the starting state).
    """
    sim = st.sim
    simIK = st.simIK
    joints = st.joints
    ik_target = st.ik_target
    ik_env = st.ik_env
    ik_group = st.ik_group

    # Snapshot joint state + IK target pose so we can fully restore
    saved_q = [sim.getJointPosition(j) for j in joints]
    saved_tp = list(sim.getObjectPosition(ik_target, sim.handle_world))
    saved_to = list(sim.getObjectOrientation(ik_target, sim.handle_world))

    start_pos = list(sim.getObjectPosition(st.tip_dummy, sim.handle_world))
    if target_ori is None:
        target_ori = list(sim.getObjectOrientation(st.tip_dummy, sim.handle_world))

    final_pos_err = None
    final_ori_err_deg = None
    try:
        for stage in range(1, stages + 1):
            t = stage / stages
            interp_pos = [
                start_pos[k] + t * (target_pos[k] - start_pos[k])
                for k in range(3)
            ]
            # Orientation: request the final target every stage, let the
            # solver gradually rotate the wrist over the stages.
            sim.setObjectPosition(
                ik_target, sim.handle_world,
                [float(x) for x in interp_pos],
            )
            sim.setObjectOrientation(
                ik_target, sim.handle_world,
                [float(x) for x in target_ori],
            )
            for _ in range(iters_per_stage):
                simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})

        goal = [float(sim.getJointPosition(j)) for j in joints]
        # Measure final convergence while still in the live state
        tip_pos = sim.getObjectPosition(st.tip_dummy, sim.handle_world)
        tip_ori = sim.getObjectOrientation(st.tip_dummy, sim.handle_world)
        final_pos_err = math.sqrt(
            sum((tip_pos[k] - target_pos[k]) ** 2 for k in range(3))
        )
        final_ori_err_deg = math.degrees(
            max(abs(_wrap_angle(tip_ori[k] - target_ori[k])) for k in range(3))
        )
    finally:
        # Restore joint state and IK target dummy — sim is back to the
        # pose it had before this function was called.
        for j_idx, j in enumerate(joints):
            sim.setJointPosition(j, saved_q[j_idx])
        sim.setObjectPosition(ik_target, sim.handle_world, saved_tp)
        sim.setObjectOrientation(ik_target, sim.handle_world, saved_to)
        simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})

    if (final_pos_err is not None
            and (final_pos_err > 0.01 or final_ori_err_deg > 2.0)):
        print(f"   [staged IK] imperfect convergence after {stages} stages: "
              f"pos={final_pos_err * 1000:.1f} mm  ori={final_ori_err_deg:.2f}°")

    return _normalize_config_close(goal, saved_q)


def move_tip_to(st, pos, ori=None, steps=SMOOTH_STEPS_TRANSIT, delay=SMOOTH_DELAY,
                use_staged_ik=True):
    """Silent IK solve → smooth joint-space move → resync IK target.

    By default uses staged IK (reliable for large moves). Pass
    use_staged_ik=False to fall back to the single-shot solve, which is
    faster but can fail on targets that are far away and/or require big
    wrist reorientations.
    """
    if use_staged_ik:
        cfg = compute_ik_staged(st, pos, ori)
    else:
        cfg = compute_ik_no_seed(st, pos, ori)
    move_joints_smooth(st.sim, st.joints, cfg, steps=steps, delay=delay)
    resync_ik_target_local(st)


def refine_tip_alignment(st, target_pos, target_ori, label="align",
                          max_iters=400, pos_tol_m=0.001, ori_tol_deg=0.5):
    """
    After a smooth move, run additional IK iterations on the LIVE joint
    state until the tip is within pos_tol_m and ori_tol_deg of the target,
    or until max_iters is exhausted. The joints are left at the converged
    state so the caller can then do setObjectParent against a perfectly
    aligned tip. Logs how many iterations it took and the final error.
    Returns True iff the tip reached the target within tolerance.
    """
    sim = st.sim
    simIK = st.simIK

    # Set IK target to the EXACT desired pose (override any prior resync)
    sim.setObjectPosition(st.ik_target, sim.handle_world,
                          [float(x) for x in target_pos])
    sim.setObjectOrientation(st.ik_target, sim.handle_world,
                              [float(x) for x in target_ori])

    pos_err = float("inf")
    ori_err_deg = float("inf")
    for i in range(max_iters):
        simIK.handleGroup(st.ik_env, st.ik_group, {"syncWorlds": True})
        tip_pos = sim.getObjectPosition(st.tip_dummy, sim.handle_world)
        tip_ori = sim.getObjectOrientation(st.tip_dummy, sim.handle_world)
        pos_err = math.sqrt(
            sum((tip_pos[k] - target_pos[k]) ** 2 for k in range(3))
        )
        ori_err_rad = max(
            abs(_wrap_angle(tip_ori[k] - target_ori[k])) for k in range(3)
        )
        ori_err_deg = math.degrees(ori_err_rad)
        if pos_err <= pos_tol_m and ori_err_deg <= ori_tol_deg:
            print(f"   [{label}] IK converged in {i + 1} iters: "
                  f"pos={pos_err * 1000:.2f} mm  ori={ori_err_deg:.2f}°")
            return True

    print(f"   [{label}] !! IK did NOT reach tolerance after {max_iters} iters: "
          f"pos={pos_err * 1000:.2f} mm  ori={ori_err_deg:.2f}°")
    return False


def approach_and_descend(st, target_pos, target_ori=None,
                           approach_height=APPROACH_HEIGHT):
    """Move to (x, y, z + approach_height), then straight down to target.
    If target_ori is given, both waypoints request that orientation so the
    tip rotates into alignment *during* the transit rather than snapping
    at the end."""
    above = [target_pos[0], target_pos[1], target_pos[2] + approach_height]
    move_tip_to(st, above, ori=target_ori, steps=SMOOTH_STEPS_TRANSIT)
    move_tip_to(st, target_pos, ori=target_ori, steps=SMOOTH_STEPS_DESCEND)


def _align_dummy_to_target_pose(st, dummy, target_pos, target_ori):
    """Compute the (pos, ori) for st.tip_dummy such that ``dummy`` ends up
    at (target_pos, target_ori) in the world frame.

    ``dummy`` MUST currently be a descendant of st.tip_dummy in the scene
    graph (i.e. the tool that owns it is attached). The returned tuple
    can be passed straight into move_tip_to / compute_ik_no_seed as
    (pos, ori) — IK will then drive the UR5 tip to a pose that places
    the bit dummy on the screw with both position and orientation matched.

    Math: tip_world * local_xform = dummy_world  (since dummy follows tip)
        → tip_world = target_world * inv(local_xform)
    where local_xform = the constant transform from tip_dummy to dummy
    while the tool is attached.
    """
    sim = st.sim

    # Constant local transform from tip_dummy to the tool dummy. Computed
    # AFTER attach so it captures the actual relationship the simulator
    # has, not what we hoped it would be.
    local_xform = sim.getObjectMatrix(dummy, st.tip_dummy)
    local_inv = sim.getMatrixInverse(local_xform)

    # Build the desired world matrix for the tool dummy via a tiny
    # throwaway helper. CoppeliaSim doesn't have a "matrix from euler" call
    # exposed via the ZMQ shim, so the easiest portable way is to set a
    # dummy's pose and then read its matrix.
    helper = sim.createDummy(0.001)
    try:
        sim.setObjectInt32Param(helper, sim.objintparam_visibility_layer, 0)
    except Exception:
        pass
    sim.setObjectPosition(helper, sim.handle_world,
                          [float(v) for v in target_pos])
    sim.setObjectOrientation(helper, sim.handle_world,
                              [float(v) for v in target_ori])
    target_world_xform = sim.getObjectMatrix(helper, sim.handle_world)
    sim.removeObjects([helper])

    # tip_world = target_world * local^-1
    tip_world_xform = sim.multiplyMatrices(target_world_xform, local_inv)

    # Extract pos / ori from the result via another throwaway helper.
    helper2 = sim.createDummy(0.001)
    try:
        sim.setObjectInt32Param(helper2, sim.objintparam_visibility_layer, 0)
    except Exception:
        pass
    sim.setObjectMatrix(helper2, sim.handle_world, tip_world_xform)
    tip_pos = list(sim.getObjectPosition(helper2, sim.handle_world))
    tip_ori = list(sim.getObjectOrientation(helper2, sim.handle_world))
    sim.removeObjects([helper2])

    return tip_pos, tip_ori


def lift_above(st, target_pos, target_ori=None, approach_height=APPROACH_HEIGHT):
    """Lift straight up from wherever we are to approach_height above target."""
    above = [target_pos[0], target_pos[1], target_pos[2] + approach_height]
    move_tip_to(st, above, ori=target_ori, steps=SMOOTH_STEPS_DESCEND)


# ─────────────────────────── Tool attach / detach ─────────────
def _print_tool_target(tool_key, tool, verb):
    """Print the target pos / ori for a tool in a human-readable form."""
    pos = tool["attach_pos"]
    ori_deg = [round(math.degrees(v), 2) for v in tool["attach_ori"]]
    print(f"\n   {verb} {tool_key} ({tool['color']}):")
    print(f"     target pos (m)   : "
          f"[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")
    print(f"     target ori (deg) : "
          f"[{ori_deg[0]:+.2f}, {ori_deg[1]:+.2f}, {ori_deg[2]:+.2f}]")


def attach_tool(st, color):
    """Attach the tool corresponding to the given color. Steps:

      1. Approach → descend with smooth joint-space motion toward the tool's
         attach dummy pose (position AND orientation).
      2. STRICT refinement: run additional IK iterations on the live joint
         state until pos ≤ 1 mm and ori ≤ 0.5° from the target, so the tip
         dummy is *actually* aligned with the tool dummy before parenting.
      3. Log pose error and parent the tool (mirrors the Lua attach).
      4. Lift clear.

    This ensures all three toolheads (grabber / screwdriver / riveter) are
    aligned in both position and orientation before attach.
    """
    tool_key = COLOR_TO_TOOL_KEY.get(color)
    if tool_key is None:
        _fail("attach_tool", f"unknown color '{color}'")
        return False

    tool = st.tools[tool_key]
    sim = st.sim
    _print_tool_target(tool_key, tool, "Attaching")

    # 1. Smooth approach → descend (position + orientation target)
    approach_and_descend(st, tool["attach_pos"], target_ori=tool["attach_ori"])

    # 2. Strict IK refinement to guarantee alignment before parenting
    aligned = refine_tip_alignment(
        st, tool["attach_pos"], tool["attach_ori"],
        label=f"attach_{tool_key}",
    )

    # 3. Log pose error (position + orientation)
    tip_pos = sim.getObjectPosition(st.tip_dummy, sim.handle_world)
    tip_ori = sim.getObjectOrientation(st.tip_dummy, sim.handle_world)
    _log_pose_error(f"attach_{tool_key}",
                    tool["attach_pos"], tool["attach_ori"],
                    tip_pos, tip_ori)
    if not aligned:
        print(f"   [attach_{tool_key}] WARNING: tip not perfectly aligned — "
              f"parenting anyway. Tool may hang at a small offset.")

    # 4. Parent the tool (mirrors Lua)
    sim.setObjectParent(tool["root"], st.tip_dummy, True)
    try:
        sim.resetDynamicObject(tool["root"])
    except Exception as e:
        print(f"   resetDynamicObject: {e}")
    st.current_tool = tool_key
    print(f"   ✓ {tool_key} attached")

    # 5. Retreat upward, keeping tip at the aligned orientation
    lift_above(st, tool["attach_pos"], target_ori=tool["attach_ori"])
    return True


def detach_tool(st):
    """Return the tool to its original pose and detach. Same alignment
    guarantees as attach_tool — the tip is strictly aligned with the tool
    dummy's original pose before the reparent call, so the tool lands in
    exactly the saved slot."""
    if st.current_tool is None:
        print("   detach_tool: nothing held")
        return False

    tool_key = st.current_tool
    tool = st.tools[tool_key]
    sim = st.sim
    _print_tool_target(tool_key, tool, "Detaching")

    # 1. Smooth approach → descend
    approach_and_descend(st, tool["attach_pos"], target_ori=tool["attach_ori"])

    # 2. Strict IK refinement
    aligned = refine_tip_alignment(
        st, tool["attach_pos"], tool["attach_ori"],
        label=f"detach_{tool_key}",
    )

    # 3. Log pose error
    tip_pos = sim.getObjectPosition(st.tip_dummy, sim.handle_world)
    tip_ori = sim.getObjectOrientation(st.tip_dummy, sim.handle_world)
    _log_pose_error(f"detach_{tool_key}",
                    tool["attach_pos"], tool["attach_ori"],
                    tip_pos, tip_ori)
    if not aligned:
        print(f"   [detach_{tool_key}] WARNING: tip not perfectly aligned — "
              f"detaching anyway.")

    # 4. Reparent to original parent (mirrors Lua)
    sim.setObjectParent(tool["root"], tool["original_parent"], True)
    try:
        sim.resetDynamicObject(tool["root"])
    except Exception as e:
        print(f"   resetDynamicObject: {e}")

    # Snap the tool exactly back to its saved pose
    sim.setObjectPosition(tool["root"], sim.handle_world, tool["original_pos"])
    sim.setObjectOrientation(tool["root"], sim.handle_world, tool["original_ori"])

    print(f"   ✓ {tool_key} detached")
    st.current_tool = None

    lift_above(st, tool["attach_pos"], target_ori=tool["attach_ori"])
    return True


# ─────────────────────────── Contact helpers ─────────────────
def check_distance(sim, entity_a, entity_b):
    """Return the minimum world-space distance between two shapes in meters,
    or None if the query failed. Robust to a couple of return-value shapes
    that different CoppeliaSim versions produce."""
    try:
        result = sim.checkDistance(entity_a, entity_b, 0.0)
    except Exception as e:
        print(f"   checkDistance error: {e}")
        return None
    if result is None:
        return None
    # Newer API: (flag, [x1,y1,z1, x2,y2,z2, dist])
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        flag = result[0]
        data = result[1]
        if flag == 1 and data and len(data) >= 7:
            return float(data[6])
    # Some versions return just the 7-element list
    if isinstance(result, (list, tuple)) and len(result) >= 7:
        return float(result[6])
    return None


def get_object_world_center(sim, handle):
    """Return the WORLD-space position of an object's bounding-box center.

    sim.getObjectPosition returns the object's *origin* (pivot), which may
    be offset from the visible center (corner pivot, CAD import, etc.).
    We read the object's own bounding box in its local frame, take the
    midpoint, then transform it to world via the object's transform matrix.
    Falls back to getObjectPosition if the bbox params aren't available.
    """
    try:
        bb_min = [
            sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_x),
            sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_y),
            sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z),
        ]
        bb_max = [
            sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_x),
            sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_y),
            sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z),
        ]
        local_center = [float((bb_min[i] + bb_max[i]) / 2.0) for i in range(3)]
        m = sim.getObjectMatrix(handle, sim.handle_world)
        return list(sim.multiplyVector(m, local_center))
    except Exception as e:
        print(f"   get_object_world_center fallback for handle {handle}: {e}")
        return list(sim.getObjectPosition(handle, sim.handle_world))


def get_grate_working_center(sim, grate, screws):
    """Estimate the grate's visual working center.

    On this scene, /Grate_Assembly has its pivot at a corner and the
    objbbox query returns a box symmetric around the pivot, so the
    'bbox center' equals the origin equals a corner — not helpful.

    A much better signal: the two screws are children of the grate,
    mounted on its working surface. Their midpoint is close to where
    the grate actually is visually, and it's closer to the UR5 base,
    which also helps reachability (less overextension).

    The screws are children of the grate so their world positions follow
    the grate as it moves, meaning this helper works both when the grate
    is in its original pose AND when it's been attached to the tip.

    Returns [cx, cy, grate_origin_z] — XY is the screws' midpoint; Z is
    the grate's origin Z because the screws sit on TOP of the grate and
    their Z isn't the right pickup / placement height.
    """
    grate_pos = list(sim.getObjectPosition(grate, sim.handle_world))
    if not screws:
        return grate_pos
    sp = [list(sim.getObjectPosition(s, sim.handle_world)) for s in screws]
    cx = sum(p[0] for p in sp) / len(sp)
    cy = sum(p[1] for p in sp) / len(sp)
    return [float(cx), float(cy), float(grate_pos[2])]


# ─────────────────────────── Grate pick / place ───────────────
def pick_grate(st):
    """Pick up the grate, with the grabber visually CENTERED over the grate.

    Three things go into the target tip position:
      1. grate_center_world  — the grate's bounding-box center in world space
         (not its origin, which may be a corner or off-center pivot).
      2. grabber_offset_xy   — the grabber's visual XY offset from the tip
         dummy, measured after rotating the tip to the target orientation.
         This accounts for the fact that the claw was attached with
         keepInPlace=True and hangs off the tip at some local offset.
      3. target_tip_xy = grate_center_xy − grabber_offset_xy

    The Z target is kept at the grate's origin Z (same as before) — feel
    free to change if you want a different drop height.
    """
    if st.current_tool != "grabber":
        _fail("pick_grate",
              f"wrong tool attached: {st.current_tool} (need grabber)")
        return False
    sim = st.sim
    # Keep the SAME tip orientation that was used for the toolbox attach.
    # This way the grabber hangs off the tip in the exact same visual
    # pose it had when the robot first attached to it — no rotation
    # between the toolbox and the grate.
    target_ori = list(st.universal_attach_ori)
    target_ori_deg = [round(math.degrees(v), 2) for v in target_ori]

    # ── 1. Grate target — use the screws' midpoint instead of the grate's
    #     origin or bbox center. The grate's mesh is imported with its
    #     pivot at a corner; bbox is symmetric around that pivot, so it
    #     doesn't reflect the visual center. The screws ARE on the grate's
    #     working surface, so their midpoint is a much better target AND
    #     it's closer to the UR5 base (less overextension).
    grate_origin = list(sim.getObjectPosition(st.grate, sim.handle_world))
    grate_bbox_center = get_object_world_center(sim, st.grate)
    grate_working_center = get_grate_working_center(sim, st.grate, st.screws)
    print(f"   grate origin            : "
          f"[{grate_origin[0]:+.4f}, {grate_origin[1]:+.4f}, {grate_origin[2]:+.4f}]")
    print(f"   grate bbox center (bad) : "
          f"[{grate_bbox_center[0]:+.4f}, {grate_bbox_center[1]:+.4f}, {grate_bbox_center[2]:+.4f}]")
    print(f"   grate working center ✓  : "
          f"[{grate_working_center[0]:+.4f}, {grate_working_center[1]:+.4f}, {grate_working_center[2]:+.4f}]  "
          f"(screws midpoint)")
    print(f"   target ori (deg)        : "
          f"[{target_ori_deg[0]:+.3f}, "
          f"{target_ori_deg[1]:+.3f}, "
          f"{target_ori_deg[2]:+.3f}]  (SAME AS TOOLBOX APPROACH)")

    # ── 2. Rotate tip to target orientation at a safe high-up XY ──
    # We do this first so we can measure the grabber's offset AT THE FINAL
    # orientation. The offset depends on orientation — if we measured it
    # before rotating, the XY offset would be wrong after the rotation.
    cur_tip = list(sim.getObjectPosition(st.tip_dummy, sim.handle_world))
    safe_z = max(cur_tip[2], grate_working_center[2] + 0.50)
    rotate_waypoint = [cur_tip[0], cur_tip[1], safe_z]
    move_tip_to(st, rotate_waypoint, ori=target_ori, steps=SMOOTH_STEPS_TRANSIT)

    # ── 3. Measure grabber offset (claw visual center − tip position) ──
    claw_root = st.tools["grabber"]["root"]
    claw_center = get_object_world_center(sim, claw_root)
    tip_now = list(sim.getObjectPosition(st.tip_dummy, sim.handle_world))
    grabber_offset = [claw_center[i] - tip_now[i] for i in range(3)]
    print(f"   grabber offset          : "
          f"[{grabber_offset[0]:+.4f}, {grabber_offset[1]:+.4f}, {grabber_offset[2]:+.4f}]  "
          f"(claw_center - tip_pos)")

    # ── 4. Compute target tip position so grabber is centered over grate ──
    target_tip_xy = [
        grate_working_center[0] - grabber_offset[0],
        grate_working_center[1] - grabber_offset[1],
    ]
    target_tip_z = grate_working_center[2] + GRATE_PICK_Z_OFFSET
    approach_pos = [target_tip_xy[0], target_tip_xy[1],
                    target_tip_z + GRATE_APPROACH_HEIGHT]
    descend_pos = [target_tip_xy[0], target_tip_xy[1], target_tip_z]
    print(f"   tip approach pos        : "
          f"[{approach_pos[0]:+.4f}, {approach_pos[1]:+.4f}, {approach_pos[2]:+.4f}]")
    print(f"   tip descend pos         : "
          f"[{descend_pos[0]:+.4f}, {descend_pos[1]:+.4f}, {descend_pos[2]:+.4f}]")

    # ── 5. Smooth approach + descend + strict alignment ──
    move_tip_to(st, approach_pos, ori=target_ori, steps=SMOOTH_STEPS_TRANSIT)
    move_tip_to(st, descend_pos, ori=target_ori, steps=SMOOTH_STEPS_DESCEND)
    refine_tip_alignment(st, descend_pos, target_ori, label="pick_grate")

    # ── 6. Verify grabber is actually centered over grate (XY error) ──
    claw_after = get_object_world_center(sim, claw_root)
    grate_after = get_grate_working_center(sim, st.grate, st.screws)
    xy_err = math.sqrt(
        (claw_after[0] - grate_after[0]) ** 2
        + (claw_after[1] - grate_after[1]) ** 2
    )
    print(f"   grabber vs grate XY center error: {xy_err * 1000:.2f} mm")
    METRICS.setdefault("grabber_centering_error_mm", {})
    METRICS["grabber_centering_error_mm"]["pick_grate"] = round(xy_err * 1000, 2)

    # ── 7. Log pose error (tip vs target) ──
    tip_pos = sim.getObjectPosition(st.tip_dummy, sim.handle_world)
    tip_ori = sim.getObjectOrientation(st.tip_dummy, sim.handle_world)
    _log_pose_error("pick_grate_align", descend_pos, target_ori, tip_pos, tip_ori)

    # ── 8. Parent grate ──
    sim.setObjectParent(st.grate, st.tip_dummy, True)
    try:
        sim.resetDynamicObject(st.grate)
    except Exception as e:
        print(f"   resetDynamicObject(grate): {e}")
    print("   grate parented to tip")

    # ── 9. Lift ──
    move_tip_to(st, approach_pos, ori=target_ori, steps=SMOOTH_STEPS_DESCEND)
    return True


def place_grate(st,
                  contact_tolerance_m=0.001,
                  fast_gap_m=0.01,
                  fine_step_m=0.002,
                  max_fine_steps=30):
    """Place the grate on the yellow base with proper XY + orientation
    alignment and a contact-driven descent. Uses sim.checkDistance between
    the grate and the panel to stop descending the moment they touch.

    Parameters
    ----------
    contact_tolerance_m : stop when distance ≤ this (default 1 mm)
    fast_gap_m          : after the single big descent, aim for this much
                          remaining clearance before starting fine steps
    fine_step_m         : increment for each fine descent step (default 2 mm)
    max_fine_steps      : give up after this many fine steps
    """
    if st.current_tool != "grabber":
        _fail("place_grate",
              f"wrong tool attached: {st.current_tool} (need grabber)")
        return False
    sim = st.sim
    # Use the SAME orientation as pick_grate (= toolbox attach orientation).
    # That way the grate's pose relative to the tip, which was frozen at
    # pick time, is preserved end-to-end and the grate lands in the same
    # world orientation it had when picked up.
    target_ori = list(st.universal_attach_ori)
    target_ori_deg = [round(math.degrees(v), 2) for v in target_ori]
    print(f"   target ori (deg) : "
          f"[{target_ori_deg[0]:+.3f}, "
          f"{target_ori_deg[1]:+.3f}, "
          f"{target_ori_deg[2]:+.3f}]  (SAME AS TOOLBOX APPROACH)")

    # ── 1. Panel target + grate-to-tip offset. The grate is attached to
    #     the tip, and we need its *working center* (screws midpoint, the
    #     same metric pick_grate used) to know where the grate actually
    #     is in world space. Because the screws are children of the grate,
    #     their world positions follow the grate as it moves, so calling
    #     get_grate_working_center now gives the grate's current world
    #     center with it attached to the tip.
    panel_origin = list(sim.getObjectPosition(st.panel, sim.handle_world))
    panel_center = get_object_world_center(sim, st.panel)

    # The grate was grabbed at its screws midpoint, NOT at its origin.
    # So the grate's origin hangs ~8 cm from the tip. To land the
    # grate's origin on the panel's center, compensate for this offset:
    #   tip_target = panel_center − (grate_origin − tip_pos)
    grate_pos_now = list(sim.getObjectPosition(st.grate, sim.handle_world))
    tip_now = list(sim.getObjectPosition(st.tip_dummy, sim.handle_world))
    grate_offset_xy = [grate_pos_now[i] - tip_now[i] for i in range(2)]
    target_tip_xy = [
        panel_center[0] - grate_offset_xy[0] + GRATE_PLACE_XY_NUDGE[0],
        panel_center[1] - grate_offset_xy[1] + GRATE_PLACE_XY_NUDGE[1],
    ]
    print(f"   panel center     : [{panel_center[0]:+.4f}, {panel_center[1]:+.4f}]")
    print(f"   grate origin now : [{grate_pos_now[0]:+.4f}, {grate_pos_now[1]:+.4f}]")
    print(f"   grate offset XY  : [{grate_offset_xy[0]:+.4f}, {grate_offset_xy[1]:+.4f}]  "
          f"(grate_origin - tip)")
    print(f"   nudge XY         : [{GRATE_PLACE_XY_NUDGE[0]:+.4f}, {GRATE_PLACE_XY_NUDGE[1]:+.4f}]")
    print(f"   tip target XY    : [{target_tip_xy[0]:+.4f}, {target_tip_xy[1]:+.4f}]  "
          f"(panel_center - offset + nudge)")

    # ── 2. Move high above panel (XY already correct for centering)
    approach_z_high = panel_origin[2] + 0.35
    high_above = [target_tip_xy[0], target_tip_xy[1], approach_z_high]
    move_tip_to(st, high_above, ori=target_ori, steps=SMOOTH_STEPS_TRANSIT)

    # ── 3. Grate-to-panel distance for contact descent
    dist = check_distance(sim, st.grate, st.panel)
    print(f"   initial grate→panel distance: "
          f"{'unavailable' if dist is None else f'{dist * 1000:.1f} mm'}")

    # ── 4. Big descent to within fast_gap_m
    if dist is not None and dist > fast_gap_m:
        fast_drop = dist - fast_gap_m
        fast_target = [target_tip_xy[0], target_tip_xy[1],
                       approach_z_high - fast_drop]
        move_tip_to(st, fast_target, ori=target_ori, steps=SMOOTH_STEPS_DESCEND)
        dist = check_distance(sim, st.grate, st.panel)
        print(f"   after fast descent:           "
              f"{'unavailable' if dist is None else f'{dist * 1000:.1f} mm'}")
    elif dist is None:
        fallback_target = [target_tip_xy[0], target_tip_xy[1],
                           panel_origin[2] + 0.05]
        move_tip_to(st, fallback_target, ori=target_ori,
                    steps=SMOOTH_STEPS_DESCEND)

    # ── 5. Fine descent loop with contact check
    contacted = False
    for step_idx in range(max_fine_steps):
        dist = check_distance(sim, st.grate, st.panel)
        if dist is None:
            print(f"   fine step {step_idx}: checkDistance unavailable, stopping")
            break
        if dist <= contact_tolerance_m:
            contacted = True
            print(f"   CONTACT at fine step {step_idx}: "
                  f"dist = {dist * 1000:.2f} mm")
            break
        step = min(fine_step_m, dist)
        cur_tip = list(sim.getObjectPosition(st.tip_dummy, sim.handle_world))
        new_target = [target_tip_xy[0], target_tip_xy[1], cur_tip[2] - step]
        move_tip_to(st, new_target, ori=target_ori, steps=20, delay=0.008)
    else:
        print(f"   WARNING: no contact after {max_fine_steps} fine steps")

    # ── 6. Verify grate is actually centered over panel (XY)
    grate_final_center = get_grate_working_center(sim, st.grate, st.screws)
    xy_err = math.sqrt(
        (grate_final_center[0] - panel_center[0]) ** 2
        + (grate_final_center[1] - panel_center[1]) ** 2
    )
    print(f"   grate vs panel XY center error: {xy_err * 1000:.2f} mm")
    METRICS.setdefault("grabber_centering_error_mm", {})
    METRICS["grabber_centering_error_mm"]["place_grate"] = round(xy_err * 1000, 2)

    # Log final pose error for the grate
    grate_final_pos = list(sim.getObjectPosition(st.grate, sim.handle_world))
    grate_final_ori = list(sim.getObjectOrientation(st.grate, sim.handle_world))
    _log_pose_error("place_grate",
                    [panel_center[0], panel_center[1], panel_origin[2]],
                    target_ori,
                    grate_final_pos, grate_final_ori)
    METRICS["place_grate_contact_distance_mm"] = (
        round(dist * 1000, 2) if dist is not None else None
    )
    METRICS["place_grate_contacted"] = contacted

    # ── 7. Release grate to the scene root
    sim.setObjectParent(st.grate, -1, True)
    try:
        sim.resetDynamicObject(st.grate)
    except Exception as e:
        print(f"   resetDynamicObject(grate): {e}")
    print(f"   grate placed at {grate_final_pos}")

    # ── 8. Lift tip clear
    lift_above(st, high_above, target_ori=target_ori,
               approach_height=0.0)
    return True


# ─────────────────────────── Screw driving ────────────────────
def drive_screw(st, screw_handle, idx):
    """Align the screwdriver over the screw, parent the screw to the tip,
    and push straight down by SCREW_PUSH_DEPTH.

    The end-effector orientation is kept at ``st.universal_attach_ori`` —
    the SAME orientation the tip had when it approached the toolbox and
    attached to the screwdriver. This means the screwdriver stays in the
    exact visual pose it was in at the toolbox; no wrist rotations happen
    between attach and screw-driving.
    """
    if st.current_tool != "screwdriver":
        _fail(f"drive_screw_{idx}",
              f"wrong tool attached: {st.current_tool} (need screwdriver)")
        return False

    sim = st.sim
    tip_ori = list(st.universal_attach_ori)
    screw_pos = list(sim.getObjectPosition(screw_handle, sim.handle_world))
    print(f"   screw[{idx}] target pos = "
          f"[{screw_pos[0]:+.4f}, {screw_pos[1]:+.4f}, {screw_pos[2]:+.4f}]")

    # ── 1. Measure screwdriver bit offset from tip ──
    # screw_dummy is the ATTACH point (top of screwdriver) — it sits
    # right at the tip, so its offset is ~0, which is useless.
    # The screwdriver ROOT is at the body/bit end. After the tip rotates
    # to the universal_attach_ori, the screwdriver hangs tilted, so the
    # root (bit end) is laterally offset from the tip. We use the ROOT's
    # world position as the proxy for where the bit actually is.
    screwdriver_root = st.tools["screwdriver"]["root"]
    bit_pos = list(sim.getObjectPosition(screwdriver_root, sim.handle_world))
    tip_now = list(sim.getObjectPosition(st.tip_dummy, sim.handle_world))
    bit_offset_xy = [bit_pos[i] - tip_now[i] for i in range(2)]
    print(f"   screwdriver root= [{bit_pos[0]:+.4f}, {bit_pos[1]:+.4f}]")
    print(f"   bit offset XY   = [{bit_offset_xy[0]:+.4f}, {bit_offset_xy[1]:+.4f}]  "
          f"(screwdriver_root - tip)")

    # ── 2. Tip target so the BIT lands on the screw + manual nudge ──
    target_tip_xy = [
        screw_pos[0] - bit_offset_xy[0] + SCREW_DRIVE_XY_NUDGE[0],
        screw_pos[1] - bit_offset_xy[1] + SCREW_DRIVE_XY_NUDGE[1],
    ]
    print(f"   nudge XY        = [{SCREW_DRIVE_XY_NUDGE[0]:+.4f}, {SCREW_DRIVE_XY_NUDGE[1]:+.4f}]")
    print(f"   tip target XY   = [{target_tip_xy[0]:+.4f}, {target_tip_xy[1]:+.4f}]")

    # ── 3. Approach above the screw ──
    above_pos = [target_tip_xy[0], target_tip_xy[1],
                 screw_pos[2] + APPROACH_HEIGHT]
    move_tip_to(st, above_pos, ori=tip_ori, steps=SMOOTH_STEPS_TRANSIT)

    # ── 4. Descend ──
    descend_pos = [target_tip_xy[0], target_tip_xy[1],
                   screw_pos[2] + SCREW_DRIVE_Z_OFFSET]
    move_tip_to(st, descend_pos, ori=tip_ori, steps=SMOOTH_STEPS_DESCEND)
    refine_tip_alignment(st, descend_pos, tip_ori,
                          label=f"drive_screw_{idx}_align")

    # ── 5. Verify bit (screwdriver root) is over the screw ──
    bit_after = list(sim.getObjectPosition(screwdriver_root, sim.handle_world))
    bit_vs_screw = math.sqrt(
        (bit_after[0] - screw_pos[0]) ** 2
        + (bit_after[1] - screw_pos[1]) ** 2
    )
    print(f"   bit vs screw XY error: {bit_vs_screw * 1000:.2f} mm")
    tip_world_pos = list(sim.getObjectPosition(st.tip_dummy, sim.handle_world))
    tip_world_ori = list(sim.getObjectOrientation(st.tip_dummy, sim.handle_world))
    _log_pose_error(f"drive_screw_{idx}_align",
                    descend_pos, tip_ori, tip_world_pos, tip_world_ori)

    # ── 6. Parent the screw ──
    sim.setObjectParent(screw_handle, st.tip_dummy, True)
    try:
        sim.resetDynamicObject(screw_handle)
    except Exception:
        pass

    # ── 7. Push straight down from the offset height ──
    pushed_pos = [target_tip_xy[0], target_tip_xy[1],
                  screw_pos[2] + SCREW_DRIVE_Z_OFFSET - SCREW_PUSH_DEPTH]
    move_tip_to(st, pushed_pos, ori=tip_ori, steps=60)
    screw_final = list(sim.getObjectPosition(screw_handle, sim.handle_world))
    _log_position_error(f"drive_screw_{idx}_pushed", pushed_pos, screw_final)

    # ── 8. Release ──
    sim.setObjectParent(screw_handle, -1, True)
    try:
        sim.resetDynamicObject(screw_handle)
    except Exception:
        pass

    # ── 9. Lift ──
    move_tip_to(st, above_pos, ori=tip_ori, steps=SMOOTH_STEPS_DESCEND)
    return True


# ─────────────────────────── VLM ──────────────────────────────
def capture_vlm_image(st, path):
    sim = st.sim
    time.sleep(0.3)
    img, res = None, None
    for _ in range(20):
        img, res = sim.getVisionSensorImg(st.vlm_camera)
        if img:
            break
        time.sleep(0.1)
    if not img:
        return None
    arr = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
    arr = cv2.flip(arr, 0)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    return bgr


def query_vlm(st, task_description, expected_color=None):
    """Capture an image, ask LLaVA which color tool to use for the given
    task, and record the result. ``expected_color`` is optional — when
    None, accuracy isn't computed for this query (useful for free-form
    --task input where we don't know the right answer in advance)."""
    fname = f"vlm_view_{len(METRICS['vlm_tool_selections'])}.jpg"
    img = capture_vlm_image(st, fname)
    if img is None:
        print("   VLM: failed to capture image")
        got = None
        reply = ""
    else:
        prompt = VLM_PROMPT_TEMPLATE.format(task=task_description)
        try:
            r = ollama.chat(
                model=VLM_MODEL,
                messages=[{"role": "user", "content": prompt, "images": [fname]}],
            )
            reply = r["message"]["content"].lower()
        except Exception as e:
            print(f"   Ollama error: {e}")
            reply = ""
        got = None
        for c in ("red", "green", "blue"):
            if c in reply:
                got = c
                break

    correct = None if expected_color is None else (got == expected_color)
    METRICS["vlm_tool_selections"].append({
        "task": task_description,
        "expected": expected_color,
        "got": got,
        "raw_reply": reply[:200],
        "correct": correct,
        "image": fname,
    })
    print(f"   VLM: task='{task_description[:60]}'")
    if expected_color is None:
        print(f"        got={got}  (no expected answer for free-form task)")
    else:
        mark = "✓" if correct else "✗"
        print(f"        expected={expected_color}, got={got}  {mark}")
    return got


# ─────────────────────────── Sub-routines per tool ─────────────
def _run_grate_routine(st):
    """Pick the grate and place it on the yellow base."""
    _begin_phase("pick_grate")
    ok = pick_grate(st)
    _end_phase("pick_grate")
    if not ok:
        return False

    _begin_phase("place_grate")
    ok = place_grate(st)
    _end_phase("place_grate")
    return ok


def _run_screw_routine(st):
    """Drive every screw in st.screws in order."""
    for idx, screw in enumerate(st.screws):
        phase = f"drive_screw_{idx}"
        _begin_phase(phase)
        ok = drive_screw(st, screw, idx)
        _end_phase(phase)
        if not ok:
            return False
    return True


def _run_riveter_routine(st):
    """Riveter sub-action — currently a placeholder. Just confirm we can
    hold the tool. Extend this when a real riveting target exists."""
    print("   (riveter routine: no rivet sub-action defined yet — "
          "tool was attached and lifted as a no-op demo)")
    return True


# ─────────────────────────── Single-task pipeline ──────────────
def _run_one_task(st, task_description, expected_color=None,
                  force_color=None, phase_suffix=""):
    """Ask LLaVA which tool to use for `task_description`, attach that
    tool, run the matching physical sub-routine, and detach.

    If ``force_color`` is provided, LLaVA is bypassed entirely and the
    given color is used directly.

    Returns True on success, False otherwise.
    """
    sfx = f"_{phase_suffix}" if phase_suffix else ""

    # ── Tool selection: VLM or forced override ──
    if force_color is not None:
        chosen = force_color
        print(f"\n=== TOOL SELECT (forced): {chosen} ===")
        # Still log the selection so the metrics file shows what was used
        METRICS["vlm_tool_selections"].append({
            "task": task_description,
            "expected": expected_color,
            "got": chosen,
            "raw_reply": "(forced via --color, VLM bypassed)",
            "correct": (None if expected_color is None
                        else chosen == expected_color),
            "image": None,
        })
    else:
        _begin_phase(f"vlm_select{sfx}")
        chosen = query_vlm(st, task_description, expected_color)
        _end_phase(f"vlm_select{sfx}")
        if chosen is None:
            _fail(f"vlm_select{sfx}", "no reply from VLM")
            return False

    # ── Attach the chosen tool ──
    _begin_phase(f"attach_tool{sfx}")
    if not attach_tool(st, chosen):
        _end_phase(f"attach_tool{sfx}")
        return False
    _end_phase(f"attach_tool{sfx}")

    # ── Dispatch physical sub-routine based on the tool color ──
    routines = {
        "green": _run_grate_routine,
        "red":   _run_screw_routine,
        "blue":  _run_riveter_routine,
    }
    routine = routines.get(chosen)
    if routine is None:
        _fail("dispatch", f"no routine for color '{chosen}'")
        return False

    routine_phase = f"routine_{COLOR_TO_TOOL_KEY[chosen]}{sfx}"
    _begin_phase(routine_phase)
    routine_ok = routine(st)
    _end_phase(routine_phase)

    # ── Always try to detach so the scene is clean for the next stage ──
    _begin_phase(f"detach_tool{sfx}")
    detach_tool(st)
    _end_phase(f"detach_tool{sfx}")

    return routine_ok


# ─────────────────────────── Main ─────────────────────────────
def main(task=None, force_color=None, no_reload=False):
    """Run the task pipeline.

    Args:
        task: Natural-language command for the VLM.  LLaVA picks a tool
              and ONLY that tool's routine runs. No automatic
              prerequisites — if you need the grate placed before
              screwing, run the grate task first then ``--no-reload``.
        force_color: 'red' | 'green' | 'blue' to bypass LLaVA and force
              that tool selection.
        no_reload: If True, skip scene loading — connect to the
              already-running simulation and pick up where the last
              ``vla_task.py`` run left off.  The simulation is NOT
              stopped at the end so subsequent runs can chain.
    """
    client = RemoteAPIClient()
    sim = client.require("sim")
    simIK = client.require("simIK")

    st = setup_scene(sim, simIK, no_reload=no_reload)

    if not no_reload:
        sim.startSimulation()
        time.sleep(0.3)

    interrupted = False
    try:
        # Let IK settle before capturing home pose
        for _ in range(IK_SETTLE_ITERS):
            simIK.handleGroup(st.ik_env, st.ik_group, {"syncWorlds": True})
            time.sleep(0.02)

        st.home_config = [sim.getJointPosition(j) for j in st.joints]
        print(f"\nHome config (rad): {[round(q, 3) for q in st.home_config]}")
        home_tip = sim.getObjectPosition(st.tip_dummy, sim.handle_world)
        print(f"Home tip world pos: {home_tip}")

        _traj_set_phase("start")
        _traj_record(sim, st.joints)

        task_ok = True

        # ── Determine what to run ──
        #
        #   --task "<text>"  → LLaVA picks ONE tool. ONLY that tool's routine
        #                      runs. No automatic prerequisites — if you need
        #                      the grate placed first, run that task in a
        #                      separate invocation, then use --no-reload.
        #
        #   --color <c>      → Same but LLaVA is bypassed; color is forced.
        #
        #   neither          → Default full demo: grate (forced green) then
        #                      screws (forced red), no LLaVA.

        if task is not None:
            # ── --task mode: LLaVA decides the tool, run ONLY that routine ──
            _begin_phase("vlm_select_user")
            chosen_color = query_vlm(st, task, expected_color=None)
            _end_phase("vlm_select_user")
            if chosen_color is None:
                _fail("vlm_select_user", "no reply from VLM")
                task_ok = False
            if task_ok:
                print(f"\n>>> USER TASK — \"{task}\"")
                if not _run_one_task(
                    st, task, expected_color=None,
                    force_color=chosen_color, phase_suffix="user",
                ):
                    task_ok = False

        elif force_color is not None:
            # ── --color mode: run ONLY that tool's routine, no LLaVA ──
            canned = {
                "red":   TASK_B_DESC,
                "green": TASK_A_DESC,
                "blue":  "I need to rivet two parts together.",
            }[force_color]
            print(f"\n>>> USER TASK (forced {force_color}) — \"{canned}\"")
            if not _run_one_task(
                st, canned, expected_color=None,
                force_color=force_color, phase_suffix="user",
            ):
                task_ok = False

        else:
            # ── Default mode: full demo, no LLaVA ──
            # Phase 1: grate (forced green)
            print(f"\n>>> PHASE 1 (forced green) — \"{TASK_A_DESC}\"")
            if not _run_one_task(
                st, TASK_A_DESC, expected_color=TASK_A_EXPECTED,
                force_color="green", phase_suffix="grate",
            ):
                task_ok = False
            # Phase 2: screws (forced red)
            if task_ok:
                print(f"\n>>> PHASE 2 (forced red) — \"{TASK_B_DESC}\"")
                if not _run_one_task(
                    st, TASK_B_DESC, expected_color=TASK_B_EXPECTED,
                    force_color="red", phase_suffix="user",
                ):
                    task_ok = False

        # ───── Return home ─────
        _begin_phase("return_home")
        current = [sim.getJointPosition(j) for j in st.joints]
        home_normalized = _normalize_config_close(st.home_config, current)
        move_joints_smooth(sim, st.joints, home_normalized,
                            steps=SMOOTH_STEPS_TRANSIT, delay=SMOOTH_DELAY)
        resync_ik_target_local(st)
        _end_phase("return_home")

        # ───── Metrics ─────
        METRICS["task_success"] = task_ok
        # Only count VLM selections that had a known expected answer when
        # computing accuracy (free-form --task queries have correct=None
        # because we don't know the right answer).
        scored = [v for v in METRICS["vlm_tool_selections"]
                  if v.get("correct") is not None]
        if scored:
            n_correct = sum(1 for v in scored if v["correct"])
            METRICS["vlm_accuracy"] = round(n_correct / len(scored), 3)
        else:
            METRICS["vlm_accuracy"] = None
        pos_errs = [e["error_mm"] for e in METRICS["position_errors_mm"]]
        if pos_errs:
            METRICS["mean_position_error_mm"] = round(sum(pos_errs) / len(pos_errs), 2)
            METRICS["max_position_error_mm"] = round(max(pos_errs), 2)
        ori_errs = [e["max_deg"] for e in METRICS["orientation_errors_deg"]]
        if ori_errs:
            METRICS["mean_orientation_error_deg"] = round(sum(ori_errs) / len(ori_errs), 2)
            METRICS["max_orientation_error_deg"] = round(max(ori_errs), 2)

        with open("task_metrics.json", "w") as f:
            json.dump(METRICS, f, indent=2)
        with open("task_trajectory.json", "w") as f:
            json.dump({"samples": TRAJECTORY, "meta": {
                "home_config": st.home_config,
                "num_samples": len(TRAJECTORY),
            }}, f, indent=2)

        print("\n" + "=" * 60)
        print(f"  task_success       : {task_ok}")
        print(f"  vlm_accuracy       : {METRICS.get('vlm_accuracy')}")
        if pos_errs:
            print(f"  mean_pos_error_mm  : {METRICS['mean_position_error_mm']}")
            print(f"  max_pos_error_mm   : {METRICS['max_position_error_mm']}")
        if ori_errs:
            print(f"  mean_ori_error_deg : {METRICS['mean_orientation_error_deg']}")
            print(f"  max_ori_error_deg  : {METRICS['max_orientation_error_deg']}")
        if METRICS["failure_reason"]:
            print(f"  failure_phase      : {METRICS['failure_phase']}")
            print(f"  failure_reason     : {METRICS['failure_reason']}")
        print(f"  metrics saved to   : task_metrics.json")
        print(f"  trajectory saved to: task_trajectory.json")
        print("=" * 60)

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n" + "!" * 60)
        print("  INTERRUPTED BY USER (Ctrl+C)")
        print("  Halting trajectory and safely stopping simulation...")
        print("!" * 60 + "\n")

    finally:
        # Determine if we should force a stop. We stop if interrupted, OR if it's the default full demo.
        should_stop = interrupted or (task is None and force_color is None)

        if should_stop:
            if not interrupted:
                time.sleep(2)
                print("\n   Simulation stopped (full demo complete).")
            
            # Ensures the simulator always stops, even if the ZMQ socket is broken from the interrupt
            try:
                sim.stopSimulation()
            except Exception:
                # The socket is out-of-sync. Spin up a fresh "rescue" client to force the stop.
                try:
                    rescue_client = RemoteAPIClient()
                    rescue_sim = rescue_client.require("sim")
                    rescue_sim.removeObjectFromSelection(rescue_sim.handle_all)
                    rescue_sim.stopSimulation()
                except Exception as rescue_e:
                    print(f"  [Warning] Could not forcefully stop simulation: {rescue_e}")
                    
            if interrupted:
                print("Simulation stopped.")
        else:
            # Leave running for --no-reload chaining
            print("\n   Simulation left running — use --no-reload on the next "
                  "run to continue from this state.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLA task runner. With --task, LLaVA picks ONE tool and "
                    "ONLY that tool's routine runs (no automatic grate "
                    "prerequisite). Chain tasks by running the grate task "
                    "first, then --task + --no-reload for screws. With no "
                    "flags, runs the full demo (grate + screws, no LLaVA).",
    )
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task",
        type=str,
        default=None,
        help="Natural-language command passed to LLaVA. LLaVA picks one "
             "of red/green/blue and ONLY that tool's routine runs. "
             "Example: \"pick and place the grate\" → green/grabber. "
             "Example: \"screw the screws\" → red/screwdriver. "
             "The simulation is left running afterwards so you can "
             "chain a second --task with --no-reload.",
    )
    task_group.add_argument(
        "--color",
        choices=["red", "green", "blue"],
        help="Force a tool color directly (LLaVA bypassed). ONLY that "
             "tool's routine runs. red=screwdriver, green=grabber, "
             "blue=riveter.",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Skip scene loading — connect to the already-running "
             "simulation and continue from where the last run left off. "
             "Use this for the second task in a chain: first run places "
             "the grate, second run (with --no-reload) drives screws.",
    )
    args = parser.parse_args()

    main(task=args.task, force_color=args.color, no_reload=args.no_reload)
