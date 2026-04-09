import argparse
import re
import time
import math
import json
import cv2
import numpy as np
import ollama
import os
import glob
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ─────────────────────────── Trajectory Recorder ──────────────
# Every IK / OMPL step appends {"phase": str, "t": float, "q": [j0..j5]}
# to this list. Saved to trajectory.json at the end of main().
TRAJECTORY = []
_TRAJ_T0 = None
_TRAJ_PHASE = "init"


def _traj_set_phase(name):
    global _TRAJ_PHASE
    _TRAJ_PHASE = name


def _traj_record(sim, joints):
    global _TRAJ_T0
    if _TRAJ_T0 is None:
        _TRAJ_T0 = time.time()
    TRAJECTORY.append(
        {
            "phase": _TRAJ_PHASE,
            "t": time.time() - _TRAJ_T0,
            "q": [float(sim.getJointPosition(j)) for j in joints],
        }
    )


# ─────────────────────────── Constants ────────────────────────
RES_X, RES_Y = 512, 512
FOV_DEG = 70
NEAR_CLIP, FAR_CLIP = 0.01, 10.0

BLOCK_HALF = 0.03
GRIPPER_CLEARANCE = 0.015  # wrist stops 1.5 cm above block top
SAFE_Z = 0.35
FLOOR_Z = 0.0
MIN_JOINT_Z = 0.05  # no part of the arm should go below 5 cm

CAM_POS = [0.35, 0.0, 2.2]
CAM_ORI = [math.pi, 0, 0]


# ─────────────────────────── Scene Setup ──────────────────────
def clear_scene(sim):
    print("[1/7] Purging the scene...")
    safe = {
        "DefaultCamera",
        "DefaultLights",
        "LightA",
        "LightB",
        "LightC",
        "LightD",
        "Floor",
    }
    all_objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    to_del = [
        o
        for o in all_objs
        if sim.getObjectAlias(o) not in safe
        and "ViewCamera" not in sim.getObjectAlias(o)
    ]
    if to_del:
        sim.removeObjects(to_del)


def setup_vla_environment(sim):
    print("[2/7] Spawning UR5 + workspace...")

    patterns = [
        "/Applications/[Cc]oppeliaSim*.app/Contents/Resources/models/robots/non-mobile/UR5.ttm",
        os.path.expanduser(
            "~/Applications/[Cc]oppeliaSim*.app/Contents/Resources/models/robots/non-mobile/UR5.ttm"
        ),
    ]
    path = next((f for p in patterns for f in glob.glob(p)), None)
    if not path:
        print("ERROR: UR5.ttm not found!")
        return None

    # Pedestal
    pedestal = sim.createPrimitiveShape(
        sim.primitiveshape_cylinder, [0.15, 0.15, 0.2], 0
    )
    sim.setObjectPosition(pedestal, sim.handle_world, [0, 0, 0.1])
    sim.setShapeColor(
        pedestal, None, sim.colorcomponent_ambient_diffuse, [0.3, 0.3, 0.3]
    )
    sim.setObjectInt32Param(pedestal, sim.shapeintparam_static, 1)

    ur5_base = sim.loadModel(path)
    sim.setObjectPosition(ur5_base, sim.handle_world, [0, 0, 0.2])

    tree = sim.getObjectsInTree(ur5_base, sim.handle_all, 0)
    joints = [o for o in tree if sim.getObjectType(o) == sim.object_joint_type]

    # NOTE: we intentionally do NOT snap to the ready pose here.
    # main() does a smooth_joint_move from the loaded-model default pose to
    # the ready pose after startSimulation, so the startup looks animated
    # instead of teleporting.

    for obj in tree:
        try:
            t = sim.getObjectType(obj)
            if t == sim.object_script_type:
                sim.removeObjects([obj])
            elif t == sim.object_shape_type:
                sim.setObjectInt32Param(obj, sim.shapeintparam_static, 1)
            elif t == sim.object_joint_type:
                sim.setJointMode(obj, sim.jointmode_kinematic)
        except Exception:
            pass
    sim.setObjectInt32Param(ur5_base, sim.modelproperty_not_dynamic, 1)

    # Floor
    floor_h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [2, 2, 0.01], 0)
    sim.setObjectPosition(floor_h, sim.handle_world, [0, 0, -0.005])
    sim.setObjectInt32Param(floor_h, sim.shapeintparam_static, 1)
    sim.setShapeColor(
        floor_h, None, sim.colorcomponent_ambient_diffuse, [0.8, 0.8, 0.8]
    )

    # Grid
    for i in range(-6, 7):
        lx = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [1.2, 0.003, 0.001], 0)
        sim.setObjectPosition(lx, sim.handle_world, [0.35, i * 0.1, 0.001])
        sim.setShapeColor(lx, None, sim.colorcomponent_ambient_diffuse, [0.2, 0.2, 0.2])
        sim.setObjectInt32Param(lx, sim.shapeintparam_static, 1)
        ly = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.003, 1.2, 0.001], 0)
        sim.setObjectPosition(ly, sim.handle_world, [0.35 + i * 0.1, 0, 0.001])
        sim.setShapeColor(ly, None, sim.colorcomponent_ambient_diffuse, [0.2, 0.2, 0.2])
        sim.setObjectInt32Param(ly, sim.shapeintparam_static, 1)

    # Tool blocks
    colors_rgb = {
        "red": [0.9, 0.1, 0.1],
        "green": [0.1, 0.9, 0.1],
        "blue": [0.1, 0.1, 0.9],
    }
    positions = {
        "red": [0.45, 0.15, 0.03],
        "green": [0.45, 0.0, 0.03],
        "blue": [0.45, -0.15, 0.03],
    }
    tools = {}
    for name, pos in positions.items():
        b = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.06] * 3, 0)
        sim.setObjectPosition(b, sim.handle_world, pos)
        sim.setObjectInt32Param(b, sim.shapeintparam_static, 1)
        sim.setShapeColor(b, None, sim.colorcomponent_ambient_diffuse, colors_rgb[name])
        tools[name] = b

    # Tip dummy
    tip_handle = joints[-1]
    for o in tree:
        try:
            if "connection" in sim.getObjectAlias(o).lower():
                tip_handle = o
                break
        except Exception:
            pass

    tip_dummy = sim.createDummy(0.01)
    sim.setObjectParent(tip_dummy, tip_handle, False)
    sim.setObjectPosition(tip_dummy, tip_handle, [0, 0, 0])
    sim.setObjectInt32Param(tip_dummy, sim.objintparam_visibility_layer, 0)

    ik_target = sim.createDummy(0.05)
    # Copy the tip_dummy's ACTUAL world pose (pos + ori) into the IK target
    # so the position-and-orientation IK constraint is perfectly satisfied at
    # startup. This prevents the arm from visibly snapping when the solver
    # first runs to "correct" a mismatch that shouldn't exist.
    sim.setObjectPosition(
        ik_target,
        sim.handle_world,
        sim.getObjectPosition(tip_dummy, sim.handle_world),
    )
    sim.setObjectOrientation(
        ik_target,
        sim.handle_world,
        sim.getObjectOrientation(tip_dummy, sim.handle_world),
    )
    sim.setObjectInt32Param(ik_target, sim.objintparam_visibility_layer, 0)

    # Overhead camera
    sensor = sim.createVisionSensor(
        0,
        [RES_X, RES_Y, 0, 0],
        [NEAR_CLIP, FAR_CLIP, FOV_DEG * math.pi / 180, 0.1, 0, 0, 0, 0, 0, 0, 0],
    )
    sim.setObjectParent(sensor, sim.handle_world, False)
    sim.setObjectPosition(sensor, sim.handle_world, CAM_POS)
    sim.setObjectOrientation(sensor, sim.handle_world, CAM_ORI)
    sim.setObjectInt32Param(sensor, sim.objintparam_visibility_layer, 0)

    return ik_target, tip_dummy, ur5_base, sensor, tools, joints, floor_h


# ─────────────────────────── Vision ───────────────────────────
def run_vision(sim, sensor, filename="robot_view.jpg"):
    print(f"   Capturing: {filename}")
    time.sleep(1.0)
    for _ in range(15):
        img, res = sim.getVisionSensorImg(sensor)
        if img:
            arr = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
            arr = cv2.flip(arr, 0)
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, img_bgr)
            return img_bgr
        time.sleep(0.2)
    return None


def get_object_pixel(img_bgr, color_name):
    """OpenCV colour detection → pixel centroid of the largest blob."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if color_name == "red":
        mask = cv2.inRange(
            hsv, np.array([0, 100, 100]), np.array([10, 255, 255])
        ) + cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    elif color_name == "green":
        mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    elif color_name == "blue":
        # Tightened to exclude the UR5's pastel-cyan joint caps (S≈55).
        # The actual blue block is deeply saturated (S≈227), so an S floor
        # of 150 and a hue range starting at 105 cleanly separates them.
        mask = cv2.inRange(hsv, np.array([105, 150, 70]), np.array([135, 255, 255]))
    else:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50:
        return None
    x, y, w, h = cv2.boundingRect(c)
    px, py = int(x + w / 2), int(y + h / 2)
    cv2.circle(img_bgr, (px, py), 6, (0, 255, 255), 2)
    cv2.putText(
        img_bgr, color_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
    )
    return px, py


# ──────────── Deprojection ──────────
def deproject_pixel(sim, sensor, px, py):
    """
    Convert pixel (px, py) → world XYZ using depth buffer + camera matrix.

    CoppeliaSim vision sensor conventions:
        - Image after cv2.flip(0): px=0 left, py=0 top
        - Image X axis is MIRRORED vs camera +X  (left in image = camera +X)
        - Image Y down = camera -Y

    CRITICAL: all values passed to sim.multiplyVector must be native
    Python float, not np.float32/64 (CBOR can't serialize numpy scalars).
    """
    print(f"\n   [Deproject] pixel ({px}, {py})")

    # 1. Read depth
    d_bytes, d_res = sim.getVisionSensorDepth(sensor)
    depth_raw = np.frombuffer(d_bytes, dtype=np.float32).reshape((d_res[1], d_res[0]))
    depth = cv2.flip(depth_raw, 0)

    px_c = max(0, min(RES_X - 1, px))
    py_c = max(0, min(RES_Y - 1, py))
    d_norm = float(depth[py_c, px_c])
    z_cam = NEAR_CLIP + d_norm * (FAR_CLIP - NEAR_CLIP)
    print(f"   depth_norm={d_norm:.6f}  z_cam={z_cam:.4f} m")

    # 2. Query actual perspective angle from the sensor (don't trust our constant)
    actual_fov = sim.getObjectFloatParam(sensor, sim.visionfloatparam_perspective_angle)
    half_fov = actual_fov / 2.0
    # CoppeliaSim projection convention: f = RES / tan(half_fov), not RES/2
    f_pix = RES_X / math.tan(half_fov)
    cx = (RES_X - 1) / 2.0
    cy = (RES_Y - 1) / 2.0
    print(f"   sensor FOV={math.degrees(actual_fov):.1f}°  f_pix={f_pix:.1f}")

    # CoppeliaSim vision sensor image is horizontally mirrored vs camera frame:
    #   image +px (right) = camera -X, so negate X
    x_cam = float(-(px_c - cx) * z_cam / f_pix)
    y_cam = float(-(py_c - cy) * z_cam / f_pix)
    z_cam_f = float(z_cam)

    print(f"   camera-frame: ({x_cam:.4f}, {y_cam:.4f}, {z_cam_f:.4f})")

    # 3. Camera frame → world frame via 4x3 matrix
    m = sim.getObjectMatrix(sensor, sim.handle_world)
    world_pt = sim.multiplyVector(m, [x_cam, y_cam, z_cam_f])

    print(f"   world: ({world_pt[0]:.4f}, {world_pt[1]:.4f}, {world_pt[2]:.4f})")
    return world_pt


def validate_deprojection(sim, sensor, img_bgr, known_positions):
    """Run deprojection on all 3 known blocks and print errors."""
    print("\n   === Deprojection Validation ===")
    for name, true_pos in known_positions.items():
        pix = get_object_pixel(img_bgr.copy(), name)
        if pix is None:
            print(f"   {name}: NOT DETECTED")
            continue
        proj = deproject_pixel(sim, sensor, pix[0], pix[1])
        err_xy = math.sqrt((proj[0] - true_pos[0]) ** 2 + (proj[1] - true_pos[1]) ** 2)
        print(
            f"   {name}: true=({true_pos[0]:.3f},{true_pos[1]:.3f}) "
            f"proj=({proj[0]:.3f},{proj[1]:.3f}) err={err_xy*100:.1f} cm"
        )
    cv2.imwrite("vision_debug.jpg", img_bgr)


# ─────────────────── IK Helpers ───────────────────────────────
def sync_ik(sim, simIK, ik_env, ik_group, n=5):
    for _ in range(n):
        simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})
        time.sleep(0.01)


def resync_ik_target(sim, simIK, ik_env, ik_group, ik_target, tip_dummy):
    """Re-align the IK target dummy to where the tip actually is (position only, keep orientation)."""
    tip_pos = sim.getObjectPosition(tip_dummy, sim.handle_world)
    sim.setObjectPosition(ik_target, sim.handle_world, tip_pos)
    # Keep pointing down
    sim.setObjectOrientation(ik_target, sim.handle_world, [math.pi, 0, 0])
    sync_ik(sim, simIK, ik_env, ik_group, n=10)


def _lerp(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(3)]


def smooth_joint_move(
    sim, joints, target_config, steps=60, dt=0.02, ik_target=None, tip_dummy=None
):
    """
    Smoothly interpolate the arm from its current joint config to target_config
    without any IK snapping. Uses a smoothstep ease so start and end are gentle.

    If ik_target + tip_dummy are provided, the IK target dummy is kept in sync
    with the tip's actual world pose throughout the motion, so that when IK is
    later enabled there is no position/orientation mismatch to "correct".
    """
    start_config = [float(sim.getJointPosition(j)) for j in joints]
    for i in range(1, steps + 1):
        t = i / steps
        s = t * t * (3.0 - 2.0 * t)  # smoothstep easing
        for j_idx, j in enumerate(joints):
            val = start_config[j_idx] + (target_config[j_idx] - start_config[j_idx]) * s
            sim.setJointPosition(j, val)
        if ik_target is not None and tip_dummy is not None:
            tp = sim.getObjectPosition(tip_dummy, sim.handle_world)
            to = sim.getObjectOrientation(tip_dummy, sim.handle_world)
            sim.setObjectPosition(ik_target, sim.handle_world, tp)
            sim.setObjectOrientation(ik_target, sim.handle_world, to)
        time.sleep(dt)


def _get_arm_lowest_z(sim, joints):
    """Return the lowest Z coordinate of any joint in the arm."""
    min_z = float("inf")
    for j in joints:
        pos = sim.getObjectPosition(j, sim.handle_world)
        if pos[2] < min_z:
            min_z = pos[2]
    return min_z


def seed_elbow_up(sim, joints, target_xy):
    """
    Seed joints in an elbow-UP configuration (upside-down V).
    This biases damped least-squares IK to land in the elbow-up branch
    instead of elbow-down when solving for a goal.

    Upper arm points up-forward, forearm comes down to the tool tip.
    """
    # Base rotation: face the target
    base_angle = math.atan2(target_xy[1], target_xy[0])
    sim.setJointPosition(joints[0], base_angle)
    # Shoulder pitched back/up: upper arm points mostly upward
    sim.setJointPosition(joints[1], -math.pi / 2)
    # Elbow set to 315° (equivalent to -45°) per user request
    sim.setJointPosition(joints[2], math.radians(315))
    # Wrist1 brings the tool back to horizontal-pointing-down
    sim.setJointPosition(joints[3], -math.pi / 2)
    # Wrist2/3: keep tool axis pointing straight down in world
    sim.setJointPosition(joints[4], math.pi / 2)
    sim.setJointPosition(joints[5], 0.0)


def _normalize_angle_close(target, ref):
    """Wrap target by ±2π so it's within ±π of ref (shortest angular path)."""
    diff = target - ref
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return ref + diff


def _normalize_config_close(target, current):
    """Elementwise shortest-angular-path normalization of a joint config."""
    return [_normalize_angle_close(target[i], current[i]) for i in range(len(current))]


def compute_ik_config_silently(
    sim,
    simIK,
    ik_env,
    ik_group,
    joints,
    ik_target,
    target_pos,
    target_ori=None,
    iters=8,
    ur5_base=None,
):
    """
    Compute the joint config that reaches target_pos, WITHOUT leaving the
    simulation in a visibly different state. Saves current joint state,
    seeds elbow-up, solves IK, captures goal config + arm clearance, then
    restores state.

    To avoid the brief flicker that was visible when IK ran in-place, the
    whole UR5 subtree is temporarily removed from the visible rendering
    layer for the duration of the solve. The solve itself is also reduced
    from 80 handleGroup round-trips to ~8, since DLS converges in a handful
    of passes for a target this close to the seed.

    Returns (goal_config_normalized, lowest_joint_z).
    """
    if target_ori is None:
        target_ori = [math.pi, 0, 0]

    # Snapshot everything we will touch
    saved_joints = [sim.getJointPosition(j) for j in joints]
    saved_tgt_pos = list(sim.getObjectPosition(ik_target, sim.handle_world))
    saved_tgt_ori = list(sim.getObjectOrientation(ik_target, sim.handle_world))

    # Hide the whole UR5 subtree on all rendering layers while we scratch-
    # compute joint angles, so any brief intermediate state is invisible.
    hidden_shapes = []
    if ur5_base is not None:
        try:
            subtree = sim.getObjectsInTree(ur5_base, sim.handle_all, 0)
            for o in subtree:
                try:
                    if sim.getObjectType(o) == sim.object_shape_type:
                        layer = sim.getObjectInt32Param(
                            o, sim.objintparam_visibility_layer
                        )
                        sim.setObjectInt32Param(o, sim.objintparam_visibility_layer, 0)
                        hidden_shapes.append((o, layer))
                except Exception:
                    pass
        except Exception:
            pass

    lowest_z = float("inf")
    try:
        # Seed the joints in an elbow-up config facing target
        seed_elbow_up(sim, joints, target_pos)
        sim.setObjectPosition(ik_target, sim.handle_world, target_pos)
        sim.setObjectOrientation(ik_target, sim.handle_world, target_ori)
        # A handful of fast IK iterations — DLS converges in ~5 passes for
        # targets this close to the seed. Keeping the loop short minimizes
        # the window during which the arm is in the "scratch" state.
        for _ in range(iters):
            simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})
        goal = [float(sim.getJointPosition(j)) for j in joints]
        # Measure clearance while the goal config is applied
        lowest_z = _get_arm_lowest_z(sim, joints)
    finally:
        # Restore joint state FIRST so the sim snaps back before any render
        for j_idx, j_handle in enumerate(joints):
            sim.setJointPosition(j_handle, saved_joints[j_idx])
        sim.setObjectPosition(ik_target, sim.handle_world, saved_tgt_pos)
        sim.setObjectOrientation(ik_target, sim.handle_world, saved_tgt_ori)
        simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})
        # Restore visibility layers
        for o, layer in hidden_shapes:
            try:
                sim.setObjectInt32Param(o, sim.objintparam_visibility_layer, layer)
            except Exception:
                pass

    # Normalize each joint angle to the shortest path from the saved state
    normalized = [
        _normalize_angle_close(goal[i], saved_joints[i]) for i in range(len(joints))
    ]
    return normalized, lowest_z


def _ease_in_out(t):
    """Cosine ease-in-out: smooth acceleration and deceleration, t in [0,1]."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def move_joints_smooth(sim, joints, target_config, steps=150, delay=0.012):
    """
    Smoothly drive joints from current to target_config using cosine
    ease-in-out. No IK is invoked during the motion — joint positions
    are set directly, so nothing fights the interpolation. The IK target
    dummy should be re-synced by the caller once the move finishes.
    """
    start = [sim.getJointPosition(j) for j in joints]
    deltas = [target_config[i] - start[i] for i in range(len(joints))]
    for i in range(1, steps + 1):
        u = _ease_in_out(i / steps)
        for j_idx, j_handle in enumerate(joints):
            sim.setJointPosition(j_handle, start[j_idx] + deltas[j_idx] * u)
        _traj_record(sim, joints)
        time.sleep(delay)


def move_ik_to(
    sim, simIK, ik_env, ik_group, ik_target, dest, steps, min_z=None, joints=None
):
    start = list(sim.getObjectPosition(ik_target, sim.handle_world))
    for i in range(1, steps + 1):
        p = _lerp(start, dest, i / steps)
        if min_z is not None:
            p[2] = max(p[2], min_z)
        sim.setObjectPosition(ik_target, sim.handle_world, p)
        sync_ik(sim, simIK, ik_env, ik_group, n=2)
        if joints is not None:
            _traj_record(sim, joints)
        time.sleep(0.015)


# ─────────── OMPL Joint-Space Path Planning ───────────────────
def plan_and_execute_ompl(
    sim,
    simIK,
    simOMPL,
    ik_env,
    ik_group,
    ik_target,
    tip_dummy,
    joints,
    ur5_base,
    start_config,
    goal_pos,
    floor_handle=None,
):
    """
    Plan collision-free joint-space path via OMPL RRTConnect:
    1. Solve IK for goal_pos → goal config
    2. Reset to start_config
    3. OMPL plan + execute
    Returns True on success.
    """
    n_joints = len(joints)

    # Find goal config via IK silently (no visible teleport).
    # Clearance check happens INSIDE the silent solve — no extra snap.
    # Passing ur5_base lets the helper hide the arm on the visibility layer
    # for the few ms the solve takes, so no flicker can leak through.
    print("   [OMPL] Solving IK for goal config (silent)...")
    goal_config, lowest_after_ik = compute_ik_config_silently(
        sim,
        simIK,
        ik_env,
        ik_group,
        joints,
        ik_target,
        goal_pos,
        ur5_base=ur5_base,
    )
    if lowest_after_ik < MIN_JOINT_Z:
        print(
            f"   [OMPL] IK landed with lowest joint z={lowest_after_ik:.3f}. Aborting OMPL."
        )
        return False
    # Joints are already back at start_config (silent helper restored them)

    # Force specific joints to rotate in a chosen direction during this
    # plan. _normalize_angle_close already gave us the shortest angular
    # path, which may go the wrong way around and cause the real arm to
    # self-collide, so we explicitly pick the direction per joint here.
    #
    # - Joint 5 (tool roll): always CCW — fixes the end-effector roll.
    # - Joint 2 (elbow / middle): direction depends on which block we
    #   are going to. Red (y ≈ +0.15) and green (y ≈ 0) need the elbow
    #   to rotate CW; blue (y ≈ -0.15) needs CCW. The split is chosen
    #   at y = -0.05 so green (y=0) lands in the red/green bucket.
    forced_direction = {5: "ccw"}
    if goal_pos[1] < -0.05:
        forced_direction[2] = "ccw"  # blue
    else:
        forced_direction[2] = "cw"  # red, green

    for _idx, _dir in forced_direction.items():
        d = goal_config[_idx] - start_config[_idx]
        # Normalize into (-π, π] first
        while d > math.pi:
            d -= 2.0 * math.pi
        while d <= -math.pi:
            d += 2.0 * math.pi
        if _dir == "ccw" and d < 0.0:
            d += 2.0 * math.pi
        elif _dir == "cw" and d > 0.0:
            d -= 2.0 * math.pi
        goal_config[_idx] = start_config[_idx] + d
        print(
            f"   [OMPL] Forcing joint {_idx} {_dir.upper()}: "
            f"start={start_config[_idx]:.3f} goal={goal_config[_idx]:.3f} "
            f"(delta={d:+.3f} rad)"
        )

    # Create OMPL task
    task = simOMPL.createTask("pick_task")
    simOMPL.setAlgorithm(task, simOMPL.Algorithm.RRTConnect)
    simOMPL.setVerboseLevel(task, 0)

    # Build a per-joint state space centered on the START config.
    # - Any joint listed in forced_direction gets a ONE-SIDED window in
    #   the chosen direction (CCW: [start-0.1, start+2π+0.1],
    #   CW: [start-2π-0.1, start+0.1]). This leaves room for the
    #   direction-adjusted goals above while blocking the opposite path.
    # - All other joints get ±π of slack — enough to reach any reasonable
    #   goal without letting OMPL plan through multi-revolution motions
    #   that make the arm "wind itself up".
    # Bounds are also extended if the goal config sits outside the window.
    ss_handles = []
    created_custom_ss = True
    try:
        for i, j in enumerate(joints):
            center = float(start_config[i])
            if i in forced_direction:
                if forced_direction[i] == "ccw":
                    low = [center - 0.1]
                    high = [center + 2.0 * math.pi + 0.1]
                else:  # cw
                    low = [center - 2.0 * math.pi - 0.1]
                    high = [center + 0.1]
            else:
                low = [max(-2.0 * math.pi, center - math.pi)]
                high = [min(2.0 * math.pi, center + math.pi)]
            # Extend toward goal_config if the goal sits outside the window
            g = float(goal_config[i])
            if g < low[0]:
                low[0] = g - 0.1
            if g > high[0]:
                high[0] = g + 0.1
            ss = simOMPL.createStateSpace(
                f"ur5_j{i}",
                simOMPL.StateSpaceType.joint_position,
                j,
                low,
                high,
                1,
            )
            ss_handles.append(ss)
        simOMPL.setStateSpace(task, ss_handles)
    except Exception as e:
        # Older CoppeliaSim builds may not expose createStateSpace the same
        # way — fall back to the convenience call and rely on the joints'
        # native ranges.
        print(f"   [OMPL] createStateSpace unavailable ({e}); using default.")
        created_custom_ss = False
        simOMPL.setStateSpaceForJoints(task, joints, [1, 1, 1, 1, 1, 1])

    # Collision pairs: (UR5 vs floor) AND (UR5 vs itself) so OMPL rejects any
    # candidate state where the arm intersects its own links. Without the
    # self-collision pair the planner happily routes joints into each other.
    collision_pairs = [ur5_base, ur5_base]  # self-collision
    if floor_handle is not None:
        collision_pairs += [ur5_base, floor_handle]
    simOMPL.setCollisionPairs(task, collision_pairs)

    simOMPL.setStartState(task, [float(v) for v in start_config])
    simOMPL.setGoalState(task, goal_config)
    simOMPL.setup(task)

    print("   [OMPL] Computing path (up to 5s)...")
    solved, path_states = simOMPL.compute(task, 5.0, -1, 80)

    if not solved:
        print("   [OMPL] No solution found.")
        simOMPL.destroyTask(task)
        return False

    n_states = len(path_states) // n_joints
    print(f"   [OMPL] Path: {n_states} waypoints. Executing smoothly...")

    # Unpack waypoints into a list of 6-vectors and unwrap each segment to
    # the shortest angular path so no joint ever takes the long way around.
    # EXCEPTION: any joint listed in forced_direction gets monotonically
    # unwrapped in its assigned direction (CCW or CW) instead, so the
    # OMPL path respects the per-joint direction choice set earlier.
    def _unwrap_joint(idx, val, ref):
        if idx in forced_direction:
            if forced_direction[idx] == "ccw":
                # Only allow positive steps: val must be >= ref
                while val < ref - 1e-6:
                    val += 2.0 * math.pi
                while val - ref > 2.0 * math.pi:
                    val -= 2.0 * math.pi
            else:  # cw
                # Only allow negative steps: val must be <= ref
                while val > ref + 1e-6:
                    val -= 2.0 * math.pi
                while ref - val > 2.0 * math.pi:
                    val += 2.0 * math.pi
            return val
        return _normalize_angle_close(val, ref)

    waypoints = []
    for i in range(n_states):
        state = [float(path_states[i * n_joints + k]) for k in range(n_joints)]
        if waypoints:
            prev = waypoints[-1]
            state = [_unwrap_joint(k, state[k], prev[k]) for k in range(n_joints)]
        waypoints.append(state)

    # Make sure the first waypoint exactly matches the current joint state
    # so we never see a jump from the starting pose into waypoint[0].
    current_q = [float(sim.getJointPosition(j)) for j in joints]
    waypoints[0] = [
        _unwrap_joint(k, waypoints[0][k], current_q[k]) for k in range(n_joints)
    ]

    # Sub-step count per segment: picked so even a big joint-space jump gets
    # broken into small increments (≤ ~3° per sub-step).
    SUB_MIN = 8
    SUB_PER_RAD = 20  # ~3° per sub-step
    DT = 0.012  # wall time per sub-step
    for seg in range(len(waypoints) - 1):
        a = waypoints[seg]
        b = waypoints[seg + 1]
        max_delta = max(abs(b[k] - a[k]) for k in range(n_joints))
        sub = max(SUB_MIN, int(math.ceil(max_delta * SUB_PER_RAD)))
        for s in range(1, sub + 1):
            u = s / sub  # linear within a segment avoids stop/start at each waypoint
            for j_idx, j_handle in enumerate(joints):
                val = a[j_idx] + (b[j_idx] - a[j_idx]) * u
                sim.setJointPosition(j_handle, val)
            # NOTE: no sync_ik here — IK would fight the OMPL-chosen joint
            # values. We drive joints directly, which is exactly what OMPL
            # planned.
            _traj_record(sim, joints)
            time.sleep(DT)

    simOMPL.destroyTask(task)

    # Re-sync IK target after OMPL moved joints directly
    resync_ik_target(sim, simIK, ik_env, ik_group, ik_target, tip_dummy)
    return True


# ─────────── Full Pick Sequence ───────────────────────────────
def pick_block(
    sim,
    simIK,
    ik_env,
    ik_group,
    ik_target,
    tip_dummy,
    joints,
    ur5_base,
    tool_handle,
    world_pos,
    initial_pose,
    start_config,
    simOMPL=None,
    floor_handle=None,
    target_color=None,
):
    """
    Pick the block at world_pos (from deprojection).
    Tries OMPL for the big move, falls back to IK waypoints.
    """
    touch_z = world_pos[2] + GRIPPER_CLEARANCE
    above_block = [world_pos[0], world_pos[1], SAFE_Z]
    touch_pos = [world_pos[0], world_pos[1], touch_z]
    # For the blue block, insert a higher intermediate "high approach"
    # waypoint so the arm rises before translating to the above-block pose,
    # giving more clearance on the way in.
    use_high_approach = target_color == "blue"
    HIGH_APPROACH_Z = 0.60  # higher than SAFE_Z (0.35)

    print(
        f"   Target world: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})"
    )
    print(f"   Touch Z: {touch_z:.3f}")

    used_ompl = False
    if simOMPL is not None:
        try:
            print("\n   --- OMPL: home → above block ---")
            _traj_set_phase("ompl_home_to_above_block")
            used_ompl = plan_and_execute_ompl(
                sim,
                simIK,
                simOMPL,
                ik_env,
                ik_group,
                ik_target,
                tip_dummy,
                joints,
                ur5_base,
                start_config,
                above_block,
                floor_handle=floor_handle,
            )
        except Exception as e:
            print(f"   OMPL error: {e}")
            used_ompl = False

    if not used_ompl:
        print("\n   --- Smooth joint-space approach ---")

        # Optional: high intermediate waypoint for blue (more clearance)
        if use_high_approach:
            high_above = [world_pos[0], world_pos[1], HIGH_APPROACH_Z]
            print(f"   Computing high-approach joint config (Z={HIGH_APPROACH_Z})...")
            high_config, _ = compute_ik_config_silently(
                sim, simIK, ik_env, ik_group, joints, ik_target, high_above
            )
            _traj_set_phase("smooth_home_to_high_approach")
            print("   Smooth joint interpolation → high approach")
            move_joints_smooth(sim, joints, high_config, steps=140, delay=0.012)

        # 1) Compute the above-block joint config silently (no visible teleport)
        print("   Computing above-block joint config...")
        above_config, _ = compute_ik_config_silently(
            sim, simIK, ik_env, ik_group, joints, ik_target, above_block
        )
        print(f"   Goal config (rad): {[round(x,3) for x in above_config]}")

        # 2) Smoothly ease joints from current to goal — no IK during motion,
        #    cosine easing, many small steps
        _traj_set_phase("smooth_home_to_above_block")
        print("   Smooth joint interpolation → above block")
        move_joints_smooth(sim, joints, above_config, steps=180, delay=0.012)

    # Sync the IK target dummy to wherever the tip actually ended up before
    # switching back to task-space motion for the descent.
    resync_ik_target(sim, simIK, ik_env, ik_group, ik_target, tip_dummy)

    _traj_set_phase("ik_phase3_descend_to_block")
    print(f"   Phase 3: Descend (Z → {touch_z:.3f})")
    move_ik_to(sim, simIK, ik_env, ik_group, ik_target, touch_pos, 40, joints=joints)

    # Report actual distance
    final_tip = sim.getObjectPosition(tip_dummy, sim.handle_world)
    block_actual = sim.getObjectPosition(tool_handle, sim.handle_world)
    dx = final_tip[0] - block_actual[0]
    dy = final_tip[1] - block_actual[1]
    dz = final_tip[2] - (block_actual[2] + BLOCK_HALF)
    print(f"   Tip→block distance: dx={dx*100:.1f} dy={dy*100:.1f} dz={dz*100:.1f} cm")

    print("   >> EQUIPPING")
    sim.setObjectParent(tool_handle, tip_dummy, True)
    time.sleep(0.4)

    # ───── Carry the block and rotate the base 90° clockwise ─────
    # 1) Lift straight up in task space so the block clears the table
    _traj_set_phase("lift_with_block")
    print("   Lift with block (straight up)")
    resync_ik_target(sim, simIK, ik_env, ik_group, ik_target, tip_dummy)
    move_ik_to(
        sim,
        simIK,
        ik_env,
        ik_group,
        ik_target,
        [world_pos[0], world_pos[1], SAFE_Z],
        40,
        joints=joints,
    )

    # 2) Rotate base (joint[0]) by -90° — clockwise when viewed from above.
    #    All other joints hold their current angles, so the whole arm
    #    (and the block parented to the tip) swings together.
    _traj_set_phase("rotate_base_90_cw")
    print("   Rotate base 90° clockwise")
    current_config = [sim.getJointPosition(j) for j in joints]
    rotated_config = list(current_config)
    rotated_config[0] = current_config[0] - math.pi / 2
    move_joints_smooth(sim, joints, rotated_config, steps=150, delay=0.012)
    resync_ik_target(sim, simIK, ik_env, ik_group, ik_target, tip_dummy)


# ─────────────────────────── Main ─────────────────────────────
def main(task="I need to tighten a screw.", force_color=None):
    client = RemoteAPIClient()
    sim = client.require("sim")
    simIK = client.require("simIK")

    try:
        simOMPL = client.require("simOMPL")
        print("simOMPL loaded")
    except Exception:
        simOMPL = None
        print("simOMPL not available — IK waypoints only")

    sim.stopSimulation()
    time.sleep(0.5)
    clear_scene(sim)

    result = setup_vla_environment(sim)
    if not result:
        return
    ik_target, tip_dummy, ur5, sensor, tools, joints, floor_h = result

    # Known positions for validation only (NOT used for targeting)
    known_positions = {
        "red": [0.45, 0.15, 0.03],
        "green": [0.45, 0.0, 0.03],
        "blue": [0.45, -0.15, 0.03],
    }

    # IK setup
    ik_env = simIK.createEnvironment()
    ik_group = simIK.createGroup(ik_env)
    simIK.setGroupCalculation(
        ik_env, ik_group, simIK.method_damped_least_squares, 0.1, 99
    )
    simIK.addElementFromScene(
        ik_env,
        ik_group,
        ur5,
        tip_dummy,
        ik_target,
        simIK.constraint_position + simIK.constraint_orientation,
    )

    sim.startSimulation()
    time.sleep(0.3)  # let the sim render the loaded-default pose once

    # BEFORE moving anything, sync the IK target to the current (loaded) tip
    # pose so the position+orientation IK constraint has nothing to correct.
    tp = sim.getObjectPosition(tip_dummy, sim.handle_world)
    to = sim.getObjectOrientation(tip_dummy, sim.handle_world)
    sim.setObjectPosition(ik_target, sim.handle_world, tp)
    sim.setObjectOrientation(ik_target, sim.handle_world, to)

    # Smoothly animate the arm from its loaded-default pose into the elbow-up
    # ready pose (~1.2 s with smoothstep easing). The ik_target dummy is
    # dragged along with the tip on every sub-step, so the IK solver never
    # sees a mismatch to snap at.
    print("\n   Smoothly moving to ready pose...")
    ready_config = [
        0.0,
        -math.pi / 2,
        math.pi / 2,
        -math.pi / 2,
        math.pi / 2,
        0.0,
    ]
    smooth_joint_move(
        sim,
        joints,
        ready_config,
        steps=60,
        dt=0.02,
        ik_target=ik_target,
        tip_dummy=tip_dummy,
    )

    # Final resync, then a couple of IK iterations so the solver state is
    # consistent with the pose the arm is now holding.
    tp = sim.getObjectPosition(tip_dummy, sim.handle_world)
    to = sim.getObjectOrientation(tip_dummy, sim.handle_world)
    sim.setObjectPosition(ik_target, sim.handle_world, tp)
    sim.setObjectOrientation(ik_target, sim.handle_world, to)
    for _ in range(3):
        simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})
        time.sleep(0.02)

    initial_pose = list(sim.getObjectPosition(ik_target, sim.handle_world))
    start_config = [float(sim.getJointPosition(j)) for j in joints]

    # ── Capture ──
    print("\n[3/7] Capturing camera view...")
    img_bgr = run_vision(sim, sensor, "robot_view.jpg")
    if img_bgr is None:
        print("ERROR: no image!")
        return

    # ── Validate deprojection on all blocks ──
    print("\n[4/7] Validating deprojection (should be < 2 cm error)...")
    validate_deprojection(sim, sensor, img_bgr.copy(), known_positions)

    # ── Decide which block ──
    # When --color is passed on the CLI, force_color is set and we skip the
    # VLM entirely for a deterministic pick. Otherwise we ask LLaVA to reason
    # about the task string and parse its reply.
    if force_color is not None:
        print(f"\n[5/7] Color override: {force_color} (skipping LLaVA)")
        vlm_reply = force_color
    else:
        print("\n[5/7] LLaVA reasoning...")
        prompt = (
            "Look at the grid. RED block = Hammer. GREEN block = Screwdriver. "
            "BLUE block = Wrench.\n"
            f"Human command: '{task}'\n"
            "Which tool color should the robot pick up? "
            "Reply with exactly one word: red, green, or blue."
        )
        print(f"   Task: {task}")
        try:
            res = ollama.chat(
                model="llava",
                messages=[
                    {"role": "user", "content": prompt, "images": ["robot_view.jpg"]}
                ],
            )
            vlm_reply = res["message"]["content"].lower()
            print(f"   LLaVA: '{vlm_reply}'")
        except Exception as e:
            print(f"   Ollama error: {e}")
            return

    # Parse LLaVA's reply robustly. The prompt asks for a single word, but
    # LLaVA often explains — e.g. "blue, not red or green" — so a naive
    # substring check biased toward whichever color is tested first would
    # misfire. Strategy:
    #   1) If the reply, stripped of punctuation, starts with a color word,
    #      trust that (handles "blue.", "Blue block", etc.).
    #   2) Otherwise, count whole-word occurrences of each color and pick
    #      the most frequent (ties broken by the last one mentioned, since
    #      LLaVA's conclusion usually comes at the end).
    matches = re.findall(r"\b(red|green|blue)\b", vlm_reply)
    target_color = None
    if matches:
        head = re.match(r"[^a-z]*(red|green|blue)\b", vlm_reply)
        if head:
            target_color = head.group(1)
        else:
            counts = {c: matches.count(c) for c in set(matches)}
            max_count = max(counts.values())
            top = [c for c in matches if counts[c] == max_count]
            target_color = top[-1]
    if not target_color:
        print("   Could not parse colour.")
        return

    # ── OpenCV: find the block in the image ──
    print(f"\n[6/7] OpenCV: locating '{target_color}' block...")
    pix = get_object_pixel(img_bgr, target_color)
    if not pix:
        print("   Block not found in image!")
        return
    print(f"   Detected at pixel ({pix[0]}, {pix[1]})")
    cv2.imwrite("vision_debug.jpg", img_bgr)

    # ── Deproject pixel → world ──
    world_pos = deproject_pixel(sim, sensor, pix[0], pix[1])

    # ── Execute pick using the perceived position ──
    print(
        f"\n[7/7] Picking block at perceived world ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})"
    )
    # Record the starting joint config before any pick motion happens
    _traj_set_phase("start")
    _traj_record(sim, joints)
    pick_block(
        sim,
        simIK,
        ik_env,
        ik_group,
        ik_target,
        tip_dummy,
        joints,
        ur5,
        tools[target_color],
        world_pos,
        initial_pose,
        start_config,
        simOMPL=simOMPL,
        floor_handle=floor_h,
        target_color=target_color,
    )

    # ── Final photo ──
    print("\n   Final verification photo...")
    run_vision(sim, sensor, "final_equipped_view.jpg")

    # ── Save the full joint-angle trajectory ──
    traj_out = {
        "meta": {
            "robot": "UR5",
            "num_joints": len(joints),
            "target_color": target_color,
            "vla_world_target": [float(v) for v in world_pos],
            "initial_pose_xyz": [float(v) for v in initial_pose],
            "start_config": start_config,
            "num_samples": len(TRAJECTORY),
        },
        "samples": TRAJECTORY,
    }
    with open("trajectory.json", "w") as f:
        json.dump(traj_out, f, indent=2)
    print(f"\n   Wrote trajectory.json ({len(TRAJECTORY)} samples)")

    time.sleep(2)
    sim.stopSimulation()
    print("\nPIPELINE COMPLETE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA pick-and-place pipeline.")
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--task",
        type=str,
        default="I need to tighten a screw.",
        help="Natural-language command passed to the VLM "
        "(e.g. 'I need to drive a nail.').",
    )
    task_group.add_argument(
        "--color",
        choices=["red", "green", "blue"],
        help="Shortcut that picks a canned task for the given block color: "
        "red=drive a nail, green=tighten a screw, blue=loosen a bolt.",
    )
    args = parser.parse_args()

    if args.color:
        # --color is a deterministic override: bypass LLaVA and pick that
        # block directly. The task string is still set to the matching
        # canned command purely for logging / trajectory metadata.
        task = {
            "red": "I need to drive a nail.",
            "green": "I need to tighten a screw.",
            "blue": "I need to loosen a bolt.",
        }[args.color]
        main(task=task, force_color=args.color)
    else:
        main(task=args.task)
