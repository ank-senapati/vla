"""
Replay a joint-angle trajectory recorded by vla.py in a fresh CoppeliaSim scene.

Usage:
    python replay_trajectory.py                 # uses trajectory.json
    python replay_trajectory.py my_traj.json    # uses custom file

Assumes:
    - A UR5 (or any 6-DOF arm) is already loaded in the running CoppeliaSim scene.
    - The arm's revolute joints are discoverable under a root object whose
      alias contains "UR5" (case-insensitive). If not, pass a root alias as
      the 2nd argument:
          python replay_trajectory.py trajectory.json MyRobot
"""
import sys
import json
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def find_arm_joints(sim, root_hint="UR5"):
    """Find the first subtree whose alias matches root_hint and return its
    revolute joints, ordered by object handle (base → tip).
    """
    all_objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    root = None
    for o in all_objs:
        try:
            alias = sim.getObjectAlias(o)
        except Exception:
            continue
        if root_hint.lower() in alias.lower():
            root = o
            break
    if root is None:
        raise RuntimeError(
            f"No object with alias containing '{root_hint}' found. "
            f"Load a UR5 in the scene first."
        )
    tree = sim.getObjectsInTree(root, sim.handle_all, 0)
    joints = [o for o in tree if sim.getObjectType(o) == sim.object_joint_type]
    return joints


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "trajectory.json"
    root_hint = sys.argv[2] if len(sys.argv) > 2 else "UR5"

    with open(path) as f:
        data = json.load(f)

    meta = data.get("meta", {})
    samples = data["samples"]
    n_joints_expected = meta.get("num_joints", len(samples[0]["q"]))

    print(f"Loaded {len(samples)} samples from {path}")
    print(f"Meta: {meta}")

    client = RemoteAPIClient()
    sim = client.require("sim")

    joints = find_arm_joints(sim, root_hint)
    if len(joints) < n_joints_expected:
        raise RuntimeError(
            f"Found {len(joints)} joints under '{root_hint}' but trajectory "
            f"has {n_joints_expected}."
        )
    joints = joints[:n_joints_expected]
    print(f"Found {len(joints)} joints on the arm. Replaying...")

    # Put arm at the first recorded pose, then play back in real time
    sim.startSimulation()
    time.sleep(0.2)

    t_prev = samples[0]["t"]
    for j_idx, j in enumerate(joints):
        sim.setJointPosition(j, float(samples[0]["q"][j_idx]))

    last_phase = samples[0].get("phase", "")
    print(f"  [{last_phase}]")
    for s in samples[1:]:
        phase = s.get("phase", "")
        if phase != last_phase:
            print(f"  [{phase}]")
            last_phase = phase
        dt = s["t"] - t_prev
        t_prev = s["t"]
        if dt > 0:
            time.sleep(min(dt, 0.2))  # cap pause so one bad sample can't hang replay
        for j_idx, j in enumerate(joints):
            sim.setJointPosition(j, float(s["q"][j_idx]))

    print("Replay complete.")
    time.sleep(1.0)
    sim.stopSimulation()


if __name__ == "__main__":
    main()
