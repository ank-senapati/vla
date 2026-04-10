"""
One-shot scene inspector. Loads the .ttt, walks the object tree, and prints
every object's alias, type, and world position so we can see what's in the
scene and decide how to wire the pick pipeline to it.

Usage:
    1. Start CoppeliaSim (empty scene)
    2. python3 inspect_scene.py
"""
import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

SCENE_PATH = os.path.abspath("3-AD6ECD9793D23BD9 1.ttt")


def type_name(sim, t):
    # Best-effort type name lookup
    for attr in dir(sim):
        if attr.startswith("object_") and attr.endswith("_type"):
            try:
                if getattr(sim, attr) == t:
                    return attr
            except Exception:
                pass
    return f"type_{t}"


def main():
    client = RemoteAPIClient()
    sim = client.require("sim")

    sim.stopSimulation()
    import time; time.sleep(0.3)

    print(f"Loading scene: {SCENE_PATH}")
    sim.loadScene(SCENE_PATH)
    time.sleep(0.5)

    all_objs = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    print(f"\nTotal objects: {len(all_objs)}\n")

    rows = []
    for h in all_objs:
        try:
            alias = sim.getObjectAlias(h, 1)  # full path alias
        except Exception:
            alias = "<no alias>"
        try:
            tname = type_name(sim, sim.getObjectType(h))
        except Exception:
            tname = "?"
        try:
            pos = sim.getObjectPosition(h, sim.handle_world)
            pos_str = f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})"
        except Exception:
            pos_str = "(?,?,?)"
        rows.append((alias, tname, pos_str, h))

    rows.sort(key=lambda r: r[0])
    width = max(len(r[0]) for r in rows)
    for alias, tname, pos, h in rows:
        print(f"  {alias:<{width}}  {tname:<24}  {pos}  h={h}")

    # Also print just the joints and shapes/dummies as a quick filter
    print("\n--- JOINTS ONLY ---")
    for alias, tname, pos, h in rows:
        if "joint" in tname:
            print(f"  {alias}")

    print("\n--- DUMMIES ONLY ---")
    for alias, tname, pos, h in rows:
        if "dummy" in tname:
            print(f"  {alias}  {pos}")

    print("\n--- VISION SENSORS ---")
    for alias, tname, pos, h in rows:
        if "visionsensor" in tname.lower() or "proximity" in tname.lower():
            print(f"  {alias}  {pos}  {tname}")


if __name__ == "__main__":
    main()
