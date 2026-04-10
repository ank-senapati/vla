"""Quick check: are the two screws children of /Grate_Assembly?"""
import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require("sim")

sim.stopSimulation()
import time; time.sleep(0.3)
sim.loadScene(os.path.abspath("3-AD6ECD9793D23BD9 1.ttt"))
time.sleep(0.4)

def alias(h):
    try:
        return sim.getObjectAlias(h, 1)
    except Exception:
        return f"<{h}>"

grate = sim.getObject("/Grate_Assembly")
print(f"Grate handle = {grate}  alias = {alias(grate)}")

for i in [0, 1]:
    s = sim.getObject(f"/91400A242_Mil__Spec__Phillips_Rounded_Head_Screws[{i}]")
    parent = sim.getObjectParent(s)
    print(f"Screw[{i}] handle = {s}  parent = {parent} ({alias(parent)})")
    print(f"  Is child of grate: {parent == grate}")

# Also get panel bounding box so we know where the grate should land
panel = sim.getObject("/Bottom_Panel")
try:
    bb = sim.getObjectFloatArrayParam(panel, sim.objfloatparam_objbbox_max_z)
    print(f"Panel bbox_max_z (local) = {bb}")
except Exception as e:
    print(f"bbox query failed: {e}")

try:
    bb_max = sim.getObjectFloatParam(panel, sim.objfloatparam_objbbox_max_z)
    bb_min = sim.getObjectFloatParam(panel, sim.objfloatparam_objbbox_min_z)
    print(f"Panel local Z range: [{bb_min}, {bb_max}]")
except Exception as e:
    print(f"bbox float-param query failed: {e}")

# Panel world pose
panel_pos = sim.getObjectPosition(panel, sim.handle_world)
print(f"Panel world pos = {panel_pos}")

# Grate world pose
grate_pos = sim.getObjectPosition(grate, sim.handle_world)
print(f"Grate world pos = {grate_pos}")

# UR5 tip dummy initial pose
tip = sim.getObject("/UR5/dummy")
tip_pos = sim.getObjectPosition(tip, sim.handle_world)
tip_ori = sim.getObjectOrientation(tip, sim.handle_world)
print(f"Tip dummy pos = {tip_pos}")
print(f"Tip dummy ori = {tip_ori}")
