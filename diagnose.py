"""
Diagnostic: test deprojection accuracy and IK reachability.
Skips VLM, uses known block positions as ground truth.
"""
import time, math, cv2, numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from vla import (
    clear_scene, setup_vla_environment, sync_ik, resync_ik_target,
    move_ik_to, run_vision, RES_X, RES_Y, SAFE_Z, GRIPPER_CLEARANCE,
    BLOCK_HALF, deproject_pixel, get_object_pixel,
)


def capture(sim, sensor, filename):
    time.sleep(0.3)
    for _ in range(15):
        img, res = sim.getVisionSensorImg(sensor)
        if img:
            arr = np.frombuffer(img, dtype=np.uint8).reshape((res[1], res[0], 3))
            arr = cv2.flip(arr, 0)
            cv2.imwrite(filename, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            print(f"   Saved {filename}")
            return
        time.sleep(0.1)


def main():
    client = RemoteAPIClient()
    sim = client.require("sim")
    simIK = client.require("simIK")

    sim.stopSimulation()
    time.sleep(0.5)
    clear_scene(sim)

    result = setup_vla_environment(sim)
    if not result:
        return
    ik_target, tip_dummy, ur5, sensor_top, tools, joints, floor_h = result

    # IK setup
    ik_env = simIK.createEnvironment()
    ik_group = simIK.createGroup(ik_env)
    simIK.setGroupCalculation(ik_env, ik_group, simIK.method_damped_least_squares, 0.1, 99)
    simIK.addElementFromScene(
        ik_env, ik_group, ur5, tip_dummy, ik_target,
        simIK.constraint_position + simIK.constraint_orientation
    )

    sim.startSimulation()
    time.sleep(0.2)

    for _ in range(50):
        simIK.handleGroup(ik_env, ik_group, {"syncWorlds": True})
        time.sleep(0.02)

    # Query actual sensor FOV
    actual_fov = sim.getObjectFloatParam(sensor_top, sim.visionfloatparam_perspective_angle)
    print(f"\n=== SENSOR ACTUAL FOV: {math.degrees(actual_fov):.2f}° (we set {70}°)")
    print(f"=== f_pix (corrected): {RES_X/math.tan(actual_fov/2.0):.1f}")
    print(f"=== f_pix (old wrong):  {(RES_X/2.0)/math.tan(actual_fov/2.0):.1f}")

    # Known positions
    known = {
        "red": [0.45, 0.15, 0.03],
        "green": [0.45, 0.0, 0.03],
        "blue": [0.45, -0.15, 0.03],
    }

    # Capture and test deprojection for ALL blocks
    print("\n=== DEPROJECTION TEST (all 3 blocks) ===")
    img_bgr = run_vision(sim, sensor_top, "diag_top.jpg")
    if img_bgr is None:
        print("No image!")
        return

    for color, true_pos in known.items():
        actual = sim.getObjectPosition(tools[color], sim.handle_world)
        pix = get_object_pixel(img_bgr.copy(), color)
        if pix is None:
            print(f"\n   {color}: NOT DETECTED")
            continue
        print(f"\n   --- {color} block ---")
        print(f"   Pixel: ({pix[0]}, {pix[1]})")
        world = deproject_pixel(sim, sensor_top, pix[0], pix[1])
        err = math.sqrt((world[0]-actual[0])**2 + (world[1]-actual[1])**2)
        print(f"   TRUE:       ({actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.4f})")
        print(f"   DEPROJECTED: ({world[0]:.4f}, {world[1]:.4f}, {world[2]:.4f})")
        print(f"   XY ERROR: {err*100:.1f} cm")

    cv2.imwrite("diag_detections.jpg", img_bgr)

    time.sleep(1)
    sim.stopSimulation()
    print("\nDIAGNOSTIC COMPLETE.")


if __name__ == "__main__":
    main()
