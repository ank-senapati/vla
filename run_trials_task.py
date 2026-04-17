"""
run_trials_task.py — Full-task trial sweep for vla_task.py.

Runs N trials of each tool-specific routine with the correct color
forced (VLM bypassed — VLM accuracy is measured separately in
run_trials_vlm.py). Records per-trial METRICS snapshots and aggregates
across trials.

Metrics captured per trial:
  - task_success (bool)
  - pose_errors at each align phase (pos_mm, ori_deg)
  - phase_timings_sec (dict)
  - screw_seating (for screw trials only)
  - ik_convergence entries (wall-time, iters, converged)

Aggregates:
  - success rate per routine
  - pose error mean ± std per phase
  - timing mean ± std per phase
  - IK convergence rate + wall-time stats
  - screw seating travel mean ± std

Output: trial_results_task.json + stdout summary.

Usage:
  python3 run_trials_task.py --n 30                    # full sweep
  python3 run_trials_task.py --n 2                     # dry run
  python3 run_trials_task.py --n 30 --tools red,green  # subset
"""
import argparse
import copy
import json
import statistics
import time
import traceback

import vla_task as vt


ROUTINE_LABEL = {
    "red":   "screwdriver",
    "green": "grabber",
    "blue":  "riveter",
}


def _reset_metrics():
    """Reset the module-level METRICS dict in vla_task to fresh state
    before each trial, so we get a clean snapshot per trial."""
    vt.METRICS.clear()
    vt.METRICS.update({
        "vlm_tool_selections": [],
        "vlm_calls": [],
        "position_errors_mm": [],
        "orientation_errors_deg": [],
        "pose_errors": [],
        "screw_seating": [],
        "ik_convergence": [],
        "phase_timings_sec": {},
        "task_success": False,
        "failure_phase": None,
        "failure_reason": None,
    })
    vt._phase_start.clear()
    vt.TRAJECTORY.clear()


def run_one_trial(color):
    """Run a single forced-color trial. Returns a METRICS snapshot dict
    plus a `_error` field if the trial threw an exception."""
    _reset_metrics()
    snap = {"color": color, "routine": ROUTINE_LABEL[color]}
    t0 = time.time()
    try:
        vt.main(task=None, force_color=color,
                no_reload=False, record_side_video=False)
        snap["wall_sec"] = round(time.time() - t0, 2)
        snap["_error"] = None
    except Exception as exc:
        snap["wall_sec"] = round(time.time() - t0, 2)
        snap["_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    # Deep-copy METRICS so subsequent trials' resets don't mutate past snaps
    snap["metrics"] = copy.deepcopy(vt.METRICS)
    return snap


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None)
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (statistics.mean(xs), statistics.stdev(xs))


def _fmt_ms(m, s):
    if m is None:
        return "n/a"
    return f"{m:.2f} ± {s:.2f}"


def summarize_per_color(trials, color):
    subset = [t for t in trials if t["color"] == color]
    if not subset:
        return None

    n = len(subset)
    successes = [t for t in subset if t["metrics"].get("task_success") is True
                   and t["_error"] is None]
    n_success = len(successes)

    # Aggregate by phase label
    phase_pos = {}  # phase -> list of pos_mm
    phase_ori = {}
    phase_time = {}
    ik_iters = []
    ik_wall = []
    ik_converged = []
    screw_travels = []
    screw_tip_tracking = []
    wall_secs = []
    exceptions = []

    for t in subset:
        wall_secs.append(t["wall_sec"])
        if t["_error"]:
            exceptions.append(t["_error"])
        m = t["metrics"]
        for e in m.get("pose_errors", []):
            phase_pos.setdefault(e["phase"], []).append(e["pos_error_mm"])
            phase_ori.setdefault(e["phase"], []).append(e["ori_error_deg"])
        for ph, sec in m.get("phase_timings_sec", {}).items():
            phase_time.setdefault(ph, []).append(sec)
        for ik in m.get("ik_convergence", []):
            ik_iters.append(ik["iters"])
            ik_wall.append(ik["wall_sec"])
            ik_converged.append(bool(ik["converged"]))
        for s in m.get("screw_seating", []):
            screw_travels.append(s["travel_z_mm"])
            screw_tip_tracking.append(s["tip_tracking_error_mm"])

    pose_summary = {}
    for ph, vals in sorted(phase_pos.items()):
        pm, ps = _mean_std(vals)
        om, os_ = _mean_std(phase_ori.get(ph, []))
        pose_summary[ph] = {
            "n": len(vals),
            "pos_mm_mean": pm, "pos_mm_std": ps,
            "ori_deg_mean": om, "ori_deg_std": os_,
        }

    timing_summary = {}
    for ph, vals in sorted(phase_time.items()):
        m, s = _mean_std(vals)
        timing_summary[ph] = {"n": len(vals), "mean_sec": m, "std_sec": s}

    wall_m, wall_s = _mean_std(wall_secs)
    ik_iters_m, ik_iters_s = _mean_std(ik_iters)
    ik_wall_m, ik_wall_s = _mean_std(ik_wall)
    screw_travel_m, screw_travel_s = _mean_std(screw_travels)
    screw_track_m, screw_track_s = _mean_std(screw_tip_tracking)

    return {
        "color": color,
        "routine": ROUTINE_LABEL[color],
        "n_trials": n,
        "n_success": n_success,
        "success_rate": round(n_success / n, 4),
        "wall_sec_mean": wall_m, "wall_sec_std": wall_s,
        "exceptions": exceptions,
        "pose_error_per_phase": pose_summary,
        "phase_timings": timing_summary,
        "ik_iters_mean": ik_iters_m, "ik_iters_std": ik_iters_s,
        "ik_wall_sec_mean": ik_wall_m, "ik_wall_sec_std": ik_wall_s,
        "ik_convergence_rate": (sum(ik_converged) / len(ik_converged)
                                 if ik_converged else None),
        "screw_travel_mm_mean": screw_travel_m,
        "screw_travel_mm_std": screw_travel_s,
        "screw_tip_tracking_mm_mean": screw_track_m,
        "screw_tip_tracking_mm_std": screw_track_s,
    }


def summarize(trials, out_path):
    per_color = {}
    for c in ("red", "green", "blue"):
        s = summarize_per_color(trials, c)
        if s is not None:
            per_color[c] = s

    result = {
        "n_trials_total": len(trials),
        "per_routine": per_color,
        "raw_trials": trials,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  TASK SWEEP SUMMARY  (N_total = {len(trials)})")
    print("=" * 70)
    for c, s in per_color.items():
        print(f"\n  [{c}] routine = {s['routine']}  (N = {s['n_trials']})")
        print(f"    success rate        : {s['success_rate']:.1%}  "
              f"({s['n_success']}/{s['n_trials']})")
        print(f"    wall time           : {_fmt_ms(s['wall_sec_mean'], s['wall_sec_std'])} s")
        print(f"    IK iters            : "
              f"{_fmt_ms(s['ik_iters_mean'], s['ik_iters_std'])}")
        print(f"    IK wall time        : "
              f"{_fmt_ms(s['ik_wall_sec_mean'], s['ik_wall_sec_std'])} s")
        if s["ik_convergence_rate"] is not None:
            print(f"    IK convergence rate : {s['ik_convergence_rate']:.1%}")
        if s["screw_travel_mm_mean"] is not None:
            print(f"    screw travel Z mm   : "
                  f"{_fmt_ms(s['screw_travel_mm_mean'], s['screw_travel_mm_std'])}")
            print(f"    screw tip-tracking  : "
                  f"{_fmt_ms(s['screw_tip_tracking_mm_mean'], s['screw_tip_tracking_mm_std'])} mm")
        print(f"    pose errors per phase:")
        for ph, d in s["pose_error_per_phase"].items():
            print(f"      {ph:40s}  pos={_fmt_ms(d['pos_mm_mean'], d['pos_mm_std'])} mm   "
                  f"ori={_fmt_ms(d['ori_deg_mean'], d['ori_deg_std'])}°   (N={d['n']})")
        if s["exceptions"]:
            print(f"    EXCEPTIONS ({len(s['exceptions'])}):")
            for e in s["exceptions"][:5]:
                print(f"      - {e}")
    print("\n" + "=" * 70)
    print(f"  results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-task trial sweep")
    parser.add_argument("--n", type=int, default=30,
                        help="trials per tool (default 30)")
    parser.add_argument("--tools", type=str, default="red,green,blue",
                        help="comma-separated colors to run (default: all)")
    parser.add_argument("--out", type=str, default="trial_results_task.json")
    args = parser.parse_args()

    colors = [c.strip() for c in args.tools.split(",") if c.strip()]
    for c in colors:
        if c not in ("red", "green", "blue"):
            raise SystemExit(f"bad color: {c}")

    total = args.n * len(colors)
    print(f"\nStarting task sweep: N={args.n} × {len(colors)} tools = "
          f"{total} trials")

    trials = []
    t_start = time.time()
    for c in colors:
        for k in range(args.n):
            trial_idx = len(trials) + 1
            print(f"\n{'#' * 70}")
            print(f"# Trial {trial_idx}/{total}  color={c}  "
                  f"routine={ROUTINE_LABEL[c]}  (run {k + 1}/{args.n})")
            print(f"{'#' * 70}")
            snap = run_one_trial(c)
            trials.append(snap)
            elapsed = time.time() - t_start
            eta = elapsed / trial_idx * (total - trial_idx)
            print(f"  trial done in {snap['wall_sec']:.1f}s   "
                  f"overall elapsed {elapsed / 60:.1f} min   "
                  f"ETA {eta / 60:.1f} min")

    summarize(trials, args.out)
    print(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")
