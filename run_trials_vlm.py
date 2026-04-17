"""
run_trials_vlm.py — VLM accuracy + latency + memory sweep.

Loads the scene ONCE, sets up the workspace (arm idle in home pose),
and fires query_vlm() N times for each (tool, phrasing) pair. Records
per-call accuracy, latency, and peak RSS. Produces:

  - trial_results_vlm.json  — raw per-call data
  - A summary block printed to stdout with mean/std/confusion matrix.

Usage:
  python3 run_trials_vlm.py              # default: N=30 × 3 tools × 3 phrasings
  python3 run_trials_vlm.py --n 5        # smoke test
  python3 run_trials_vlm.py --n 30 --out trial_results_vlm.json

Preconditions:
  - CoppeliaSim running
  - Ollama running with `llava` pulled
"""
import argparse
import json
import statistics
import time

# Import from vla_task so we reuse the exact query_vlm, setup_scene,
# METRICS schema that the paper results should reflect. We also use
# the already-instrumented ollama timing + RSS capture there.
import vla_task as vt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# 3 phrasings × 3 tools = 9 unique prompts; repeated N times each.
PHRASINGS = {
    "red": [  # screwdriver
        "I need to tighten a screw.",
        "Drive in the fastener here.",
        "I need to screw this panel down.",
    ],
    "green": [  # grabber
        "Pick up the grate and place it on the base.",
        "I need to lift and move this metal part.",
        "Grab that piece for me.",
    ],
    "blue": [  # riveter
        "I need to rivet these two parts together.",
        "Permanently join these panels with a rivet.",
        "Fasten these parts with a rivet.",
    ],
}


def run_sweep(n_per_prompt):
    client = RemoteAPIClient()
    sim = client.require("sim")
    simIK = client.require("simIK")
    st = vt.setup_scene(sim, simIK, no_reload=False)

    # Start sim so the vision sensor has depth etc. initialized
    sim.startSimulation()
    time.sleep(0.4)

    # Let IK settle so the arm is in a stable pose
    for _ in range(vt.IK_SETTLE_ITERS):
        simIK.handleGroup(st.ik_env, st.ik_group, {"syncWorlds": True})
        time.sleep(0.02)

    # Clear METRICS so our per-call entries are fresh
    vt.METRICS["vlm_tool_selections"].clear()
    vt.METRICS["vlm_calls"].clear()

    # Warmup call — LLaVA shows ~4.5s on first inference (model load) and
    # ~0.2s thereafter once resident. We do ONE throwaway call so every
    # timed trial measures warm-state latency. Cold-start latency from
    # this call is recorded separately.
    print("\n=== WARMUP CALL (ignored in statistics) ===")
    warmup_t0 = time.time()
    vt.query_vlm(st, "warmup prompt, ignore", expected_color=None)
    cold_start_sec = time.time() - warmup_t0
    warm_rss = vt.METRICS["vlm_tool_selections"][-1].get("peak_rss_mb", 0.0)
    print(f"Cold-start latency: {cold_start_sec:.2f}s   "
          f"post-load ollama RSS: {warm_rss:.0f} MB")
    vt.METRICS["vlm_tool_selections"].clear()
    vt.METRICS["vlm_calls"].clear()

    calls = []  # [{expected, prompt, got, correct, latency_sec, peak_rss_mb}]
    total = sum(len(p) for p in PHRASINGS.values()) * n_per_prompt
    done = 0
    t0 = time.time()
    for expected, prompts in PHRASINGS.items():
        for prompt in prompts:
            for trial in range(n_per_prompt):
                print(f"\n--- trial {done + 1}/{total}  expected={expected}  "
                      f"prompt={prompt!r}  (run {trial + 1}/{n_per_prompt}) ---")
                got = vt.query_vlm(st, prompt, expected_color=expected)
                last = vt.METRICS["vlm_tool_selections"][-1]
                calls.append({
                    "expected": expected,
                    "prompt": prompt,
                    "got": got,
                    "correct": got == expected,
                    "raw_reply": last.get("raw_reply", "")[:200],
                    "latency_sec": last.get("latency_sec"),
                    "peak_rss_mb": last.get("peak_rss_mb"),
                })
                done += 1

    wall_total = time.time() - t0

    try:
        sim.stopSimulation()
    except Exception:
        pass

    return calls, wall_total, cold_start_sec, warm_rss


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (0.0, 0.0)
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (statistics.mean(xs), statistics.stdev(xs))


def summarize(calls, out_path, cold_start_sec=None, post_load_rss_mb=None):
    total = len(calls)
    correct = sum(1 for c in calls if c["correct"])
    overall_acc = correct / total if total else 0.0

    per_tool = {}
    for expected in ("red", "green", "blue"):
        subset = [c for c in calls if c["expected"] == expected]
        if subset:
            n = len(subset)
            ok = sum(1 for c in subset if c["correct"])
            per_tool[expected] = {
                "n": n,
                "correct": ok,
                "accuracy": round(ok / n, 4),
            }

    per_prompt = {}
    for c in calls:
        key = f"{c['expected']}::{c['prompt']}"
        d = per_prompt.setdefault(key, {"expected": c["expected"],
                                          "prompt": c["prompt"],
                                          "n": 0, "correct": 0})
        d["n"] += 1
        if c["correct"]:
            d["correct"] += 1
    for d in per_prompt.values():
        d["accuracy"] = round(d["correct"] / d["n"], 4) if d["n"] else 0.0

    # Confusion: rows = expected, cols = got (including None)
    labels = ["red", "green", "blue", None]
    confusion = {str(e): {str(g): 0 for g in labels} for e in labels[:3]}
    for c in calls:
        e = c["expected"]
        g = c["got"]
        confusion[str(e)][str(g)] += 1

    lat_mean, lat_std = _mean_std([c["latency_sec"] for c in calls])
    rss_mean, rss_std = _mean_std([c["peak_rss_mb"] for c in calls])

    summary = {
        "n_total": total,
        "n_correct": correct,
        "overall_accuracy": round(overall_acc, 4),
        "per_tool_accuracy": per_tool,
        "per_prompt_accuracy": list(per_prompt.values()),
        "confusion_matrix": confusion,
        "latency_sec": {
            "mean": round(lat_mean, 3),
            "std": round(lat_std, 3),
            "min": round(min((c["latency_sec"] for c in calls), default=0.0), 3),
            "max": round(max((c["latency_sec"] for c in calls), default=0.0), 3),
        },
        "peak_rss_mb": {
            "mean": round(rss_mean, 1),
            "std": round(rss_std, 1),
            "min": round(min((c["peak_rss_mb"] for c in calls), default=0.0), 1),
            "max": round(max((c["peak_rss_mb"] for c in calls), default=0.0), 1),
        },
        "cold_start_latency_sec": (round(cold_start_sec, 3)
                                    if cold_start_sec is not None else None),
        "post_load_ollama_rss_mb": (round(post_load_rss_mb, 1)
                                     if post_load_rss_mb is not None else None),
        "phrasings": PHRASINGS,
    }

    result = {"summary": summary, "calls": calls}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  VLM SWEEP SUMMARY  (N_total = {total})")
    print("=" * 60)
    print(f"  overall accuracy     : {overall_acc:.1%}  "
          f"({correct}/{total})")
    print(f"  per-tool accuracy    :")
    for tool, d in per_tool.items():
        print(f"      {tool:5s}: {d['accuracy']:.1%}  "
              f"({d['correct']}/{d['n']})")
    print(f"  per-prompt accuracy  :")
    for d in summary["per_prompt_accuracy"]:
        print(f"      [{d['expected']:5s}] {d['prompt'][:50]:50s}  "
              f"{d['accuracy']:.1%}  ({d['correct']}/{d['n']})")
    print(f"  latency mean ± std   : "
          f"{lat_mean:.2f} ± {lat_std:.2f} s  "
          f"[min {summary['latency_sec']['min']}, "
          f"max {summary['latency_sec']['max']}]")
    print(f"  ollama RSS mean ± std: "
          f"{rss_mean:.0f} ± {rss_std:.0f} MB  "
          f"[max {summary['peak_rss_mb']['max']}]")
    if cold_start_sec is not None:
        print(f"  cold-start latency   : {cold_start_sec:.2f} s  (one-time)")
    if post_load_rss_mb is not None:
        print(f"  post-load ollama RSS : {post_load_rss_mb:.0f} MB  "
              f"(resident LLaVA footprint)")
    print(f"  confusion matrix     : (row = expected, col = got)")
    header = "                "
    for g in ["red", "green", "blue", "None"]:
        header += f"{g:>8s}"
    print(header)
    for e in ["red", "green", "blue"]:
        row = f"      {e:5s}      "
        for g in ["red", "green", "blue", "None"]:
            row += f"{confusion[e][g]:>8d}"
        print(row)
    print(f"  results saved        : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM accuracy + latency sweep")
    parser.add_argument("--n", type=int, default=30,
                        help="trials per (tool, phrasing) pair (default 30)")
    parser.add_argument("--out", type=str, default="trial_results_vlm.json")
    args = parser.parse_args()

    print(f"\nStarting VLM sweep: N={args.n} per prompt × "
          f"{sum(len(p) for p in PHRASINGS.values())} prompts = "
          f"{args.n * sum(len(p) for p in PHRASINGS.values())} total calls")
    t_start = time.time()
    calls, wall_total, cold_start, post_load_rss = run_sweep(args.n)
    print(f"\nSweep wall time: {wall_total:.1f} s "
          f"({wall_total / max(len(calls), 1):.2f} s/call avg)")
    summarize(calls, args.out, cold_start_sec=cold_start,
              post_load_rss_mb=post_load_rss)
    print(f"Total elapsed: {time.time() - t_start:.1f} s")
