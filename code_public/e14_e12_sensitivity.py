"""E14: threshold and seed sensitivity of the E12 structural-anchor result.

The question is whether the headline result, that two anchors halve the
identification set and the effect then saturates, survives the two arbitrary
choices in E12: the pool split thresholds and the random anchor draw.

Thresholds swept: the near-vertex pool is the top q_near fraction by closeness
to a vertex, the ambiguous pool is the bottom q_amb fraction.
Anchor draws: five independent seeds per configuration.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations
import json, os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e12_structural_anchor_realdata import load, radius, L4, FAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e14_e12_sensitivity_matched.json")
RAD_SEEDS = [42, 123, 456, 789, 1024, 20260804]  # matched to E12
N_DIRS = 3000  # matched to E12
DRAW_SEEDS = [7, 101, 202, 303, 404]
THRESHOLDS = [(0.05, 0.50), (0.10, 0.50), (0.20, 0.50), (0.10, 0.30), (0.10, 0.70)]

def main() -> None:
    judges, fam_of, P, g = load()
    kern = [np.array(json.load(open(L4))["kernels_from_gold"][f]) for f in FAMS]
    vdist = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vdist)
    n = len(order)
    res = {"env": {"script": "e14_e12_sensitivity.py", "n_dirs_per_seed": N_DIRS,
                   "radius_seeds": RAD_SEEDS, "draw_seeds": DRAW_SEEDS,
                   "gpu_used": False, "api_calls": 0, "cost_usd": 0.0},
           "configs": []}
    for q_near, q_amb in THRESHOLDS:
        near = order[: int(q_near * n)]
        amb = order[int((1 - q_amb) * n):]
        rows = []
        for ds in DRAW_SEEDS:
            rng = np.random.default_rng(ds)
            base = rng.choice(amb, size=min(400, len(amb)), replace=False)
            r0 = radius(kern, P[base], RAD_SEEDS, N_DIRS)
            anch = rng.choice(near, size=2, replace=False)
            r2 = radius(kern, np.vstack([P[base], P[anch]]), RAD_SEEDS, N_DIRS)
            anch8 = rng.choice(near, size=8, replace=False)
            r8 = radius(kern, np.vstack([P[base], P[anch8]]), RAD_SEEDS, N_DIRS)
            rows.append({"draw_seed": ds, "radius_0": r0, "radius_2": r2,
                         "radius_8": r8, "ratio_0_over_2": r0 / r2 if r2 > 0 else None,
                         "saturated_2_to_8": bool(abs(r2 - r8) < 1e-6)})
        rr = np.array([x["ratio_0_over_2"] for x in rows])
        res["configs"].append({
            "q_near": q_near, "q_amb": q_amb,
            "near_pool_size": int(len(near)), "amb_pool_size": int(len(amb)),
            "near_max_l1_vertex_distance": float(vdist[near].max()),
            "per_draw": rows,
            "reduction_ratio_mean": float(rr.mean()),
            "reduction_ratio_sd": float(rr.std(ddof=1)),
            "reduction_ratio_min": float(rr.min()),
            "reduction_ratio_max": float(rr.max()),
            "saturation_holds_all_draws": bool(all(x["saturated_2_to_8"] for x in rows)),
        })
        c = res["configs"][-1]
        print(f"q_near={q_near} q_amb={q_amb}: ratio {c['reduction_ratio_mean']:.3f} "
              f"± {c['reduction_ratio_sd']:.3f} (min {c['reduction_ratio_min']:.3f}, "
              f"max {c['reduction_ratio_max']:.3f}) saturate={c['saturation_holds_all_draws']}", flush=True)
    allr = [r for c in res["configs"] for r in [x["ratio_0_over_2"] for x in c["per_draw"]]]
    res["summary"] = {
        "n_configurations": len(THRESHOLDS), "n_draws_per_config": len(DRAW_SEEDS),
        "reduction_ratio_overall_mean": float(np.mean(allr)),
        "reduction_ratio_overall_min": float(np.min(allr)),
        "reduction_ratio_overall_max": float(np.max(allr)),
        "always_reduces": bool(np.min(allr) > 1.0),
        "saturation_holds_everywhere": bool(all(c["saturation_holds_all_draws"] for c in res["configs"])),
    }
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + json.dumps(res["summary"], indent=1))

if __name__ == "__main__":
    main()
