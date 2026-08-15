"""E16: does the diameter estimate converge in the number of random directions?

The identification set diameter is estimated as a maximum over randomly drawn
gauge directions, so every reported value is a Monte Carlo lower bound.  The
claim that the diameter is unchanged from m = 1 to m = 3 is therefore open to
the objection that a larger direction budget might still reveal a reduction.

This script sweeps the budget over roughly two orders of magnitude and records
the estimate at m = 0, 1, 2 and 3, so the plateau (or its absence) can be read
off directly.  Seeds are nested: each larger budget contains every direction
drawn at the smaller ones, so the sequence is non-decreasing by construction
and the question is only where it stops rising.

Setup is identical to E2c: gold-conditioned mean prototypes, family kernels
estimated from the LLMJudge gold labels, L = 4.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_criterion_and_unified_radius import unified_radius, FAMS
from e6_sensitivity_and_vossler import load, item_profiles, kernels_from_gold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e16_direction_budget_convergence.json")

SEEDS = [42, 123, 456, 789, 1024, 20260804]
L = 4

# (number of seeds, directions per seed) -> total budget
SCHEDULE = [
    (1, 250),
    (1, 500),
    (1, 1000),
    (2, 1000),
    (3, 1000),
    (4, 1500),
    (6, 2000),
    (6, 4000),
    (6, 8000),
    (6, 16000),
]


def main() -> None:
    judges, runs, gold, fam_of, keys = load()
    P = item_profiles(judges, runs, keys)
    g = np.array([gold[k] for k in keys])
    C = kernels_from_gold(judges, runs, gold, fam_of, keys)
    kern = [C[f] for f in FAMS]
    protos = np.vstack([P[g == k].mean(axis=0) for k in range(L)])

    out = {
        "env": {
            "script": "e16_direction_budget_convergence.py",
            "seeds": SEEDS,
            "prototypes": "gold-conditioned mean (identical to E2c)",
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "eta": float(protos.min()),
        "floor_2eta": 2 * float(protos.min()),
        "rows": [],
    }

    for n_seeds, per in SCHEDULE:
        seeds = SEEDS[:n_seeds]
        row = {"n_seeds": n_seeds, "per_seed": per, "total_directions": n_seeds * per}
        for m in (0, 1, 2, 3):
            r, n = unified_radius(kern[:m], protos, seeds, per)
            row[f"m={m}"] = r
            row["measured_directions"] = n
        row["gap_m1_m3"] = row["m=1"] - row["m=3"]
        out["rows"].append(row)
        print(
            f"budget={row['total_directions']:>6}  "
            f"m0={row['m=0']:.4f}  m1={row['m=1']:.4f}  "
            f"m2={row['m=2']:.4f}  m3={row['m=3']:.4f}  "
            f"gap(m1-m3)={row['gap_m1_m3']:.2e}",
            flush=True,
        )

    largest = out["rows"][-1]
    ref = next(r for r in out["rows"] if r["total_directions"] == 24000)
    out["summary"] = {
        "largest_budget": largest["total_directions"],
        "m=1_at_largest": largest["m=1"],
        "m=3_at_largest": largest["m=3"],
        "m1_minus_m3_at_largest": largest["gap_m1_m3"],
        "relative_change_m3_from_24k_to_largest": (
            (largest["m=3"] - ref["m=3"]) / ref["m=3"]
        ),
        "flat_m1_to_m3_at_every_budget": all(
            abs(r["gap_m1_m3"]) < 1e-9 for r in out["rows"]
        ),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nsummary:", json.dumps(out["summary"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
