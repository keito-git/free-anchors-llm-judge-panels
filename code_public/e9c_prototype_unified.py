"""E9c: prototype sensitivity recomputed with the unified radius estimator.

E6a measured the three prototype constructions with a different direction
budget from the main table, which left two numbers in the paper for the same
quantity.  This script repeats E6a using the single estimator adopted in E9b:
the maximum over the pooled directions from six seeds, 4,000 directions each.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_criterion_and_unified_radius import unified_radius, FAMS
from e6_sensitivity_and_vossler import load, item_profiles, kernels_from_gold, em_profiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e9c_prototype_unified.json")
SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 4000
L = 4

def main() -> None:
    judges, runs, gold, fam_of, keys = load()
    P = item_profiles(judges, runs, keys)
    g = np.array([gold[k] for k in keys])
    C = kernels_from_gold(judges, runs, gold, fam_of, keys)
    kern = [C[f] for f in FAMS]
    rng = np.random.default_rng(99)

    variants = {"gold_mean": np.vstack([P[g == k].mean(axis=0) for k in range(L)])}
    tr = []
    for k in range(L):
        b = P[g == k]
        lo, hi = np.percentile(b, [10, 90], axis=0)
        m = np.all((b >= lo) & (b <= hi), axis=1)
        sel = b[m] if m.sum() > 20 else b
        v = sel.mean(axis=0)
        tr.append(v / v.sum())
    variants["gold_trimmed_mean"] = np.vstack(tr)
    variants["em_latent_K4"] = em_profiles(P, L, rng)

    out = {"env": {"script": "e9c_prototype_unified.py", "seeds": SEEDS,
                   "n_dirs_per_seed": PER, "estimator": "pooled max over 6 seeds x 4000 dirs",
                   "gpu_used": False, "api_calls": 0, "cost_usd": 0.0}}
    for name, protos in variants.items():
        row = {}
        for m in (0, 1, 2, 3):
            r, n = unified_radius(kern[:m], protos, SEEDS, PER)
            row[f"m={m}"] = r
            row["n_directions"] = n
        row["eta"] = float(protos.min())
        row["floor_2eta"] = 2 * float(protos.min())
        seq = [row["m=1"], row["m=2"], row["m=3"]]
        row["flat_from_m1"] = bool(max(seq) - min(seq) < 1e-6)
        row["kernels_bind"] = bool(row["m=0"] - row["m=1"] > 1e-3)
        out[name] = row
        print(f"{name:20s} eta={row['eta']:.4f} m0={row['m=0']:.4f} m1={row['m=1']:.4f} "
              f"m2={row['m=2']:.4f} m3={row['m=3']:.4f} flat={row['flat_from_m1']}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)

if __name__ == "__main__":
    main()
