"""E10: does the E2c radius conclusion depend on the latent class count?

E5b shows the model fits at K = 8 but leaves a small excess at K = 4, which is
the resolution E2c uses for its gold-conditioned prototypes.  This script keeps
the gold-estimated L x L kernels fixed and replaces the prototypes by the EM
latent class profiles at several K, then remeasures the radius.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5b_latent_class_gof import load as load5b, em_fit, FAMS
from e9_criterion_and_unified_radius import unified_radius

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L4 = os.path.join(ROOT, "results", "e2c_realdata_L4.json")
OUT = os.path.join(ROOT, "results", "e10_radius_at_fitting_K.json")
SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 4000
K_GRID = [4, 5, 6, 8]

def main() -> None:
    kern = [np.array(json.load(open(L4))["kernels_from_gold"][f]) for f in FAMS]
    judges, fam_of, Y = load5b()
    rng = np.random.default_rng(20260813)
    out = {"env": {"script": "e10_radius_at_fitting_K.py",
                   "kernels": "gold-estimated L x L, identical to E2c",
                   "estimator": "pooled max over 6 seeds x 4000 dirs",
                   "seeds": SEEDS, "gpu_used": False, "api_calls": 0, "cost_usd": 0.0}}
    for K in K_GRID:
        best = None
        for _ in range(20):
            pi, C, post, ll = em_fit(Y, fam_of, K, rng)
            if best is None or ll > best[3]:
                best = (pi, C, post, ll)
        pi, C, post, ll = best
        P = np.stack([np.mean([C[f][:, k] for f in FAMS], axis=0) for k in range(K)])
        P = P / P.sum(axis=1, keepdims=True)
        row = {"K": K, "loglik": float(ll), "eta": float(P.min())}
        for m in (0, 1, 2, 3):
            r, _ = unified_radius(kern[:m], P, SEEDS, PER)
            row[f"m={m}"] = r
        seq = [row["m=1"], row["m=2"], row["m=3"]]
        row["flat_from_m1"] = bool(max(seq) - min(seq) < 1e-6)
        row["kernels_bind"] = bool(row["m=0"] - row["m=1"] > 1e-3)
        out[f"K={K}"] = row
        print(f"K={K}: m0={row['m=0']:.4f} m1={row['m=1']:.4f} m2={row['m=2']:.4f} "
              f"m3={row['m=3']:.4f} flat={row['flat_from_m1']}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)

if __name__ == "__main__":
    main()
