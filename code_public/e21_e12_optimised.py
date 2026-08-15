"""E21: repeat the structural-anchor sweep of E12 with the optimised estimator.

The anchor rows in E20 were computed against a pool that did not reproduce
E12's: they omitted the family kernels, so the sampled baseline came out at
1.2339 instead of E12's 0.7196 and the sweep degenerated.  Those rows are void.
Here the pool, the seeds and the kernels are taken from E12 itself, so the
sampled column reproduces E12 exactly and the optimised column is directly
comparable to it.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e12_structural_anchor_realdata import (
    L, FAMS, SEEDS, N_DIRS, L4, load, radius, max_step,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e21_e12_optimised.json")

N_STARTS = 10
MAXITER = 700
FRACTIONS = [0.0, 0.005, 0.20]          # baseline, first minimum, largest rate


def to_matrix(v: np.ndarray) -> np.ndarray:
    D = np.zeros((L, L))
    D[: L - 1, :] = v.reshape(L - 1, L)
    D[L - 1, :] = -D[: L - 1, :].sum(axis=0)
    n = np.linalg.norm(D)
    return D / n if n > 1e-12 else D


def displacement(D, kernels, protos) -> float:
    t = max_step(kernels, D, protos)
    A = np.eye(L) + t * D
    return float(np.abs(protos @ A.T - protos).sum(axis=1).max())


def optimised(kernels, protos) -> float:
    scored = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        for _ in range(N_DIRS):
            D = rng.normal(size=(L, L))
            D -= D.mean(axis=0, keepdims=True)
            n = np.linalg.norm(D)
            if n < 1e-12:
                continue
            D /= n
            scored.append((displacement(D, kernels, protos), D))
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0]
    for _, D in scored[:N_STARTS]:
        res = minimize(
            lambda v: -displacement(to_matrix(v), kernels, protos),
            D[: L - 1, :].ravel(),
            method="Nelder-Mead",
            options={"maxiter": MAXITER, "xatol": 1e-7, "fatol": 1e-9},
        )
        best = max(best, -res.fun)
    return best


def main() -> None:
    judges, fam_of, P, g = load()
    src = json.load(open(L4))
    kern = [np.array(src["kernels_from_gold"][f]) for f in FAMS]

    vertex_dist = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vertex_dist)
    near_vertex_idx = order[: int(0.10 * len(order))]
    interior_idx = order[int(0.50 * len(order)):]

    rng0 = np.random.default_rng(7)
    n_base = 400
    base = rng0.choice(interior_idx, size=min(n_base, len(interior_idx)), replace=False)

    out = {
        "env": {
            "script": "e21_e12_optimised.py",
            "seeds": SEEDS,
            "n_dirs_per_seed": N_DIRS,
            "optimiser": f"Nelder-Mead, {N_STARTS} starts, maxiter {MAXITER}",
            "pool": "identical construction, seeds and kernels to E12",
            "note": "optimised values are lower bounds (local search)",
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "rows": [],
    }

    # The sweep draws anchors in sequence, exactly as E12 does.
    for frac in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
        n_anchor = int(round(frac * len(base)))
        if n_anchor > 0:
            anchors = rng0.choice(near_vertex_idx,
                                  size=min(n_anchor, len(near_vertex_idx)),
                                  replace=False)
            protos = np.vstack([P[base], P[anchors]])
        else:
            protos = P[base]
        if frac not in FRACTIONS:
            continue                     # keep the draw sequence, skip the measurement
        r_s = radius(kern, protos, SEEDS, N_DIRS)
        r_o = optimised(kern, protos)
        out["rows"].append({
            "anchor_fraction": frac,
            "n_anchor_items": int(n_anchor),
            "n_prototypes": int(protos.shape[0]),
            "sampled": r_s,
            "optimised": r_o,
        })
        print(f"rate={frac:<6} anchors={n_anchor:<3} sampled={r_s:.4f} "
              f"optimised={r_o:.4f}", flush=True)

    rows = {r["anchor_fraction"]: r for r in out["rows"]}
    out["summary"] = {
        "sampled_reduction_factor_2_anchors": rows[0.0]["sampled"] / rows[0.005]["sampled"],
        "optimised_reduction_factor_2_anchors": rows[0.0]["optimised"] / rows[0.005]["optimised"],
        "sampled_matches_e12_baseline": abs(rows[0.0]["sampled"] - 0.7196474426601712) < 1e-6,
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nsummary:", json.dumps(out["summary"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
