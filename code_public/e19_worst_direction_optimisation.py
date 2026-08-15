"""E19: is the sampled diameter tight, or only a loose lower bound?

Every diameter in this paper is a maximum over randomly drawn gauge directions,
so it under-estimates by construction.  E16 shows the estimate stops rising as
the budget grows, which is convergence evidence but not proof that no direction
does better.  Here the direction is optimised rather than sampled.

For each family count m the gauge set is

    G_m = { A : 1'A = 1', A p^(k) in simplex, C_f A^{-1} column-stochastic }

and the diameter is the largest l1 displacement max_k || A p^(k) - p^(k) ||_1
over A in G_m.  Writing A = I + t D with 1'D = 0, the feasible step t along a
direction D is found by bisection exactly as in the sampling estimator, so the
objective f(D) is the displacement attained at that step.  We maximise f by
multi-start Nelder-Mead, seeded from the best directions the sampler found, and
compare the result with the sampled value.

If optimisation cannot beat sampling, the reported diameters are tight and the
Monte Carlo objection is closed.  If it can, the reported diameters are too
small and the paper must say so; in particular the m = 1 and m = 3 values are
re-examined here, since their equality is the paper's central numerical claim.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_criterion_and_unified_radius import unified_radius, FAMS
from e6_sensitivity_and_vossler import load, item_profiles, kernels_from_gold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e19_worst_direction_optimisation.json")

SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 4000
L = 4
N_STARTS = 24          # local searches per m, seeded from the best samples
MAXITER = 1200


def to_matrix(v: np.ndarray) -> np.ndarray:
    """12 free numbers -> L x L matrix whose columns sum to zero, unit norm."""
    D = np.zeros((L, L))
    D[: L - 1, :] = v.reshape(L - 1, L)
    D[L - 1, :] = -D[: L - 1, :].sum(axis=0)
    n = np.linalg.norm(D)
    return D / n if n > 1e-12 else D


def displacement(D: np.ndarray, kernels, protos) -> float:
    """l1 displacement at the largest feasible step along D."""
    eye = np.eye(L)
    lo, hi = 0.0, 2.0
    for _ in range(45):
        t = 0.5 * (lo + hi)
        A = eye + t * D
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            hi = t
            continue
        ok = all((Ck @ Ainv >= -1e-12).all() for Ck in kernels)
        if ok:
            ok = bool((protos @ A.T >= -1e-12).all())
        lo, hi = (t, hi) if ok else (lo, t)
    A = eye + lo * D
    return float(np.abs(protos @ A.T - protos).sum(axis=1).max())


def best_random_starts(kernels, protos, n_keep: int):
    """Sample directions and keep the best few as optimisation seeds."""
    scored = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        for _ in range(PER // len(SEEDS)):
            D = rng.normal(size=(L, L))
            D -= D.mean(axis=0, keepdims=True)
            n = np.linalg.norm(D)
            if n < 1e-12:
                continue
            D /= n
            scored.append((displacement(D, kernels, protos), D))
    scored.sort(key=lambda x: -x[0])
    return scored[:n_keep]


def optimise(kernels, protos, starts):
    best_val, best_D = 0.0, None
    for val, D in starts:
        v0 = D[: L - 1, :].ravel()
        res = minimize(
            lambda v: -displacement(to_matrix(v), kernels, protos),
            v0,
            method="Nelder-Mead",
            options={"maxiter": MAXITER, "xatol": 1e-8, "fatol": 1e-10},
        )
        got = -res.fun
        if got > best_val:
            best_val, best_D = got, to_matrix(res.x)
    return best_val, best_D


def main() -> None:
    judges, runs, gold, fam_of, keys = load()
    P = item_profiles(judges, runs, keys)
    g = np.array([gold[k] for k in keys])
    C = kernels_from_gold(judges, runs, gold, fam_of, keys)
    kern = [C[f] for f in FAMS]
    protos = np.vstack([P[g == k].mean(axis=0) for k in range(L)])

    out = {
        "env": {
            "script": "e19_worst_direction_optimisation.py",
            "seeds": SEEDS,
            "sampled_directions": PER,
            "optimiser": f"Nelder-Mead, {N_STARTS} starts, maxiter {MAXITER}",
            "prototypes": "gold-conditioned mean (identical to E2c)",
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "rows": [],
    }

    for m in (0, 1, 2, 3):
        kernels = kern[:m]
        sampled, n_dirs = unified_radius(kernels, protos, SEEDS, PER)
        starts = best_random_starts(kernels, protos, N_STARTS)
        opt, _ = optimise(kernels, protos, starts)
        gain = opt - sampled
        row = {
            "m": m,
            "sampled_diameter": sampled,
            "sampled_directions": n_dirs,
            "optimised_diameter": opt,
            "absolute_gain": gain,
            "relative_gain": gain / sampled if sampled > 0 else 0.0,
        }
        out["rows"].append(row)
        print(
            f"m={m}  sampled={sampled:.6f}  optimised={opt:.6f}  "
            f"gain={gain:+.6f} ({100 * row['relative_gain']:+.3f}%)",
            flush=True,
        )

    r = {x["m"]: x for x in out["rows"]}
    out["summary"] = {
        "max_relative_gain": max(x["relative_gain"] for x in out["rows"]),
        "optimised_m1": r[1]["optimised_diameter"],
        "optimised_m3": r[3]["optimised_diameter"],
        "optimised_m1_minus_m3": r[1]["optimised_diameter"] - r[3]["optimised_diameter"],
        "flatness_survives_optimisation": bool(
            abs(r[1]["optimised_diameter"] - r[3]["optimised_diameter"]) < 1e-6
        ),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nsummary:", json.dumps(out["summary"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
