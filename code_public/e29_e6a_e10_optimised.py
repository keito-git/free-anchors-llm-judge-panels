"""E29: put E6a and E10 on the same estimator as the rest of the paper.

After E19 the headline numbers were recomputed with the optimised estimator,
but the prototype-construction spread (E6a) and the fitting-K sweep (E10) were
left on the sampler.  That is an inconsistency inside one table: E19 showed
sampling both understates the diameter and exaggerates the effect of anything
that adds constraints, so a spread measured by sampling is not comparable with
a reduction measured by optimisation.

This script recomputes both with the optimised estimator, reusing the exact
prototype constructions of e9c and e10 so that only the estimator changes.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_criterion_and_unified_radius import FAMS
from e6_sensitivity_and_vossler import (
    load, item_profiles, kernels_from_gold, em_profiles,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e29_e6a_e10_optimised.json")

L = 4
SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 4000
N_STARTS = 16
MAXITER = 1000


def to_matrix(v):
    D = np.zeros((L, L))
    D[: L - 1, :] = v.reshape(L - 1, L)
    D[L - 1, :] = -D[: L - 1, :].sum(axis=0)
    n = np.linalg.norm(D)
    return D / n if n > 1e-12 else D


def displacement(D, kernels, protos) -> float:
    eye = np.eye(L)
    lo, hi = 0.0, 2.0
    for _ in range(45):
        t = 0.5 * (lo + hi)
        A = eye + t * D
        ok = True
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            ok = False
        if ok:
            ok = all((Ck @ Ainv >= -1e-12).all() for Ck in kernels)
        if ok:
            ok = bool((protos @ A.T >= -1e-12).all())
        lo, hi = (t, hi) if ok else (lo, t)
    A = eye + lo * D
    return float(np.abs(protos @ A.T - protos).sum(axis=1).max())


def optimised(kernels, protos) -> float:
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
    best = scored[0][0]
    for _, D in scored[:N_STARTS]:
        res = minimize(lambda v: -displacement(to_matrix(v), kernels, protos),
                       D[: L - 1, :].ravel(), method="Nelder-Mead",
                       options={"maxiter": MAXITER, "xatol": 1e-8, "fatol": 1e-10})
        best = max(best, -res.fun)
    return best


def main() -> None:
    judges, runs, gold, fam_of, keys = load()
    P = item_profiles(judges, runs, keys)
    g = np.array([gold[k] for k in keys])
    C = kernels_from_gold(judges, runs, gold, fam_of, keys)
    kern = [C[f] for f in FAMS]
    rng = np.random.default_rng(99)

    out = {
        "env": {
            "script": "e29_e6a_e10_optimised.py",
            "seeds": SEEDS, "sampled_directions": PER,
            "optimiser": f"Nelder-Mead, {N_STARTS} starts, maxiter {MAXITER}",
            "gpu_used": False, "api_calls": 0, "cost_usd": 0.0,
        },
        "E6a_prototype_constructions": {},
        "E10_fitting_K": {},
    }

    # ---- E6a: three prototype constructions, identical to e9c ---------------
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

    vals = {}
    for name, protos in variants.items():
        d1 = optimised(kern[:1], protos)
        d3 = optimised(kern, protos)
        vals[name] = d3
        out["E6a_prototype_constructions"][name] = {
            "eta": float(protos.min()),
            "optimised_m1": d1, "optimised_m3": d3,
            "reduction_m1_to_m3_pct": 100 * (1 - d3 / d1),
        }
        print(f"[E6a {name:20s}] m1={d1:.6f} m3={d3:.6f} "
              f"drop={100*(1-d3/d1):.2f}%", flush=True)
    v = list(vals.values())
    out["E6a_prototype_constructions"]["spread_pct"] = 100 * (max(v) - min(v)) / max(v)

    # ---- E10: EM prototypes at several K, kernels fixed ---------------------
    for K in (4, 5, 6, 8):
        protos = em_profiles(P, K, np.random.default_rng(99))
        d1 = optimised(kern[:1], protos)
        d3 = optimised(kern, protos)
        out["E10_fitting_K"][f"K={K}"] = {
            "optimised_m1": d1, "optimised_m3": d3,
            "reduction_m1_to_m3_pct": 100 * (1 - d3 / d1),
        }
        print(f"[E10 K={K}] m1={d1:.6f} m3={d3:.6f} "
              f"drop={100*(1-d3/d1):.2f}%", flush=True)

    ks = out["E10_fitting_K"]
    out["summary"] = {
        "E6a_spread_pct": out["E6a_prototype_constructions"]["spread_pct"],
        "E6a_max_reduction_pct": max(
            r["reduction_m1_to_m3_pct"] for k, r in
            out["E6a_prototype_constructions"].items() if isinstance(r, dict)),
        "E10_max_reduction_pct": max(r["reduction_m1_to_m3_pct"]
                                     for r in ks.values()),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nsummary:", json.dumps(out["summary"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
