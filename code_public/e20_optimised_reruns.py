"""E20: re-run the main real-data measurements with the optimised estimator.

E19 showed that the sampling estimator understates the identification set
diameter by up to 40%, and that the equality of the m = 1 and m = 3 diameters
it reported as exact is really a 0.30% reduction at the second family.  Every
other real-data number in the paper was produced by the same sampler, so this
script repeats the measurements that carry a claim, using the same
worst-direction optimisation as E19, and reports sampled and optimised values
side by side.

Covered here:
  E17'  four family partitions              (does the qualification survive?)
  E12'  structural anchor injection sweep   (VOID, see below)
  E18'  contract A vs contract B at m = 0   (does the comparison survive?)

The E12' rows in this file are void: the pool was rebuilt here without the
family kernels, so the sampled baseline came out at 1.2339 instead of E12's
0.7196 and the sweep degenerated.  e21 repeats that measurement correctly by
reusing E12's own construction; use its numbers, not these.

Optimiser settings are held fixed across all three and are looser than E19's,
so that the whole set finishes in reasonable time on one CPU; they are recorded
in the output.  Any value here is still a lower bound, since the optimiser is a
local search.

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
from e6_sensitivity_and_vossler import (
    L, VALID, SUB_DIR, GOLD, MAP, read_qrel, load, item_profiles,
    kernels_from_gold,
)
from e17_partition_robustness import assign, ORDER, DESC, kernels_for
from e18_contract_diameter_contrast import ntcir_L4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e20_optimised_reruns.json")

SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 4000
N_STARTS = 10
MAXITER = 700


def to_matrix(v: np.ndarray) -> np.ndarray:
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


def optimised_diameter(kernels, protos) -> float:
    """Sample, keep the best starts, then refine each by Nelder-Mead."""
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
        res = minimize(
            lambda v: -displacement(to_matrix(v), kernels, protos),
            D[: L - 1, :].ravel(),
            method="Nelder-Mead",
            options={"maxiter": MAXITER, "xatol": 1e-7, "fatol": 1e-9},
        )
        best = max(best, -res.fun)
    return best


def main() -> None:
    out = {
        "env": {
            "script": "e20_optimised_reruns.py",
            "seeds": SEEDS,
            "sampled_directions": PER,
            "optimiser": f"Nelder-Mead, {N_STARTS} starts, maxiter {MAXITER}",
            "note": "optimised values are still lower bounds (local search)",
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        }
    }

    judges = sorted(f[:-4] for f in os.listdir(SUB_DIR) if f.endswith(".txt"))
    runs = {j: read_qrel(os.path.join(SUB_DIR, j + ".txt")) for j in judges}
    gold = read_qrel(GOLD)
    meta = json.load(open(MAP))
    keys = sorted(set(gold) & set.intersection(*[set(runs[j]) for j in judges]))
    keys = [k for k in keys if gold[k] in VALID]
    P = item_profiles(judges, runs, keys)
    g = np.array([gold[k] for k in keys])
    protos = np.vstack([P[g == k].mean(axis=0) for k in range(L)])
    tags = [assign(meta.get(j) or {}) for j in judges]

    # ---- E17': family partitions -------------------------------------------
    print("== E17' partitions ==", flush=True)
    out["E17_partitions"] = {}
    for key, fams in ORDER.items():
        labels = [t[key] for t in tags]
        present = [f for f in fams if sum(1 for x in labels if x == f) > 0]
        C = kernels_for(judges, runs, gold, labels, keys, present)
        kern = [C[f] for f in present]
        row = {"description": DESC[key], "families": present,
               "sampled": {}, "optimised": {}}
        for m in range(len(present) + 1):
            row["sampled"][f"m={m}"] = unified_radius(kern[:m], protos, SEEDS, PER)[0]
            row["optimised"][f"m={m}"] = optimised_diameter(kern[:m], protos)
        o = [row["optimised"][f"m={m}"] for m in range(len(present) + 1)]
        row["optimised_step_reduction_pct"] = [
            100 * (1 - o[i] / o[i - 1]) for i in range(1, len(o))
        ]
        row["binds_after_first_optimised"] = bool(
            any(r > 0.5 for r in row["optimised_step_reduction_pct"][1:])
        )
        out["E17_partitions"][key] = row
        print(f"[{key}] opt " + "  ".join(f"m{m}={o[m]:.4f}" for m in range(len(o)))
              + "  step%=" + ", ".join(f"{r:.2f}" for r in row["optimised_step_reduction_pct"]),
              flush=True)

    # ---- E12': anchor injection sweep --------------------------------------
    print("== E12' anchors ==", flush=True)
    e12 = json.load(open(os.path.join(ROOT, "results",
                                      "e12_structural_anchor_realdata.json")))
    dist = np.abs(P - np.eye(L)[P.argmax(axis=1)]).sum(axis=1)
    order = np.argsort(dist)
    near = P[order[: int(0.10 * len(order))]]
    amb = P[order[int(0.50 * len(order)):]]
    rng = np.random.default_rng(7)
    base = amb[rng.choice(amb.shape[0], size=400, replace=False)]
    out["E12_anchor_sweep"] = {"sampled_source": "results/e12_structural_anchor_realdata.json",
                               "rows": []}
    for frac, n_anchor in ((0.0, 0), (0.005, 2), (0.20, 80)):
        pool = base if n_anchor == 0 else np.vstack(
            [base, near[rng.choice(near.shape[0], size=n_anchor, replace=False)]])
        r_s = unified_radius([], pool, SEEDS, PER)[0]
        r_o = optimised_diameter([], pool)
        out["E12_anchor_sweep"]["rows"].append(
            {"injection_rate": frac, "n_anchors": n_anchor,
             "n_prototypes": int(pool.shape[0]),
             "sampled": r_s, "optimised": r_o})
        print(f"  rate={frac:<6} anchors={n_anchor:<3} sampled={r_s:.4f} "
              f"optimised={r_o:.4f}", flush=True)
    rows = out["E12_anchor_sweep"]["rows"]
    out["E12_anchor_sweep"]["optimised_reduction_factor_2_anchors"] = (
        rows[0]["optimised"] / rows[1]["optimised"])

    # ---- E18': contract A vs B at m = 0 ------------------------------------
    print("== E18' contracts ==", flush=True)
    out["E18_contracts"] = {"LLMJudge": {"sampled": unified_radius([], protos, SEEDS, PER)[0],
                                         "optimised": optimised_diameter([], protos)}}
    print(f"  LLMJudge optimised={out['E18_contracts']['LLMJudge']['optimised']:.4f}",
          flush=True)
    for lang in ("Japanese", "Chinese"):
        M = ntcir_L4(lang)
        dom = M.argmax(axis=1)
        rows_b = [M[dom == k].mean(axis=0) for k in range(L) if (dom == k).any()]
        pb = np.vstack([v / v.sum() for v in rows_b])
        out["E18_contracts"][lang] = {
            "sampled": unified_radius([], pb, SEEDS, PER)[0],
            "optimised": optimised_diameter([], pb)}
        print(f"  {lang} optimised={out['E18_contracts'][lang]['optimised']:.4f}",
              flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
