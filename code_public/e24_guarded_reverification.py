"""E24: re-verify the optimised diameters with invertibility enforced.

E22 exposed a defect in how the optimised diameters were obtained.  The gauge
group requires A to be invertible, but the feasibility test only rejected
matrices that numpy refused to invert, so nearly singular A were admitted.  At
m = 0 nothing else restrains them, and the optimum found there is attained by
the rank-one matrix A = e_j 1', which collapses every profile onto one vertex.
Its displacement is exactly 2(1 - eta), which is what both the search and the
linear program returned.

Two consequences.  First, the m = 0 diameter is not an informative baseline: as
a supremum over the invertible matrices it equals 2(1 - eta) in closed form,
determined by eta alone and by nothing about the panel, and it is approached
only in the singular limit.  Second, the m >= 1 diameters should be unaffected,
because C_f A^{-1} >= 0 blows up as A approaches singularity, but that needs
checking rather than assuming.

This script checks it: the closed form is verified at m = 0, and the m >= 1
diameters are recomputed with an explicit floor on the smallest singular value
of A, so that nearly singular matrices are rejected outright.  If the guarded
values match the unguarded ones, the m >= 1 numbers stand.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6_sensitivity_and_vossler import load, item_profiles, kernels_from_gold
from e9_criterion_and_unified_radius import FAMS
from e12_structural_anchor_realdata import L4 as E2C_JSON, load as load12

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e24_guarded_reverification.json")

L = 4
SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 4000
N_STARTS = 16
MAXITER = 1000
SV_FLOOR = 1e-3          # smallest admissible singular value of A


def to_matrix(v):
    D = np.zeros((L, L))
    D[: L - 1, :] = v.reshape(L - 1, L)
    D[L - 1, :] = -D[: L - 1, :].sum(axis=0)
    n = np.linalg.norm(D)
    return D / n if n > 1e-12 else D


def displacement(D, kernels, protos, guard: bool) -> float:
    eye = np.eye(L)
    lo, hi = 0.0, 2.0
    for _ in range(45):
        t = 0.5 * (lo + hi)
        A = eye + t * D
        ok = True
        if guard and np.linalg.svd(A, compute_uv=False)[-1] < SV_FLOOR:
            ok = False
        if ok:
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


def optimised(kernels, protos, guard: bool) -> float:
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
            scored.append((displacement(D, kernels, protos, guard), D))
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0]
    for _, D in scored[:N_STARTS]:
        res = minimize(lambda v: -displacement(to_matrix(v), kernels, protos, guard),
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
    protos = np.vstack([P[g == k].mean(axis=0) for k in range(L)])
    eta = float(protos.min())

    out = {
        "env": {
            "script": "e24_guarded_reverification.py",
            "seeds": SEEDS, "sampled_directions": PER,
            "optimiser": f"Nelder-Mead, {N_STARTS} starts, maxiter {MAXITER}",
            "singular_value_floor": SV_FLOOR,
            "gpu_used": False, "api_calls": 0, "cost_usd": 0.0,
        },
        "m0_closed_form": {
            "eta": eta,
            "two_times_one_minus_eta": 2 * (1 - eta),
            "reported_m0_optimum": 1.929614528,
            "matches": abs(2 * (1 - eta) - 1.929614528) < 1e-6,
            "attained_by": "A = e_j 1', which is singular and outside GL(L,R)",
            "reading": ("the m = 0 diameter is a supremum in closed form, fixed "
                        "by eta alone, and is not an informative baseline"),
        },
        "e2c_prototypes": {},
    }
    print(f"eta={eta:.6f}  2(1-eta)={2*(1-eta):.6f}  "
          f"matches reported m=0: {out['m0_closed_form']['matches']}", flush=True)

    for m in (1, 2, 3):
        u = optimised(kern[:m], protos, guard=False)
        gd = optimised(kern[:m], protos, guard=True)
        out["e2c_prototypes"][f"m={m}"] = {
            "unguarded": u, "guarded": gd, "difference": u - gd}
        print(f"m={m}  unguarded={u:.6f}  guarded={gd:.6f}  diff={u - gd:+.2e}",
              flush=True)

    r = out["e2c_prototypes"]
    out["e2c_prototypes"]["reduction_m1_to_m3_guarded_pct"] = 100 * (
        1 - r["m=3"]["guarded"] / r["m=1"]["guarded"])

    # anchor pair, kernels from E2c exactly as E12 uses them
    judges2, fam2, P2, g2 = load12()
    src = json.load(open(E2C_JSON))
    kern12 = [np.array(src["kernels_from_gold"][f]) for f in FAMS]
    vd = 2.0 * (1.0 - P2.max(axis=1))
    order = np.argsort(vd)
    near_idx = order[: int(0.10 * len(order))]
    int_idx = order[int(0.50 * len(order)):]
    rng0 = np.random.default_rng(7)
    base = rng0.choice(int_idx, size=400, replace=False)
    anchors = rng0.choice(near_idx, size=2, replace=False)

    out["anchor_pair"] = {}
    for tag, pool in (("no_anchor", P2[base]),
                      ("two_anchors", np.vstack([P2[base], P2[anchors]]))):
        u = optimised(kern12, pool, guard=False)
        gd = optimised(kern12, pool, guard=True)
        out["anchor_pair"][tag] = {"unguarded": u, "guarded": gd}
        print(f"{tag:12s} unguarded={u:.6f}  guarded={gd:.6f}", flush=True)
    a0 = out["anchor_pair"]["no_anchor"]["guarded"]
    a2 = out["anchor_pair"]["two_anchors"]["guarded"]
    out["anchor_pair"]["reduction_pct_guarded"] = 100 * (1 - a2 / a0)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nm=1->3 reduction (guarded): "
          f"{out['e2c_prototypes']['reduction_m1_to_m3_guarded_pct']:.3f}%")
    print(f"anchor reduction (guarded): {out['anchor_pair']['reduction_pct_guarded']:.3f}%")
    print("saved", OUT)


if __name__ == "__main__":
    main()
