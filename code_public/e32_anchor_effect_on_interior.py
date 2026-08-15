"""E32: measure what anchors do to the quantity anchors are supposed to fix.

E30 injected near-vertex items into the second panel's pool and the diameter
went up.  E31 explains why that is not a failure of the rule.  On this panel the
free pool reaches all three vertices, so the precondition of Theorem 4 is met,
and the theorem's conclusion is that the gauge group collapses to permutations.
A permutation moves a vertex to another vertex, which is the largest l1 move
there is, so once vertices are in the prototype set they become the extremal
objects and the diameter measures the permutation orbit rather than the
continuum the anchors removed.  Remark 4.x already states that set shrinkage
and diameter shrinkage are different things; E30 is that remark in the data.

The right measurement separates the two roles an anchor plays.  Vertex items
enter the constraint set, so they restrict the admissible A, but the
displacement is measured only over the interior prototypes, which are the
profiles the evaluation actually wants to estimate.  If Theorem 4 is doing what
it claims, that quantity should fall sharply as vertices enter, on a panel whose
free pool covers every label.

For contrast the same measurement is run on the first panel, whose free pool
reaches only one label, so the precondition fails there and the fall should be
smaller.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e32_anchor_effect_on_interior.json")
SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 3000
N_STARTS = 16
MAXITER = 1000


def make_ops(L):
    def to_matrix(v):
        D = np.zeros((L, L))
        D[: L - 1, :] = v.reshape(L - 1, L)
        D[L - 1, :] = -D[: L - 1, :].sum(axis=0)
        n = np.linalg.norm(D)
        return D / n if n > 1e-12 else D

    def displacement(D, kernels, constrain, measure):
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
                ok = bool((constrain @ A.T >= -1e-12).all())
            lo, hi = (t, hi) if ok else (lo, t)
        A = eye + lo * D
        return float(np.abs(measure @ A.T - measure).sum(axis=1).max())

    def optimised(kernels, constrain, measure):
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
                scored.append((displacement(D, kernels, constrain, measure), D))
        scored.sort(key=lambda x: -x[0])
        best = scored[0][0]
        for _, D in scored[:N_STARTS]:
            res = minimize(
                lambda v: -displacement(to_matrix(v), kernels, constrain, measure),
                D[: L - 1, :].ravel(), method="Nelder-Mead",
                options={"maxiter": MAXITER, "xatol": 1e-8, "fatol": 1e-10})
            best = max(best, -res.fun)
        return best

    return optimised


def sweep(name, P, kern, L, n_base, fracs, out):
    optimised = make_ops(L)
    vd = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vd)
    near_idx = order[: int(0.10 * len(order))]
    amb_idx = order[int(0.50 * len(order)):]
    rng0 = np.random.default_rng(7)
    base = rng0.choice(amb_idx, size=min(n_base, len(amb_idx)), replace=False)
    interior = P[base]

    rows = []
    for frac in fracs:
        n_anchor = int(round(frac * len(base)))
        if n_anchor > 0:
            anchors = rng0.choice(near_idx, size=min(n_anchor, len(near_idx)),
                                  replace=False)
            constrain = np.vstack([interior, P[anchors]])
        else:
            constrain = interior
        d = optimised(kern, constrain, interior)      # measure on interior only
        rows.append({"anchor_fraction": frac, "n_anchor_items": int(n_anchor),
                     "interior_diameter": d})
        print(f"  [{name}] rate={frac:<6} anchors={n_anchor:<3} "
              f"interior diameter={d:.6f}", flush=True)
    base_d = rows[0]["interior_diameter"]
    out[name] = {
        "n_interior_prototypes": int(len(base)),
        "sweep": rows,
        "reduction_pct_largest": 100 * (1 - rows[-1]["interior_diameter"] / base_d),
        "monotone_non_increasing": all(
            rows[i]["interior_diameter"] <= rows[i - 1]["interior_diameter"] + 1e-9
            for i in range(1, len(rows))),
    }


def second_panel():
    SAMPLE = os.path.join(ROOT, "data", "second_panel_treccovid", "sample.json")
    PANEL = os.path.join(ROOT, "data", "second_panel_treccovid", "panel_labels.json")
    sample = {i["item_id"]: i for i in json.load(open(SAMPLE))["items"]}
    panel = json.load(open(PANEL))
    fam_of = {j: v["family"] for j, v in panel["env"]["judges"].items()}
    judges = list(panel["labels"])
    keys = sorted(set.intersection(*[set(panel["labels"][j]) for j in judges]))
    gold = np.array([sample[k]["gold"] for k in keys])
    L = 3
    P = np.zeros((len(keys), L))
    for i, k in enumerate(keys):
        c = np.zeros(L)
        for j in judges:
            c[panel["labels"][j][k]] += 1
        P[i] = c / c.sum()
    fams = []
    for j in judges:
        if fam_of[j] not in fams:
            fams.append(fam_of[j])
    kern = []
    for f in fams:
        cols = [j for j in judges if fam_of[j] == f]
        M = np.zeros((L, L))
        for i, k in enumerate(keys):
            for j in cols:
                M[panel["labels"][j][k], gold[i]] += 1
        s = M.sum(axis=0, keepdims=True)
        s[s == 0] = 1.0
        kern.append(M / s)
    return P, kern, L


def first_panel():
    from e12_structural_anchor_realdata import load as load12, FAMS, L4
    _, _, P, _ = load12()
    src = json.load(open(L4))
    kern = [np.array(src["kernels_from_gold"][f]) for f in FAMS]
    return P, kern, 4


def main() -> None:
    out = {"env": {"script": "e32_anchor_effect_on_interior.py",
                   "seeds": SEEDS, "n_dirs_per_seed": PER,
                   "estimator": f"optimised, {N_STARTS} starts, maxiter {MAXITER}",
                   "measurement": ("anchors constrain A; displacement is measured "
                                   "on the interior prototypes only"),
                   "gpu_used": False, "api_calls": 0, "cost_usd": 0.0}}

    P2, k2, L2 = second_panel()
    sweep("second_panel_all_vertices_covered", P2, k2, L2, 120,
          [0.0, 0.02, 0.10, 0.20], out)

    P1, k1, L1 = first_panel()
    sweep("first_panel_one_vertex_covered", P1, k1, L1, 400,
          [0.0, 0.005, 0.05, 0.20], out)

    a = out["second_panel_all_vertices_covered"]["reduction_pct_largest"]
    b = out["first_panel_one_vertex_covered"]["reduction_pct_largest"]
    out["summary"] = {
        "second_panel_reduction_pct": a,
        "first_panel_reduction_pct": b,
        "reading": ("Theorem 4 needs every label represented among the vertex "
                    "items; the second panel meets that and the first does not."),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nsummary:", json.dumps(out["summary"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
