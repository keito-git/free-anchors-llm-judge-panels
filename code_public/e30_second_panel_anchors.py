"""E30: does the structural-anchor rule hold on the second panel too?

E28 showed on the independently judged TREC-COVID panel that families after the
first contribute nothing, which reproduces half of the paper's design rule.
The other half, that unambiguous items already in the pool do shrink the
identification set, has so far been tested only on the first panel.  If it
fails here the rule is collection-specific; if it holds, both halves of the
rule are supported on two panels that share nothing.

The construction follows E12: order items by l1 distance to the nearest simplex
vertex, take the ambiguous half as prototypes and the closest tenth as the
anchor pool, then inject anchors and measure the diameter with the family
kernels in place.  Diameters use the optimised estimator throughout.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "data", "second_panel_treccovid", "sample.json")
PANEL = os.path.join(ROOT, "data", "second_panel_treccovid", "panel_labels.json")
OUT = os.path.join(ROOT, "results", "e30_second_panel_anchors.json")

L = 3
SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 3000
N_STARTS = 16
MAXITER = 1000
FRACTIONS = [0.0, 0.005, 0.02, 0.10, 0.20]
N_BASE = 120                # ambiguous prototypes; the panel has 294 items


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
    sample = {i["item_id"]: i for i in json.load(open(SAMPLE))["items"]}
    panel = json.load(open(PANEL))
    fam_of = {j: v["family"] for j, v in panel["env"]["judges"].items()}
    judges = list(panel["labels"])
    keys = sorted(set.intersection(*[set(panel["labels"][j]) for j in judges]))
    gold = np.array([sample[k]["gold"] for k in keys])

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

    vertex_dist = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vertex_dist)
    near_idx = order[: int(0.10 * len(order))]
    amb_idx = order[int(0.50 * len(order)):]

    rng0 = np.random.default_rng(7)
    base = rng0.choice(amb_idx, size=min(N_BASE, len(amb_idx)), replace=False)

    out = {
        "env": {
            "script": "e30_second_panel_anchors.py",
            "panel": "TREC-COVID, nine judges, seven families",
            "n_items": len(keys), "L": L,
            "n_base_prototypes": int(len(base)),
            "near_vertex_pool": int(len(near_idx)),
            "near_vertex_max_l1": float(vertex_dist[near_idx].max()),
            "seeds": SEEDS, "n_dirs_per_seed": PER,
            "estimator": f"optimised, {N_STARTS} starts, maxiter {MAXITER}",
            "gpu_used": False, "api_calls": 0, "cost_usd": 0.0,
        },
        "sweep": [],
    }

    for frac in FRACTIONS:
        n_anchor = int(round(frac * len(base)))
        if n_anchor > 0:
            anchors = rng0.choice(near_idx, size=min(n_anchor, len(near_idx)),
                                  replace=False)
            protos = np.vstack([P[base], P[anchors]])
        else:
            protos = P[base]
        d = optimised(kern, protos)
        out["sweep"].append({"anchor_fraction": frac, "n_anchor_items": int(n_anchor),
                             "n_prototypes": int(protos.shape[0]),
                             "eta": float(protos.min()), "diameter": d})
        print(f"rate={frac:<6} anchors={n_anchor:<3} diameter={d:.6f}", flush=True)

    r = {x["anchor_fraction"]: x["diameter"] for x in out["sweep"]}
    out["summary"] = {
        "diameter_no_anchor": r[0.0],
        "diameter_first_injection": r[0.005],
        "reduction_pct_first_injection": 100 * (1 - r[0.005] / r[0.0]),
        "reduction_pct_largest": 100 * (1 - r[0.20] / r[0.0]),
        "monotone_non_increasing": all(
            out["sweep"][i]["diameter"] <= out["sweep"][i - 1]["diameter"] + 1e-9
            for i in range(1, len(out["sweep"]))),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nsummary:", json.dumps(out["summary"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
