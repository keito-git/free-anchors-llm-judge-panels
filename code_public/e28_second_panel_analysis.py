"""E28: repeat the family-count analysis on the independently judged panel.

This is the second panel the paper needed.  Nine judges from seven provider
families labelled the same 294 TREC-COVID items on three levels, so the
collection, the judges and the family structure are all independent of the
LLMJudge panel used elsewhere.  Human gold labels come with the collection, so
the family kernels are estimated the same way as in E2c.

The question is whether the pattern established on the first panel reappears:
does the identification set stop shrinking once the first family is in place,
and does whether a family binds track panel domination rather than provenance?

Diameters use the optimised estimator of E19, since E19 showed sampling to be
loose by about 40% and to exaggerate every intervention.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "data", "second_panel_treccovid", "sample.json")
PANEL = os.path.join(ROOT, "data", "second_panel_treccovid", "panel_labels.json")
OUT = os.path.join(ROOT, "results", "e28_second_panel_analysis.json")

L = 3                       # TREC-COVID levels 0, 1, 2
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


def dominated(Cg, others) -> bool:
    """Is every row of Cg in the convex cone spanned by the rows of the others?"""
    from scipy.optimize import linprog
    rows = np.vstack([C for C in others]) if others else np.zeros((0, L))
    if rows.shape[0] == 0:
        return False
    for r in Cg:
        res = linprog(np.zeros(rows.shape[0]), A_eq=rows.T, b_eq=r,
                      bounds=[(0, None)] * rows.shape[0], method="highs")
        if not res.success:
            return False
    return True


def main() -> None:
    sample = {i["item_id"]: i for i in json.load(open(SAMPLE))["items"]}
    panel = json.load(open(PANEL))
    fam_of = {j: v["family"] for j, v in panel["env"]["judges"].items()}

    judges = list(panel["labels"])
    keys = sorted(set.intersection(*[set(panel["labels"][j]) for j in judges]))
    gold = np.array([sample[k]["gold"] for k in keys])

    # per-item judge profile and per-family kernels, exactly as in E2c
    P = np.zeros((len(keys), L))
    for i, k in enumerate(keys):
        c = np.zeros(L)
        for j in judges:
            c[panel["labels"][j][k]] += 1
        P[i] = c / c.sum()
    protos = np.vstack([P[gold == k].mean(axis=0) for k in range(L)])

    fams = []
    for j in judges:
        if fam_of[j] not in fams:
            fams.append(fam_of[j])
    C = {}
    for f in fams:
        cols = [j for j in judges if fam_of[j] == f]
        M = np.zeros((L, L))
        for i, k in enumerate(keys):
            for j in cols:
                M[panel["labels"][j][k], gold[i]] += 1
        s = M.sum(axis=0, keepdims=True)
        s[s == 0] = 1.0
        C[f] = M / s

    out = {
        "env": {
            "script": "e28_second_panel_analysis.py",
            "collection": "BEIR TREC-COVID, independent of TREC DL 2023",
            "n_items": len(keys), "L": L,
            "judges": len(judges), "families": fams,
            "judges_per_family": {f: sum(1 for j in judges if fam_of[j] == f)
                                  for f in fams},
            "estimator": f"optimised, {N_STARTS} starts, maxiter {MAXITER}",
            "gpu_used": False, "api_calls": 0, "cost_usd": 0.0,
        },
        "eta": float(protos.min()),
        "floor_2eta": 2 * float(protos.min()),
        "diameter": {}, "step_reduction_pct": [], "domination": {},
    }

    kern = [C[f] for f in fams]
    prev = None
    for m in range(0, len(fams) + 1):
        d = optimised(kern[:m], protos)
        out["diameter"][f"m={m}"] = d
        if prev is not None:
            out["step_reduction_pct"].append(100 * (1 - d / prev))
        prev = d
        print(f"m={m} ({'none' if m == 0 else fams[m-1]:>9s})  diameter={d:.6f}",
              flush=True)

    for i, f in enumerate(fams[1:], start=1):
        out["domination"][f] = bool(dominated(C[f], [C[g] for g in fams[:i]]))

    out["summary"] = {
        "first_family_reduction_pct": out["step_reduction_pct"][0],
        "later_family_reductions_pct": out["step_reduction_pct"][1:],
        "max_later_reduction_pct": max(out["step_reduction_pct"][1:]),
        "n_dominated_later_families": sum(out["domination"].values()),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nstep reductions (%):",
          [f"{x:.2f}" for x in out["step_reduction_pct"]])
    print("dominated later families:", out["domination"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
