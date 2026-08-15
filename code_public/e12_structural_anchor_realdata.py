"""E12: does the structural-anchor rule work on the real LLMJudge panel?

The design rule the paper proposes is that adding unambiguous items to the
judging pool collapses the gauge and restores identification without any extra
human labels.  So far that rule has been demonstrated on synthetic data (E1)
and on TREC-derived semi-synthetic profiles (E3).  It has never been shown on
the real panel, which is exactly where a reader would want it.

Here we take the 33 LLMJudge judges and the human gold labels, split the items
by how close their judge profile sits to a vertex of the simplex, and measure
what happens to the identification set as near-vertex items enter the
prototype set.

  Ambiguous-only pool : prototypes built from interior items alone
  Mixed pool          : the same, with a fraction of near-vertex items added

If the rule transfers, the gauge radius should fall sharply as the fraction of
near-vertex items grows, and the fall should already be visible at small
fractions.

CPU only.  No API calls.  No GPU.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_DIR = os.path.join(ROOT, "data", "llmjudge_submissions")
GOLD = os.path.join(ROOT, "data", "llmjudge_gold", "llmjudge_test_qrel_idx.txt")
MAP = os.path.join(ROOT, "data", "family_mapping.json")
L4 = os.path.join(ROOT, "results", "e2c_realdata_L4.json")
OUT = os.path.join(ROOT, "results", "e12_structural_anchor_realdata.json")

L = 4
VALID = set(range(L))
FAMS = ["GPT", "Llama", "T5"]
SEEDS = [42, 123, 456, 789, 1024, 20260804]
N_DIRS = 3000
FRACTIONS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def read_qrel(path: str) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 4:
            try:
                out[(p[0], p[2])] = int(p[3])
            except ValueError:
                pass
    return out


def load():
    judges = sorted(f[:-4] for f in os.listdir(SUB_DIR) if f.endswith(".txt"))
    runs = {j: read_qrel(os.path.join(SUB_DIR, j + ".txt")) for j in judges}
    gold = read_qrel(GOLD)
    fam_raw = json.load(open(MAP))
    fam = {j: (fam_raw.get(j) or {}).get("family", "unknown") for j in judges}
    keys = sorted(set(gold) & set.intersection(*[set(runs[j]) for j in judges]))
    keys = [k for k in keys if gold[k] in VALID]
    P = np.zeros((len(keys), L))
    g = np.zeros(len(keys), dtype=int)
    for i, k in enumerate(keys):
        c = np.zeros(L)
        n = 0
        for j in judges:
            v = runs[j][k]
            if v in VALID:
                c[v] += 1
                n += 1
        P[i] = c / max(n, 1)
        g[i] = gold[k]
    return judges, [fam[j] for j in judges], P, g


def max_step(kernels, D, protos):
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
    return lo


def radius(kernels, protos, seeds, n_dirs) -> float:
    eye = np.eye(L)
    best = 0.0
    for s in seeds:
        rng = np.random.default_rng(s)
        for _ in range(n_dirs):
            D = rng.normal(size=(L, L))
            D -= D.mean(axis=0, keepdims=True)
            nn = np.linalg.norm(D)
            if nn < 1e-12:
                continue
            D /= nn
            t = max_step(kernels, D, protos)
            best = max(best, float(np.abs(protos @ (eye + t * D).T - protos).sum(axis=1).max()))
    return best


def main() -> None:
    judges, fam_of, P, g = load()
    src = json.load(open(L4))
    kern = [np.array(src["kernels_from_gold"][f]) for f in FAMS]

    # distance from the nearest vertex, in l1
    vertex_dist = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vertex_dist)
    near_vertex_idx = order[: int(0.10 * len(order))]  # the 10% closest to a vertex
    interior_idx = order[int(0.50 * len(order)) :]     # the most ambiguous half

    res = {
        "env": {
            "script": "e12_structural_anchor_realdata.py",
            "n_items": int(P.shape[0]),
            "n_judges": len(judges),
            "L": L,
            "seeds": SEEDS,
            "n_dirs_per_seed": N_DIRS,
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "pool_characterisation": {
            "l1_vertex_distance_min": float(vertex_dist.min()),
            "l1_vertex_distance_median": float(np.median(vertex_dist)),
            "l1_vertex_distance_max": float(vertex_dist.max()),
            "near_vertex_pool_size": int(len(near_vertex_idx)),
            "near_vertex_max_l1_distance": float(vertex_dist[near_vertex_idx].max()),
            "interior_pool_size": int(len(interior_idx)),
            "interior_min_l1_distance": float(vertex_dist[interior_idx].min()),
        },
        "sweep": [],
    }

    rng0 = np.random.default_rng(7)
    n_base = 400
    base = rng0.choice(interior_idx, size=min(n_base, len(interior_idx)), replace=False)

    for frac in FRACTIONS:
        n_anchor = int(round(frac * len(base)))
        if n_anchor > 0:
            anchors = rng0.choice(near_vertex_idx, size=min(n_anchor, len(near_vertex_idx)), replace=False)
            protos = np.vstack([P[base], P[anchors]])
        else:
            protos = P[base]
        r = radius(kern, protos, SEEDS, N_DIRS)
        eta = float(protos.min())
        res["sweep"].append(
            {
                "anchor_fraction": frac,
                "n_anchor_items": int(n_anchor),
                "n_prototypes": int(protos.shape[0]),
                "eta": eta,
                "radius": r,
            }
        )
        print(f"frac={frac:.3f} n_anchor={n_anchor:3d} eta={eta:.5f} radius={r:.6f}", flush=True)

    r0 = res["sweep"][0]["radius"]
    rl = res["sweep"][-1]["radius"]
    res["summary"] = {
        "radius_no_anchor": r0,
        "radius_max_fraction": rl,
        "reduction_factor": float(r0 / rl) if rl > 1e-12 else None,
        "monotone_non_increasing": bool(
            all(
                res["sweep"][i]["radius"] >= res["sweep"][i + 1]["radius"] - 1e-9
                for i in range(len(res["sweep"]) - 1)
            )
        ),
        "reading": (
            "The design rule is that unambiguous items entering the pool shrink the "
            "identification set without any new human labels. This sweep measures "
            "the effect on the real LLMJudge panel rather than on synthetic data."
        ),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    s = res["summary"]
    print(f"\nno anchor: {s['radius_no_anchor']:.6f} -> max fraction: {s['radius_max_fraction']:.6f}")
    print(f"reduction factor: {s['reduction_factor']}")
    print(f"monotone: {s['monotone_non_increasing']}")


if __name__ == "__main__":
    main()
