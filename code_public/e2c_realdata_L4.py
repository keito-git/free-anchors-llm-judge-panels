"""E2c: Real-data common-confusion floor at L=4 (graded relevance).

Improvements over e2b_realdata_floor.py:
  1. L=4 graded labels instead of binarising at threshold 2.
     Corollary 1(ii) could not be tested at L=2 because the signed gauge
     component had no room to act; L=4 gives a 12-dimensional gauge tangent.
  2. Family kernels C_f are estimated against HUMAN GOLD labels
     (llmjudge_test_qrel_idx.txt), not against a majority-vote proxy.
     The majority vote is exactly the quantity Corollary 1(i) predicts to be
     contaminated by shared confusion, so using it as a truth surrogate was
     circular.
  3. eta is reported as a distribution (min, 5th pct, median), not only the
     panel-size-pinned minimum 1/n_judges.

CPU only. No API calls. No GPU.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_DIR = os.path.join(ROOT, "data", "llmjudge_submissions")
GOLD = os.path.join(ROOT, "data", "llmjudge_gold", "llmjudge_test_qrel_idx.txt")
MAP = os.path.join(ROOT, "data", "family_mapping.json")
OUT = os.path.join(ROOT, "results", "e2c_realdata_L4.json")

L = 4  # graded relevance levels 0..3
VALID = set(range(L))
SEEDS = [42, 123, 456, 789, 1024]
N_DIRS = 4000  # random gauge directions per configuration


def read_qrel(path: str) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            score = int(parts[3])
        except ValueError:
            continue
        out[(parts[0], parts[2])] = score
    return out


def load_all() -> Tuple[List[str], Dict[str, Dict], Dict, Dict[str, str], Dict]:
    judges = sorted(f[:-4] for f in os.listdir(SUB_DIR) if f.endswith(".txt"))
    runs = {j: read_qrel(os.path.join(SUB_DIR, j + ".txt")) for j in judges}
    gold = read_qrel(GOLD)
    fam_raw = json.load(open(MAP))
    fam = {}
    for j in judges:
        entry = fam_raw.get(j)
        fam[j] = entry.get("family", "unknown") if isinstance(entry, dict) else "unknown"
    return judges, runs, gold, fam, fam_raw


def out_of_range(runs: Dict[str, Dict]) -> Dict[str, Dict[int, int]]:
    """Locate the score=5 / score=10 anomalies flagged in the L=2 run."""
    bad: Dict[str, Dict[int, int]] = {}
    for j, r in runs.items():
        counts: Dict[int, int] = defaultdict(int)
        for v in r.values():
            if v not in VALID:
                counts[v] += 1
        if counts:
            bad[j] = dict(counts)
    return bad


def build_profiles(judges, runs, gold):
    """Per-item empirical label distribution over judges (L=4), plus gold."""
    keys = sorted(set(gold) & set.intersection(*[set(runs[j]) for j in judges]))
    P = np.zeros((len(keys), L))
    g = np.zeros(len(keys), dtype=int)
    dropped = 0
    for i, k in enumerate(keys):
        counts = np.zeros(L)
        n = 0
        for j in judges:
            v = runs[j][k]
            if v in VALID:
                counts[v] += 1
                n += 1
            else:
                dropped += 1
        P[i] = counts / max(n, 1)
        g[i] = gold[k]
    return keys, P, g, dropped


def kernels_from_gold(judges, runs, gold, fam, keys, families):
    """C_f[l, k] = P(family f assigns label l | gold label k). Column-stochastic."""
    C = {}
    for f in families:
        members = [j for j in judges if fam[j] == f]
        M = np.zeros((L, L))
        for k in keys:
            gk = gold[k]
            if gk not in VALID:
                continue
            for j in members:
                v = runs[j][k]
                if v in VALID:
                    M[v, gk] += 1
        col = M.sum(axis=0, keepdims=True)
        col[col == 0] = 1.0
        C[f] = M / col
    return C


def gauge_radius(protos: np.ndarray, kernels: List[np.ndarray], rng, n_dirs: int):
    """Max l1 displacement ||A p - p||_1 over admissible A with 1^T A = 1^T.

    Admissibility: C_f A^{-1} >= 0 for every kernel, and A p in the simplex
    for every prototype. Line search along random tangent directions.
    """
    best = 0.0
    eye = np.eye(L)
    for _ in range(n_dirs):
        D = rng.normal(size=(L, L))
        D -= D.mean(axis=0, keepdims=True)  # columns sum to 0 => 1^T A = 1^T
        nrm = np.linalg.norm(D)
        if nrm < 1e-12:
            continue
        D /= nrm
        lo, hi = 0.0, 2.0
        for _ in range(40):  # bisection on step size
            t = 0.5 * (lo + hi)
            A = eye + t * D
            try:
                Ainv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                hi = t
                continue
            ok = all((Ck @ Ainv >= -1e-12).all() for Ck in kernels)
            if ok:
                Ap = protos @ A.T
                ok = bool((Ap >= -1e-12).all())
            if ok:
                lo = t
            else:
                hi = t
        A = eye + lo * D
        disp = float(np.abs(protos @ A.T - protos).sum(axis=1).max())
        best = max(best, disp)
    return best


def main() -> None:
    judges, runs, gold, fam, fam_raw = load_all()
    anomalies = out_of_range(runs)
    keys, P, g, dropped = build_profiles(judges, runs, gold)

    eta_item = P.min(axis=1)
    interior = eta_item > 0
    families = ["GPT", "Llama", "T5"]
    C = kernels_from_gold(judges, runs, gold, fam, keys, families)

    # Prototypes: gold-label-conditioned mean judge profile (K = L = 4).
    protos = np.vstack([P[g == k].mean(axis=0) for k in range(L)])
    eta_proto = float(protos.min())

    res = {
        "env": {
            "script": "e2c_realdata_L4.py",
            "L": L,
            "n_judges": len(judges),
            "n_items": int(len(keys)),
            "gold_source": "llmjudge_test_qrel_idx.txt (human, LLMJudge benchmark)",
            "kernel_estimation": "conditioned on human gold (NOT majority vote)",
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
            "n_dirs": N_DIRS,
            "seeds": SEEDS,
        },
        "data_quality": {
            "out_of_range_scores_by_judge": anomalies,
            "n_out_of_range_entries_dropped": int(dropped),
        },
        "eta": {
            "definition": "eta_i = min_l p_i[l] over the L=4 judge label distribution",
            "interior_fraction": float(interior.mean()),
            "n_interior": int(interior.sum()),
            "min": float(eta_item.min()),
            "pct5": float(np.percentile(eta_item, 5)),
            "median": float(np.median(eta_item)),
            "mean": float(eta_item.mean()),
            "interior_min": float(eta_item[interior].min()) if interior.any() else None,
            "interior_pct5": float(np.percentile(eta_item[interior], 5)) if interior.any() else None,
            "note": "min is pinned to the panel-size lattice 1/n_judges; report the distribution.",
        },
        "prototypes": {
            "definition": "gold-label-conditioned mean judge profile (K=L=4)",
            "profiles": protos.tolist(),
            "eta_proto": eta_proto,
            "l1_floor_2eta": 2.0 * eta_proto,
        },
        "kernels_from_gold": {f: C[f].tolist() for f in families},
    }

    # Common-confusion floor (G^0): A^{-1} = S nonneg column-stochastic.
    # S_t = (1-t) I + t e_{j*} 1^T is admissible for every kernel and any m.
    t_max = eta_proto
    res["G0_floor"] = {
        "construction": "S_t = (1-t) I + t e_{j*} 1^T, admissible for all m",
        "t_max_analytic": t_max,
        "l1_displacement_2t": 2.0 * t_max,
        "m_independent": True,
    }

    # Full gauge radius as a function of m (signed A^{-1} allowed).
    radii = {}
    for m in (1, 2, 3):
        ks = [np.array(C[f]) for f in families[:m]]
        vals = []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            vals.append(gauge_radius(protos, ks, rng, N_DIRS))
        radii[f"m={m}"] = {
            "families": families[:m],
            "radius_mean": float(np.mean(vals)),
            "radius_std": float(np.std(vals)),
            "per_seed": [float(v) for v in vals],
        }
    res["full_gauge_radius_vs_m"] = radii

    seq = [radii[f"m={m}"]["radius_mean"] for m in (1, 2, 3)]
    res["corollary_1ii_check"] = {
        "radius_sequence": seq,
        "monotone_non_increasing": bool(all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))),
        "strictly_shrinks_somewhere": bool(any(seq[i] > seq[i + 1] + 1e-6 for i in range(len(seq) - 1))),
        "floor_respected": bool(min(seq) >= 2.0 * t_max - 1e-6),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "kernels_from_gold"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
