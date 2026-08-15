"""E13: contract B with measured intent probabilities, at L=5.

Two limitations of E3 are removed here.

  1. E3 had to assume a uniform intent prior, because TREC Web diversity qrels
     carry no intent probabilities.  NTCIR-10 INTENT-2 publishes intent
     probabilities estimated from assessor voting, so the prior is measured
     rather than assumed.

  2. E3 ran at L=2, which is where the gauge group is thinnest, since its
     dimension is L(L-1).  NTCIR provides per-intent graded relevance on five
     levels, so the induced truth profile lives on a five-point simplex and the
     gauge has dimension 20.

The construction is the same one the paper uses for the distribution contract.
Given a topic q with intent distribution P(i|q) and a document d judged at
level rel(d|i) under each intent, the relevance level of d is a random variable
whose distribution is

    p_{q,d}[l] = sum_{i : rel(d|i) = l} P(i | q)

which is a point of the simplex.  Documents whose relevance does not depend on
the intent sit at a vertex; documents whose relevance changes with the intent
sit in the interior.

Both the Japanese and the Chinese subtask are processed, giving two independent
collections.

CPU only.  No API calls.  No GPU.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NT = os.path.join(ROOT, "data", "ntcir_intent2", "INTENT2ResearchPurposeData")
OUT = os.path.join(ROOT, "results", "e13_ntcir_measured_intent.json")

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEV_IDX = {v: i for i, v in enumerate(LEVELS)}
L = len(LEVELS)
SUBTASKS = {"Japanese": ("INTENT-2DRJ.Iprob", "INTENT-2DRJ.Dqrels"),
            "Chinese": ("INTENT-2DRC.Iprob", "INTENT-2DRC.Dqrels")}


def load_iprob(path: str) -> Dict[str, Dict[str, float]]:
    P: Dict[str, Dict[str, float]] = defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) >= 3:
            P[p[0]][p[1]] = float(p[2])
    return P


def load_dqrels(path: str) -> Dict[str, Dict[str, Dict[str, int]]]:
    """rel[topic][doc][intent] = level index."""
    rel: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for line in open(path):
        p = line.split()
        if len(p) >= 4 and p[3] in LEV_IDX:
            rel[p[0]][p[2]][p[1]] = LEV_IDX[p[3]]
    return rel


def profiles(P, rel, uniform: bool):
    """p_{q,d}[l] = sum over intents with level l of the intent prior."""
    rows, keys, coverage = [], [], []
    for q, docs in rel.items():
        if q not in P:
            continue
        prior = P[q]
        intents = sorted(prior)
        if len(intents) < 2:
            continue
        if uniform:
            w = {i: 1.0 / len(intents) for i in intents}
        else:
            s = sum(prior[i] for i in intents)
            w = {i: prior[i] / s for i in intents}
        for d, per_intent in docs.items():
            judged = [i for i in intents if i in per_intent]
            if len(judged) < 2:
                continue
            mass = sum(w[i] for i in judged)
            if mass <= 0:
                continue
            v = np.zeros(L)
            for i in judged:
                v[per_intent[i]] += w[i] / mass
            rows.append(v)
            keys.append((q, d))
            coverage.append(len(judged) / len(intents))
    return np.array(rows), keys, float(np.mean(coverage)) if coverage else 0.0


def summarise(Pm: np.ndarray) -> Dict[str, float]:
    interior = (Pm > 0).sum(axis=1) > 1  # mass on more than one level
    strict = (Pm > 0).all(axis=1)        # support on every level
    eta = Pm.min(axis=1)
    eta_i = Pm[interior].min(axis=1) if interior.any() else np.array([0.0])
    return {
        "n_pairs": int(Pm.shape[0]),
        "interior_fraction": float(interior.mean()),
        "n_interior": int(interior.sum()),
        "full_support_fraction": float(strict.mean()),
        "eta_min_over_all": float(eta.min()),
        "eta_mean_interior": float(eta_i.mean()),
        "eta_pct5_interior": float(np.percentile(eta_i, 5)),
        "eta_median_interior": float(np.median(eta_i)),
        "l1_floor_2eta_mean_interior": float(2 * eta_i.mean()),
        "mean_levels_with_mass": float((Pm > 0).sum(axis=1).mean()),
    }


def main() -> None:
    res = {
        "env": {
            "script": "e13_ntcir_measured_intent.py",
            "source": "NTCIR-10 INTENT-2 Document Ranking subtask",
            "L": L,
            "levels": LEVELS,
            "gauge_dimension_L_times_L_minus_1": L * (L - 1),
            "intent_prior": "measured from assessor voting (Iprob), not assumed",
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "subtasks": {},
    }

    for name, (ip, dq) in SUBTASKS.items():
        P = load_iprob(os.path.join(NT, ip))
        rel = load_dqrels(os.path.join(NT, dq))
        Pm_meas, keys, cov = profiles(P, rel, uniform=False)
        Pm_unif, _, _ = profiles(P, rel, uniform=True)

        block = {
            "n_topics": len(set(k[0] for k in keys)),
            "judged_intent_coverage_mean": cov,
            "measured_prior": summarise(Pm_meas),
            "uniform_prior": summarise(Pm_unif),
        }
        # how much does the uniform assumption distort the profiles?
        diff = np.abs(Pm_meas - Pm_unif).sum(axis=1)
        block["uniform_vs_measured"] = {
            "l1_difference_mean": float(diff.mean()),
            "l1_difference_median": float(np.median(diff)),
            "l1_difference_p95": float(np.percentile(diff, 95)),
            "fraction_with_l1_diff_above_0p10": float((diff > 0.10).mean()),
            "eta_mean_measured": block["measured_prior"]["eta_mean_interior"],
            "eta_mean_uniform": block["uniform_prior"]["eta_mean_interior"],
        }
        res["subtasks"][name] = block
        m, u = block["measured_prior"], block["uniform_prior"]
        print(
            f"{name}: pairs={m['n_pairs']} topics={block['n_topics']} "
            f"interior={100*m['interior_fraction']:.1f}% "
            f"full_support={100*m['full_support_fraction']:.1f}% "
            f"eta_int={m['eta_mean_interior']:.4f} "
            f"(uniform {u['eta_mean_interior']:.4f}) "
            f"L1diff_mean={block['uniform_vs_measured']['l1_difference_mean']:.4f}",
            flush=True,
        )

    both = [res["subtasks"][k]["measured_prior"]["interior_fraction"] for k in SUBTASKS]
    res["summary"] = {
        "interior_fraction_by_subtask": dict(zip(SUBTASKS, both)),
        "distribution_contract_supported_in_both": bool(all(x > 0.05 for x in both)),
        "reading": (
            "The distribution contract needs truth profiles that genuinely sit in the "
            "interior of the simplex. Here the intent prior is measured rather than "
            "assumed, and the profile lives on five levels rather than two, so the "
            "gauge has dimension twenty."
        ),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n" + json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
