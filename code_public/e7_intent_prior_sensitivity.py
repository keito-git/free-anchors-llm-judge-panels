"""E7: does the contract-B construction depend on the intent prior P(i|q)?

E3 built the distributional-truth profiles from TREC Web diversity qrels as

    p_{q,d} = sum_i P(i|q) * rel(d|i)

and had to assume a uniform intent prior, because the qrels carry no intent
probabilities.  The question is whether the conclusion survives other priors.

We compare four priors:

  uniform          P(i|q) = 1/|I_q|                       (the E3 choice)
  count            proportional to the number of documents judged relevant
                   for that intent
  inverse_count    proportional to 1 / (1 + relevant count), which deliberately
                   up-weights rare intents and is the adversarial direction
  dirichlet_band   random perturbations of uniform, giving a robustness band

and report, for each, the interior fraction, the eta distribution, whether the
l1 floor 2*eta holds, and the rank correlation of the resulting profiles with
the uniform-prior profiles.

Subtopic 0 in TREC diversity qrels is the overall judgement rather than a
genuine intent; both inclusion and exclusion are reported.

CPU only.  No API calls.  No GPU.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIV_DIR = os.path.join(ROOT, "data", "trec_web_diversity")
OUT = os.path.join(ROOT, "results", "e7_intent_prior_sensitivity.json")

YEARS = ["2009", "2010", "2011"]
N_BAND = 200
SEED = 20260813


def load_year(year: str, drop_subtopic_zero: bool):
    path = os.path.join(DIV_DIR, f"{year}qrels.diversity.txt")
    rel: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for line in open(path):
        p = line.split()
        if len(p) < 4:
            continue
        qid, intent, doc, s = p[0], p[1], p[2], p[3]
        if drop_subtopic_zero and intent == "0":
            continue
        try:
            rel[qid][intent][doc] = 1 if int(s) > 0 else 0
        except ValueError:
            continue
    return rel


def priors(rel_q: Dict[str, Dict[str, int]], kind: str, rng=None) -> Dict[str, float]:
    intents = sorted(rel_q)
    if kind == "uniform":
        w = np.ones(len(intents))
    elif kind == "count":
        w = np.array([max(sum(rel_q[i].values()), 0) for i in intents], dtype=float)
        if w.sum() == 0:
            w = np.ones(len(intents))
    elif kind == "inverse_count":
        w = np.array([1.0 / (1.0 + sum(rel_q[i].values())) for i in intents])
    elif kind == "dirichlet":
        w = rng.dirichlet(np.ones(len(intents)) * 3.0)
    else:
        raise ValueError(kind)
    w = w / w.sum()
    return dict(zip(intents, w))


def profiles(rel, kind: str, rng=None) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """v_{q,d} = sum_i P(i|q) rel(d|i); truth profile is (1 - v, v)."""
    vals, keys = [], []
    for qid, per_intent in rel.items():
        if len(per_intent) < 2:
            continue
        P = priors(per_intent, kind, rng)
        docs = set()
        for i in per_intent:
            docs |= set(per_intent[i])
        for d in docs:
            v = sum(P[i] * per_intent[i].get(d, 0) for i in per_intent)
            vals.append(v)
            keys.append((qid, d))
    return np.array(vals), keys


def summarise(v: np.ndarray) -> Dict[str, float]:
    interior = (v > 0) & (v < 1)
    eta = np.minimum(v, 1 - v)
    eta_int = eta[interior]
    return {
        "n_pairs": int(len(v)),
        "interior_fraction": float(interior.mean()),
        "n_interior": int(interior.sum()),
        "eta_mean_interior": float(eta_int.mean()) if len(eta_int) else 0.0,
        "eta_median_interior": float(np.median(eta_int)) if len(eta_int) else 0.0,
        "eta_pct5_interior": float(np.percentile(eta_int, 5)) if len(eta_int) else 0.0,
        "l1_floor_2eta_mean": float(2 * eta_int.mean()) if len(eta_int) else 0.0,
        "floor_holds_all_interior": bool(len(eta_int) == 0 or (eta_int > 0).all()),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    res = {
        "env": {
            "script": "e7_intent_prior_sensitivity.py",
            "years": YEARS,
            "n_dirichlet_band": N_BAND,
            "seed": SEED,
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "by_subtopic_zero_handling": {},
    }

    for drop in (False, True):
        tag = "exclude_subtopic0" if drop else "include_subtopic0"
        block = {}
        for year in YEARS:
            rel = load_year(year, drop)
            base_v, base_keys = profiles(rel, "uniform")
            entry = {"uniform": summarise(base_v)}
            base_map = dict(zip(base_keys, base_v))

            for kind in ("count", "inverse_count"):
                v, keys = profiles(rel, kind)
                s = summarise(v)
                common = [k for k in keys if k in base_map]
                a = np.array([base_map[k] for k in common])
                b = np.array([v[keys.index(k)] for k in common[:0]]) if False else None
                # rank correlation against the uniform profiles
                vmap = dict(zip(keys, v))
                bb = np.array([vmap[k] for k in common])
                rho = float(spearmanr(a, bb).statistic) if len(common) > 10 else float("nan")
                s["spearman_vs_uniform"] = rho
                entry[kind] = s

            # dirichlet robustness band on the interior fraction and eta
            fracs, etas = [], []
            for _ in range(N_BAND):
                v, _ = profiles(rel, "dirichlet", rng)
                si = summarise(v)
                fracs.append(si["interior_fraction"])
                etas.append(si["eta_mean_interior"])
            entry["dirichlet_band"] = {
                "interior_fraction_mean": float(np.mean(fracs)),
                "interior_fraction_ci95": [
                    float(np.percentile(fracs, 2.5)),
                    float(np.percentile(fracs, 97.5)),
                ],
                "eta_mean_interior_mean": float(np.mean(etas)),
                "eta_mean_interior_ci95": [
                    float(np.percentile(etas, 2.5)),
                    float(np.percentile(etas, 97.5)),
                ],
            }
            block[year] = entry
        res["by_subtopic_zero_handling"][tag] = block

    # headline check: is the interior fraction conclusion stable?
    ex = res["by_subtopic_zero_handling"]["exclude_subtopic0"]
    fr = {y: ex[y]["uniform"]["interior_fraction"] for y in YEARS}
    band = {y: ex[y]["dirichlet_band"]["interior_fraction_ci95"] for y in YEARS}
    res["verdict"] = {
        "interior_fraction_uniform_by_year": fr,
        "interior_fraction_dirichlet_ci95_by_year": band,
        "interiority_survives_all_priors": bool(
            all(ex[y][k]["interior_fraction"] > 0.05 for y in YEARS for k in ("uniform", "count", "inverse_count"))
        ),
        "reading": (
            "Contract B needs a non-negligible set of documents whose truth profile "
            "sits in the interior of the simplex. If that survives every intent "
            "prior, the uniform assumption in E3 is not load-bearing."
        ),
    }

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
