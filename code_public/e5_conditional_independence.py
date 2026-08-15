"""E5: does between-family conditional independence hold on real data?

This is the decisive gap in the panel model.  The paper assumes

    judges in different families are independent GIVEN the latent truth,
    while judges inside a family may depend on each other arbitrarily.

E2 reported raw agreement (within phi = 0.589, cross phi = 0.494).  Raw
agreement cannot separate two explanations:

  A  the model is right, and the cross-family agreement is entirely induced by
     the shared latent truth;
  B  the model is misspecified, and families share extra dependence (common
     pre-training corpora), in which case the radius invariance found in E2c
     is an artefact of misspecification rather than evidence for Theorem 3.

The test that separates them is the residual correlation AFTER conditioning on
the truth label.  Under A the cross-family residual correlation is zero; under
B it stays positive.

Residuals are formed against the family-and-gold-specific mean, so that all
first-order family bias is removed:

    r_ij = y_ij - E[ y | family(j), gold_i ]

Significance uses a cluster (item-level) bootstrap, because judgements on the
same item are not independent observations.

CPU only.  No API calls.  No GPU.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUB_DIR = os.path.join(ROOT, "data", "llmjudge_submissions")
GOLD = os.path.join(ROOT, "data", "llmjudge_gold", "llmjudge_test_qrel_idx.txt")
MAP = os.path.join(ROOT, "data", "family_mapping.json")
OUT = os.path.join(ROOT, "results", "e5_conditional_independence.json")

L = 4
VALID = set(range(L))
FAMS = ["GPT", "Llama", "T5"]
N_BOOT = 2000
SEED = 20260804


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
    return judges, runs, gold, fam, keys


def build_matrix(judges, runs, gold, keys):
    """Y[i, j] = label, G[i] = gold. Entries outside 0..3 become NaN."""
    Y = np.full((len(keys), len(judges)), np.nan)
    G = np.zeros(len(keys), dtype=int)
    for i, k in enumerate(keys):
        G[i] = gold[k]
        for jj, j in enumerate(judges):
            v = runs[j][k]
            if v in VALID:
                Y[i, jj] = v
    return Y, G


def residualise(Y: np.ndarray, G: np.ndarray, fam_of: List[str]) -> np.ndarray:
    """Remove E[y | family, gold]: all first-order family bias per truth class."""
    R = np.full_like(Y, np.nan)
    for f in FAMS:
        cols = [c for c, ff in enumerate(fam_of) if ff == f]
        if not cols:
            continue
        for g in range(L):
            rows = np.where(G == g)[0]
            if len(rows) == 0:
                continue
            block = Y[np.ix_(rows, cols)]
            mu = np.nanmean(block)
            R[np.ix_(rows, cols)] = block - mu
    return R


def pair_corr(R: np.ndarray, a: int, b: int) -> float:
    x, y = R[:, a], R[:, b]
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 30:
        return np.nan
    xs, ys = x[m], y[m]
    sx, sy = xs.std(), ys.std()
    if sx < 1e-12 or sy < 1e-12:
        return np.nan
    return float(((xs - xs.mean()) * (ys - ys.mean())).mean() / (sx * sy))


def summarise(R: np.ndarray, fam_of: List[str]):
    within, cross = [], []
    n = R.shape[1]
    for a, b in itertools.combinations(range(n), 2):
        c = pair_corr(R, a, b)
        if np.isnan(c):
            continue
        (within if fam_of[a] == fam_of[b] else cross).append(c)
    return np.array(within), np.array(cross)


def main() -> None:
    judges, runs, gold, fam, keys = load()
    fam_of = [fam[j] for j in judges]
    Y, G = build_matrix(judges, runs, gold, keys)

    # raw (unconditioned) correlations, for comparison with E2
    raw_within, raw_cross = summarise(Y - np.nanmean(Y), fam_of)

    # residual (gold- and family-conditioned) correlations: the decisive quantity
    R = residualise(Y, G, fam_of)
    res_within, res_cross = summarise(R, fam_of)

    rng = np.random.default_rng(SEED)
    n_items = Y.shape[0]
    boot_cross, boot_within, boot_gap = [], [], []
    for _ in range(N_BOOT):
        idx = rng.choice(n_items, size=n_items, replace=True)  # cluster on items
        Rb = residualise(Y[idx], G[idx], fam_of)
        w, c = summarise(Rb, fam_of)
        if len(w) and len(c):
            boot_within.append(w.mean())
            boot_cross.append(c.mean())
            boot_gap.append(w.mean() - c.mean())
    boot_cross = np.array(boot_cross)
    boot_within = np.array(boot_within)
    boot_gap = np.array(boot_gap)

    res = {
        "env": {
            "script": "e5_conditional_independence.py",
            "n_items": int(n_items),
            "n_judges": len(judges),
            "n_bootstrap": N_BOOT,
            "bootstrap_unit": "item (cluster bootstrap)",
            "seed": SEED,
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "raw_correlation": {
            "within_mean": float(raw_within.mean()),
            "cross_mean": float(raw_cross.mean()),
            "gap": float(raw_within.mean() - raw_cross.mean()),
            "n_within_pairs": int(len(raw_within)),
            "n_cross_pairs": int(len(raw_cross)),
            "note": "comparable to E2 measurement 3; cannot separate interpretations A and B",
        },
        "residual_correlation": {
            "definition": "r_ij = y_ij - E[y | family(j), gold_i]",
            "within_mean": float(res_within.mean()),
            "cross_mean": float(res_cross.mean()),
            "gap": float(res_within.mean() - res_cross.mean()),
            "cross_ci95": [float(np.percentile(boot_cross, 2.5)), float(np.percentile(boot_cross, 97.5))],
            "within_ci95": [float(np.percentile(boot_within, 2.5)), float(np.percentile(boot_within, 97.5))],
            "gap_ci95": [float(np.percentile(boot_gap, 2.5)), float(np.percentile(boot_gap, 97.5))],
            "cross_excludes_zero": bool(
                np.percentile(boot_cross, 2.5) > 0 or np.percentile(boot_cross, 97.5) < 0
            ),
        },
        "per_family_pair_residual_cross": {},
    }

    for fa, fb in itertools.combinations(FAMS, 2):
        vals = []
        for a in range(len(judges)):
            for b in range(a + 1, len(judges)):
                if {fam_of[a], fam_of[b]} == {fa, fb}:
                    c = pair_corr(R, a, b)
                    if not np.isnan(c):
                        vals.append(c)
        if vals:
            res["per_family_pair_residual_cross"][f"{fa}-{fb}"] = {
                "mean": float(np.mean(vals)),
                "n_pairs": len(vals),
            }

    cm = res["residual_correlation"]["cross_mean"]
    lo, hi = res["residual_correlation"]["cross_ci95"]
    res["verdict"] = {
        "interpretation_A_supported": bool(abs(cm) < 0.05 and lo < 0.05),
        "interpretation_B_supported": bool(lo > 0.10),
        "residual_cross_mean": cm,
        "reading": (
            "Interpretation A (model correct, agreement induced by the shared truth) "
            "predicts a residual cross-family correlation of zero. Interpretation B "
            "(extra shared dependence, model misspecified) predicts it stays clearly "
            "positive. The measured value and its item-level bootstrap interval "
            "decide between them."
        ),
    }

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
