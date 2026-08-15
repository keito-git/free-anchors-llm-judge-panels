"""E5b: is the residual cross-family correlation evidence of misspecification?

E5 conditioned on the human gold label and found a residual cross-family
correlation of 0.45.  That number is not decisive on its own: the model asserts
conditional independence given the LATENT class, and the gold label is a noisy
one-assessor proxy for it.  Conditioning on a noisy proxy leaves residual
correlation even when the model is correct.

This script runs the proper goodness-of-fit test.

  1. Fit a latent-class model with K classes and family-level confusion
     kernels by EM (judges inside a family share a kernel, which is what the
     paper assumes).
  2. Compute the observed cross-family residual correlation, residualising
     against the posterior-weighted family expectation.
  3. Draw parametric bootstrap replicates FROM THE FITTED MODEL and compute
     the same statistic.  Under the model the replicates give the reference
     distribution of the statistic, including whatever residual correlation is
     induced purely by estimation noise.
  4. Compare observed with reference, sweeping K.

If the observed statistic sits inside the reference band for some K, the data
are consistent with between-family conditional independence at that K, and the
E5 number was an artefact of conditioning on a noisy proxy.  If it stays far
above the band for every K, the assumption really fails.

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
OUT = os.path.join(ROOT, "results", "e5b_latent_class_gof_v2.json")

L = 4
VALID = set(range(L))
FAMS = ["GPT", "Llama", "T5"]
K_GRID = [2, 3, 4, 5, 6, 8]
N_SIM = 1000
EM_ITERS = 200
EM_TOL = 1e-7
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
    Y = np.full((len(keys), len(judges)), -1, dtype=int)
    for i, k in enumerate(keys):
        for jj, j in enumerate(judges):
            v = runs[j][k]
            if v in VALID:
                Y[i, jj] = v
    return judges, [fam[j] for j in judges], Y


def counts_by_family(Y: np.ndarray, fam_of: List[str]):
    """N[f][i, l] = number of judges of family f giving label l to item i."""
    out = {}
    for f in FAMS:
        cols = [c for c, ff in enumerate(fam_of) if ff == f]
        M = np.zeros((Y.shape[0], L))
        for c in cols:
            v = Y[:, c]
            for l in range(L):
                M[:, l] += (v == l)
        out[f] = M
    return out


def em_fit(Y: np.ndarray, fam_of: List[str], K: int, rng, iters=EM_ITERS):
    """Latent class model with family-level kernels. Returns pi, C, posterior."""
    N = counts_by_family(Y, fam_of)
    n = Y.shape[0]
    pi = rng.dirichlet(np.ones(K) * 5.0)
    C = {f: rng.dirichlet(np.ones(L) * 3.0, size=K).T for f in FAMS}  # C[f][l, k]
    prev = -np.inf
    post = np.full((n, K), 1.0 / K)
    for _ in range(iters):
        logp = np.log(pi + 1e-300)[None, :] * np.ones((n, 1))
        for f in FAMS:
            logp = logp + N[f] @ np.log(C[f] + 1e-300)
        mx = logp.max(axis=1, keepdims=True)
        w = np.exp(logp - mx)
        s = w.sum(axis=1, keepdims=True)
        post = w / s
        ll = float((np.log(s) + mx).sum())
        pi = post.mean(axis=0)
        for f in FAMS:
            num = N[f].T @ post  # (L, K)
            C[f] = (num + 1e-9) / (num.sum(axis=0, keepdims=True) + 1e-9 * L)
        if abs(ll - prev) < EM_TOL * max(1.0, abs(prev)):
            break
        prev = ll
    return pi, C, post, prev


def cross_residual_corr(Y: np.ndarray, fam_of: List[str], C, post) -> float:
    """Mean pairwise residual correlation across families.

    Residual uses the posterior-weighted family expectation, which is the
    model's own prediction for that judge on that item.
    """
    lev = np.arange(L)
    exp_f = {f: post @ (C[f].T @ lev) for f in FAMS}  # (n,)
    n_j = Y.shape[1]
    R = np.full(Y.shape, np.nan)
    for c in range(n_j):
        v = Y[:, c].astype(float)
        m = Y[:, c] >= 0
        R[m, c] = v[m] - exp_f[fam_of[c]][m]
    vals = []
    for a, b in itertools.combinations(range(n_j), 2):
        if fam_of[a] == fam_of[b]:
            continue
        x, y = R[:, a], R[:, b]
        msk = ~np.isnan(x) & ~np.isnan(y)
        if msk.sum() < 30:
            continue
        xs, ys = x[msk], y[msk]
        sx, sy = xs.std(), ys.std()
        if sx < 1e-12 or sy < 1e-12:
            continue
        vals.append(((xs - xs.mean()) * (ys - ys.mean())).mean() / (sx * sy))
    return float(np.mean(vals)) if vals else np.nan


def simulate(pi, C, fam_of, n_items, rng) -> np.ndarray:
    """Draw a dataset from the fitted model: latent class, then independent judges."""
    K = len(pi)
    k = rng.choice(K, size=n_items, p=pi)
    Y = np.zeros((n_items, len(fam_of)), dtype=int)
    for c, f in enumerate(fam_of):
        probs = C[f][:, k].T  # (n_items, L)
        u = rng.random(n_items)[:, None]
        Y[:, c] = (probs.cumsum(axis=1) < u).sum(axis=1).clip(0, L - 1)
    return Y


def main() -> None:
    judges, fam_of, Y = load()
    rng = np.random.default_rng(SEED)
    res = {
        "env": {
            "script": "e5b_latent_class_gof.py",
            "n_items": int(Y.shape[0]),
            "n_judges": len(judges),
            "K_grid": K_GRID,
            "n_sim_per_K": N_SIM,
            "seed": SEED,
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "per_K": {},
    }

    for K in K_GRID:
        best = None
        for _ in range(20):  # restarts
            pi, C, post, ll = em_fit(Y, fam_of, K, rng)
            if best is None or ll > best[3]:
                best = (pi, C, post, ll)
        pi, C, post, ll = best
        obs = cross_residual_corr(Y, fam_of, C, post)

        sims = []
        for _ in range(N_SIM):
            Ys = simulate(pi, C, fam_of, Y.shape[0], rng)
            _, Cs, posts, _ = em_fit(Ys, fam_of, K, rng)  # same iters as the main fit
            sims.append(cross_residual_corr(Ys, fam_of, Cs, posts))
        sims_a = np.array([s for s in sims if not np.isnan(s)])

        lo, hi = float(np.percentile(sims_a, 2.5)), float(np.percentile(sims_a, 97.5))
        res["per_K"][str(K)] = {
            "loglik": float(ll),
            "observed_cross_residual_corr": obs,
            "reference_mean": float(sims_a.mean()),
            "reference_ci95": [lo, hi],
            "inside_reference_band": bool(lo <= obs <= hi),
            "excess_over_reference": float(obs - sims_a.mean()),
        }
        print(
            f"K={K}: obs={obs:+.4f} ref={sims_a.mean():+.4f} "
            f"CI=[{lo:+.4f},{hi:+.4f}] inside={lo <= obs <= hi}",
            flush=True,
        )

    excesses = {k: v["excess_over_reference"] for k, v in res["per_K"].items()}
    any_inside = any(v["inside_reference_band"] for v in res["per_K"].values())
    res["verdict"] = {
        "any_K_consistent_with_model": any_inside,
        "min_excess": float(min(excesses.values())),
        "argmin_K": min(excesses, key=excesses.get),
        "reading": (
            "If no K puts the observed statistic inside the reference band, the "
            "between-family conditional independence assumption fails on this "
            "panel and the radius invariance in E2c cannot be read as evidence "
            "for Theorem 3."
        ),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
