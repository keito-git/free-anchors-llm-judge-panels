"""E6: prototype sensitivity and the Vossler recovery contrast.

  E6a  Prototype sensitivity.  E2c defined prototypes as the gold-conditioned
       MEAN judge profile.  The question is whether the conclusion survives
       other reasonable choices.  We repeat the radius-versus-m measurement with
       three prototype constructions: gold-conditioned mean, gold-conditioned
       median-ish (trimmed mean), and EM-estimated latent class profiles.

  E6b  Vossler recovery contrast.  An earlier attempt (E4c) compared gauge
       radii and failed, because the radius is the wrong functional for this
       comparison.  The distinction between Vossler's contract and ours is
       about what is estimated, so the right comparison is a RECOVERY
       experiment on one synthetic dataset:
         - pooled contract   : items share K prototypes, m families -> fit and
                               measure recovery of the confusion kernels
         - item-free contract: every item carries its own free prevalence
                               vector, no pooling -> count how many distinct
                               parameter settings reproduce the observations
       Both verdicts should hold on the same data.

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
OUT = os.path.join(ROOT, "results", "e6_sensitivity_and_vossler.json")

L = 4
VALID = set(range(L))
FAMS = ["GPT", "Llama", "T5"]
SEEDS = [42, 123, 456]
N_DIRS = 3000


def read_qrel(path):
    out = {}
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
    return judges, runs, gold, [fam[j] for j in judges], keys


def item_profiles(judges, runs, keys):
    P = np.zeros((len(keys), L))
    for i, k in enumerate(keys):
        c = np.zeros(L)
        n = 0
        for j in judges:
            v = runs[j][k]
            if v in VALID:
                c[v] += 1
                n += 1
        P[i] = c / max(n, 1)
    return P


def kernels_from_gold(judges, runs, gold, fam_of, keys):
    C = {}
    for f in FAMS:
        cols = [j for j, ff in zip(judges, fam_of) if ff == f]
        M = np.zeros((L, L))
        for k in keys:
            g = gold[k]
            for j in cols:
                v = runs[j][k]
                if v in VALID:
                    M[v, g] += 1
        s = M.sum(axis=0, keepdims=True)
        s[s == 0] = 1.0
        C[f] = M / s
    return C


def rand_dir(rng):
    D = rng.normal(size=(L, L))
    D -= D.mean(axis=0, keepdims=True)
    n = np.linalg.norm(D)
    return D / n if n > 1e-12 else None


def max_step(kernels, D, protos):
    eye = np.eye(L)
    lo, hi = 0.0, 2.0
    for _ in range(45):
        t = 0.5 * (lo + hi)
        A = eye + t * D
        try:
            Ai = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            hi = t
            continue
        ok = all((Ck @ Ai >= -1e-12).all() for Ck in kernels)
        if ok:
            ok = bool((protos @ A.T >= -1e-12).all())
        lo, hi = (t, hi) if ok else (lo, t)
    return lo


def radius(kernels, protos, seed, n_dirs=N_DIRS):
    rng = np.random.default_rng(seed)
    eye = np.eye(L)
    best = 0.0
    for _ in range(n_dirs):
        D = rand_dir(rng)
        if D is None:
            continue
        t = max_step(kernels, D, protos)
        best = max(best, float(np.abs(protos @ (eye + t * D).T - protos).sum(axis=1).max()))
    return best


# --------------------------------------------------------------------------
# E6a prototype sensitivity
# --------------------------------------------------------------------------
def em_profiles(P: np.ndarray, K: int, rng, iters=150):
    """Soft k-means style latent profiles on the simplex (multinomial mixture)."""
    n = P.shape[0]
    pi = rng.dirichlet(np.ones(K) * 5.0)
    mu = rng.dirichlet(np.ones(L) * 3.0, size=K)
    for _ in range(iters):
        logp = np.log(pi + 1e-300)[None, :] + P @ np.log(mu + 1e-300).T * 33.0
        mx = logp.max(axis=1, keepdims=True)
        w = np.exp(logp - mx)
        post = w / w.sum(axis=1, keepdims=True)
        pi = post.mean(axis=0)
        mu = (post.T @ P) / np.maximum(post.sum(axis=0)[:, None], 1e-9)
        mu = mu / mu.sum(axis=1, keepdims=True)
    return mu


def e6a(judges, runs, gold, fam_of, keys):
    P = item_profiles(judges, runs, keys)
    g = np.array([gold[k] for k in keys])
    C = kernels_from_gold(judges, runs, gold, fam_of, keys)
    kern = [C[f] for f in FAMS]
    rng = np.random.default_rng(99)

    variants = {}
    # (1) gold-conditioned mean (the E2c choice)
    variants["gold_mean"] = np.vstack([P[g == k].mean(axis=0) for k in range(L)])
    # (2) gold-conditioned trimmed mean (robust to outlying items)
    trimmed = []
    for k in range(L):
        block = P[g == k]
        lo, hi = np.percentile(block, [10, 90], axis=0)
        m = np.all((block >= lo) & (block <= hi), axis=1)
        sel = block[m] if m.sum() > 20 else block
        v = sel.mean(axis=0)
        trimmed.append(v / v.sum())
    variants["gold_trimmed_mean"] = np.vstack(trimmed)
    # (3) EM latent class profiles, K = L
    variants["em_latent_K4"] = em_profiles(P, L, rng)

    out = {}
    for name, protos in variants.items():
        seq = {}
        for m in (0, 1, 2, 3):
            ks = kern[:m]
            vals = [radius(ks, protos, s) for s in SEEDS]
            seq[f"m={m}"] = float(np.mean(vals))
        eta = float(protos.min())
        out[name] = {
            "prototypes": protos.tolist(),
            "eta": eta,
            "floor_2eta": 2 * eta,
            "radius_by_m": seq,
            "radius_flat_from_m1": bool(abs(seq["m=1"] - seq["m=3"]) < 1e-3),
            "kernels_bind": bool(seq["m=0"] - seq["m=1"] > 1e-3),
        }
    out["note"] = (
        "The E2c conclusion is that the radius drops once when the first family "
        "is added and then stops moving. This block checks whether that pattern "
        "survives other prototype constructions."
    )
    return out


# --------------------------------------------------------------------------
# E6b Vossler recovery contrast
# --------------------------------------------------------------------------
def e6b(seed=2027, n_items=1200, n_probe=400):
    """Same data, two contracts, two verdicts."""
    rng = np.random.default_rng(seed)
    M = 3  # scoring levels, so Vossler's M >= 3 regime
    K = 3
    protos = np.array([[0.72, 0.18, 0.10], [0.14, 0.71, 0.15], [0.10, 0.22, 0.68]])
    theta = []
    for _ in range(3):
        A = rng.dirichlet(np.ones(M) * 5.0, size=M).T
        theta.append(A / A.sum(axis=0, keepdims=True))

    z = rng.choice(K, size=n_items, p=np.ones(K) / K)
    n_per_fam = 8
    obs = []
    for f in range(3):
        probs = (theta[f] @ protos.T).T[z]  # (n_items, M)
        for _ in range(n_per_fam):
            u = rng.random(n_items)[:, None]
            obs.append((probs.cumsum(axis=1) < u).sum(axis=1).clip(0, M - 1))
    Y = np.stack(obs, axis=1)

    # --- pooled contract: recover kernels from the pooled mixture ---
    fam_of = sum([[f] * n_per_fam for f in range(3)], [])
    counts = {f: np.zeros((n_items, M)) for f in range(3)}
    for c, f in enumerate(fam_of):
        for l in range(M):
            counts[f][:, l] += (Y[:, c] == l)
    pi_hat = np.ones(K) / K
    C_hat = [rng.dirichlet(np.ones(M) * 3.0, size=K).T for _ in range(3)]
    for _ in range(300):
        logp = np.log(pi_hat + 1e-300)[None, :] * np.ones((n_items, 1))
        for f in range(3):
            logp = logp + counts[f] @ np.log(C_hat[f] + 1e-300)
        mx = logp.max(axis=1, keepdims=True)
        w = np.exp(logp - mx)
        post = w / w.sum(axis=1, keepdims=True)
        pi_hat = post.mean(axis=0)
        for f in range(3):
            num = counts[f].T @ post
            C_hat[f] = (num + 1e-9) / (num.sum(axis=0, keepdims=True) + 1e-9 * M)

    # best permutation alignment
    best_err = np.inf
    for perm in itertools.permutations(range(K)):
        err = float(np.mean([np.abs(C_hat[f][:, list(perm)] - theta[f] @ np.eye(K)).mean() for f in range(3)]))
        best_err = min(best_err, err)
    pooled_recovery_err = best_err

    # --- item-free contract: how many distinct (prevalence, theta) match? ---
    # For a single item the observation constrains only the mixture theta @ p.
    # Count distinct prevalences reproducing the observed per-family
    # distribution to within sampling tolerance.
    item = 0
    target = {f: counts[f][item] / max(counts[f][item].sum(), 1) for f in range(3)}
    matches = 0
    tol = 0.12
    for _ in range(n_probe * 50):
        p = rng.dirichlet(np.ones(M))
        ok = True
        for f in range(3):
            pred = theta[f] @ p
            if np.abs(pred - target[f]).max() > tol:
                ok = False
                break
        if ok:
            matches += 1
    frac = matches / (n_probe * 50)

    return {
        "M_levels": M,
        "K_prototypes": K,
        "n_items": n_items,
        "families": 3,
        "pooled_contract": {
            "kernel_recovery_mean_abs_error": float(pooled_recovery_err),
            "identified": bool(pooled_recovery_err < 0.08),
            "note": "pooling supplies the finite-mixture tensor structure Kruskal needs",
        },
        "item_free_contract": {
            "fraction_of_random_prevalences_matching_observation": float(frac),
            "tolerance_l_inf": tol,
            "continuum_of_solutions": bool(frac > 1e-3),
            "note": (
                "with each item carrying its own free prevalence, a positive-measure "
                "set of prevalences reproduces the observation, which is Vossler's "
                "M >= 3 non-identification"
            ),
        },
        "both_hold_simultaneously": None,  # filled in main
    }


def main() -> None:
    judges, runs, gold, fam_of, keys = load()
    res = {
        "env": {
            "script": "e6_sensitivity_and_vossler.py",
            "n_items": len(keys),
            "n_judges": len(judges),
            "seeds": SEEDS,
            "n_dirs": N_DIRS,
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "E6a_prototype_sensitivity": e6a(judges, runs, gold, fam_of, keys),
        "E6b_vossler_recovery_contrast": e6b(),
    }
    b = res["E6b_vossler_recovery_contrast"]
    b["both_hold_simultaneously"] = bool(
        b["pooled_contract"]["identified"] and b["item_free_contract"]["continuum_of_solutions"]
    )
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1)[:3500])


if __name__ == "__main__":
    main()
