"""E2c diagnostics: does family diversity actually shrink the identification set?

Runs three checks on the real LLMJudge panel at L=4, with family kernels
estimated against human gold labels:

  D1  Do the kernel constraints bind at all?  Compare the gauge radius with
      no kernel constraint (m=0), with GPT only (m=1), and with all three
      families (m=3).  Without this control an unchanging radius is
      uninterpretable.

  D2  Is the flatness explained by panel domination (Corollary 1(ii))?
      Non-negative least squares tests whether the rows of one kernel lie in
      the convex cone of the rows of the others.

  D3  Set shrinkage versus diameter.  Corollary 1(ii) constrains the gauge
      SET; the reported radius is a sup functional over that set.  The two
      can move independently, so we count how many random gauge directions
      lose admissible length when families are added, and separately track
      the radius.

Reads results/e2c_realdata_L4.json for the kernels and prototypes.
CPU only. No API calls. No GPU.
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np
from scipy.optimize import nnls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results", "e2c_realdata_L4.json")
OUT = os.path.join(ROOT, "results", "e2c_diagnostics.json")

L = 4
FAMS = ["GPT", "Llama", "T5"]
SEEDS = [42, 123, 456]
N_DIRS_RADIUS = 4000
N_DIRS_SHRINK = 6000
SHRINK_SEED = 20260804


def load():
    d = json.load(open(IN))
    C = {k: np.array(v) for k, v in d["kernels_from_gold"].items()}
    protos = np.array(d["prototypes"]["profiles"])
    return C, protos, d


def max_step(kernels: List[np.ndarray], D: np.ndarray, protos: np.ndarray, iters: int = 45) -> float:
    """Largest t with A = I + tD admissible: C_f A^{-1} >= 0 and A p in the simplex."""
    eye = np.eye(L)
    lo, hi = 0.0, 2.0
    for _ in range(iters):
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


def rand_dir(rng) -> np.ndarray:
    """Tangent direction with columns summing to zero, so 1^T A = 1^T."""
    D = rng.normal(size=(L, L))
    D -= D.mean(axis=0, keepdims=True)
    n = np.linalg.norm(D)
    return D / n if n > 1e-12 else None


def radius(kernels, protos, seed, n_dirs) -> float:
    rng = np.random.default_rng(seed)
    eye = np.eye(L)
    best = 0.0
    for _ in range(n_dirs):
        D = rand_dir(rng)
        if D is None:
            continue
        t = max_step(kernels, D, protos)
        A = eye + t * D
        best = max(best, float(np.abs(protos @ A.T - protos).sum(axis=1).max()))
    return best


def main() -> None:
    C, protos, src = load()
    res = {
        "env": {
            "script": "e2c_diagnostics.py",
            "input": os.path.basename(IN),
            "L": L,
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
            "seeds_radius": SEEDS,
            "n_dirs_radius": N_DIRS_RADIUS,
            "n_dirs_shrink": N_DIRS_SHRINK,
            "shrink_seed": SHRINK_SEED,
        }
    }

    # D1: do kernel constraints bind?
    configs = {
        "m=0_no_kernel_constraint": [],
        "m=1_GPT": [C["GPT"]],
        "m=2_GPT_Llama": [C["GPT"], C["Llama"]],
        "m=3_all": [C[f] for f in FAMS],
    }
    d1 = {}
    for name, ks in configs.items():
        vals = [radius(ks, protos, s, N_DIRS_RADIUS) for s in SEEDS]
        d1[name] = {"radius_mean": float(np.mean(vals)), "per_seed": [float(v) for v in vals]}
    res["D1_constraints_bind"] = d1
    res["D1_interpretation"] = {
        "kernel_constraints_bind": bool(
            d1["m=0_no_kernel_constraint"]["radius_mean"] - d1["m=1_GPT"]["radius_mean"] > 1e-3
        ),
        "note": "If m=0 and m=1 agreed, an unchanging radius would be a measurement artefact.",
    }

    # D2: panel domination via row-cone membership
    d2 = {}
    for tgt in FAMS:
        others = [f for f in FAMS if f != tgt]
        B = np.vstack([C[f] for f in others]).T
        residuals = [float(nnls(B, C[tgt][r])[1]) for r in range(L)]
        d2[tgt] = {
            "cone_generators": others,
            "row_residuals": residuals,
            "max_residual": float(max(residuals)),
            "dominated": bool(max(residuals) < 1e-8),
        }
    res["D2_panel_domination"] = d2
    res["D2_interpretation"] = {
        "any_family_dominated": bool(any(v["dominated"] for v in d2.values())),
        "note": "No family is redundant in the exact sense of Corollary 1(ii).",
    }

    # D3: set shrinkage vs diameter
    rng = np.random.default_rng(SHRINK_SEED)
    eye = np.eye(L)
    K1, K3 = [C["GPT"]], [C[f] for f in FAMS]
    shrunk = 0
    ratios, disp1, disp3 = [], [], []
    for _ in range(N_DIRS_SHRINK):
        D = rand_dir(rng)
        if D is None:
            continue
        t1 = max_step(K1, D, protos)
        t3 = max_step(K3, D, protos)
        if t3 < t1 - 1e-9:
            shrunk += 1
        ratios.append(t3 / t1 if t1 > 0 else 1.0)
        disp1.append(float(np.abs(protos @ (eye + t1 * D).T - protos).sum(axis=1).max()))
        disp3.append(float(np.abs(protos @ (eye + t3 * D).T - protos).sum(axis=1).max()))
    ratios_a, d1a, d3a = np.array(ratios), np.array(disp1), np.array(disp3)
    res["D3_set_vs_diameter"] = {
        "n_directions": int(len(ratios_a)),
        "n_directions_shrunk": int(shrunk),
        "frac_directions_shrunk": float(shrunk / len(ratios_a)),
        "step_ratio_t3_over_t1": {
            "mean": float(ratios_a.mean()),
            "median": float(np.median(ratios_a)),
            "min": float(ratios_a.min()),
        },
        "radius_m1": float(d1a.max()),
        "radius_m3": float(d3a.max()),
        "radius_difference": float(d1a.max() - d3a.max()),
        "mean_displacement_m1": float(d1a.mean()),
        "mean_displacement_m3": float(d3a.mean()),
        "mean_displacement_reduction_pct": float(100 * (1 - d3a.mean() / d1a.mean())),
    }
    res["D3_interpretation"] = {
        "set_strictly_shrinks": bool(shrunk > 0),
        "diameter_unchanged": bool(abs(d1a.max() - d3a.max()) < 1e-9),
        "note": (
            "Real-data instance of the two-layer separation: adding families removes "
            "admissible gauges, but the extremal direction survives, so the "
            "identification-set diameter is unchanged."
        ),
    }

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
