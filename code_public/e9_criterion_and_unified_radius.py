"""E9: the L=2 checkable criterion and the unified radius estimator.

  E9a  Demonstration of the L=2 checkable criterion (Proposition F2 binary
       form).  The paper claims that for two labels, whether a new family
       shrinks the gauge set can be decided by arithmetic on likelihood
       ratios:

           shrinks  <=>  r1(g) > max_f r1(f)   or   r0(g) < min_f r0(f)

       where r1 = C[1,1]/C[1,0] and r0 = C[0,1]/C[0,0].  Until now no worked
       instance existed.  We sweep synthetic panels, apply the criterion, and
       check it against the directly measured gauge set.

  E9b  A single radius estimator.  Table 3 currently carries two numbers for
       the same quantity, 1.091 and 1.054, because two blocks used different
       direction budgets and seed handling.  We recompute the radius with one
       estimator, the maximum over the pooled set of directions from all
       seeds, and report the direction budget alongside it.

CPU only.  No API calls.  No GPU.
"""

from __future__ import annotations

import json
import os
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L4_IN = os.path.join(ROOT, "results", "e2c_realdata_L4.json")
OUT = os.path.join(ROOT, "results", "e9_criterion_and_unified_radius.json")

FAMS = ["GPT", "Llama", "T5"]


# --------------------------------------------------------------------------
# E9a: the binary checkable criterion
# --------------------------------------------------------------------------
def ratios(C: np.ndarray):
    """r1 = C[1,1]/C[1,0], r0 = C[0,1]/C[0,0] for a 2x2 column-stochastic C."""
    return C[1, 1] / max(C[1, 0], 1e-12), C[0, 1] / max(C[0, 0], 1e-12)


def criterion_says_shrinks(C_new: np.ndarray, existing: List[np.ndarray]) -> bool:
    r1g, r0g = ratios(C_new)
    r1s = [ratios(c)[0] for c in existing]
    r0s = [ratios(c)[1] for c in existing]
    return bool(r1g > max(r1s) + 1e-12 or r0g < min(r0s) - 1e-12)


def gauge_set_size(kernels: List[np.ndarray], protos: np.ndarray, n_grid: int = 260) -> int:
    """Count admissible gauges on a grid over the 2x2 sum-zero tangent.

    A = I + [[a, -b], [-a, b]] keeps column sums at one.  Admissible means
    C_f A^{-1} >= 0 for every kernel and A p in the simplex for every
    prototype.
    """
    eye = np.eye(2)
    grid = np.linspace(-0.9, 0.9, n_grid)
    count = 0
    for a in grid:
        for b in grid:
            A = eye + np.array([[a, -b], [-a, b]])
            det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
            if abs(det) < 1e-9:
                continue
            Ainv = np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det
            if not all((Ck @ Ainv >= -1e-12).all() for Ck in kernels):
                continue
            if not (protos @ A.T >= -1e-12).all():
                continue
            count += 1
    return count


def e9a(n_cases: int = 60, seed: int = 31415):
    rng = np.random.default_rng(seed)
    protos = np.array([[0.75, 0.25], [0.30, 0.70], [0.55, 0.45]])
    rows = []
    agree = 0
    for _ in range(n_cases):
        base = []
        for _ in range(2):
            d = rng.uniform(0.6, 0.92)
            e = rng.uniform(0.55, 0.9)
            base.append(np.array([[d, 1 - e], [1 - d, e]]))
        d = rng.uniform(0.55, 0.95)
        e = rng.uniform(0.5, 0.95)
        newk = np.array([[d, 1 - e], [1 - d, e]])

        pred = criterion_says_shrinks(newk, base)
        before = gauge_set_size(base, protos)
        after = gauge_set_size(base + [newk], protos)
        observed = after < before
        rows.append(
            {
                "criterion_predicts_shrink": pred,
                "gauge_cells_before": before,
                "gauge_cells_after": after,
                "observed_shrink": bool(observed),
                "match": bool(pred == observed),
            }
        )
        agree += int(pred == observed)
    return {
        "n_cases": n_cases,
        "agreement": agree,
        "agreement_rate": agree / n_cases,
        "n_predicted_shrink": int(sum(r["criterion_predicts_shrink"] for r in rows)),
        "n_observed_shrink": int(sum(r["observed_shrink"] for r in rows)),
        "disagreements": [r for r in rows if not r["match"]][:10],
        "grid": 260,
        "note": (
            "The criterion is compared against a direct grid count of the "
            "admissible gauge set. Disagreements at the boundary are expected "
            "because the grid resolves the set only to its spacing."
        ),
    }


# --------------------------------------------------------------------------
# E9b: one radius estimator
# --------------------------------------------------------------------------
def unified_radius(kernels, protos, seeds, n_dirs_per_seed, dim=4):
    """Maximum displacement over the pooled directions from all seeds."""
    eye = np.eye(dim)
    best = 0.0
    total = 0
    for s in seeds:
        rng = np.random.default_rng(s)
        for _ in range(n_dirs_per_seed):
            D = rng.normal(size=(dim, dim))
            D -= D.mean(axis=0, keepdims=True)
            n = np.linalg.norm(D)
            if n < 1e-12:
                continue
            D /= n
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
            A = eye + lo * D
            best = max(best, float(np.abs(protos @ A.T - protos).sum(axis=1).max()))
            total += 1
    return best, total


def e9b():
    src = json.load(open(L4_IN))
    C = {k: np.array(v) for k, v in src["kernels_from_gold"].items()}
    protos = np.array(src["prototypes"]["profiles"])
    seeds = [42, 123, 456, 789, 1024, 20260804]
    per = 4000
    out = {}
    for m in (0, 1, 2, 3):
        ks = [C[f] for f in FAMS[:m]]
        r, n = unified_radius(ks, protos, seeds, per)
        out[f"m={m}"] = {"radius": r, "n_directions": n}
    seq = [out[f"m={m}"]["radius"] for m in (1, 2, 3)]
    out["estimator"] = (
        "maximum l1 displacement over the pooled directions from six seeds, "
        f"{per} directions each"
    )
    out["kernel_constraints_bind"] = bool(out["m=0"]["radius"] - out["m=1"]["radius"] > 1e-3)
    out["flat_from_m1"] = bool(max(seq) - min(seq) < 1e-6)
    out["supersedes"] = (
        "Table 3 previously reported 1.091 from a three-seed mean of per-seed "
        "maxima and 1.054 from a single-seed maximum. Both were lower bounds on "
        "the same supremum under different budgets. This value replaces them."
    )
    return out


def main() -> None:
    res = {
        "env": {
            "script": "e9_criterion_and_unified_radius.py",
            "answers": "L=2 checkable criterion; unified radius estimator",
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "E9a_binary_criterion": e9a(),
        "E9b_unified_radius": e9b(),
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    a = res["E9a_binary_criterion"]
    b = res["E9b_unified_radius"]
    print(f"E9a: agreement {a['agreement']}/{a['n_cases']} = {a['agreement_rate']:.3f}")
    print(f"     predicted shrink {a['n_predicted_shrink']}, observed {a['n_observed_shrink']}")
    for m in (0, 1, 2, 3):
        print(f"E9b: m={m} radius={b[f'm={m}']['radius']:.6f} (dirs={b[f'm={m}']['n_directions']})")
    print(f"     bind={b['kernel_constraints_bind']} flat_from_m1={b['flat_from_m1']}")


if __name__ == "__main__":
    main()
