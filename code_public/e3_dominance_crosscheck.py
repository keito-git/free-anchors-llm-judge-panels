"""
E3 Dominance Crosscheck: Independent verification of the analytic panel-domination claim

Three verification points (per arbitration document §2.6):
  1. n_removed = 0 for seed3 6->7 and seed4 8->9 (dominated steps)
  2. n_removed for seed3 2->3 (non-dominated; G~ should shrink at prototype-free level)
  3. LR-range criterion (panel domination) independently computed and cross-tabulated

Independence principle: this script does NOT look at the e3_dominance_verify.py output
to adjust results. Any disagreement is reported as-is.

Theory (analytic derivation):
    Panel domination (L=2):
        r1(f) = beta_f / alpha_f       (likelihood ratio, output "1")
        r0(f) = (1-beta_f)/(1-alpha_f) (likelihood ratio, output "0")
        new kernel g is dominated iff:
            r1(g) <= max_f r1(f)  AND  r0(g) >= min_f r0(f)

    Prediction (unconditional, Proposition F2(a)):
        dominated => G_{m+1} = G_m => n_removed = 0

    Prediction (Proposition F2(b), prototype-free set G~):
        non-dominated => G~_{m+1} strictly contained in G~_m
        But prototype constraints can rescue (n_removed may still be 0 in full G).

Date: 2026-08-04
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))
from e3_experiment import is_valid_gauge, default_kernels

CONV_JSON = BASE / "results" / "e3_convexhull_check.json"
OUT_JSON  = BASE / "results" / "e3_dominance_crosscheck.json"


# ---------------------------------------------------------------------------
# (A) LR-range panel-domination criterion (independent implementation)
# ---------------------------------------------------------------------------

def lr_dominated(
    new_alpha: float,
    new_beta: float,
    existing: List[Dict],
    tol: float = 1e-12,
) -> Dict:
    """Check if new kernel is panel-dominated using LR-range criterion.

    Panel domination (L=2, §2.3 of arbitration document):
        r1(g) = beta_g / alpha_g
        r0(g) = (1-beta_g) / (1-alpha_g)
        dominated iff r1(g) <= max_f r1(f)  AND  r0(g) >= min_f r0(f)

    Args:
        new_alpha, new_beta: parameters of the new kernel
        existing: list of {"alpha": float, "beta": float} for existing kernels
        tol: numerical tolerance

    Returns:
        dict with r1_new, r0_new, r1_max, r0_min, dominated (bool)
    """
    r1_new = new_beta / new_alpha
    r0_new = (1.0 - new_beta) / (1.0 - new_alpha)
    r1_max = max(p["beta"] / p["alpha"] for p in existing)
    r0_min = min((1.0 - p["beta"]) / (1.0 - p["alpha"]) for p in existing)
    dominated = (r1_new <= r1_max + tol) and (r0_new >= r0_min - tol)
    return {
        "r1_new": r1_new, "r1_max": r1_max,
        "r0_new": r0_new, "r0_min": r0_min,
        "dominated": dominated,
    }


# ---------------------------------------------------------------------------
# (B) n_removed measurement (gauge transform count comparison)
# ---------------------------------------------------------------------------

def measure_n_removed(
    kernels_prev: List[Tuple[float, float]],
    kernels_curr: List[Tuple[float, float]],
    n_grid: int = 200,
    eps: float = 1e-4,
) -> Dict:
    """Count gauge transforms removed by adding a new kernel.

    Scans (a, b) grid and counts transforms valid in kernels_prev but not kernels_curr.

    Args:
        kernels_prev: list of (alpha, beta) for m-1 kernels
        kernels_curr: list of (alpha, beta) for m kernels (= kernels_prev + new)
        n_grid: grid resolution per axis
        eps: boundary epsilon

    Returns:
        dict with n_prev, n_curr, n_removed, n_added (should be 0)
    """
    a_vals = np.linspace(eps, 1.0 - eps, n_grid)
    b_vals = np.linspace(eps, 1.0 - eps, n_grid)

    n_prev = n_curr = n_removed = n_added = 0
    for a in a_vals:
        for b in b_vals:
            if abs(a - b) < eps:
                continue
            vp = is_valid_gauge(a, b, kernels_prev)
            vc = is_valid_gauge(a, b, kernels_curr)
            if vp:
                n_prev += 1
            if vc:
                n_curr += 1
            if vp and not vc:
                n_removed += 1
            if vc and not vp:
                n_added += 1  # should always be 0 (monotone)

    return {
        "n_prev": n_prev,
        "n_curr": n_curr,
        "n_removed": n_removed,
        "n_added": n_added,  # sanity: must be 0
        "n_grid": n_grid,
    }


# ---------------------------------------------------------------------------
# Main crosscheck
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== E3 Dominance Crosscheck (independent implementation) ===\n")
    t0 = time.time()

    # Load convexhull check data for kernel values and step metadata
    with open(CONV_JSON) as f:
        conv = json.load(f)

    # -----------------------------------------------------------------------
    # PART 1: LR-range criterion independently computed for all 40 steps
    # -----------------------------------------------------------------------
    print("[Part 1] LR-range panel-domination over all 40 steps")
    print(f"  {'seed':>4} {'step':>7} {'r1_new':>8} {'r1_max':>8} {'r0_new':>8} {'r0_min':>8} "
          f"{'dom(this run)':>9} {'dom(analytic)':>19}")

    my_rows = []
    for seed_str, sd in conv["per_seed"].items():
        kernels_list = sd["kernels"]  # list of {"alpha": float, "beta": float}
        for st in sd["steps"]:
            m_prev = st["m_prev"]
            a_g = st["new_kernel_alpha"]
            b_g = st["new_kernel_beta"]
            existing = kernels_list[:m_prev]
            lr = lr_dominated(a_g, b_g, existing)
            my_dominated = lr["dominated"]

            # Analytic dominance is not directly stored in convexhull_check.json,
            # so we compute it independently; the analytic prediction is 3 dominated steps.
            my_rows.append({
                "seed": seed_str,
                "step": st["step"],
                "m_prev": m_prev,
                "r1_new": lr["r1_new"], "r1_max": lr["r1_max"],
                "r0_new": lr["r0_new"], "r0_min": lr["r0_min"],
                "dominated_mine": my_dominated,
                "diam_change": st["mean_diam_change"],
                "diam_shrank": st["diameter_shrank"],
            })

            print(f"  {seed_str:>4} {st['step']:>7} {lr['r1_new']:8.4f} {lr['r1_max']:8.4f} "
                  f"{lr['r0_new']:8.4f} {lr['r0_min']:8.4f} "
                  f"{'dom' if my_dominated else 'non':>9}")

    # Cross-tab (mine)
    n_dom = sum(1 for r in my_rows if r["dominated_mine"])
    n_nondom = len(my_rows) - n_dom
    print(f"\n  Dominated:     {n_dom}")
    print(f"  Non-dominated: {n_nondom}")

    dom_steps = [(r["seed"], r["step"]) for r in my_rows if r["dominated_mine"]]
    print(f"  Dominated steps: {dom_steps}")

    # Consistency: dominated => no shrink
    viol = [r for r in my_rows if r["dominated_mine"] and r["diam_shrank"]]
    print(f"  Violations 'dominated => no shrink': {len(viol)}")

    # Non-dominated + diam unchanged
    nd_nochange = [r for r in my_rows if not r["dominated_mine"] and not r["diam_shrank"]]
    print(f"  Non-dominated but diam unchanged: {len(nd_nochange)}: "
          + str([(r["seed"], r["step"]) for r in nd_nochange]))

    # -----------------------------------------------------------------------
    # PART 2: n_removed for the two unreported dominated steps
    # (seed3 6->7 and seed4 8->9)
    # -----------------------------------------------------------------------
    print("\n[Part 2] n_removed for dominated steps not yet measured")
    dominated_to_check = [
        ("3", 6, 7),  # seed3 6->7
        ("4", 8, 9),  # seed4 8->9
    ]

    n_removed_results = {}
    for seed_str, m_prev, m_curr in dominated_to_check:
        rng = np.random.default_rng(int(seed_str))
        kernels_max = default_kernels(10, rng, seed=int(seed_str))
        kp = kernels_max[:m_prev]
        kc = kernels_max[:m_curr]
        print(f"  seed={seed_str} step {m_prev}->{m_curr}: scanning (n_grid=200)...")
        result = measure_n_removed(kp, kc, n_grid=200)
        n_removed_results[f"seed{seed_str}_step{m_prev}to{m_curr}"] = result
        print(f"    n_prev={result['n_prev']}, n_curr={result['n_curr']}, "
              f"n_removed={result['n_removed']}, n_added={result['n_added']}")
        theory_ok = (result["n_removed"] == 0)
        print(f"    Theory (dominated => n_removed=0): {'CONFIRMED' if theory_ok else 'VIOLATED'}")

    # -----------------------------------------------------------------------
    # PART 3: n_removed for seed3 2->3 (non-dominated but diam unchanged)
    # -----------------------------------------------------------------------
    print("\n[Part 3] n_removed for seed3 2->3 (non-dominated, diam=0)")
    seed_str, m_prev, m_curr = "3", 2, 3
    rng = np.random.default_rng(3)
    kernels_max = default_kernels(10, rng, seed=3)
    kp = kernels_max[:m_prev]
    kc = kernels_max[:m_curr]
    print(f"  seed={seed_str} step {m_prev}->{m_curr}: scanning (n_grid=200)...")
    result_23 = measure_n_removed(kp, kc, n_grid=200)
    n_removed_results[f"seed{seed_str}_step{m_prev}to{m_curr}"] = result_23
    print(f"    n_prev={result_23['n_prev']}, n_curr={result_23['n_curr']}, "
          f"n_removed={result_23['n_removed']}, n_added={result_23['n_added']}")

    if result_23["n_removed"] > 0:
        print(f"    Interpretation: G~_3 strictly smaller than G~_2 (n_removed={result_23['n_removed']}). "
              "Diameter unchanged because removed transforms are not on extremal orbits. "
              "Same pattern as seed0 5->6 (14 removed, diam=0). "
              "Consistent with Proposition F2(b) + §2.5 two-layer separation.")
    else:
        print(f"    Interpretation: n_removed=0. Either prototype constraints already excluded "
              "the region (Proposition F2(c)), or grid resolution insufficient. "
              "Would need LP/finer grid to distinguish from F2(c) case.")

    # -----------------------------------------------------------------------
    # Comparison with the analytic prediction
    # -----------------------------------------------------------------------
    print("\n[Cross-check vs analytic prediction]")
    analytic_dominated_steps = {("0", "3 -> 4"), ("3", "6 -> 7"), ("4", "8 -> 9")}
    my_dominated_set = {(r["seed"], r["step"]) for r in my_rows if r["dominated_mine"]}
    matches = (my_dominated_set == analytic_dominated_steps)
    print(f"  Analytic dominated steps: {sorted(analytic_dominated_steps)}")
    print(f"  My calculation:             {sorted(my_dominated_set)}")
    print(f"  Exact match: {matches}")
    if not matches:
        print("  DISCREPANCY DETECTED — reporting as-is (not adjusting).")
        print(f"  Only in analytic: {analytic_dominated_steps - my_dominated_set}")
        print(f"  Only in mine:    {my_dominated_set - analytic_dominated_steps}")

    analytic_violations = 0
    my_violations = len(viol)
    print(f"\n  Analytic violations count: {analytic_violations}")
    print(f"  My violations count:    {my_violations}")
    print(f"  Agree on zero violations: {my_violations == analytic_violations}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    elapsed = round(time.time() - t0, 2)
    output = {
        "script": "e3_dominance_crosscheck.py (independent implementation)",
        "date": "2026-08-04",
        "elapsed_seconds": elapsed,
        "part1_lr_range": {
            "all_steps": my_rows,
            "n_dominated": n_dom,
            "n_nondominated": n_nondom,
            "dominated_steps": [(r["seed"], r["step"]) for r in my_rows if r["dominated_mine"]],
            "violations_dominated_implies_no_shrink": len(viol),
            "nd_diam_unchanged": [(r["seed"], r["step"]) for r in my_rows if not r["dominated_mine"] and not r["diam_shrank"]],
        },
        "part2_n_removed_dominated_steps": n_removed_results,
        "part3_n_removed_seed3_2to3": result_23,
        "crosscheck_vs_analytic": {
            "dominated_sets_match": matches,
            "analytic_dominated": [list(s) for s in sorted(analytic_dominated_steps)],
            "mine_dominated": [list(s) for s in sorted(my_dominated_set)],
            "violations_mine": my_violations,
            "violations_analytic": analytic_violations,
            "zero_violations_agree": my_violations == analytic_violations,
        },
        "verdict": {
            "check1_seed3_67": (
                f"seed3 6->7: n_removed={n_removed_results.get('seed3_step6to7', {}).get('n_removed', 'N/A')}. "
                "Theory predicts 0. " + (
                    "CONFIRMED" if n_removed_results.get("seed3_step6to7", {}).get("n_removed") == 0
                    else "VIOLATED"
                )
            ),
            "check1_seed4_89": (
                f"seed4 8->9: n_removed={n_removed_results.get('seed4_step8to9', {}).get('n_removed', 'N/A')}. "
                "Theory predicts 0. " + (
                    "CONFIRMED" if n_removed_results.get("seed4_step8to9", {}).get("n_removed") == 0
                    else "VIOLATED"
                )
            ),
            "check2_seed3_23": (
                f"seed3 2->3: n_removed={result_23['n_removed']}. Non-dominated. "
                "If > 0: G~ shrinks (diam unchanged = layer-2 separation, consistent with §2.5). "
                "If = 0: prototype constraint case (F2(c))."
            ),
            "check3_lr_criterion": (
                f"LR criterion: {n_dom} dominated steps identified, exact match with analytic prediction: {matches}. "
                f"Zero violations of 'dominated => no shrink': {my_violations == 0}."
            ),
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUT_JSON}")
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
