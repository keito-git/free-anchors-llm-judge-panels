"""
E3 Convex Hull Check: Verify that G_m shrinkage matches convex hull structure

Theory (Corollary 1(i), systematic formulation):
    Adding a new family f_new to the panel shrinks the gauge set G_m ONLY IF
    C_{f_new} is outside the convex hull of existing kernels {C_{f_1}, ..., C_{f_{m-1}}}.
    If C_{f_new} is inside the convex hull, the new constraint is redundant
    (dominated by an existing constraint), and G_m = G_{m-1}.

    For L=2, C_f is parameterized by (alpha, beta):
        C_f = [[1-alpha, 1-beta], [alpha, beta]]
    C_f is affine in (alpha, beta), so the convex hull in matrix space corresponds
    exactly to the convex hull in (alpha, beta) parameter space.

    Key prediction: gauge diameter decreases <=> new kernel outside hull.

Experimental check:
    - Use default_kernels(m_max=10, rng, seed) for each of 5 seeds
    - For each step m-1 -> m: check if kernel_{m-1} (0-indexed) is outside
      the convex hull of {kernel_0, ..., kernel_{m-2}}
    - Measure gauge diameter change delta_diam = diam(m-1) - diam(m)
    - Verify: outside hull <=> delta_diam > 0 (diameter shrinks)

Note on kernel generation:
    default_kernels uses linspace + jitter, placing kernels approximately
    along a diagonal in (alpha, beta) space. Each added kernel extends beyond
    the previous set, so all additions should be outside-hull.

Date: 2026-08-04
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE       = Path(__file__).resolve().parents[1]
OUT_JSON   = BASE / "results" / "e3_convexhull_check.json"
AGG_PATH   = BASE / "results" / "e3_aggregated.json"

# Import from sibling modules
sys.path.insert(0, str(BASE / "code"))
from e3_experiment import gauge_orbit_diameter, default_kernels


# ---------------------------------------------------------------------------
# Kernel convex hull utilities
# ---------------------------------------------------------------------------

def point_in_convex_hull_2d(
    point: Tuple[float, float],
    hull_points: List[Tuple[float, float]],
    tol: float = 1e-8,
) -> bool:
    """Check if a 2D point is inside (or on boundary of) the convex hull.

    Handles degenerate cases:
      - 0 points: always outside
      - 1 point:  inside iff coincident
      - 2 points: inside iff on the line segment
      - 3+ points: use scipy ConvexHull

    Args:
        point: (alpha, beta) to check
        hull_points: existing (alpha, beta) values
        tol: numerical tolerance for boundary membership

    Returns:
        True if point is inside (or on boundary of) the convex hull.
    """
    n = len(hull_points)
    px, py = point

    if n == 0:
        return False

    if n == 1:
        hx, hy = hull_points[0]
        return abs(px - hx) < tol and abs(py - hy) < tol

    if n == 2:
        # Check if point is on segment [p1, p2]
        x1, y1 = hull_points[0]
        x2, y2 = hull_points[1]
        # Parametric: (x1+t*(x2-x1), y1+t*(y2-y1)) = (px, py)
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < tol and abs(dy) < tol:
            # Degenerate: both endpoints same
            return abs(px - x1) < tol and abs(py - y1) < tol
        if abs(dx) > abs(dy):
            t = (px - x1) / dx
            return (0.0 - tol <= t <= 1.0 + tol) and abs(y1 + t * dy - py) < tol
        else:
            t = (py - y1) / dy
            return (0.0 - tol <= t <= 1.0 + tol) and abs(x1 + t * dx - px) < tol

    # n >= 3: check using convex hull half-plane constraints
    pts = np.array(hull_points)
    try:
        hull = ConvexHull(pts)
    except QhullError:
        # Degenerate: all points collinear — treat as n=2 segment check
        # with min/max endpoints
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        # Check if point is within bounding segment
        p = np.array([px, py])
        # Project and check
        for dim in range(2):
            if maxs[dim] - mins[dim] > tol:
                # Use this dimension as the primary
                t_vals = (pts[:, dim] - mins[dim]) / (maxs[dim] - mins[dim])
                t_p = (p[dim] - mins[dim]) / (maxs[dim] - mins[dim])
                if not (-tol <= t_p <= 1.0 + tol):
                    return False
                # Check cross-dimension
                other_dim = 1 - dim
                other_interp = np.interp(p[dim], pts[:, dim], pts[:, other_dim])
                if abs(p[other_dim] - other_interp) > tol * 10:
                    return False
                return True
        return True  # all points same

    # Check half-plane constraints: hull.equations is (n_facets, 3)
    # Constraint: equations @ [x, y, 1] <= 0 for interior
    p_ext = np.array([px, py, 1.0])
    return bool(np.all(hull.equations @ p_ext <= tol))


def classify_kernels(
    kernels_max: List[Tuple[float, float]],
    m_values: List[int],
    n_grid: int = 80,
) -> List[Dict]:
    """For each step m-1 -> m, classify new kernel as inside/outside hull.

    Also measures gauge orbit diameter at m-1 and m using a representative pi.

    Args:
        kernels_max: full list (length = max(m_values)) of (alpha, beta) tuples
        m_values: sorted list of m values to step through [2, 3, 4, ...]
        n_grid: grid for gauge diameter measurement

    Returns:
        List of per-step dicts.
    """
    representative_pis = [0.20, 0.25, 0.33, 0.40, 0.50]  # typical interior values
    results = []

    # Compute diameter at m=1 baseline (just 1 kernel, largest feasible orbit)
    prev_m = m_values[0]
    prev_kernels = kernels_max[:prev_m]

    for m in m_values[1:]:
        new_kernel = kernels_max[m - 1]  # 0-indexed
        existing_kernels = kernels_max[:m - 1]

        is_inside = point_in_convex_hull_2d(new_kernel, existing_kernels)
        is_outside = not is_inside

        # Measure gauge diameter change at a range of pi values
        diam_changes = []
        for pi in representative_pis:
            r_prev = gauge_orbit_diameter(prev_kernels, pi, n_grid=n_grid)
            r_curr = gauge_orbit_diameter(kernels_max[:m], pi, n_grid=n_grid)
            diam_changes.append(r_prev["max_delta"] - r_curr["max_delta"])

        mean_diam_change = float(np.mean(diam_changes))
        diameter_shrank = bool(mean_diam_change > 1e-4)

        # Prediction: outside hull <=> diameter shrinks
        prediction_matches = (is_outside == diameter_shrank)

        results.append({
            "step": f"{prev_m} -> {m}",
            "m_prev": prev_m,
            "m_curr": m,
            "new_kernel_alpha": float(new_kernel[0]),
            "new_kernel_beta": float(new_kernel[1]),
            "existing_hull_size": len(existing_kernels),
            "new_kernel_outside_hull": is_outside,
            "mean_diam_change": mean_diam_change,
            "diameter_shrank": diameter_shrank,
            "prediction_matches": prediction_matches,
        })

        prev_m = m
        prev_kernels = kernels_max[:m]

    return results


# ---------------------------------------------------------------------------
# Check convex hull positions for all seeds
# ---------------------------------------------------------------------------

def run_convexhull_check(
    seeds: List[int],
    m_max: int = 10,
    m_values: List[int] = None,
) -> Dict:
    """Run convex hull check across all seeds.

    Args:
        seeds: list of random seeds
        m_max: total number of kernels to generate
        m_values: m values to step through (default: 2..m_max)

    Returns:
        dict with per-seed and aggregate results
    """
    if m_values is None:
        m_values = list(range(2, m_max + 1))

    per_seed = {}
    all_step_results = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        kernels_max = default_kernels(m_max, rng, seed=seed)
        step_results = classify_kernels(kernels_max, m_values, n_grid=80)
        per_seed[str(seed)] = {
            "kernels": [{"alpha": a, "beta": b} for a, b in kernels_max],
            "steps": step_results,
            "all_outside_hull": all(s["new_kernel_outside_hull"] for s in step_results),
            "all_predictions_match": all(s["prediction_matches"] for s in step_results),
        }
        all_step_results.extend(step_results)

    # Aggregate across seeds
    n_outside = sum(1 for s in all_step_results if s["new_kernel_outside_hull"])
    n_inside  = sum(1 for s in all_step_results if not s["new_kernel_outside_hull"])
    n_match   = sum(1 for s in all_step_results if s["prediction_matches"])
    n_total   = len(all_step_results)

    return {
        "per_seed": per_seed,
        "aggregate": {
            "n_steps_total": n_total,
            "n_outside_hull": n_outside,
            "n_inside_hull": n_inside,
            "n_predictions_match": n_match,
            "pct_outside_hull": round(100.0 * n_outside / n_total, 1),
            "pct_predictions_match": round(100.0 * n_match / n_total, 1),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== E3 Convex Hull Check ===")
    t0 = time.time()

    SEEDS    = [0, 1, 2, 3, 4]
    M_MAX    = 10
    M_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"  Seeds: {SEEDS}, M_MAX: {M_MAX}")
    print(f"  Steps: {M_VALUES[0]-1} -> {M_VALUES[-1]}")

    results = run_convexhull_check(SEEDS, M_MAX, M_VALUES)

    agg = results["aggregate"]
    print(f"\n[Results]")
    print(f"  Total steps: {agg['n_steps_total']}")
    print(f"  Outside hull: {agg['n_outside_hull']} ({agg['pct_outside_hull']}%)")
    print(f"  Inside hull:  {agg['n_inside_hull']}")
    print(f"  Predictions match: {agg['n_predictions_match']} ({agg['pct_predictions_match']}%)")

    # Per-seed summary
    print(f"\n  Per-seed summary:")
    for seed, sr in results["per_seed"].items():
        all_out = sr["all_outside_hull"]
        all_match = sr["all_predictions_match"]
        print(f"    seed={seed}: all_outside_hull={all_out}, all_predictions_match={all_match}")

    # Detailed step view (seed 0)
    print(f"\n  Step detail (seed=0):")
    print(f"  {'Step':12s} {'Outside?':10s} {'DeltaDiam':10s} {'Match?':8s}")
    for step in results["per_seed"]["0"]["steps"]:
        print(
            f"  {step['step']:12s} "
            f"{'YES' if step['new_kernel_outside_hull'] else 'NO ':10s} "
            f"{step['mean_diam_change']:+.4f}   "
            f"{'OK' if step['prediction_matches'] else 'FAIL':8s}"
        )

    # Theory verdict
    n_in = agg["n_inside_hull"]
    n_match = agg["n_predictions_match"]
    n_total = agg["n_steps_total"]

    if n_in == 0:
        verdict = (
            "All kernel additions are OUTSIDE the convex hull of previous kernels "
            f"({n_total}/{n_total} steps). "
            "This confirms default_kernels generates an expanding sequence, "
            "consistent with monotone gauge shrinkage observed in M2. "
            "The condition for Corollary 1(i) is satisfied in E3's experimental setup. "
            "However, this means E3 does NOT test the 'inside-hull -> no shrinkage' case. "
            "A dedicated inside-hull test would be needed to fully verify Corollary 1(i)."
        )
    elif n_in > 0 and n_match == n_total:
        verdict = (
            f"{n_in} inside-hull additions found. "
            "Theory prediction matches in ALL cases: "
            "outside-hull -> diameter shrinks, inside-hull -> diameter does not shrink. "
            "Corollary 1(i) is confirmed."
        )
    else:
        n_mismatch = n_total - n_match
        verdict = (
            f"WARNING: {n_mismatch} prediction mismatches found. "
            f"Inside-hull additions: {n_in}. "
            "Theory may need revision."
        )

    results["verdict"] = verdict
    results["elapsed_seconds"] = round(time.time() - t0, 2)

    print(f"\n[Verdict]")
    print(f"  {verdict}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {OUT_JSON}")
    print(f"Done in {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
