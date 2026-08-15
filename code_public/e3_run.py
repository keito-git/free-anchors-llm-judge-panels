"""
E3: Main experiment runner

Runs all three measurements (M2, M3, M4) with 5 seeds.
Saves results to results/e3_*.json.
Produces env_log manifest for reproducibility.

Usage:
    python e3_run.py [--dry-run] [--n-interior N] [--n-vertex N]

Date: 2026-08-04
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add code dir to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent))

from e3_profiles import (
    load_diversity_qrels_2009,
    build_profiles_uniform,
    build_profiles_count_weighted,
    summarize_profiles,
    sample_items,
)
from e3_experiment import (
    default_kernels,
    load_external_kernels,
    measurement_gauge_nonidentification,
    measurement_anchor_budget,
    measurement_structural_anchor,
    gauge_loglikelihood_flatness,
    fit_power_law,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "trec_web_diversity"
RESULTS_DIR = BASE_DIR / "results"
CODE_DIR = BASE_DIR / "code"

SEEDS = [0, 1, 2, 3, 4]  # 5 independent seeds (requirement)

# Experiment parameters
N_INTERIOR = 200        # interior items to include in experiments
N_VERTEX_0 = 100        # vertex-0 items (for structural anchor pool)
N_VERTEX_1 = 20         # vertex-1 items (rare in 2009 data: only 28 exist)
N_FAMILIES_MAX = 10     # maximum evaluator families

M_VALUES = [2, 3, 5, 7, 10]   # m values for gauge diameter test
N_ANCHOR_VALUES = [5, 10, 20, 50, 100, 200]  # anchor counts for budget curve
N_VERTEX_VALUES = [2, 4, 6, 10, 20]          # structural anchor counts

N_JUDGMENTS = 100        # judgments per (family, item)
N_GAUGE_GRID = 120       # grid resolution for gauge orbit scan
N_REP_BUDGET = 30        # repetitions for anchor budget
N_REP_STRUCTURAL = 20   # repetitions for structural anchor


# ---------------------------------------------------------------------------
# Environment logging
# ---------------------------------------------------------------------------


def get_data_hash(path: Path) -> str:
    """SHA256 of first 64KB of file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(65536))
    return h.hexdigest()[:16]


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(BASE_DIR)
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_env_log(seeds: List[int], qrel_path: Path) -> Dict:
    """Build reproducibility manifest."""
    return {
        "experiment": "E3_contract_b_semi_synthetic",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": _get_version("scipy"),
        "matplotlib_version": _get_version("matplotlib"),
        "git_commit": get_git_commit(),
        "seeds": seeds,
        "data_files": {
            "2009qrels.diversity": {
                "path": str(qrel_path),
                "sha256_prefix": get_data_hash(qrel_path),
                "license": "U.S. Government work (NIST TREC) — public domain",
            }
        },
        "gpu": "NONE — CPU only (GPU not authorized)",
        "api_calls": "NONE — zero cost",
        "p_intent_assumption": "uniform: P(i|q) = 1/|I_q| for all intents i",
        "note_ntcir_intent2": (
            "NTCIR INTENT-2 not yet available (application pending). "
            "When acquired, replace uniform P(i|q) with voted probabilities. "
            "Sensitivity check with count-weighted P(i|q) included."
        ),
        "note_clueweb": (
            "No ClueWeb document text used. "
            "Experiment uses qrels (document IDs + judgments) only."
        ),
    }


def _get_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------


def run_seed(
    seed: int,
    profiles_uniform: list,
    profiles_count: list,
    n_interior: int,
    n_vertex_0: int,
    n_vertex_1: int,
    m_values: List[int],
    external_kernels_path: Optional[str],
    dry_run: bool = False,
) -> Dict:
    """Run full E3 experiment for a single seed."""
    rng = np.random.default_rng(seed)
    logger.info("=== Seed %d ===", seed)

    # Sample items
    items = sample_items(
        profiles_uniform,
        n_interior=n_interior,
        n_vertex_0=n_vertex_0,
        n_vertex_1=n_vertex_1,
        rng=rng,
    )
    interior = items["interior"]
    vertex_0 = items["vertex_0"]
    vertex_1 = items["vertex_1"]

    logger.info(
        "Items: %d interior, %d vertex_0, %d vertex_1",
        len(interior), len(vertex_0), len(vertex_1),
    )

    if len(vertex_1) == 0:
        logger.warning(
            "ZERO vertex_1 items available — structural anchor M4 cannot run for seed %d. "
            "This is a data limitation: only 28 vertex_1 docs exist in 2009 qrels.",
            seed,
        )

    # Build kernels (injection point for E2 calibration)
    ext = load_external_kernels(external_kernels_path) if external_kernels_path else None
    if ext:
        logger.info("Using external (E2-calibrated) kernels: %d families", len(ext))
        kernels_max = ext
    else:
        kernels_max = default_kernels(N_FAMILIES_MAX, rng, seed=seed)
    logger.info("Kernel alpha/beta ranges: alpha=%.2f–%.2f, beta=%.2f–%.2f",
                min(k[0] for k in kernels_max), max(k[0] for k in kernels_max),
                min(k[1] for k in kernels_max), max(k[1] for k in kernels_max))

    if dry_run:
        logger.info("Dry-run mode: skipping computations")
        return {"seed": seed, "dry_run": True}

    # ------- M1: interiority (computed globally, just record for this seed) -------
    stats_u = summarize_profiles(profiles_uniform)
    stats_c = summarize_profiles(profiles_count)
    m1_result = {
        "uniform_assumption": stats_u,
        "count_weighted_sensitivity": stats_c,
        "interior_pct_uniform": stats_u["pct_interior"],
        "interior_pct_count": stats_c["pct_interior"],
    }

    # ------- M2: gauge non-identification -------
    logger.info("M2: Gauge non-identification (m values: %s)", m_values)
    m_vals_valid = [m for m in m_values if m <= len(kernels_max)]
    m2_result = measurement_gauge_nonidentification(
        interior_profiles=interior[:50],  # use 50 interior items
        kernels=kernels_max,
        m_values=m_vals_valid,
        n_judgments=N_JUDGMENTS,
        n_grid=N_GAUGE_GRID,
    )

    # Gauge flatness demonstration (for one representative interior item)
    rep_profile = interior[0]
    flatness = gauge_loglikelihood_flatness(
        pi=rep_profile.pi,
        kernels=kernels_max[:3],
        n_judgments=N_JUDGMENTS * 5,
        rng=rng,
        n_gauge_points=30,
        n_grid=80,
    )
    m2_result["flatness_demo"] = {
        "pi_true": rep_profile.pi,
        "max_abs_loglik_diff": flatness["max_abs_loglik_diff"],
        "mean_abs_loglik_diff": flatness["mean_abs_loglik_diff"],
        "n_gauge_points": flatness["n_gauge_points"],
    }

    # ------- M3: anchor budget curve -------
    logger.info("M3: Anchor budget curve")
    anchor_pool = interior + vertex_0 + vertex_1  # can anchor any type
    n_anchor_vals = [n for n in N_ANCHOR_VALUES if n <= len(anchor_pool)]
    m3_result = measurement_anchor_budget(
        anchor_pool=anchor_pool,
        interior_profiles=interior[:50],
        kernels_true=kernels_max[:3],
        n_anchor_values=n_anchor_vals,
        n_judgments=N_JUDGMENTS,
        n_rep=N_REP_BUDGET,
        rng=rng,
    )

    # Fit power law
    n_vals_sorted = sorted(m3_result.keys())
    rmse_means = [m3_result[n]["mean_rmse"] for n in n_vals_sorted]
    c_fit, r2_fit = fit_power_law(n_vals_sorted, rmse_means)
    m3_result["power_law_fit"] = {
        "c": c_fit,
        "r_squared": r2_fit,
        "model": "RMSE = c * n^{-0.5}",
        "note": "Theoretical prediction from T2 (Le Cam lower bound)",
    }
    logger.info("M3 power law fit: c=%.4f, R²=%.4f", c_fit, r2_fit)

    # ------- M4: structural anchor effect -------
    logger.info("M4: Structural anchor effect")
    if len(vertex_1) >= 1:
        n_v_vals = [n for n in N_VERTEX_VALUES if n // 2 <= min(len(vertex_0), len(vertex_1))]
        m4_result = measurement_structural_anchor(
            interior_profiles=interior[:50],
            vertex_0=vertex_0,
            vertex_1=vertex_1,
            kernels_true=kernels_max[:3],
            n_vertex_values=n_v_vals,
            n_judgments=N_JUDGMENTS,
            n_rep=N_REP_STRUCTURAL,
            rng=rng,
        )
    else:
        m4_result = {
            "WARNING": (
                "Cannot run M4: no vertex_1 items sampled in this seed. "
                "Only 28 vertex_1 docs exist in 2009 qrels (0.1% of profiles). "
                "NTCIR INTENT-2 acquisition would provide more balanced vertex coverage. "
                "Using 2010 or 2011 qrels for M4 is recommended if vertex_1 is critical."
            )
        }
        logger.warning("M4 skipped for seed %d: no vertex_1 items", seed)

    return {
        "seed": seed,
        "n_interior": len(interior),
        "n_vertex_0": len(vertex_0),
        "n_vertex_1": len(vertex_1),
        "kernels_true": kernels_max,
        "M1_interiority": m1_result,
        "M2_gauge_nonidentification": m2_result,
        "M3_anchor_budget": m3_result,
        "M4_structural_anchor": m4_result,
    }


# ---------------------------------------------------------------------------
# Aggregate across seeds
# ---------------------------------------------------------------------------


def aggregate_results(seed_results: List[Dict]) -> Dict:
    """Compute mean ± std across seeds for key metrics."""

    def mean_std(vals: List[float]) -> Dict:
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    agg: Dict[str, Any] = {}

    # M1: interiority
    pct_interior = [r["M1_interiority"]["interior_pct_uniform"] for r in seed_results if not r.get("dry_run")]
    if pct_interior:
        agg["M1_pct_interior"] = mean_std(pct_interior)

    # M2: gauge diameter vs m
    m_vals_all: Dict[int, List[float]] = {}
    for r in seed_results:
        if r.get("dry_run"):
            continue
        m2 = r.get("M2_gauge_nonidentification", {})
        for m, info in m2.items():
            if isinstance(info, dict) and "mean_diameter" in info:
                m_vals_all.setdefault(int(m), []).append(info["mean_diameter"])
    agg["M2_gauge_diameter"] = {
        str(m): mean_std(vals) for m, vals in sorted(m_vals_all.items())
    }

    # Flatness max abs diff
    flatness_diffs = [
        r["M2_gauge_nonidentification"].get("flatness_demo", {}).get("max_abs_loglik_diff", None)
        for r in seed_results if not r.get("dry_run")
    ]
    flatness_diffs = [d for d in flatness_diffs if d is not None]
    if flatness_diffs:
        agg["M2_flatness_max_abs_loglik_diff"] = mean_std(flatness_diffs)

    # M3: RMSE vs n_anchors
    n_vals_all: Dict[int, List[float]] = {}
    for r in seed_results:
        if r.get("dry_run"):
            continue
        m3 = r.get("M3_anchor_budget", {})
        for n_a, info in m3.items():
            if isinstance(info, dict) and "mean_rmse" in info:
                n_vals_all.setdefault(int(n_a), []).append(info["mean_rmse"])
    agg["M3_rmse_vs_n"] = {
        str(n): mean_std(vals) for n, vals in sorted(n_vals_all.items())
    }

    # Power law fit across seeds
    c_vals = [
        r["M3_anchor_budget"].get("power_law_fit", {}).get("c", None)
        for r in seed_results if not r.get("dry_run")
    ]
    r2_vals = [
        r["M3_anchor_budget"].get("power_law_fit", {}).get("r_squared", None)
        for r in seed_results if not r.get("dry_run")
    ]
    c_vals = [v for v in c_vals if v is not None]
    r2_vals = [v for v in r2_vals if v is not None]
    if c_vals:
        agg["M3_power_law"] = {
            "c": mean_std(c_vals),
            "r_squared": mean_std(r2_vals),
        }

    # M4: structural anchor RMSE vs n_vertex
    nv_vals_with: Dict[int, List[float]] = {}
    nv_vals_without: Dict[int, List[float]] = {}
    for r in seed_results:
        if r.get("dry_run"):
            continue
        m4 = r.get("M4_structural_anchor", {})
        for nv, info in m4.items():
            if isinstance(info, dict) and "mean_rmse_with" in info:
                nv_vals_with.setdefault(int(nv), []).append(info["mean_rmse_with"])
                nv_vals_without.setdefault(int(nv), []).append(info["mean_rmse_without"])
    agg["M4_rmse_with_structural"] = {
        str(nv): mean_std(vals) for nv, vals in sorted(nv_vals_with.items())
    }
    agg["M4_rmse_without_structural"] = {
        str(nv): mean_std(vals) for nv, vals in sorted(nv_vals_without.items())
    }

    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="E3 Contract B experiment runner")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without running experiments")
    parser.add_argument("--n-interior", type=int, default=N_INTERIOR)
    parser.add_argument("--n-vertex", type=int, default=None, help="Override N_VERTEX_0")
    parser.add_argument("--external-kernels", type=str, default=None,
                        help="Path to E2-calibrated kernels JSON")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # CPU-only assertion
    try:
        import torch
        if torch.cuda.is_available():
            logger.error("CUDA is available but GPU use is NOT authorized for E3. Abort.")
            sys.exit(1)
    except ImportError:
        pass  # No torch → no GPU risk

    # Load data
    qrel_path = DATA_DIR / "2009qrels.diversity.txt"
    if not qrel_path.exists():
        logger.error("Qrel file not found: %s", qrel_path)
        sys.exit(1)

    logger.info("Loading TREC Web diversity 2009 qrels from %s", qrel_path)
    df = load_diversity_qrels_2009(qrel_path)

    profiles_uniform = build_profiles_uniform(df)
    profiles_count = build_profiles_count_weighted(df)

    stats_u = summarize_profiles(profiles_uniform)
    logger.info("Profiles (uniform): %d total, %.1f%% interior", stats_u["n_total"], stats_u["pct_interior"])

    if dry_run := args.dry_run:
        logger.info("Dry-run: profiles loaded. Stopping before experiments.")
        env_log = build_env_log(args.seeds, qrel_path)
        out_path = RESULTS_DIR / "e3_env_log.json"
        with open(out_path, "w") as fh:
            json.dump(env_log, fh, indent=2)
        logger.info("Env log written to %s", out_path)
        return

    n_interior = args.n_interior
    n_vertex_0 = args.n_vertex if args.n_vertex else N_VERTEX_0
    n_vertex_1 = N_VERTEX_1

    logger.info("Starting E3 experiments: %d seeds, n_interior=%d", len(args.seeds), n_interior)

    all_seed_results: List[Dict] = []

    for seed in args.seeds:
        result = run_seed(
            seed=seed,
            profiles_uniform=profiles_uniform,
            profiles_count=profiles_count,
            n_interior=n_interior,
            n_vertex_0=n_vertex_0,
            n_vertex_1=n_vertex_1,
            m_values=M_VALUES,
            external_kernels_path=args.external_kernels,
            dry_run=False,
        )
        all_seed_results.append(result)

        # Save per-seed result
        seed_path = RESULTS_DIR / f"e3_seed{seed}_results.json"
        with open(seed_path, "w") as fh:
            json.dump(result, fh, indent=2)
        logger.info("Saved seed %d results to %s", seed, seed_path)

    # Aggregate and save
    agg = aggregate_results(all_seed_results)
    agg_path = RESULTS_DIR / "e3_aggregated.json"
    with open(agg_path, "w") as fh:
        json.dump(agg, fh, indent=2)
    logger.info("Saved aggregated results to %s", agg_path)

    # Environment log
    env_log = build_env_log(args.seeds, qrel_path)
    env_path = RESULTS_DIR / "e3_env_log.json"
    with open(env_path, "w") as fh:
        json.dump(env_log, fh, indent=2)
    logger.info("Saved env log to %s", env_path)

    # Print summary
    print("\n" + "=" * 60)
    print("E3 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nM1 — Profile interiority (Measurement 1):")
    u = stats_u
    print(f"  Total profiles: {u['n_total']:,}")
    print(f"  Interior (0<pi<1): {u['n_interior']:,} ({u['pct_interior']:.1f}%)")
    print(f"  Vertex 0 (pi=0):   {u['n_vertex_0']:,}")
    print(f"  Vertex 1 (pi=1):   {u['n_vertex_1']:,}")
    print(f"  Interior pi: mean={u['interior_pi_mean']:.3f} ± {u['interior_pi_std']:.3f}")
    print(f"  Min vertex distance: {u['interior_min_vertex_dist_min']:.3f}–{u['interior_min_vertex_dist_mean']:.3f}")

    print(f"\nM2 — Gauge diameter vs m (mean ± std across {len(args.seeds)} seeds):")
    for m_str, v in sorted(agg.get("M2_gauge_diameter", {}).items(), key=lambda x: int(x[0])):
        print(f"  m={m_str}: diameter = {v['mean']:.4f} ± {v['std']:.4f}")
    if "M2_flatness_max_abs_loglik_diff" in agg:
        v = agg["M2_flatness_max_abs_loglik_diff"]
        print(f"  Log-likelihood flatness (max |diff|): {v['mean']:.2e} ± {v['std']:.2e}  (should ≈ 0)")

    print(f"\nM3 — Anchor budget curve (mean RMSE ± std):")
    for n_str, v in sorted(agg.get("M3_rmse_vs_n", {}).items(), key=lambda x: int(x[0])):
        print(f"  n={n_str:>4}: RMSE = {v['mean']:.4f} ± {v['std']:.4f}")
    if "M3_power_law" in agg:
        pl = agg["M3_power_law"]
        print(f"  Power law fit: RMSE = {pl['c']['mean']:.4f} * n^{{-0.5}}, R² = {pl['r_squared']['mean']:.3f}")

    print(f"\nM4 — Structural anchor effect:")
    for nv_str, v_with in sorted(agg.get("M4_rmse_with_structural", {}).items(), key=lambda x: int(x[0])):
        v_wo = agg.get("M4_rmse_without_structural", {}).get(nv_str, {"mean": float("nan"), "std": 0})
        print(f"  n_v={nv_str}: RMSE_with={v_with['mean']:.4f}±{v_with['std']:.4f}"
              f"  RMSE_without={v_wo['mean']:.4f}±{v_wo['std']:.4f}")

    print(f"\nResult files:")
    print(f"  Per-seed: {RESULTS_DIR}/e3_seed*_results.json")
    print(f"  Aggregate: {agg_path}")
    print(f"  Env log: {env_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
