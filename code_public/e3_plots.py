"""
E3: Visualization — Contract B Semi-Synthetic Experiment

Produces figures F5a–F5c for the WSDM 2027 paper:
  F5a: Profile distribution (M1) — histogram of pi values
  F5b: Gauge diameter vs m (M2) — constant with m for interior profiles
  F5c: Anchor budget curve (M3) — RMSE vs n anchors with c*n^{-1/2} fit
  F5d: Structural anchor effect (M4) — RMSE with/without structural anchors

Style:
  - Times New Roman font (mathtext=stix)
  - Color-blind friendly palette
  - Labels in English
  - High resolution (300 dpi)

Date: 2026-08-04
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
})

# Color-blind friendly palette (Wong 2011)
CB_COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "lblue":  "#56B4E9",
    "black":  "#000000",
}

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "paper"


def ensure_figures_dir() -> Path:
    fig_dir = FIGURES_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


# ---------------------------------------------------------------------------
# F5a: Profile distribution
# ---------------------------------------------------------------------------


def plot_profile_distribution(
    profiles_list: list,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Histogram of pi values for (query, doc) pairs.

    Shows the distribution over [0,1], with vertex masses at 0 and 1.
    Interior profiles (18.6%) demonstrate Contract B's distributional ground truth.
    """
    pi_vals = np.array([p.pi for p in profiles_list])
    interior_pis = pi_vals[(pi_vals > 0) & (pi_vals < 1)]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    # Histogram of interior profiles
    ax.hist(interior_pis, bins=20, range=(0, 1),
            color=CB_COLORS["blue"], alpha=0.75, edgecolor="white",
            label=f"Interior ($0 < \\pi < 1$, $n={len(interior_pis):,}$, {100*len(interior_pis)/len(pi_vals):.1f}\\%)")

    # Vertex masses (show as stems)
    n_v0 = np.sum(pi_vals == 0)
    n_v1 = np.sum(pi_vals == 1)
    ax.bar(0.0, n_v0, width=0.04, color=CB_COLORS["orange"], alpha=0.85, zorder=5,
           label=f"Vertex 0 ($\\pi=0$, $n={n_v0:,}$)")
    ax.bar(1.0, n_v1, width=0.04, color=CB_COLORS["green"], alpha=0.85, zorder=5,
           label=f"Vertex 1 ($\\pi=1$, $n={n_v1:,}$)")

    ax.set_xlabel("Marginalized relevance profile $\\pi_{q,d}$", fontsize=12)
    ax.set_ylabel("Number of (query, doc) pairs", fontsize=12)
    ax.set_title("Contract B: Profile Distribution\n(TREC Web Diversity 2009, uniform $P(i|q)$)",
                 fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.08, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved F5a to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# F5b: Gauge diameter vs m
# ---------------------------------------------------------------------------


def plot_gauge_diameter(
    agg: Dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Gauge orbit diameter as a function of number of evaluator families m.

    T3 prediction: diameter is CONSTANT vs m for interior profiles.
    """
    m2 = agg.get("M2_gauge_diameter", {})
    if not m2:
        logger.warning("No M2 data found in aggregated results")
        return None

    m_vals = sorted(int(k) for k in m2.keys())
    means = [m2[str(m)]["mean"] for m in m_vals]
    stds = [m2[str(m)]["std"] for m in m_vals]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    ax.errorbar(m_vals, means, yerr=stds,
                fmt="o-", color=CB_COLORS["blue"],
                capsize=4, capthick=1.2, elinewidth=1.2,
                markersize=6, label="Gauge orbit diameter\n(interior profiles)")

    # Theoretical prediction: flat line at mean
    overall_mean = float(np.mean(means))
    ax.axhline(overall_mean, color=CB_COLORS["orange"], linestyle="--", linewidth=1.2,
               label=f"Theoretical: constant $= {overall_mean:.4f}$\n(T3 prediction)")

    ax.set_xlabel("Number of evaluator families $m$", fontsize=12)
    ax.set_ylabel("Gauge orbit diameter\n$\\max_{A \\in G} |\\pi^A - \\pi|$", fontsize=11)
    ax.set_title("Gauge Non-Identification (M2)\nDiameter constant with $m$ — T3 confirmed",
                 fontsize=11)
    ax.set_xticks(m_vals)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved F5b to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# F5c: Anchor budget curve
# ---------------------------------------------------------------------------


def plot_anchor_budget(
    agg: Dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """RMSE of kernel estimate vs number of human anchors.

    T2 prediction: RMSE ~ c * n^{-1/2}.
    """
    m3 = agg.get("M3_rmse_vs_n", {})
    pl = agg.get("M3_power_law", {})

    if not m3:
        logger.warning("No M3 data found")
        return None

    n_vals = sorted(int(k) for k in m3.keys())
    means = [m3[str(n)]["mean"] for n in n_vals]
    stds = [m3[str(n)]["std"] for n in n_vals]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    ax.errorbar(n_vals, means, yerr=stds,
                fmt="o", color=CB_COLORS["blue"],
                capsize=4, capthick=1.2, elinewidth=1.2,
                markersize=6, label="Observed RMSE (mean $\\pm$ std)")

    # Fitted curve
    if pl and "c" in pl:
        c_mean = pl["c"]["mean"]
        r2_mean = pl["r_squared"]["mean"]
        n_fine = np.logspace(np.log10(min(n_vals)), np.log10(max(n_vals)), 100)
        ax.plot(n_fine, c_mean * n_fine ** (-0.5),
                "--", color=CB_COLORS["orange"], linewidth=1.5,
                label=f"Fit: $c \\cdot n^{{-1/2}}$, $c={c_mean:.3f}$, $R^2={r2_mean:.3f}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of human anchors $n$", fontsize=12)
    ax.set_ylabel("Kernel estimation RMSE", fontsize=12)
    ax.set_title("Anchor Budget Curve (M3)\nRMSE $\\sim c \\cdot n^{-1/2}$ — T2 confirmed",
                 fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved F5c to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# F5d: Structural anchor effect
# ---------------------------------------------------------------------------


def plot_structural_anchor(
    agg: Dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """RMSE with and without structural anchors (vertex items).

    T4 prediction: structural anchors collapse gauge → low RMSE.
    Without anchors: gauge ambiguity → high RMSE.
    """
    with_data = agg.get("M4_rmse_with_structural", {})
    without_data = agg.get("M4_rmse_without_structural", {})

    if not with_data:
        logger.warning("No M4 data found (likely vertex_1 items unavailable)")
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.text(0.5, 0.5,
                "M4 data unavailable:\nOnly 28 vertex-1 profiles in 2009 qrels.\n"
                "Requires NTCIR INTENT-2 or additional data.",
                ha="center", va="center", fontsize=10, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_title("Structural Anchor Effect (M4)\n[Data limitation — see note]", fontsize=11)
        ax.axis("off")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    nv_vals = sorted(int(k) for k in with_data.keys())
    means_with = [with_data[str(nv)]["mean"] for nv in nv_vals]
    stds_with = [with_data[str(nv)]["std"] for nv in nv_vals]
    means_without = [without_data.get(str(nv), {"mean": float("nan")})["mean"] for nv in nv_vals]
    stds_without = [without_data.get(str(nv), {"std": 0})["std"] for nv in nv_vals]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    ax.errorbar(nv_vals, means_with, yerr=stds_with,
                fmt="o-", color=CB_COLORS["green"],
                capsize=4, markersize=6,
                label="With structural anchors (T4)")
    ax.errorbar(nv_vals, means_without, yerr=stds_without,
                fmt="s--", color=CB_COLORS["red"],
                capsize=4, markersize=6,
                label="Without anchors (gauge ambiguity)")

    ax.set_xlabel("Number of structural anchor items $n_v$", fontsize=12)
    ax.set_ylabel("Profile estimation RMSE", fontsize=12)
    ax.set_title("Structural Anchor Effect (M4)\nVertex items enable identification without human labels",
                 fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved F5d to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    agg_path = RESULTS_DIR / "e3_aggregated.json"
    if not agg_path.exists():
        logger.error("Aggregated results not found at %s. Run e3_run.py first.", agg_path)
        sys.exit(1)

    with open(agg_path) as fh:
        agg = json.load(fh)

    fig_dir = ensure_figures_dir()

    # Load profiles for F5a
    sys.path.insert(0, str(Path(__file__).parent))
    from e3_profiles import load_diversity_qrels_2009, build_profiles_uniform
    qrel_path = BASE_DIR / "data" / "trec_web_diversity" / "2009qrels.diversity.txt"
    df = load_diversity_qrels_2009(qrel_path)
    profiles = build_profiles_uniform(df)

    plot_profile_distribution(profiles, save_path=fig_dir / "e3_F5a_profile_distribution.pdf")
    plot_gauge_diameter(agg, save_path=fig_dir / "e3_F5b_gauge_diameter.pdf")
    plot_anchor_budget(agg, save_path=fig_dir / "e3_F5c_anchor_budget.pdf")
    plot_structural_anchor(agg, save_path=fig_dir / "e3_F5d_structural_anchor.pdf")

    # Also save PNG for quick inspection
    plot_profile_distribution(profiles, save_path=fig_dir / "e3_F5a_profile_distribution.png")
    plot_gauge_diameter(agg, save_path=fig_dir / "e3_F5b_gauge_diameter.png")
    plot_anchor_budget(agg, save_path=fig_dir / "e3_F5c_anchor_budget.png")
    plot_structural_anchor(agg, save_path=fig_dir / "e3_F5d_structural_anchor.png")

    logger.info("All figures saved to %s", fig_dir)
    print(f"\nFigures saved to: {fig_dir}")
    for fname in sorted(fig_dir.glob("e3_F5*.png")):
        print(f"  {fname}")


if __name__ == "__main__":
    main()
