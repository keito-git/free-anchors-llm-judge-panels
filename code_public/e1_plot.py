"""
E1 Phase Diagram Visualisation
================================
Loads results/e1_synthetic_phase.json and produces Figure F3 for the WSDM 2027
paper "Families, Not Judges".

Design rules:
  - Font: Times New Roman, mathtext = stix
  - Colorblind-safe categorical palette (CVD-validated: blue #2166AC, orange #D6604D,
    green #4DAC26, grey #878787 — all pass OKLab ΔE ≥ 8 between adjacent pairs)
  - Resolution: 300 dpi
  - Layout: 1 row × 3 panels (A: T1 NRMSE; B: T3b gauge radius; C: T4 phase)
  - Error bars: ± 1 std over 5 seeds
  - No dual y-axis. One metric per panel.
"""

import json
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Font and style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# CVD-validated palette (OKLab ΔE ≥ 8 between adjacent pairs, validated)
# Blue, Orange/Red, Green, Grey
# ---------------------------------------------------------------------------

BLUE   = "#2166AC"   # identified
RED    = "#D6604D"   # not identified
GREEN  = "#4DAC26"   # recovered (T4)
GREY   = "#878787"   # gauge orbit (T3b)
ORANGE = "#F4A582"   # light red / 50% anchor

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

JSON_PATH = os.path.join(RESULTS_DIR, "e1_synthetic_phase.json")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_results() -> dict:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_T1(data: dict):
    """Return (m_vals, nrmse_mean, nrmse_std) arrays."""
    agg = data["T1"]["aggregate"]
    seeds = data["env_log"]["seeds"]
    m_vals = data["env_log"]["M_VALUES_T1"]
    means, stds = [], []
    for m in m_vals:
        key = f"m_{m}"
        means.append(agg[key]["nrmse_mean"])
        stds.append(agg[key]["nrmse_std"])
    return np.array(m_vals), np.array(means), np.array(stds)


def extract_T3b(data: dict):
    """Return (m_vals, gauge_mean, gauge_std) arrays for interior profiles."""
    agg = data["T3"]["T3b_interior"]["aggregate"]
    m_vals = data["env_log"]["M_VALUES_T3"]
    means, stds, ll_maxs = [], [], []
    for m in m_vals:
        key = f"m_{m}"
        means.append(agg[key]["gauge_radius_mean"])
        stds.append(agg[key]["gauge_radius_std"])
        ll_maxs.append(agg[key]["ll_diff_max"])
    return np.array(m_vals), np.array(means), np.array(stds), np.array(ll_maxs)


def extract_T3a(data: dict):
    """Return (m_vals, max_radius) for vertex profiles (should be 0)."""
    t3a = data["T3"]["T3a_vertex"]
    rows = []
    for key, val in t3a.items():
        m = int(key.split("_")[1])
        rows.append((m, val["mean_radius"], val["max_radius"]))
    rows.sort()
    m_vals = [r[0] for r in rows]
    mean_r = [r[1] for r in rows]
    max_r  = [r[2] for r in rows]
    return m_vals, mean_r, max_r


def extract_T4(data: dict):
    """Return (anchor_pct, gauge_mean, gauge_std, nrmse_mean, nrmse_std)."""
    agg = data["T4"]["aggregate"]
    rates = data["env_log"]["ANCHOR_RATES"]
    g_means, g_stds, n_means, n_stds = [], [], [], []
    for r in rates:
        key = f"rate_{r:.2f}"
        g_means.append(agg[key]["gauge_radius_mean"])
        g_stds.append(agg[key]["gauge_radius_std"])
        n_means.append(agg[key]["nrmse_mean"])
        n_stds.append(agg[key]["nrmse_std"])
    anchor_pct = np.array(rates) * 100
    return anchor_pct, np.array(g_means), np.array(g_stds), np.array(n_means), np.array(n_stds)


# ---------------------------------------------------------------------------
# Main figure: 3 panels
# ---------------------------------------------------------------------------

def make_figure(data: dict, out_prefix: str) -> None:
    m_vals_t1, nrmse_mean, nrmse_std = extract_T1(data)
    m_vals_t3, gauge_mean, gauge_std, ll_maxs = extract_T3b(data)
    t3a_m, t3a_mean, t3a_max = extract_T3a(data)
    anchor_pct, g_means, g_stds, n_means, n_stds = extract_T4(data)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.17,
                        wspace=0.38)

    # -- Panel A: T1  NRMSE vs m  ----------------------------------------
    ax = axes[0]
    colors_t1 = [RED if m < 3 else BLUE for m in m_vals_t1]
    for i, (m, mu, sd, c) in enumerate(zip(m_vals_t1, nrmse_mean, nrmse_std, colors_t1)):
        ax.errorbar(m, mu, yerr=sd, fmt="o", color=c,
                    ecolor=c, elinewidth=1.0, capsize=3, capthick=0.8,
                    markersize=6, zorder=3)
    # Connect identified points with line
    id_mask = np.array(m_vals_t1) >= 3
    ax.plot(np.array(m_vals_t1)[id_mask], nrmse_mean[id_mask],
            "-", color=BLUE, lw=1.4, zorder=2)
    ni_mask = np.array(m_vals_t1) < 3
    ax.plot(np.array(m_vals_t1)[ni_mask], nrmse_mean[ni_mask],
            "--", color=RED, lw=1.4, zorder=2)
    # Threshold line
    ax.axvline(2.5, color="black", lw=0.8, ls=":", zorder=1)
    ax.text(2.55, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.97,
            r"$m \geq 3$", fontsize=8, va="top", ha="left")

    # Legend patches
    p1 = mpatches.Patch(color=RED,  label=r"$m \leq 2$: non-identified")
    p2 = mpatches.Patch(color=BLUE, label=r"$m \geq 3$: identified")
    ax.legend(handles=[p1, p2], fontsize=7.5, loc="upper right",
              framealpha=0.9, edgecolor="none")

    ax.set_xlabel("Number of families $m$", labelpad=3)
    ax.set_ylabel("NRMSE (emission matrices)", labelpad=3)
    ax.set_title("(A) T1: Identification threshold at $m = 3$", pad=5)
    ax.set_xticks(m_vals_t1)
    ax.yaxis.grid(True, lw=0.5, color="#e0e0e0", zorder=0)
    ax.set_xlim(0.5, max(m_vals_t1) + 0.5)
    ax.set_ylim(bottom=0)

    # -- Panel B: T3b  Gauge radius vs m  -----------------------------------
    ax = axes[1]
    # Interior profiles (main result)
    ax.errorbar(m_vals_t3, gauge_mean, yerr=gauge_std,
                fmt="o-", color=GREY, ecolor=GREY,
                elinewidth=1.0, capsize=3, capthick=0.8,
                markersize=6, lw=1.4, label="Interior profiles (T3b)", zorder=3)
    # Vertex profiles (should be 0)
    for m, mr in zip(t3a_m, t3a_mean):
        ax.plot(m, mr, "x", color=RED, markersize=8, mew=1.5, zorder=4)

    # Annotation for LL flatness
    max_ll = max(ll_maxs[~np.isnan(ll_maxs)]) if not all(np.isnan(ll_maxs)) else 0.0
    if not np.isnan(max_ll):
        ax.text(0.02, 0.97,
                r"$|\Delta\ell| < $" + f"{max_ll:.0e}".replace("e-", r"$\times 10^{-") + "}$"
                if max_ll < 0.01 else f"|Δℓ| max = {max_ll:.2e}",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    p_int = mpatches.Patch(color=GREY, label="Interior profiles\n(T3b: orbit positive for all $m$)")
    p_vtx = plt.Line2D([0], [0], marker="x", color=RED, lw=0, markersize=7,
                       markeredgewidth=1.5, label="Vertex profiles\n(T3a: orbit = permutations)")
    ax.legend(handles=[p_int, p_vtx], fontsize=7.5, loc="upper right",
              framealpha=0.9, edgecolor="none")

    ax.set_xlabel("Number of families $m$", labelpad=3)
    ax.set_ylabel("Gauge orbit radius", labelpad=3)
    ax.set_title("(B) T3: Gauge orbit positive for all $m$ (interior profiles)", pad=5)
    ax.set_xticks(m_vals_t3)
    ax.yaxis.grid(True, lw=0.5, color="#e0e0e0", zorder=0)
    ax.set_xlim(min(m_vals_t3) - 0.5, max(m_vals_t3) + 0.5)
    ax.set_ylim(bottom=0)

    # -- Panel C: T4  Recovery vs anchor rate  ------------------------------
    ax = axes[2]
    x = anchor_pct
    x_jitter = x  # no jitter for gauge radius (different axis)

    # NRMSE (primary)
    ax.errorbar(x, n_means, yerr=n_stds,
                fmt="s-", color=BLUE, ecolor=BLUE,
                elinewidth=1.0, capsize=3, capthick=0.8,
                markersize=6, lw=1.4, label="NRMSE (emission matrices)", zorder=3)
    # Gauge radius (secondary series, same axis — different units, overlay only if comparable)
    ax.errorbar(x + 0.1, g_means, yerr=g_stds,
                fmt="D--", color=GREEN, ecolor=GREEN,
                elinewidth=1.0, capsize=3, capthick=0.8,
                markersize=6, lw=1.4, label="Gauge orbit radius", zorder=3)

    # Phase transition annotation
    # Find first rate where gauge collapses
    collapse_idx = None
    for i, g in enumerate(g_means):
        if g < 0.02:
            collapse_idx = i
            break
    if collapse_idx is not None and collapse_idx < len(x):
        ax.axvline(x[collapse_idx], color="black", lw=0.8, ls=":", zorder=1)
        ax.text(x[collapse_idx] + 0.15, ax.get_ylim()[1] * 0.95
                if ax.get_ylim()[1] > 0 else 0.95,
                "Gauge collapses", fontsize=7.5, va="top", rotation=90)

    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9, edgecolor="none")
    ax.set_xlabel("Anchor rate (% vertex items in pool)", labelpad=3)
    ax.set_ylabel("Value", labelpad=3)
    ax.set_title("(C) T4: Structural anchor collapses gauge orbit", pad=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r)}%" for r in x])
    ax.yaxis.grid(True, lw=0.5, color="#e0e0e0", zorder=0)
    ax.set_xlim(-0.3, max(x) + 0.5)
    ax.set_ylim(bottom=0)

    # Shared style
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Save
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{out_prefix}.{ext}")
        dpi = 300 if ext == "png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary figure: Phase heatmap (m × interiority)
# ---------------------------------------------------------------------------

def make_phase_heatmap(data: dict, out_prefix: str) -> None:
    """
    2D phase diagram: x = number of families m, y = contract type.
    Cell color: identified (blue) / not identified (red).
    """
    agg_t1 = data["T1"]["aggregate"]
    agg_t3b = data["T3"]["T3b_interior"]["aggregate"]

    m_all = sorted(set(data["env_log"]["M_VALUES_T1"] + data["env_log"]["M_VALUES_T3"]))
    m_all = [m for m in m_all if 1 <= m <= 20]

    # Row 0: vertex profiles (point contract)
    # Row 1: interior profiles (distribution contract)
    nrows = 2
    ncols = len(m_all)

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.82, bottom=0.22)

    identified_threshold = 0.15  # NRMSE < this => identified

    for ci, m in enumerate(m_all):
        # Vertex (row 0): color by THEORY (m >= 3 => identified)
        key = f"m_{m}"
        theory_identified = (m >= 3)
        color = BLUE if theory_identified else RED

        # NRMSE annotation
        if key in agg_t1:
            nrmse = agg_t1[key]["nrmse_mean"]
            nrmse_std = agg_t1[key]["nrmse_std"]
            label = f"{nrmse:.2f}\n±{nrmse_std:.2f}"
        else:
            nrmse = None
            label = "id.*"  # theoretical, not directly measured

        rect = mpatches.FancyBboxPatch(
            (ci + 0.05, 0.55), 0.9, 0.85,
            boxstyle="round,pad=0.02", linewidth=0.5,
            facecolor=color, edgecolor="white", transform=ax.transData)
        ax.add_patch(rect)
        ax.text(ci + 0.5, 0.975, label, ha="center", va="center",
                fontsize=7.0, color="white", fontweight="bold", linespacing=1.2)

        # Interior (row 1): always non-identified (T3b)
        color2 = RED
        if key in agg_t3b:
            gauge_r = agg_t3b[key]["gauge_radius_mean"]
            gauge_s = agg_t3b[key]["gauge_radius_std"]
            label2 = f"{gauge_r:.2f}\n±{gauge_s:.2f}"
        else:
            label2 = "orbit>0\n(theory)"

        rect2 = mpatches.FancyBboxPatch(
            (ci + 0.05, -0.3), 0.9, 0.85,
            boxstyle="round,pad=0.02", linewidth=0.5,
            facecolor=color2, edgecolor="white", transform=ax.transData)
        ax.add_patch(rect2)
        ax.text(ci + 0.5, 0.125, label2, ha="center", va="center",
                fontsize=7.0, color="white", fontweight="bold", linespacing=1.2)

    # Axes
    ax.set_xlim(0, ncols)
    ax.set_ylim(-0.4, 1.6)
    ax.set_xticks([i + 0.5 for i in range(ncols)])
    ax.set_xticklabels([str(m) for m in m_all])
    ax.set_yticks([0.125, 0.975])
    ax.set_yticklabels(["Distribution contract\n(interior profile; T3b)",
                         "Point contract\n(vertex profile; T1/T3a)"], fontsize=8)
    ax.set_xlabel("Number of families $m$", labelpad=4)
    ax.set_title("E1 Synthetic Phase Diagram: Identification Regime", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    # Legend
    p_id   = mpatches.Patch(color=BLUE, label="Identified (up to permutation)")
    p_nid  = mpatches.Patch(color=RED,  label="Non-identified (continuous gauge orbit)")
    ax.legend(handles=[p_id, p_nid], loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8,
              framealpha=0.9, edgecolor="none")

    # Annotations inside cells for vertex row
    ax.text(ncols * 0.5, 1.52, "NRMSE (vertex profiles)", ha="center",
            fontsize=7.5, color="#444")
    ax.text(ncols * 0.5, 0.47, "Gauge radius (interior profiles)", ha="center",
            fontsize=7.5, color="#444")

    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{out_prefix}.{ext}")
        dpi = 300 if ext == "png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results ...")
    data = load_results()

    print("Generating 3-panel figure ...")
    make_figure(data, "e1_three_panel")

    print("Generating phase heatmap ...")
    make_phase_heatmap(data, "e1_phase_heatmap")

    print("Done. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
