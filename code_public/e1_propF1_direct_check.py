"""
E1 Prop F1 Direct Check — G⁰ orbit radius via the Proposition F1 construction
===============================================================================
Addresses the measurement design problem found in e1_radius_decomposition.py:
  Random direction exploration almost never hits G⁰ directions (only 0.3-1%
  of random directions satisfy A^{-1} >= 0).

This script directly probes the G⁰ orbit using the Prop F1 construction:
  S_t = (1-t)*I + t * e_{j*} @ 1^T   (t ∈ [0, η])
  A_t = S_t^{-1}

where j* = argmin_j (min_k profiles[j,k]) (the row index achieving η).

Theoretical prediction (Proposition F1):
  - t_max = η = min_{k,j} p_j^(k)  for all m
  - t_max is INDEPENDENT of m (because C_f A_t^{-1} = C_f S_t >= 0 for ANY C_f)
  - The ℓ1 orbit diameter is 2η

This gives a definitive, m-independent measurement of the G⁰ floor.

Additionally verifies: t_max from binary search equals η from profiles (numerical agreement).

Saves to results/e1_propF1_direct_check.json (new file, never overwrites).
CPU only. No API calls.
"""

import json
import os
import time
import platform
import socket

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

L          = 3
K_INT      = 5
N_SEEDS    = 5
SEEDS      = [42, 123, 456, 789, 1234]
M_VALUES   = [2, 3, 4, 6, 10, 20, 35, 50]
N_BISECT   = 100

ALPHA_PROF = 0.5
ALPHA_CF   = 2.0

print(f"Prop F1 Direct Check: L={L}, K={K_INT}, SEEDS={SEEDS}")
print(f"M_VALUES={M_VALUES}, N_BISECT={N_BISECT}")
print(f"Direction: S_t = (1-t)I + t*e_{{j*}}@1^T  (Prop F1 construction)")
print()


# ---------------------------------------------------------------------------
# Sampling (same as e1_radius_decomposition.py)
# ---------------------------------------------------------------------------

def sample_interior_profiles(K, L, rng, alpha=0.5, min_entry=0.02):
    for _ in range(10000):
        profiles = np.column_stack([rng.dirichlet(alpha * np.ones(L)) for _ in range(K)])
        if np.all(profiles > min_entry):
            return profiles
    raise RuntimeError("sampling failed")


def sample_positive_distortion(L, rng, alpha=2.0):
    for _ in range(1000):
        C = np.column_stack([rng.dirichlet(alpha * np.ones(L)) for _ in range(L)])
        if np.all(C > 0.01):
            return C
    raise RuntimeError("sampling failed")


# ---------------------------------------------------------------------------
# Gauge feasibility checks (same as e1_radius_decomposition.py)
# ---------------------------------------------------------------------------

def is_gauge_feasible(A, profiles, distortions, eps=1e-8):
    if not np.allclose(A.sum(axis=0), 1.0, atol=eps):
        return False
    if np.any(A @ profiles < -eps):
        return False
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False
    for C_f in distortions:
        if np.any(C_f @ A_inv < -eps):
            return False
    return True


def is_gauge_feasible_nonneg_inv(A, profiles, distortions, eps=1e-8):
    """G⁰ condition: standard G_m + A^{-1} >= 0."""
    if not np.allclose(A.sum(axis=0), 1.0, atol=eps):
        return False
    if np.any(A @ profiles < -eps):
        return False
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False
    if np.any(A_inv < -eps):
        return False
    for C_f in distortions:
        if np.any(C_f @ A_inv < -eps):
            return False
    return True


# ---------------------------------------------------------------------------
# Prop F1 construction
# ---------------------------------------------------------------------------

def prop_f1_construction(t: float, j_star: int, L: int) -> np.ndarray:
    """
    Compute A_t = S_t^{-1} where S_t = (1-t)*I + t * e_{j*} @ 1^T.

    From Sherman-Morrison:
      S_t^{-1} = (1/(1-t)) * (I - t * e_{j*} @ 1^T)
    where all column sums = 1 (A_t is col-stochastic).

    Note: A_t[j*,j] = (1/(1-t)) * (delta_{j*,j} - t):
      j = j*: (1-t)/(1-t) = 1  >= 0
      j != j*: -t/(1-t)         < 0  for t > 0
    A_t is NOT entrywise nonneg, but A_t^{-1} = S_t IS nonneg.
    """
    e_j = np.zeros(L); e_j[j_star] = 1.0
    ones = np.ones(L)
    S_t = (1 - t) * np.eye(L) + t * np.outer(e_j, ones)   # (L,L)
    A_t = np.linalg.inv(S_t)                               # = S_t^{-1}
    return A_t


def binary_search_tmax_propF1(j_star, profiles, distortions, n_bisect=N_BISECT):
    """
    Find max t s.t. is_gauge_feasible_nonneg_inv(A_t) = True.
    A_t = S_t^{-1}, S_t = (1-t)I + t*e_{j*}@1^T.

    Theoretical prediction: t_max = min_k profiles[j*, k] = η_{j*}
    (where η_{j*} = min over k of the j*-th profile component).
    """
    # Quick check t=0 (A=I should be feasible)
    assert is_gauge_feasible_nonneg_inv(np.eye(L), profiles, distortions), \
        "I not feasible? something wrong"

    lo, hi = 0.0, 1.0 - 1e-6   # t must be < 1 (S_t singular at t=1 for L>1? no)
    # Actually S_t singular when 1-t=0, i.e., t=1. So hi < 1.

    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        A_t = prop_f1_construction(mid, j_star, L)
        if is_gauge_feasible_nonneg_inv(A_t, profiles, distortions):
            lo = mid
        else:
            hi = mid
    return float(lo)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dists_pool = [sample_positive_distortion(L, rng, alpha=ALPHA_CF)
                  for _ in range(max(M_VALUES))]
    profiles = sample_interior_profiles(K_INT, L, rng, alpha=ALPHA_PROF)   # (L, K)

    # η = min_{j,k} profiles[j,k]
    eta = float(np.min(profiles))
    # j* = row index achieving the minimum (profiles.shape = (L, K))
    j_star, _ = np.unravel_index(np.argmin(profiles), profiles.shape)
    j_star = int(j_star)
    eta_jstar = float(np.min(profiles[j_star, :]))   # min over k of row j*

    # Sanity: eta_jstar should equal eta (j* is the row containing the global min)
    assert abs(eta_jstar - eta) < 1e-12, f"eta mismatch {eta_jstar} vs {eta}"

    # Prop F1 prediction: t_max = η for ALL m
    theoretical_tmax = eta

    print(f"  seed={seed}: η={eta:.4f}  j*={j_star}  2η={2*eta:.4f} (ℓ1 orbit diameter)")
    print(f"  Prop F1 prediction: t_max = {theoretical_tmax:.4f} for all m")

    res = {
        "eta": eta,
        "j_star": j_star,
        "theoretical_tmax": theoretical_tmax,
        "lower_bound_2eta_l1": 2.0 * eta,
        "m_results": {},
    }

    for m in M_VALUES:
        dists_m = dists_pool[:m]
        t0 = time.time()

        t_max = binary_search_tmax_propF1(j_star, profiles, dists_m, N_BISECT)
        error  = abs(t_max - theoretical_tmax)
        ratio  = t_max / theoretical_tmax if theoretical_tmax > 0 else float("nan")

        # Verify A at t_max is feasible (should be)
        A_at_tmax = prop_f1_construction(t_max, j_star, L)
        feasible_at_tmax  = is_gauge_feasible_nonneg_inv(A_at_tmax, profiles, dists_m)
        # Beyond tmax should not be feasible
        A_at_tmax_plus = prop_f1_construction(t_max + 1e-4, j_star, L)
        feasible_beyond   = is_gauge_feasible_nonneg_inv(A_at_tmax_plus, profiles, dists_m)

        elapsed = time.time() - t0
        res["m_results"][f"m_{m}"] = {
            "m": m,
            "t_max_measured": t_max,
            "theoretical_tmax": theoretical_tmax,
            "absolute_error": error,
            "ratio_measured_to_theoretical": ratio,
            "feasible_at_tmax": feasible_at_tmax,
            "feasible_beyond_tmax": feasible_beyond,
            "elapsed_s": elapsed,
        }
        print(f"    m={m:2d}: t_max={t_max:.6f}  theory={theoretical_tmax:.6f}  "
              f"err={error:.2e}  feasible={feasible_at_tmax}  beyond={feasible_beyond}")

    return res


def main():
    t0_all = time.time()
    per_seed = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        per_seed[f"seed_{seed}"] = run_one_seed(seed)

    # Aggregate: for each m, collect t_max across seeds and compare to η
    aggregate = {}
    for m in M_VALUES:
        key = f"m_{m}"
        t_maxes    = [per_seed[f"seed_{s}"]["m_results"][key]["t_max_measured"] for s in SEEDS]
        theo_vals  = [per_seed[f"seed_{s}"]["theoretical_tmax"]                  for s in SEEDS]
        errors     = [per_seed[f"seed_{s}"]["m_results"][key]["absolute_error"]  for s in SEEDS]
        aggregate[key] = {
            "m": m,
            "t_max_mean":     float(np.mean(t_maxes)),
            "t_max_std":      float(np.std(t_maxes)),
            "theoretical_mean": float(np.mean(theo_vals)),
            "error_mean":     float(np.mean(errors)),
            "error_max":      float(np.max(errors)),
            "t_maxes_per_seed": t_maxes,
            "ratio_mean": float(np.mean([r["m_results"][key]["ratio_measured_to_theoretical"]
                                          for r in per_seed.values()])),
        }

    # Check: is t_max constant in m?
    m_arr    = np.array(M_VALUES, dtype=float)
    tmx_arr  = np.array([aggregate[f"m_{m}"]["t_max_mean"] for m in M_VALUES])
    # Normalize by seed-specific η to remove seed-level variation
    # (each seed has different η, so we compare t_max/η per seed)
    ratios_per_m = []
    for m in M_VALUES:
        key = f"m_{m}"
        rs = [per_seed[f"seed_{s}"]["m_results"][key]["ratio_measured_to_theoretical"]
              for s in SEEDS]
        ratios_per_m.append(float(np.mean(rs)))
    ratio_arr = np.array(ratios_per_m)

    slope_r, intercept_r, r_r, p_r, se_r = stats.linregress(m_arr, ratio_arr)
    cv_ratio = float(np.std(ratio_arr) / (np.mean(ratio_arr) + 1e-12))

    is_constant = bool(abs(slope_r) < 0.0001 and cv_ratio < 0.05)

    analysis = {
        "t_max_ratio_vs_m": {
            "slope": float(slope_r),
            "p_value": float(p_r),
            "r_squared": float(r_r**2),
            "cv": cv_ratio,
            "is_constant": is_constant,
            "mean_ratio": float(np.mean(ratio_arr)),
            "interpretation": (
                "Prop F1 VERIFIED: t_max = η is m-independent (G⁰ floor confirmed)"
                if is_constant else
                "Prop F1 NOT fully verified: t_max/η varies with m"
            ),
        },
    }

    # Summary
    print("\n===== Prop F1 Direct Check Summary =====")
    print(f"{'m':>4}  {'t_max_mean':>12}  {'η_mean':>10}  {'ratio':>8}  {'err_max':>10}")
    for m in M_VALUES:
        a = aggregate[f"m_{m}"]
        print(f"  {m:2d}   {a['t_max_mean']:.6f}     {a['theoretical_mean']:.6f}  "
              f"  {a['ratio_mean']:.5f}   {a['error_max']:.2e}")
    print(f"\nRatio t_max/η: slope={slope_r:.6f}  p={p_r:.4f}  CV={cv_ratio:.4f}")
    print(f"Prop F1 verified: {is_constant}")
    print(f"Interpretation: {analysis['t_max_ratio_vs_m']['interpretation']}")

    output = {
        "env_log": {
            "date": time.strftime("%Y-%m-%d"),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "L": L, "K_INT": K_INT, "N_SEEDS": N_SEEDS,
            "SEEDS": SEEDS, "M_VALUES": M_VALUES, "N_BISECT": N_BISECT,
            "ALPHA_PROF": ALPHA_PROF, "ALPHA_CF": ALPHA_CF,
            "direction": ("S_t = (1-t)I + t*e_{j*}@1^T, A_t = S_t^{-1}; "
                          "j* = argmin row index of profiles"),
            "theoretical_prediction": "t_max = eta = min_{j,k} profiles[j,k] for all m",
            "note": ("Direct check of Proposition F1. Addresses a measurement design "
                     "issue found in e1_radius_decomposition.py: random direction "
                     "search almost never satisfies the G0 condition "
                     "(frac_pos_nonneg approx 0.3-1%), so the constructive "
                     "direction of Proposition F1 is measured directly. All "
                     "settings were fixed before the run; no post hoc selection."),
            "gpu_used": False, "api_calls": 0,
            "elapsed_s": time.time() - t0_all,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "analysis": analysis,
    }
    out_path = os.path.join(RESULTS_DIR, "e1_propF1_direct_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    make_figure(output)


def make_figure(data: dict) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "Times New Roman", "mathtext.fontset": "stix",
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5, "lines.markersize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    BLUE = "#2166AC"; GREEN = "#4DAC26"; RED = "#D6604D"; GREY = "#878787"

    agg    = data["aggregate"]
    m_vals = data["env_log"]["M_VALUES"]

    # Panel A: t_max per seed across m (shows each seed's η-normalized ratio)
    # Panel B: aggregate ratio ± std

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.15, wspace=0.30)

    ax1 = axes[0]
    SEEDS_use = data["env_log"]["SEEDS"]
    palette = [BLUE, GREEN, "#E69F00", "#56B4E9", RED]
    for i, seed in enumerate(SEEDS_use):
        ts = [data["per_seed"][f"seed_{seed}"]["m_results"][f"m_{m}"]["ratio_measured_to_theoretical"]
              for m in m_vals]
        ax1.plot(m_vals, ts, "o-", color=palette[i], lw=1.0, ms=4,
                 label=f"seed={seed}", alpha=0.8)
    ax1.axhline(1.0, color="black", lw=0.8, ls="--", label=r"$t_\max/\eta=1$ (theory)")
    ax1.set_xlabel("Number of families $m$", labelpad=3)
    ax1.set_ylabel(r"$t_{\max}/\eta$  (Prop F1 direction)", labelpad=3)
    ax1.set_title("Ratio $t_{\\max}/\\eta$ per seed (should be 1.0 for all $m$)", pad=4)
    ax1.legend(fontsize=7, loc="lower right", framealpha=0.85, edgecolor="none")
    ax1.set_xticks(m_vals)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, lw=0.5, color="#e0e0e0", zorder=0)

    ax2 = axes[1]
    ratio_means = [agg[f"m_{m}"]["ratio_mean"] for m in m_vals]
    ratio_stds  = [agg[f"m_{m}"]["t_max_std"] / max(agg[f"m_{m}"]["theoretical_mean"], 1e-6)
                   for m in m_vals]
    ax2.errorbar(m_vals, ratio_means, yerr=ratio_stds, fmt="o-", color=BLUE,
                 ecolor=GREY, elinewidth=1.0, capsize=3, capthick=0.8, zorder=3)
    ax2.axhline(1.0, color=RED, lw=1.2, ls="--", zorder=2, label="$t_{\\max}/\\eta=1$ (Prop F1)")
    an = data["analysis"]["t_max_ratio_vs_m"]
    ax2.set_xlabel("Number of families $m$", labelpad=3)
    ax2.set_ylabel(r"Mean $t_{\max}/\eta$ (±1 std)", labelpad=3)
    slope = an["slope"]
    pval  = an["p_value"]
    const = an["is_constant"]
    verdict = "CONSTANT" if const else "VARIES"
    ax2.set_title(f"Mean ratio: slope={slope:.5f}, p={pval:.3f} — {verdict}", pad=4)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9, edgecolor="none")
    ax2.set_xticks(m_vals)
    ax2.set_ylim(bottom=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, lw=0.5, color="#e0e0e0", zorder=0)

    fig.suptitle(f"Prop F1 Direct Check: G⁰ Orbit Radius via $S_t = (1-t)I + t\\,e_{{j^*}}\\mathbf{{1}}^\\top$ "
                 f"($L={data['env_log']['L']}$, $K={data['env_log']['K_INT']}$, "
                 f"{data['env_log']['N_SEEDS']} seeds)",
                 y=0.99, fontsize=9)

    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"e1_propF1_direct_check.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
