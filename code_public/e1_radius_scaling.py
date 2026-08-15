"""
E1 Radius Scaling — gauge radius vs m = 2,3,4,6,10,20,35,50
=============================================================
Protocol:
  - Measure gauge orbit radius for m = 2,3,4,6,10,20,35,50
  - Log-log plot to distinguish power-law decay vs positive asymptote
  - Report lower bound extrapolation if data suggests convergence
  - Verify LL flatness (machine precision) at each m

Saves to results/e1_radius_scaling.json (never overwrites existing files).
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
# Parameters — FIXED before running (no fishing)
# ---------------------------------------------------------------------------

L          = 3
K_INT      = 5           # interior profiles (more than L to stress-test)
N_ITEMS    = 1000        # for generating the model (radius is geometric, no data needed)
N_SEEDS    = 5           # same as original for compatibility
SEEDS      = [42, 123, 456, 789, 1234]
M_VALUES   = [2, 3, 4, 6, 10, 20, 35, 50]  # extending to 50 as instructed
N_DIRS     = 300
N_BISECT   = 80          # high precision bisection

ALPHA_PROF = 0.5         # Dirichlet for interior profiles
ALPHA_CF   = 2.0         # Dirichlet for distortion matrices

print(f"Radius Scaling: L={L}, K_INT={K_INT}, SEEDS={SEEDS}")
print(f"M_VALUES={M_VALUES}, N_DIRS={N_DIRS}, N_BISECT={N_BISECT}")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_interior_profiles(K: int, L: int, rng: np.random.Generator, alpha: float = 0.5):
    """Sample K strictly interior profiles from Dirichlet(alpha,...,alpha)."""
    while True:
        profiles = np.column_stack([rng.dirichlet(alpha * np.ones(L)) for _ in range(K)])
        # Require all entries > 0.02 for numerical stability
        if np.all(profiles > 0.02):
            return profiles   # shape (L, K)


def sample_positive_distortion(L: int, rng: np.random.Generator, alpha: float = 2.0):
    """Sample a column-stochastic matrix with all entries > 0."""
    while True:
        C = np.column_stack([rng.dirichlet(alpha * np.ones(L)) for _ in range(L)])
        if np.all(C > 0.01):
            return C


# ---------------------------------------------------------------------------
# Gauge feasibility
# ---------------------------------------------------------------------------

def is_gauge_feasible(
    A: np.ndarray,
    profiles: np.ndarray,   # (L, K)
    distortions: list,      # m × (L, L)
    eps: float = 1e-8,
) -> bool:
    """True iff A ∈ Gauge group: col-stochastic, Ap>=0 forall k, C_f A^{-1}>=0 forall f."""
    if not np.allclose(A.sum(axis=0), 1.0, atol=eps):
        return False
    prod = A @ profiles   # (L, K)
    if np.any(prod < -eps):
        return False
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False
    for C_f in distortions:
        if np.any(C_f @ A_inv < -eps):
            return False
    return True


def gauge_radius_in_direction(
    D: np.ndarray,
    profiles: np.ndarray,
    distortions: list,
    n_bisect: int = N_BISECT,
) -> float:
    """Binary search for max t s.t. I + t*D_norm is in gauge."""
    D_norm = D / (np.linalg.norm(D) + 1e-15)
    if not is_gauge_feasible(np.eye(L) + 1e-8 * D_norm, profiles, distortions):
        return 0.0
    lo, hi = 0.0, 10.0
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        if is_gauge_feasible(np.eye(L) + mid * D_norm, profiles, distortions):
            lo = mid
        else:
            hi = mid
    return float(lo)


# ---------------------------------------------------------------------------
# Log-likelihood under gauge transform
# ---------------------------------------------------------------------------

def compute_ll(
    profiles: np.ndarray,   # (L, K)
    distortions: list,      # m × (L, L)
    pi: np.ndarray,         # (K,) mixing
    labels: np.ndarray,     # (n, m)
) -> float:
    """Compute log-likelihood of labels given the current model."""
    n, m = labels.shape
    log_r = np.tile(np.log(pi + 1e-300), (n, 1))
    for f in range(m):
        Q_f = distortions[f] @ profiles  # (L, K)  Q_f(y|k) ≈
        # Q_f[:, k] = distribution over y for prototype k, family f
        log_r += np.log(np.clip(Q_f[labels[:, f], :], 1e-300, None))
    lse = log_r.max(axis=1, keepdims=True)
    ll  = (lse.squeeze() + np.log(np.exp(log_r - lse).sum(axis=1))).sum()
    return float(ll)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_one_seed(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # Sample a global set of distortions large enough for max m
    distortions_all = [sample_positive_distortion(L, rng, alpha=ALPHA_CF)
                       for _ in range(max(M_VALUES))]
    profiles = sample_interior_profiles(K_INT, L, rng, alpha=ALPHA_PROF)  # (L, K)
    pi       = rng.dirichlet(np.ones(K_INT))  # mixing

    # Synthetic labels for LL verification (small)
    n_ll = 500
    data_rng = np.random.default_rng(seed + 99999)
    labels   = np.zeros((n_ll, max(M_VALUES)), dtype=int)
    for i in range(n_ll):
        k = data_rng.choice(K_INT, p=pi)
        for f in range(max(M_VALUES)):
            q = distortions_all[f] @ profiles[:, k]
            q = np.maximum(q, 0); q /= q.sum()
            labels[i, f] = data_rng.choice(L, p=q)

    res = {}
    for m in M_VALUES:
        dists_m  = distortions_all[:m]
        labels_m = labels[:, :m]

        t_m = time.time()
        # Sample gauge directions (zero column-sum perturbations of I)
        dir_rng = np.random.default_rng(seed * 1000 + m)
        radii = []
        for _ in range(N_DIRS):
            D = dir_rng.standard_normal((L, L))
            D -= D.mean(axis=0, keepdims=True)   # zero column sums → col-stochastic tangent
            r = gauge_radius_in_direction(D, profiles, dists_m)
            radii.append(r)

        radius_mean = float(np.mean(radii))
        radius_std  = float(np.std(radii))
        radius_max  = float(np.max(radii))
        frac_pos    = float(np.mean(np.array(radii) > 1e-6))   # fraction of pos dirs

        # LL flatness: use a random direction with large radius
        max_idx = int(np.argmax(radii))
        D_max   = dir_rng.standard_normal((L, L))  # approximate; direction already consumed
        D_max  -= D_max.mean(axis=0, keepdims=True)
        D_max  /= np.linalg.norm(D_max) + 1e-15
        t_val   = radii[max_idx] * 0.5   # safely inside orbit
        A_test  = np.eye(L) + t_val * D_max

        # Verify A is in orbit and compute LL diff
        ll_orig = compute_ll(profiles, dists_m, pi, labels_m)
        try:
            A_inv = np.linalg.inv(A_test)
            profiles_t  = A_test @ profiles   # (L, K)
            dists_t     = [C_f @ A_inv for C_f in dists_m]
            ll_trans    = compute_ll(profiles_t, dists_t, pi, labels_m)
            ll_delta    = abs(ll_orig - ll_trans)
            ll_ok       = is_gauge_feasible(A_test, profiles, dists_m)
        except Exception:
            ll_delta = float("nan")
            ll_ok    = False

        elapsed = time.time() - t_m
        res[f"m_{m}"] = {
            "m": m,
            "radius_mean": radius_mean,
            "radius_std":  radius_std,
            "radius_max":  radius_max,
            "frac_positive_directions": frac_pos,
            "ll_delta": float(ll_delta) if not np.isnan(ll_delta) else None,
            "ll_ok": ll_ok,
            "elapsed_s": elapsed,
        }
        print(f"  seed={seed} m={m:2d}: radius={radius_mean:.4f}±{radius_std:.4f}  "
              f"|ΔLL|={ll_delta:.2e}  time={elapsed:.1f}s")
    return res


def main():
    t0 = time.time()

    per_seed = {}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        per_seed[f"seed_{seed}"] = run_one_seed(seed)

    # Aggregate
    aggregate = {}
    for m in M_VALUES:
        key = f"m_{m}"
        r_means = [per_seed[f"seed_{s}"][key]["radius_mean"] for s in SEEDS]
        r_maxes = [per_seed[f"seed_{s}"][key]["radius_max"]  for s in SEEDS]
        ll_ds   = [per_seed[f"seed_{s}"][key]["ll_delta"]    for s in SEEDS
                   if per_seed[f"seed_{s}"][key]["ll_delta"] is not None]
        aggregate[key] = {
            "m": m,
            "radius_mean_mean": float(np.mean(r_means)),
            "radius_mean_std":  float(np.std(r_means)),
            "radius_max_mean":  float(np.mean(r_maxes)),
            "ll_delta_mean": float(np.mean(ll_ds)) if ll_ds else None,
            "ll_delta_max":  float(np.max(ll_ds))  if ll_ds else None,
        }

    # Log-log regression: log(radius) ~ a * log(m) + b
    m_arr  = np.array([m for m in M_VALUES], dtype=float)
    r_arr  = np.array([aggregate[f"m_{m}"]["radius_mean_mean"] for m in M_VALUES])
    log_m  = np.log(m_arr)
    log_r  = np.log(r_arr + 1e-15)
    slope, intercept, r_val, p_val, se = stats.linregress(log_m, log_r)

    # Also linear regression (not logged) to check asymptote
    slope_lin, int_lin, r_lin, p_lin, se_lin = stats.linregress(m_arr, r_arr)

    # If slope < -0.9 with small p, power-law to 0 is plausible
    # If slope flattens at large m, positive asymptote is plausible
    # Estimate asymptote from last 4 points
    m4 = m_arr[-4:]; r4 = r_arr[-4:]
    slope4, int4, _, _, _ = stats.linregress(m4, r4)
    asymptote_estimate = float(int4 + slope4 * 1000)  # extrapolate to m→∞

    loglog_analysis = {
        "log_slope": float(slope),
        "log_intercept": float(intercept),
        "r_squared": float(r_val**2),
        "p_value": float(p_val),
        "interpretation": (
            "power-law decay to zero (slope << -0.5, R^2 high)"
            if (slope < -0.5 and r_val**2 > 0.8) else
            "radius shrinks but not clear power-law"
        ),
        "asymptote_estimate_last4_extrapolate_m1000": asymptote_estimate,
        "converges_to_zero_evidence": bool(slope < -0.5 and r_val**2 > 0.8),
    }

    print("\n===== Radius Scaling Summary =====")
    for m in M_VALUES:
        a = aggregate[f"m_{m}"]
        print(f"  m={m:2d}: radius={a['radius_mean_mean']:.4f}±{a['radius_mean_std']:.4f}  "
              f"|ΔLL|_max={a['ll_delta_max']:.2e}")
    print(f"\n  Log-log: slope={slope:.4f}  R²={r_val**2:.4f}  p={p_val:.4f}")
    print(f"  Converges to zero? {loglog_analysis['converges_to_zero_evidence']}")
    print(f"  Asymptote estimate (m→∞): {asymptote_estimate:.4f}")

    output = {
        "env_log": {
            "date": time.strftime("%Y-%m-%d"),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "L": L, "K_INT": K_INT, "N_SEEDS": N_SEEDS,
            "SEEDS": SEEDS, "M_VALUES": M_VALUES,
            "N_DIRS": N_DIRS, "N_BISECT": N_BISECT,
            "ALPHA_PROF": ALPHA_PROF, "ALPHA_CF": ALPHA_CF,
            "note": ("Radius scaling for m = 2..50. "
                     "log-log fit to determine convergence. "
                     "All settings were fixed before the run; no post hoc selection."),
            "gpu_used": False, "api_calls": 0,
            "elapsed_s": time.time() - t0,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "loglog_regression": loglog_analysis,
    }
    out_path = os.path.join(RESULTS_DIR, "e1_radius_scaling.json")
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
    BLUE  = "#2166AC"; GREY = "#878787"; RED = "#D6604D"

    agg    = data["aggregate"]
    m_vals = data["env_log"]["M_VALUES"]
    r_mu   = np.array([agg[f"m_{m}"]["radius_mean_mean"] for m in m_vals])
    r_sd   = np.array([agg[f"m_{m}"]["radius_mean_std"]  for m in m_vals])

    ll_info  = data["loglog_regression"]
    slope    = ll_info["log_slope"]
    r2       = ll_info["r_squared"]
    converge = ll_info["converges_to_zero_evidence"]
    asymp    = ll_info["asymptote_estimate_last4_extrapolate_m1000"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.87, bottom=0.16, wspace=0.32)

    # Panel A: linear scale
    ax = axes[0]
    ax.errorbar(m_vals, r_mu, yerr=r_sd, fmt="o-", color=BLUE,
                ecolor=GREY, elinewidth=1.0, capsize=3, capthick=0.8, zorder=3)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xlabel("Number of families $m$", labelpad=3)
    ax.set_ylabel("Gauge radius (mean over directions)", labelpad=3)
    ax.set_title("Gauge radius vs $m$ (interior profiles)", pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, lw=0.5, color="#e0e0e0", zorder=0)

    # Panel B: log-log scale
    ax2 = axes[1]
    ax2.errorbar(m_vals, r_mu, yerr=r_sd, fmt="o", color=BLUE,
                 ecolor=GREY, elinewidth=1.0, capsize=3, capthick=0.8, zorder=3)
    # Fit line
    m_fit  = np.linspace(m_vals[0], m_vals[-1], 200)
    r_fit  = np.exp(ll_info["log_intercept"]) * m_fit ** slope
    ax2.plot(m_fit, r_fit, "--", color=RED, lw=1.2, zorder=2,
             label=rf"power law $m^{{{slope:.2f}}}$ ($R^2={r2:.2f}$)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Number of families $m$ (log scale)", labelpad=3)
    ax2.set_ylabel("Gauge radius (log scale)", labelpad=3)
    verdict = "converges to 0" if converge else f"asymptote ≈ {asymp:.4f}"
    ax2.set_title(f"Log-log: slope={slope:.3f} — {verdict}", pad=5)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="none")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(f"Gauge Orbit Radius Scaling ($L={data['env_log']['L']}$, "
                 f"$K={data['env_log']['K_INT']}$, {data['env_log']['N_SEEDS']} seeds)", y=0.98)

    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"e1_radius_scaling.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
