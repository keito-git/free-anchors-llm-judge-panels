"""
E1 T1 Strengthened — Identification threshold at m = 3 (point-contract case)
=============================================================================
Strengthened settings relative to e1_synthetic_phase.py:
  - N >= 2000 items total (was 600)
  - EM initialisations >= 100 (was 30)
  - seeds >= 10 (was 5)
  - Test whether the m=2 vs m=3 phase transition is statistically significant

Saves to results/e1_t1_strengthened.json (never overwrites e1_synthetic_phase.json).
CPU only. No API calls.

Protocol (fishing prohibition):
  - Same Dirichlet parameters as e1_synthetic_phase.py (ALPHA_CF=2.0)
  - Same vertex profiles (L=3, K=3)
  - Settings fixed BEFORE running; no post-hoc tuning
  - Reports Welch t-test m=2 vs m=3 regardless of sign
"""

import json
import os
import time
import platform
import socket
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind, ttest_rel

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
K_T1       = 3
N_ITEMS    = 2000        # >=2000 as instructed
N_SEEDS    = 10
SEEDS      = list(range(100, 100 + N_SEEDS))   # new seed block, separate from E1
N_INIT_EM  = 100         # >=100 as instructed
MAX_ITER   = 500
TOL_EM     = 1e-9
M_VALUES   = [1, 2, 3, 4, 5]

DIRICHLET_ALPHA_CF = 2.0   # same as e1_synthetic_phase.py

print(f"T1 Strengthened: L={L}, K={K_T1}, N_ITEMS={N_ITEMS}, "
      f"N_SEEDS={N_SEEDS}, N_INIT_EM={N_INIT_EM}")
print(f"SEEDS={SEEDS}")


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_col_stochastic(L: int, rng: np.random.Generator, alpha: float = 2.0):
    return np.column_stack([rng.dirichlet(alpha * np.ones(L)) for _ in range(L)])


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_vertex_data(
    distortions: List[np.ndarray],  # m (L,L) matrices; Q_f = C_f for vertex profiles
    n_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate (n_items, m) labels where true prototype k ~ Uniform{0,1,2}."""
    L = distortions[0].shape[0]
    K = L  # vertex profiles: K = L
    m = len(distortions)
    proto_idx = rng.choice(K, size=n_items)
    labels = np.zeros((n_items, m), dtype=int)
    for i in range(n_items):
        k = proto_idx[i]
        for f, C_f in enumerate(distortions):
            p = C_f[:, k].copy()
            p = np.maximum(p, 0); p /= p.sum()
            labels[i, f] = rng.choice(L, p=p)
    return labels


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------

def em_lcm(
    labels: np.ndarray,
    K: int, L: int,
    n_init: int = N_INIT_EM,
    max_iter: int = MAX_ITER,
    tol: float = TOL_EM,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[np.ndarray], float, float]:
    """
    Returns best (pi, Q_list), best_ll, ll_std_across_inits.
    Q_list[f] has shape (L, K).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, m = labels.shape
    best_ll = -np.inf
    best_pi = None
    best_Q  = None
    all_lls = []

    for _ in range(n_init):
        pi = rng.dirichlet(np.ones(K))
        Q  = [np.column_stack([rng.dirichlet(np.ones(L)) for _ in range(K)]) for _ in range(m)]

        prev_ll = -np.inf
        for _ in range(max_iter):
            # E-step
            log_r = np.tile(np.log(pi + 1e-300), (n, 1))
            for f in range(m):
                log_r += np.log(Q[f][labels[:, f], :] + 1e-300)
            lse = log_r.max(axis=1, keepdims=True)
            r   = np.exp(log_r - lse); r /= r.sum(axis=1, keepdims=True)
            ll  = (lse.squeeze() + np.log(np.exp(log_r - lse).sum(axis=1))).sum()

            # M-step
            r_sum = r.sum(axis=0) + 1e-10
            pi    = r_sum / r_sum.sum()
            for f in range(m):
                Qf = np.zeros((L, K))
                for y in range(L):
                    Qf[y, :] = r[labels[:, f] == y, :].sum(axis=0) + 1e-12
                Q[f] = Qf / Qf.sum(axis=0, keepdims=True)

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        all_lls.append(float(ll))
        if ll > best_ll:
            best_ll = ll
            best_pi = pi.copy()
            best_Q  = [q.copy() for q in Q]

    return best_pi, best_Q, float(best_ll), float(np.std(all_lls))


# ---------------------------------------------------------------------------
# Alignment and NRMSE
# ---------------------------------------------------------------------------

def compute_nrmse(Q_true_list, Q_est_list):
    """Match columns by Hungarian, return mean NRMSE over systems."""
    # Use first system for permutation
    K = Q_true_list[0].shape[1]
    cost = np.array([[np.sum((Q_true_list[0][:, j] - Q_est_list[0][:, k])**2)
                      for k in range(K)] for j in range(K)])
    _, col = linear_sum_assignment(cost)
    errs = []
    for Qt, Qe in zip(Q_true_list, Q_est_list):
        Qep = Qe[:, col]
        denom = np.sqrt(np.mean(Qt**2))
        errs.append(float(np.sqrt(np.mean((Qt - Qep)**2)) / (denom + 1e-12)))
    return float(np.mean(errs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    per_seed = {}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # True parameters: vertex profiles => Q_f_true = C_f
        dists_all = [sample_col_stochastic(L, rng, alpha=DIRICHLET_ALPHA_CF)
                     for _ in range(max(M_VALUES))]
        Q_true_all = [C_f.copy() for C_f in dists_all]

        seed_res = {}
        for m in M_VALUES:
            dists_m  = dists_all[:m]
            Q_true_m = Q_true_all[:m]

            # Data
            data_rng = np.random.default_rng(seed + m * 1000)
            labels   = generate_vertex_data(dists_m, N_ITEMS, data_rng)

            # EM
            em_rng = np.random.default_rng(seed + m * 2000)
            _, Q_est, best_ll, ll_std = em_lcm(labels, K_T1, L, n_init=N_INIT_EM, rng=em_rng)

            nrmse = compute_nrmse(Q_true_m, Q_est)
            seed_res[f"m_{m}"] = {
                "nrmse": nrmse,
                "best_ll": best_ll,
                "ll_std": ll_std,
            }
            print(f"  seed={seed} m={m}: NRMSE={nrmse:.4f}  ll_std={ll_std:.3f}")

        per_seed[f"seed_{seed}"] = seed_res

    # Aggregate
    aggregate = {}
    for m in M_VALUES:
        key = f"m_{m}"
        vals = [per_seed[f"seed_{s}"][key]["nrmse"] for s in SEEDS]
        lls  = [per_seed[f"seed_{s}"][key]["ll_std"] for s in SEEDS]
        aggregate[key] = {
            "m": m,
            "nrmse_mean": float(np.mean(vals)),
            "nrmse_std":  float(np.std(vals)),
            "nrmse_values": vals,
            "ll_std_mean": float(np.mean(lls)),
        }

    # Welch t-test: m=2 vs m=3
    v2 = [per_seed[f"seed_{s}"]["m_2"]["nrmse"] for s in SEEDS]
    v3 = [per_seed[f"seed_{s}"]["m_3"]["nrmse"] for s in SEEDS]
    t_23, p_23 = ttest_ind(v2, v3, equal_var=False, alternative="greater")

    # Also m=1 vs m=3
    v1 = [per_seed[f"seed_{s}"]["m_1"]["nrmse"] for s in SEEDS]
    t_13, p_13 = ttest_ind(v1, v3, equal_var=False, alternative="greater")

    # m=2 vs m=5
    v5 = [per_seed[f"seed_{s}"]["m_5"]["nrmse"] for s in SEEDS]
    t_25, p_25 = ttest_ind(v2, v5, equal_var=False, alternative="greater")

    stats = {
        "welch_m2_vs_m3": {"t": float(t_23), "p": float(p_23),
                            "mean_m2": float(np.mean(v2)), "mean_m3": float(np.mean(v3)),
                            "significant_p05": bool(p_23 < 0.05)},
        "welch_m1_vs_m3": {"t": float(t_13), "p": float(p_13),
                            "mean_m1": float(np.mean(v1)), "mean_m3": float(np.mean(v3)),
                            "significant_p05": bool(p_13 < 0.05)},
        "welch_m2_vs_m5": {"t": float(t_25), "p": float(p_25),
                            "mean_m2": float(np.mean(v2)), "mean_m5": float(np.mean(v5)),
                            "significant_p05": bool(p_25 < 0.05)},
    }

    print("\n===== T1 Strengthened Summary =====")
    for m in M_VALUES:
        a = aggregate[f"m_{m}"]
        print(f"  m={m}: NRMSE={a['nrmse_mean']:.4f}±{a['nrmse_std']:.4f}  "
              f"ll_std={a['ll_std_mean']:.3f}")
    print(f"\n  Welch t-test m=2 vs m=3: t={t_23:.3f}  p={p_23:.4f}  "
          f"significant={p_23<0.05}")
    print(f"  Welch t-test m=1 vs m=3: t={t_13:.3f}  p={p_13:.4f}  "
          f"significant={p_13<0.05}")
    print(f"  Welch t-test m=2 vs m=5: t={t_25:.3f}  p={p_25:.4f}  "
          f"significant={p_25<0.05}")

    # Save
    output = {
        "env_log": {
            "date": time.strftime("%Y-%m-%d"),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "N_ITEMS": N_ITEMS, "N_SEEDS": N_SEEDS,
            "SEEDS": SEEDS, "N_INIT_EM": N_INIT_EM,
            "MAX_ITER": MAX_ITER, "TOL_EM": TOL_EM,
            "M_VALUES": M_VALUES, "L": L, "K_T1": K_T1,
            "DIRICHLET_ALPHA_CF": DIRICHLET_ALPHA_CF,
            "note": ("Strengthened settings: N>=2000, n_init>=100, seeds>=10. "
                     "All settings were fixed before the run; no post hoc selection."),
            "gpu_used": False, "api_calls": 0,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "statistical_tests": stats,
    }
    out_path = os.path.join(RESULTS_DIR, "e1_t1_strengthened.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}  (elapsed: {time.time()-t0:.1f}s)")

    # Figure
    make_figure(output)


def make_figure(data: dict) -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    matplotlib.rcParams.update({
        "font.family": "Times New Roman", "mathtext.fontset": "stix",
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5, "lines.markersize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    BLUE = "#2166AC"; RED = "#D6604D"

    agg = data["aggregate"]
    m_vals  = data["env_log"]["M_VALUES"]
    r_means = [agg[f"m_{m}"]["nrmse_mean"] for m in m_vals]
    r_stds  = [agg[f"m_{m}"]["nrmse_std"]  for m in m_vals]
    colors  = [RED if m < 3 else BLUE for m in m_vals]
    stats   = data["statistical_tests"]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.15)

    for m, mu, sd, c in zip(m_vals, r_means, r_stds, colors):
        ax.errorbar(m, mu, yerr=sd, fmt="o", color=c, ecolor=c,
                    elinewidth=1.0, capsize=3, capthick=0.8,
                    markersize=7, zorder=3)
    ax.plot([m for m in m_vals if m < 3], [r for m, r in zip(m_vals, r_means) if m < 3],
            "--", color=RED, lw=1.4, zorder=2)
    ax.plot([m for m in m_vals if m >= 3], [r for m, r in zip(m_vals, r_means) if m >= 3],
            "-",  color=BLUE, lw=1.4, zorder=2)
    ax.axvline(2.5, color="black", lw=0.8, ls=":", zorder=1)

    p_val = stats["welch_m2_vs_m3"]["p"]
    sig   = stats["welch_m2_vs_m3"]["significant_p05"]
    ax.text(2.55, max(r_means) * 0.99,
            f"m=2 vs m=3:\np={p_val:.3f} {'*' if sig else '(n.s.)'}",
            fontsize=8, va="top")

    p1 = mpatches.Patch(color=RED,  label=r"$m\leq2$: non-identified")
    p2 = mpatches.Patch(color=BLUE, label=r"$m\geq3$: identified (T1)")
    ax.legend(handles=[p1, p2], fontsize=8, loc="upper right",
              framealpha=0.9, edgecolor="none")
    ax.set_xlabel("Number of families $m$", labelpad=3)
    ax.set_ylabel("NRMSE (emission matrices)", labelpad=3)
    ax.set_title(f"T1 Strengthened ($N={data['env_log']['N_ITEMS']}$, "
                 f"{data['env_log']['N_SEEDS']} seeds, "
                 f"{data['env_log']['N_INIT_EM']} EM inits)", pad=5)
    ax.set_xticks(m_vals)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, lw=0.5, color="#e0e0e0", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"e1_t1_strengthened.{ext}")
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
