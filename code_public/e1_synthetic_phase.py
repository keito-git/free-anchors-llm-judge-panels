"""
E1 Synthetic Phase Diagram Experiment
======================================
Numerically validates T1, T3 (Gauge Theorem), and T4 (Structural Anchor) for
the WSDM 2027 paper "Families, Not Judges".

Model:
  - K prototypes with true profiles p^(k) in Δ^{L-1}
  - m families (systems), each with distortion matrix C_f (L×L column-stochastic)
  - Response rule: Q_f(y | k) = (C_f p^(k))_y
  - Families are conditionally independent given prototype; within-family dependence
    is unrestricted (handled by treating the multi-judge output as a single token
    from the product alphabet).

Experiments:
  T1 – m >= 3 identifies (C_f, p^(k)) up to label permutation;  m <= 2 does not.
  T3 – Gauge Theorem: vertex profiles => gauge orbit = permutations (T3a);
       interior profiles => gauge orbit has dimension L(L-1), radius invariant in m (T3b).
  T4 – Structural Anchor: mixing in vertex items at rates 0/1/5/10% collapses the
       continuous gauge to permutations and restores identifiability.

CPU only.  No API calls.  Results saved to results/e1_synthetic_phase.json.
Figures saved to results/figures/.
"""

import json
import os
import platform
import socket
import hashlib
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Experimental constants
# ---------------------------------------------------------------------------

L = 3                          # number of labels
K_T1 = 3                      # prototypes for T1 (= L, vertex case)
K_INT = 3                      # interior prototypes for T3/T4
K_VERT = L                     # vertex prototypes for T4 (one per label)
N_ITEMS_T1 = 600               # total items for T1
N_ITEMS_T3 = 1000              # total items for T3
N_ITEMS_T4 = 800               # total items for T4
SEEDS = [42, 123, 456, 789, 1234]
N_INIT_EM = 30                 # EM random restarts per run
MAX_ITER_EM = 500
TOL_EM = 1e-9
N_GAUGE_DIRS = 200             # random directions for gauge radius

M_VALUES_T1 = [1, 2, 3, 4, 5]
M_VALUES_T3 = [3, 4, 6, 10, 20]
M_T4 = 3                       # fixed number of systems for T4
ANCHOR_RATES = [0.0, 0.01, 0.05, 0.10]

DIRICHLET_ALPHA_PROFILE = 0.5  # concentration for interior profiles (pushes toward interior)
DIRICHLET_ALPHA_CF = 2.0       # concentration for C_f columns (keeps entries positive)

# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def sample_interior(L: int, rng: np.random.Generator, alpha: float = 0.5) -> np.ndarray:
    """Sample a strictly interior point of the L-simplex."""
    eps = 0.05
    x = rng.dirichlet(alpha * np.ones(L))
    # Mix with uniform to guarantee strict interior
    return (1.0 - L * eps) * x + eps / L


def sample_column_stochastic(L: int, rng: np.random.Generator,
                              alpha: float = 2.0) -> np.ndarray:
    """Sample an L×L column-stochastic matrix with all positive entries."""
    return np.column_stack([rng.dirichlet(alpha * np.ones(L)) for _ in range(L)])


def make_vertex_profiles(L: int) -> np.ndarray:
    """Return L×L identity (L vertex profiles as rows)."""
    return np.eye(L)  # shape (L, L), row k = profile for prototype k


def make_interior_profiles(K: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Return (K, L) array of strictly interior profiles."""
    return np.array([sample_interior(L, rng, alpha=DIRICHLET_ALPHA_PROFILE)
                     for _ in range(K)])


def compute_emission_matrices(
    profiles: np.ndarray,       # (K, L)
    distortions: List[np.ndarray],  # list of m arrays (L, L)
) -> np.ndarray:
    """
    Return emission_probs of shape (K, m, L).
    emission_probs[k, f, y] = (C_f p^(k))_y
    """
    K, L = profiles.shape
    m = len(distortions)
    Q = np.zeros((K, m, L))
    for k in range(K):
        for f, C_f in enumerate(distortions):
            Q[k, f, :] = C_f @ profiles[k]
    return Q


def generate_data(
    profiles: np.ndarray,
    distortions: List[np.ndarray],
    mixing_weights: np.ndarray,
    n_items: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_items items from the panel model.

    Returns
    -------
    labels : (n_items, m) int array
    proto_idx : (n_items,) int array
    """
    K = profiles.shape[0]
    m = len(distortions)
    emission = compute_emission_matrices(profiles, distortions)  # (K, m, L)
    proto_idx = rng.choice(K, size=n_items, p=mixing_weights)
    labels = np.zeros((n_items, m), dtype=int)
    for i in range(n_items):
        k = proto_idx[i]
        for f in range(m):
            p = emission[k, f, :]
            p = np.maximum(p, 0.0)
            p = p / p.sum()  # renormalize to fix floating-point drift
            labels[i, f] = rng.choice(L, p=p)
    return labels, proto_idx


# ---------------------------------------------------------------------------
# Log-likelihood (unstructured LCM)
# ---------------------------------------------------------------------------


def log_likelihood_lcm(
    labels: np.ndarray,    # (n, m)
    pi: np.ndarray,        # (K,)
    Q_list: List[np.ndarray],  # m lists of (L, K)
) -> float:
    """Log-likelihood of data under the unstructured latent class model."""
    n, m = labels.shape
    K = pi.shape[0]
    log_pi = np.log(pi + 1e-300)  # (K,)
    # Accumulate log P(y | k) summed over systems
    log_joint = np.tile(log_pi, (n, 1))  # (n, K)
    for f in range(m):
        # Q_list[f] has shape (L, K); index by labels[:, f] → (n, K)
        log_joint += np.log(Q_list[f][labels[:, f], :] + 1e-300)
    # Log-sum-exp over K
    lse_max = log_joint.max(axis=1, keepdims=True)
    ll = (lse_max.squeeze() + np.log(np.exp(log_joint - lse_max).sum(axis=1))).sum()
    return float(ll)


# ---------------------------------------------------------------------------
# EM algorithm (unstructured LCM)
# ---------------------------------------------------------------------------


def em_lcm(
    labels: np.ndarray,   # (n, m)
    K: int,
    L: int,
    n_init: int = N_INIT_EM,
    max_iter: int = MAX_ITER_EM,
    tol: float = TOL_EM,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, List[np.ndarray], float, List[float]]:
    """
    EM for the unstructured latent class model.

    Parameters
    ----------
    labels : (n, m) int array of observed labels
    K : number of latent prototypes
    L : number of label categories
    n_init : number of random restarts
    max_iter : max EM iterations per restart
    tol : convergence tolerance on log-likelihood change

    Returns
    -------
    pi_best : (K,) mixing weights
    Q_best : list of m (L, K) emission matrices
    ll_best : best log-likelihood found
    all_lls : final log-likelihood of every restart (for variance analysis)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, m = labels.shape
    best_ll = -np.inf
    best_pi = None
    best_Q = None
    all_lls: List[float] = []

    for _ in range(n_init):
        # Random initialisation
        pi = rng.dirichlet(np.ones(K))
        Q = [np.column_stack([rng.dirichlet(np.ones(L)) for _ in range(K)])
             for _ in range(m)]  # list of m arrays (L, K)

        prev_ll = -np.inf
        for it in range(max_iter):
            # --- E-step ---
            log_r = np.tile(np.log(pi + 1e-300), (n, 1))  # (n, K)
            for f in range(m):
                log_r += np.log(Q[f][labels[:, f], :] + 1e-300)
            lse_max = log_r.max(axis=1, keepdims=True)
            r = np.exp(log_r - lse_max)
            r /= r.sum(axis=1, keepdims=True)   # (n, K) posterior

            ll = (lse_max.squeeze() + np.log(
                np.exp(log_r - lse_max).sum(axis=1))).sum()

            # --- M-step ---
            r_sum = r.sum(axis=0) + 1e-10  # (K,)
            pi = r_sum / r_sum.sum()

            for f in range(m):
                new_Q = np.zeros((L, K))
                for y in range(L):
                    mask = labels[:, f] == y
                    new_Q[y, :] = r[mask, :].sum(axis=0) + 1e-12
                new_Q /= new_Q.sum(axis=0, keepdims=True)
                Q[f] = new_Q

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        all_lls.append(float(ll))
        if ll > best_ll:
            best_ll = ll
            best_pi = pi.copy()
            best_Q = [q.copy() for q in Q]

    return best_pi, best_Q, float(best_ll), all_lls


# ---------------------------------------------------------------------------
# Permutation alignment and NRMSE
# ---------------------------------------------------------------------------


def align_by_permutation(
    Q_true: np.ndarray,  # (L, K)
    Q_est: np.ndarray,   # (L, K)
) -> np.ndarray:
    """Return column-permuted Q_est that minimises Frobenius distance to Q_true."""
    K = Q_true.shape[1]
    cost = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            cost[j, k] = np.sum((Q_true[:, j] - Q_est[:, k]) ** 2)
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


def nrmse_Q(
    Q_true_list: List[np.ndarray],
    Q_est_list: List[np.ndarray],
    perm: np.ndarray,
) -> float:
    """Normalised RMSE between true and estimated emission matrices (after permutation)."""
    errors = []
    for Qt, Qe in zip(Q_true_list, Q_est_list):
        Qe_perm = Qe[:, perm]
        denom = np.sqrt(np.mean(Qt ** 2))
        errors.append(np.sqrt(np.mean((Qt - Qe_perm) ** 2)) / (denom + 1e-12))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# T1 experiment: m=1..5, vertex profiles
# ---------------------------------------------------------------------------


def run_T1(rng_master: np.random.Generator) -> Dict:
    """
    Validate that m >= 3 enables identification (vertex / point-contract case).
    """
    print("  Running T1 ...")
    results_per_seed = {}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        seed_results = {}

        # Generate fixed true parameters for this seed
        profiles = make_vertex_profiles(L)   # (L, L)
        # We need up to max(M_VALUES_T1) = 5 distortions
        distortions_all = [sample_column_stochastic(L, rng, alpha=DIRICHLET_ALPHA_CF)
                           for _ in range(max(M_VALUES_T1))]
        mixing_weights = np.ones(K_T1) / K_T1

        # True Q matrices (= distortion matrices for vertex profiles)
        # Q_f[:, k] = C_f e_k = column k of C_f
        Q_true_all = [C_f.copy() for C_f in distortions_all]

        for m in M_VALUES_T1:
            distortions_m = distortions_all[:m]
            Q_true_m = Q_true_all[:m]

            # Generate data
            data_rng = np.random.default_rng(seed + m * 10000)
            labels, _ = generate_data(
                profiles, distortions_m, mixing_weights, N_ITEMS_T1, data_rng)

            # Run EM
            em_rng = np.random.default_rng(seed + m * 20000)
            _, Q_est_list, best_ll, all_lls = em_lcm(
                labels, K_T1, L, n_init=N_INIT_EM, rng=em_rng)

            # Align and compute NRMSE (use first system for alignment)
            perm = align_by_permutation(Q_true_m[0], Q_est_list[0])
            nrmse = nrmse_Q(Q_true_m, Q_est_list, perm)

            # Solution variance: std of all_lls (flat landscape = non-identified)
            ll_std = float(np.std(all_lls))
            ll_range = float(np.max(all_lls) - np.min(all_lls))

            seed_results[f"m_{m}"] = {
                "nrmse": nrmse,
                "best_ll": best_ll,
                "ll_std": ll_std,
                "ll_range": ll_range,
            }
            print(f"    seed={seed} m={m}: NRMSE={nrmse:.4f}  ll_std={ll_std:.2f}")

        results_per_seed[f"seed_{seed}"] = seed_results

    # Aggregate over seeds
    aggregate = {}
    for m in M_VALUES_T1:
        key = f"m_{m}"
        nrmse_vals = [results_per_seed[f"seed_{s}"][key]["nrmse"] for s in SEEDS]
        ll_std_vals = [results_per_seed[f"seed_{s}"][key]["ll_std"] for s in SEEDS]
        aggregate[key] = {
            "nrmse_mean": float(np.mean(nrmse_vals)),
            "nrmse_std": float(np.std(nrmse_vals)),
            "ll_std_mean": float(np.mean(ll_std_vals)),
            "identified": bool(np.mean(nrmse_vals) < 0.10),
        }

    return {"per_seed": results_per_seed, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# T3 experiment: gauge analysis with interior profiles
# ---------------------------------------------------------------------------


def is_gauge_feasible(
    A: np.ndarray,
    profiles: np.ndarray,   # (K, L)
    distortions: List[np.ndarray],
    eps: float = 1e-8,
) -> bool:
    """
    Check whether A belongs to the gauge group G:
      (i)  1^T A = 1^T  (column sums = 1)
      (ii) A p^(k) >= 0  for all k
      (iii) C_f A^{-1} >= 0  for all f
    """
    # (i) column sums
    if not np.allclose(A.sum(axis=0), 1.0, atol=eps):
        return False
    # (ii) profiles map into Δ
    for p_k in profiles:
        if np.any(A @ p_k < -eps):
            return False
    # (iii) distortions remain non-negative after right-multiply by A^{-1}
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False
    for C_f in distortions:
        if np.any(C_f @ A_inv < -eps):
            return False
    return True


def gauge_radius_in_direction(
    D: np.ndarray,             # (L, L) direction matrix (column sums = 0)
    profiles: np.ndarray,      # (K, L)
    distortions: List[np.ndarray],
    n_bisect: int = 60,
) -> float:
    """
    Binary-search for the maximum t > 0 such that I + t*D is in G.
    Returns 0.0 if even t=1e-8 is infeasible.
    """
    L = D.shape[0]
    D_norm = D / (np.linalg.norm(D) + 1e-15)
    lo, hi = 0.0, 10.0
    # Quick check: if even a tiny step is infeasible, return 0
    if not is_gauge_feasible(np.eye(L) + 1e-8 * D_norm, profiles, distortions):
        return 0.0
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        A = np.eye(L) + mid * D_norm
        if is_gauge_feasible(A, profiles, distortions):
            lo = mid
        else:
            hi = mid
    return float(lo)


def compute_gauge_diameter(
    profiles: np.ndarray,
    distortions: List[np.ndarray],
    n_dirs: int = N_GAUGE_DIRS,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Estimate the gauge orbit radius using N_GAUGE_DIRS random directions.
    Returns (mean_radius, max_radius, std_radius).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    L = profiles.shape[1]
    radii = []
    for _ in range(n_dirs):
        D = rng.standard_normal((L, L))
        D -= D.mean(axis=0, keepdims=True)  # zero column sums
        r = gauge_radius_in_direction(D, profiles, distortions)
        radii.append(r)
    radii_arr = np.array(radii)
    return (float(radii_arr.mean()), float(radii_arr.max()), float(radii_arr.std()))


def verify_ll_flatness(
    labels: np.ndarray,
    pi: np.ndarray,
    profiles: np.ndarray,
    distortions: List[np.ndarray],
    A: np.ndarray,
) -> float:
    """
    Verify that log-likelihood is invariant under gauge transform A.
    Returns the absolute difference |LL_original - LL_transformed|.
    Mathematically should be 0 (machine precision).
    """
    def ll_structured(profs: np.ndarray, dists: List[np.ndarray]) -> float:
        n, m = labels.shape
        K = profs.shape[0]
        log_pi = np.log(pi + 1e-300)
        log_joint = np.tile(log_pi, (n, 1))
        for f, C_f in enumerate(dists):
            # Q_f[:, k] = C_f @ profs[k]  =>  Q_f shape (L, K)
            Q_f = (C_f @ profs.T)  # (L, K)
            log_joint += np.log(Q_f[labels[:, f], :] + 1e-300)
        lse_max = log_joint.max(axis=1, keepdims=True)
        return float((lse_max.squeeze() + np.log(
            np.exp(log_joint - lse_max).sum(axis=1))).sum())

    ll_orig = ll_structured(profiles, distortions)

    # Apply gauge transform: p^(k) -> A p^(k), C_f -> C_f A^{-1}
    A_inv = np.linalg.inv(A)
    profiles_transformed = (A @ profiles.T).T      # (K, L)
    distortions_transformed = [C_f @ A_inv for C_f in distortions]

    ll_trans = ll_structured(profiles_transformed, distortions_transformed)
    return abs(ll_orig - ll_trans)


def sample_random_gauge_A(
    profiles: np.ndarray,
    distortions: List[np.ndarray],
    rng: np.random.Generator,
    n_try: int = 100,
) -> Optional[np.ndarray]:
    """
    Sample a random feasible gauge transform A != I.
    Returns None if no feasible non-identity A found.
    """
    L = profiles.shape[1]
    for _ in range(n_try):
        D = rng.standard_normal((L, L))
        D -= D.mean(axis=0, keepdims=True)
        D_norm = D / (np.linalg.norm(D) + 1e-15)
        r = gauge_radius_in_direction(D, profiles, distortions, n_bisect=40)
        if r > 1e-4:
            t = r * 0.5  # use half the radius for safety
            return np.eye(L) + t * D_norm
    return None


def run_T3(rng_master: np.random.Generator) -> Dict:
    """
    Validate the Gauge Theorem (T3):
      (a) Vertex profiles => gauge orbit = permutations (radius 0).
      (b) Interior profiles => gauge orbit dimension L(L-1); radius invariant in m.
    """
    print("  Running T3 ...")
    results = {}

    # --- T3(a): Vertex profiles — gauge radius should be 0 ---
    print("    T3(a): vertex profiles")
    vertex_profiles = make_vertex_profiles(L)  # (L, L)
    t3a_radii = {}
    for m in [3, 10, 20]:
        rng = np.random.default_rng(SEEDS[0] + m)
        dists = [sample_column_stochastic(L, rng, DIRICHLET_ALPHA_CF) for _ in range(m)]
        gauge_rng = np.random.default_rng(SEEDS[0] + m + 99)
        mean_r, max_r, std_r = compute_gauge_diameter(
            vertex_profiles, dists, n_dirs=50, rng=gauge_rng)
        t3a_radii[f"m_{m}"] = {"mean_radius": mean_r, "max_radius": max_r}
        print(f"      m={m}: mean_radius={mean_r:.6f}  max_radius={max_r:.6f}")
    results["T3a_vertex"] = t3a_radii

    # --- T3(b): Interior profiles — gauge radius invariant in m ---
    print("    T3(b): interior profiles")
    t3b_per_seed = {}
    # Pre-generate up to 20 distortions per seed
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        profiles = make_interior_profiles(K_INT, L, rng)
        distortions_all = [sample_column_stochastic(L, rng, DIRICHLET_ALPHA_CF)
                           for _ in range(max(M_VALUES_T3))]
        mixing_weights = np.ones(K_INT) / K_INT

        seed_results = {}
        for m in M_VALUES_T3:
            dists_m = distortions_all[:m]
            gauge_rng = np.random.default_rng(seed + m * 777)

            # Gauge radius
            mean_r, max_r, std_r = compute_gauge_diameter(
                profiles, dists_m, n_dirs=N_GAUGE_DIRS, rng=gauge_rng)

            # Log-likelihood flatness check
            data_rng = np.random.default_rng(seed + m * 31)
            labels, _ = generate_data(
                profiles, dists_m, mixing_weights, N_ITEMS_T3, data_rng)

            pi_true = mixing_weights.copy()
            A_sample = sample_random_gauge_A(profiles, dists_m, gauge_rng)
            if A_sample is not None:
                ll_diff = verify_ll_flatness(labels, pi_true, profiles, dists_m, A_sample)
            else:
                ll_diff = float("nan")

            seed_results[f"m_{m}"] = {
                "gauge_radius_mean": mean_r,
                "gauge_radius_max": max_r,
                "gauge_radius_std": std_r,
                "ll_diff_under_gauge": ll_diff,
            }
            print(f"    seed={seed} m={m}: gauge_mean={mean_r:.4f}  "
                  f"gauge_max={max_r:.4f}  ll_diff={ll_diff:.2e}")
        t3b_per_seed[f"seed_{seed}"] = seed_results

    # Aggregate T3b
    t3b_aggregate = {}
    for m in M_VALUES_T3:
        key = f"m_{m}"
        radii = [t3b_per_seed[f"seed_{s}"][key]["gauge_radius_mean"] for s in SEEDS]
        diffs = [t3b_per_seed[f"seed_{s}"][key]["ll_diff_under_gauge"] for s in SEEDS]
        diffs_clean = [d for d in diffs if not np.isnan(d)]
        t3b_aggregate[key] = {
            "gauge_radius_mean": float(np.mean(radii)),
            "gauge_radius_std": float(np.std(radii)),
            "ll_diff_mean": float(np.mean(diffs_clean)) if diffs_clean else float("nan"),
            "ll_diff_max": float(np.max(diffs_clean)) if diffs_clean else float("nan"),
        }

    results["T3b_interior"] = {"per_seed": t3b_per_seed, "aggregate": t3b_aggregate}
    return results


# ---------------------------------------------------------------------------
# T4 experiment: structural anchor — mixing vertex items
# ---------------------------------------------------------------------------


def run_T4(rng_master: np.random.Generator) -> Dict:
    """
    Validate the Structural Anchor theorem (T4):
    Mixing in vertex items at rates 0/1/5/10% collapses the gauge orbit and
    restores identifiability.

    Two metrics:
      (A) Gauge radius: how large is the feasible gauge orbit for the mixed prototype set.
      (B) NRMSE of recovered Q matrices after running EM.
    """
    print("  Running T4 ...")
    results_per_seed = {}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        seed_results = {}

        # True parameters (shared across anchor rates)
        interior_profiles = make_interior_profiles(K_INT, L, rng)  # (K_INT, L)
        vertex_profiles = make_vertex_profiles(L)                   # (L, L)
        distortions = [sample_column_stochastic(L, rng, DIRICHLET_ALPHA_CF)
                       for _ in range(M_T4)]

        for anchor_rate in ANCHOR_RATES:
            # Build mixed prototype set
            if anchor_rate == 0.0:
                # No vertex items: K = K_INT interior prototypes only
                profiles_mix = interior_profiles.copy()     # (K_INT, L)
                K_mix = K_INT
                weights = np.ones(K_INT) / K_INT
            else:
                # Vertex prototypes included; anchor_rate = fraction of items from them
                # Each vertex prototype gets weight anchor_rate/L
                # Each interior prototype gets weight (1-anchor_rate)/K_INT
                profiles_mix = np.vstack([vertex_profiles, interior_profiles])  # (L+K_INT, L)
                K_mix = L + K_INT
                w_vert = anchor_rate / L
                w_int = (1.0 - anchor_rate) / K_INT
                weights = np.array([w_vert] * L + [w_int] * K_INT)
                weights /= weights.sum()

            # True emission matrices Q_f (L × K_mix)
            Q_true_list = [(C_f @ profiles_mix.T) for C_f in distortions]  # list of (L, K_mix)

            # Gauge radius analysis
            gauge_rng = np.random.default_rng(seed + int(anchor_rate * 100) + 55)
            gauge_mean, gauge_max, gauge_std = compute_gauge_diameter(
                profiles_mix, distortions, n_dirs=N_GAUGE_DIRS // 2, rng=gauge_rng)

            # EM recovery
            data_rng = np.random.default_rng(seed + int(anchor_rate * 10000) + 100)
            labels, _ = generate_data(
                profiles_mix, distortions, weights, N_ITEMS_T4, data_rng)

            em_rng = np.random.default_rng(seed + int(anchor_rate * 10000) + 200)
            _, Q_est_list, best_ll, all_lls = em_lcm(
                labels, K_mix, L, n_init=N_INIT_EM, rng=em_rng)

            # Permutation alignment and NRMSE
            perm = align_by_permutation(Q_true_list[0], Q_est_list[0])
            nrmse = nrmse_Q(Q_true_list, Q_est_list, perm)

            # Solution variance: how spread are different EM restarts?
            ll_std = float(np.std(all_lls))
            ll_range = float(np.max(all_lls) - np.min(all_lls))

            rate_key = f"rate_{anchor_rate:.2f}"
            seed_results[rate_key] = {
                "anchor_rate": anchor_rate,
                "K_mix": K_mix,
                "gauge_radius_mean": gauge_mean,
                "gauge_radius_max": gauge_max,
                "gauge_radius_std": gauge_std,
                "nrmse": nrmse,
                "best_ll": best_ll,
                "ll_std": ll_std,
                "ll_range": ll_range,
            }
            print(f"    seed={seed} rate={anchor_rate:.2f}: "
                  f"gauge_mean={gauge_mean:.4f}  nrmse={nrmse:.4f}  "
                  f"ll_std={ll_std:.2f}")

        results_per_seed[f"seed_{seed}"] = seed_results

    # Aggregate over seeds
    aggregate = {}
    for anchor_rate in ANCHOR_RATES:
        rate_key = f"rate_{anchor_rate:.2f}"
        nrmse_vals = [results_per_seed[f"seed_{s}"][rate_key]["nrmse"] for s in SEEDS]
        gauge_vals = [results_per_seed[f"seed_{s}"][rate_key]["gauge_radius_mean"]
                      for s in SEEDS]
        ll_std_vals = [results_per_seed[f"seed_{s}"][rate_key]["ll_std"] for s in SEEDS]
        aggregate[rate_key] = {
            "anchor_rate": anchor_rate,
            "nrmse_mean": float(np.mean(nrmse_vals)),
            "nrmse_std": float(np.std(nrmse_vals)),
            "gauge_radius_mean": float(np.mean(gauge_vals)),
            "gauge_radius_std": float(np.std(gauge_vals)),
            "ll_std_mean": float(np.mean(ll_std_vals)),
        }

    return {"per_seed": results_per_seed, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Environment log
# ---------------------------------------------------------------------------


def compute_data_hash(labels: np.ndarray) -> str:
    return hashlib.md5(labels.tobytes()).hexdigest()


def build_env_log() -> Dict:
    return {
        "date": time.strftime("%Y-%m-%d"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "seeds": SEEDS,
        "L": L,
        "K_T1": K_T1,
        "K_INT": K_INT,
        "K_VERT": K_VERT,
        "N_ITEMS_T1": N_ITEMS_T1,
        "N_ITEMS_T3": N_ITEMS_T3,
        "N_ITEMS_T4": N_ITEMS_T4,
        "N_INIT_EM": N_INIT_EM,
        "MAX_ITER_EM": MAX_ITER_EM,
        "TOL_EM": TOL_EM,
        "N_GAUGE_DIRS": N_GAUGE_DIRS,
        "M_VALUES_T1": M_VALUES_T1,
        "M_VALUES_T3": M_VALUES_T3,
        "M_T4": M_T4,
        "ANCHOR_RATES": ANCHOR_RATES,
        "DIRICHLET_ALPHA_PROFILE": DIRICHLET_ALPHA_PROFILE,
        "DIRICHLET_ALPHA_CF": DIRICHLET_ALPHA_CF,
        "gpu_used": False,
        "api_calls": 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start_time = time.time()
    print("E1 Synthetic Phase Diagram Experiment")
    print("=" * 50)
    print(f"L={L}, seeds={SEEDS}, N_init_EM={N_INIT_EM}, N_gauge_dirs={N_GAUGE_DIRS}")
    print()

    rng_master = np.random.default_rng(0)

    print("[ T1 ] Vertex profiles — identification requires m >= 3")
    t1_results = run_T1(rng_master)
    print()

    print("[ T3 ] Gauge Theorem — interior profiles, orbit invariant in m")
    t3_results = run_T3(rng_master)
    print()

    print("[ T4 ] Structural Anchor — vertex item mixing collapses gauge")
    t4_results = run_T4(rng_master)
    print()

    # Assemble and save JSON
    output = {
        "env_log": build_env_log(),
        "T1": t1_results,
        "T3": t3_results,
        "T4": t4_results,
    }

    out_path = os.path.join(RESULTS_DIR, "e1_synthetic_phase.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"Results saved to: {out_path}")
    print(f"Total elapsed: {elapsed:.1f} s")

    # Print summary to stdout
    print("\n===== SUMMARY =====")
    print("T1 (vertex profiles, identification threshold at m=3):")
    for m in M_VALUES_T1:
        agg = t1_results["aggregate"][f"m_{m}"]
        print(f"  m={m}: NRMSE={agg['nrmse_mean']:.4f} ± {agg['nrmse_std']:.4f}"
              f"  identified={agg['identified']}")

    print("\nT3b (interior profiles, gauge radius vs m):")
    t3b_agg = t3_results["T3b_interior"]["aggregate"]
    for m in M_VALUES_T3:
        agg = t3b_agg[f"m_{m}"]
        print(f"  m={m}: gauge_radius={agg['gauge_radius_mean']:.4f} ± "
              f"{agg['gauge_radius_std']:.4f}  "
              f"ll_diff_max={agg['ll_diff_max']:.2e}")

    print("\nT3a (vertex profiles, gauge radius should be 0):")
    for m in [3, 10, 20]:
        agg = t3_results["T3a_vertex"][f"m_{m}"]
        print(f"  m={m}: gauge_radius_mean={agg['mean_radius']:.6f}  "
              f"max={agg['max_radius']:.6f}")

    print("\nT4 (structural anchor, NRMSE and gauge radius vs anchor rate):")
    t4_agg = t4_results["aggregate"]
    for r in ANCHOR_RATES:
        agg = t4_agg[f"rate_{r:.2f}"]
        print(f"  rate={r:.2f}: NRMSE={agg['nrmse_mean']:.4f} ± {agg['nrmse_std']:.4f}"
              f"  gauge={agg['gauge_radius_mean']:.4f} ± {agg['gauge_radius_std']:.4f}")


if __name__ == "__main__":
    main()
