"""
E3: Semi-Synthetic Identification Experiment (Contract B)

Implements three measurements:
  M2: Gauge non-identification — gauge orbit diameter constant vs m
  M3: Anchor budget curve     — error ~ c * n^{-1/2}
  M4: Structural anchor effect— vertex items enable identification

Theory background (T3, T4 from the paper):
  T3 (Gauge Theorem): For distribution contract (interior profiles),
      the map (C_f, p) -> (C_f A^{-1}, A p) is observationally equivalent
      for any column-stochastic invertible A with C_f A^{-1} >= 0.
      The gauge group has dimension L(L-1); for L=2 this is 2D.

  T4 (Structural Anchor): Including L vertex items (one per relevance level)
      collapses the gauge group to permutations, enabling identification
      without human labels.

For L=2 (binary relevance), the kernel C_f is a 2x2 column-stochastic matrix:
    C_f = [[1-alpha_f, 1-beta_f],
           [alpha_f,     beta_f]]
  where:
    alpha_f = P(judge=1 | true_label=0)  (false positive rate)
    beta_f  = P(judge=1 | true_label=1)  (true positive rate)

The observation model:
    y_{f,k} ~ Bernoulli(alpha_f * (1-pi_k) + beta_f * pi_k)

External kernel injection:
    Pass a dict of {family_id: (alpha, beta)} to override synthetic kernels.
    This is the injection point for E2-calibrated kernels.

Date: 2026-08-04
"""

from __future__ import annotations

import collections
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import bootstrap

from e3_profiles import QueryDoc

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel utilities (L=2)
# ---------------------------------------------------------------------------


def make_kernel(alpha: float, beta: float) -> np.ndarray:
    """Construct 2x2 column-stochastic confusion kernel.

    C_f[:,0] = [1-alpha, alpha]  (P(judge | true=0))
    C_f[:,1] = [1-beta,  beta]   (P(judge | true=1))
    """
    return np.array([[1.0 - alpha, 1.0 - beta],
                     [alpha,       beta]])


def obs_prob(pi: float, alpha: float, beta: float) -> float:
    """P(judge=1 | pi, alpha, beta) = alpha*(1-pi) + beta*pi."""
    return alpha * (1.0 - pi) + beta * pi


def is_valid_gauge(
    a: float, b: float,
    kernels: List[Tuple[float, float]],
    eps: float = 1e-8,
) -> bool:
    """Check if A = [[a,b],[1-a,1-b]] is a valid gauge transform.

    Validity conditions:
      1. a, b in (0, 1)
      2. a != b (invertible)
      3. For all kernels (alpha_f, beta_f): C_f A^{-1} has non-negative entries
    """
    if not (eps < a < 1 - eps and eps < b < 1 - eps):
        return False
    if abs(a - b) < eps:
        return False

    det = a - b
    for alpha, beta in kernels:
        # C_f A^{-1}: compute each entry
        # C_f = [[1-alpha, 1-beta], [alpha, beta]]
        # A^{-1} = 1/(a-b) * [[1-b, -b], [-(1-a), a]]
        # Entries of C_f @ A^{-1}:
        c00 = 1.0 - alpha
        c01 = 1.0 - beta
        c10 = alpha
        c11 = beta
        # (C_f A^{-1})[i,j] = ...
        entry_00 = (c00 * (1 - b) - c01 * (1 - a)) / det
        entry_01 = (-c00 * b + c01 * a) / det
        entry_10 = (c10 * (1 - b) - c11 * (1 - a)) / det
        entry_11 = (-c10 * b + c11 * a) / det
        if entry_00 < -eps or entry_01 < -eps or entry_10 < -eps or entry_11 < -eps:
            return False
    return True


def gauge_transform_profile(a: float, b: float, pi: float) -> float:
    """Apply A = [[a,b],[1-a,1-b]] to profile [1-pi, pi].

    Returns pi' = a*(1-pi) + b*pi  (the new 'pi' value).
    Wait: A @ [1-pi, pi] = [a*(1-pi)+b*pi, (1-a)*(1-pi)+(1-b)*pi]
    New profile: pi' = (1-a)*(1-pi) + (1-b)*pi = 1 - a*(1-pi) - b*pi

    Actually let me be careful. A = [[a, b], [1-a, 1-b]].
    v = [1-pi, pi] (column vector)
    Av = [a*(1-pi) + b*pi, (1-a)*(1-pi) + (1-b)*pi]

    The 'relevant' probability is the second component:
    pi' = (1-a)*(1-pi) + (1-b)*pi = 1 - a*(1-pi) - b*pi - pi + pi
         = (1-a)*(1-pi) + (1-b)*pi

    Let's verify: if pi=0, pi' = (1-a)*1 + (1-b)*0 = 1-a. Expected: Av = [a, 1-a], so pi'=1-a. ✓
    If pi=1, pi' = (1-a)*0 + (1-b)*1 = 1-b. Expected: Av = [b, 1-b], so pi'=1-b. ✓
    """
    return (1.0 - a) * (1.0 - pi) + (1.0 - b) * pi


def gauge_orbit_diameter(
    kernels: List[Tuple[float, float]],
    pi: float,
    n_grid: int = 200,
    eps: float = 1e-4,
) -> Dict:
    """Numerically estimate gauge orbit diameter for a profile pi.

    Scans (a, b) in [eps, 1-eps]^2 for valid gauge transforms,
    measures max |pi' - pi| over valid transforms.

    Returns dict with: max_delta, min_pi_prime, max_pi_prime, n_valid.
    """
    a_grid = np.linspace(eps, 1 - eps, n_grid)
    b_grid = np.linspace(eps, 1 - eps, n_grid)

    pi_primes: List[float] = [pi]  # identity transform
    n_valid = 0
    for a in a_grid:
        for b in b_grid:
            if abs(a - b) < eps:
                continue  # singular
            if is_valid_gauge(a, b, kernels, eps=eps / 10):
                pi_p = gauge_transform_profile(a, b, pi)
                if 0.0 <= pi_p <= 1.0:
                    pi_primes.append(pi_p)
                    n_valid += 1

    return {
        "max_delta": float(max(abs(p - pi) for p in pi_primes)),
        "min_pi_prime": float(min(pi_primes)),
        "max_pi_prime": float(max(pi_primes)),
        "n_valid_transforms": n_valid,
    }


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_observations(
    profiles: List[QueryDoc],
    kernels: List[Tuple[float, float]],
    n_judgments: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic binary judgments.

    Parameters
    ----------
    profiles : list of QueryDoc (K items)
    kernels  : list of (alpha_f, beta_f) for m families
    n_judgments : N judgments per (family, item)
    rng : random generator

    Returns
    -------
    obs : shape (m, K, 2) uint32
        obs[f, k, 1] = count of '1' judgments
        obs[f, k, 0] = count of '0' judgments
        obs[f, k, 0] + obs[f, k, 1] = n_judgments
    """
    m = len(kernels)
    K = len(profiles)
    obs = np.zeros((m, K, 2), dtype=np.int32)

    for f, (alpha, beta) in enumerate(kernels):
        for k, prof in enumerate(profiles):
            p1 = obs_prob(prof.pi, alpha, beta)
            count_1 = rng.binomial(n_judgments, p1)
            obs[f, k, 1] = count_1
            obs[f, k, 0] = n_judgments - count_1

    return obs


# ---------------------------------------------------------------------------
# M2: Gauge non-identification
# ---------------------------------------------------------------------------


def measurement_gauge_nonidentification(
    interior_profiles: List[QueryDoc],
    kernels: List[Tuple[float, float]],
    m_values: List[int],
    n_judgments: int = 100,
    n_grid: int = 150,
) -> Dict:
    """Measure gauge orbit diameter vs number of families m.

    The diameter should be CONSTANT vs m (theoretical prediction of T3).

    Uses a representative subset of interior profiles.

    Returns
    -------
    dict: {m: {'mean_diameter', 'std_diameter', 'diameters'}}
    """
    if not interior_profiles:
        raise ValueError("No interior profiles provided")

    results: Dict[int, Dict] = {}

    for m in m_values:
        if m > len(kernels):
            logger.warning("m=%d > len(kernels)=%d; skipping", m, len(kernels))
            continue
        current_kernels = kernels[:m]

        diameters = []
        for prof in interior_profiles:
            info = gauge_orbit_diameter(current_kernels, prof.pi, n_grid=n_grid)
            diameters.append(info["max_delta"])

        results[m] = {
            "mean_diameter": float(np.mean(diameters)),
            "std_diameter": float(np.std(diameters)),
            "diameters": diameters,
            "n_interior": len(interior_profiles),
        }
        logger.info(
            "M2 m=%d: mean_diameter=%.4f ± %.4f",
            m, results[m]["mean_diameter"], results[m]["std_diameter"],
        )

    return results


def gauge_loglikelihood_flatness(
    pi: float,
    kernels: List[Tuple[float, float]],
    n_judgments: int,
    rng: np.random.Generator,
    n_gauge_points: int = 30,
    n_grid: int = 100,
    eps: float = 1e-4,
) -> Dict:
    """Demonstrate exact log-likelihood flatness over gauge orbit.

    For a given interior profile pi and kernels, generate observations
    and show that the log-likelihood is IDENTICAL for all points in
    the gauge orbit. This is T3's core claim: observational equivalence.

    Returns
    -------
    dict with 'a_values', 'b_values', 'loglik_diffs' (vs identity),
    and 'max_abs_loglik_diff' (should be ~0).
    """
    m = len(kernels)

    # Generate observations once (from true model, A=identity)
    obs = np.zeros((m, 2), dtype=np.int32)
    for f, (alpha, beta) in enumerate(kernels):
        p1 = obs_prob(pi, alpha, beta)
        count_1 = rng.binomial(n_judgments, p1)
        obs[f, 1] = count_1
        obs[f, 0] = n_judgments - count_1

    # True log-likelihood
    def loglik_at(pi_eval: float, kernels_eval: List[Tuple[float, float]]) -> float:
        ll = 0.0
        for f, (al, be) in enumerate(kernels_eval):
            p1 = obs_prob(pi_eval, al, be)
            p1 = np.clip(p1, 1e-15, 1 - 1e-15)
            ll += obs[f, 1] * np.log(p1) + obs[f, 0] * np.log(1 - p1)
        return float(ll)

    ll_true = loglik_at(pi, kernels)

    # Scan gauge transforms
    a_grid = np.linspace(eps, 1 - eps, n_grid)
    b_grid = np.linspace(eps, 1 - eps, n_grid)

    valid_a, valid_b, ll_diffs = [], [], []

    for a in a_grid:
        for b in b_grid:
            if abs(a - b) < eps:
                continue
            if not is_valid_gauge(a, b, kernels, eps=eps / 10):
                continue
            # Transform model: (C_f', pi') = (C_f A^{-1}, A pi)
            pi_new = gauge_transform_profile(a, b, pi)
            if not (eps < pi_new < 1 - eps):
                continue
            # New kernels: C_f' = C_f A^{-1}
            # alpha' = C_f'[1,0], beta' = C_f'[1,1]
            det = a - b
            new_kernels = []
            for alpha, beta in kernels:
                c10 = alpha
                c11 = beta
                alpha_new = (c10 * (1 - b) - c11 * (1 - a)) / det
                beta_new = (-c10 * b + c11 * a) / det
                alpha_new = float(np.clip(alpha_new, 1e-10, 1 - 1e-10))
                beta_new = float(np.clip(beta_new, 1e-10, 1 - 1e-10))
                new_kernels.append((alpha_new, beta_new))

            ll_new = loglik_at(pi_new, new_kernels)
            ll_diffs.append(ll_new - ll_true)
            valid_a.append(a)
            valid_b.append(b)

            if len(valid_a) >= n_gauge_points:
                break
        if len(valid_a) >= n_gauge_points:
            break

    return {
        "n_gauge_points": len(ll_diffs),
        "loglik_diffs": ll_diffs,
        "max_abs_loglik_diff": float(max(abs(d) for d in ll_diffs)) if ll_diffs else None,
        "mean_abs_loglik_diff": float(np.mean(np.abs(ll_diffs))) if ll_diffs else None,
        "valid_a": valid_a,
        "valid_b": valid_b,
    }


# ---------------------------------------------------------------------------
# M3: Anchor budget curve
# ---------------------------------------------------------------------------


def estimate_kernel_from_anchors(
    anchor_profiles: List[QueryDoc],
    kernels_true: List[Tuple[float, float]],
    n_judgments: int,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    """Estimate kernels from anchor items with known true profiles.

    MLE for (alpha_f, beta_f) given known pi_k values.
    Uses scipy minimize on negative log-likelihood.

    Returns estimated [(alpha_0, beta_0), ..., (alpha_{m-1}, beta_{m-1})].
    """
    m = len(kernels_true)
    K_a = len(anchor_profiles)
    pi_vals = np.array([p.pi for p in anchor_profiles])

    # Generate observations for anchor items
    obs = np.zeros((m, K_a, 2), dtype=np.int32)
    for f, (al, be) in enumerate(kernels_true):
        for k, pi in enumerate(pi_vals):
            p1 = obs_prob(pi, al, be)
            count_1 = rng.binomial(n_judgments, p1)
            obs[f, k, 1] = count_1
            obs[f, k, 0] = n_judgments - count_1

    estimated: List[Tuple[float, float]] = []
    for f in range(m):
        counts_1 = obs[f, :, 1]
        counts_0 = obs[f, :, 0]

        def neg_ll(params: np.ndarray) -> float:
            al, be = params
            if al <= 0 or be <= 0 or al >= 1 or be >= 1:
                return 1e10
            ll = 0.0
            for k, pi in enumerate(pi_vals):
                p1 = obs_prob(pi, al, be)
                p1 = np.clip(p1, 1e-10, 1 - 1e-10)
                ll += counts_1[k] * np.log(p1) + counts_0[k] * np.log(1 - p1)
            return -ll

        al0, be0 = kernels_true[f]  # warm start (not used in blind setting)
        # Random warm start to avoid oracle bias
        al_init = float(rng.uniform(0.1, 0.4))
        be_init = float(rng.uniform(0.5, 0.9))
        res = minimize(
            neg_ll,
            x0=[al_init, be_init],
            method="L-BFGS-B",
            bounds=[(1e-3, 1 - 1e-3), (1e-3, 1 - 1e-3)],
        )
        if res.success or res.fun < 1e8:
            estimated.append((float(res.x[0]), float(res.x[1])))
        else:
            estimated.append((al_init, be_init))

    return estimated


def measurement_anchor_budget(
    anchor_pool: List[QueryDoc],
    interior_profiles: List[QueryDoc],
    kernels_true: List[Tuple[float, float]],
    n_anchor_values: List[int],
    n_judgments: int = 100,
    n_rep: int = 30,
    rng: np.random.Generator = None,
) -> Dict:
    """Measure kernel estimation error vs number of anchors.

    Expects RMSE ~ c * n^{-1/2} (Le Cam lower bound prediction).

    Returns dict: {n_anchors: {'mean_rmse', 'std_rmse', 'all_rmse'}}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    results: Dict[int, Dict] = {}

    for n_a in n_anchor_values:
        if n_a > len(anchor_pool):
            logger.warning("n_anchors=%d > pool size=%d; capping", n_a, len(anchor_pool))
            n_a = len(anchor_pool)

        rmse_all: List[float] = []
        for rep in range(n_rep):
            # Sample anchor items
            idx = rng.choice(len(anchor_pool), size=n_a, replace=False)
            anchors = [anchor_pool[i] for i in idx]

            # Estimate kernels
            est_kernels = estimate_kernel_from_anchors(
                anchors, kernels_true, n_judgments, rng
            )

            # Compute RMSE over (alpha, beta) pairs
            sq_errors = []
            for (a_est, b_est), (a_true, b_true) in zip(est_kernels, kernels_true):
                sq_errors.append((a_est - a_true) ** 2 + (b_est - b_true) ** 2)
            rmse = float(np.sqrt(np.mean(sq_errors)))
            rmse_all.append(rmse)

        results[n_a] = {
            "mean_rmse": float(np.mean(rmse_all)),
            "std_rmse": float(np.std(rmse_all)),
            "all_rmse": rmse_all,
            "n_rep": n_rep,
        }
        logger.info(
            "M3 n_anchors=%d: RMSE=%.4f ± %.4f",
            n_a, results[n_a]["mean_rmse"], results[n_a]["std_rmse"],
        )

    return results


def fit_power_law(n_values: List[int], rmse_means: List[float]) -> Tuple[float, float]:
    """Fit RMSE = c * n^{-0.5} via log-log linear regression.

    Returns (c, r_squared).
    """
    log_n = np.log(np.array(n_values, dtype=float))
    log_rmse = np.log(np.array(rmse_means, dtype=float))
    # Fit: log_rmse = log_c - 0.5 * log_n
    # Regress log_rmse on log_n with slope fixed at -0.5
    # Residuals: log_rmse - (-0.5 * log_n)
    residuals = log_rmse + 0.5 * log_n  # = log_c
    log_c = float(np.mean(residuals))
    c = float(np.exp(log_c))
    # R^2 for the -0.5 slope model
    y_pred = log_c - 0.5 * log_n
    ss_res = np.sum((log_rmse - y_pred) ** 2)
    ss_tot = np.sum((log_rmse - np.mean(log_rmse)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return c, float(r2)


# ---------------------------------------------------------------------------
# M4: Structural anchor effect
# ---------------------------------------------------------------------------


def estimate_kernel_from_vertex_items(
    vertex_0: List[QueryDoc],
    vertex_1: List[QueryDoc],
    kernels_true: List[Tuple[float, float]],
    n_judgments: int,
    rng: np.random.Generator,
) -> List[Tuple[float, float]]:
    """Estimate kernels using ONLY structural anchors (vertex items).

    For vertex_0 items (pi=0): P(judge=1) = alpha_f
    For vertex_1 items (pi=1): P(judge=1) = beta_f

    No human labels needed — the item type (vertex 0 or 1) is structural
    information from the evaluation design.

    Returns estimated [(alpha_f, beta_f)] for each family.
    """
    m = len(kernels_true)
    estimated: List[Tuple[float, float]] = []

    for f, (al_true, be_true) in enumerate(kernels_true):
        # Generate observations for vertex items
        count_1_v0 = 0
        total_v0 = 0
        for item in vertex_0:
            p1 = obs_prob(item.pi, al_true, be_true)  # pi=0 -> p1=alpha
            c1 = rng.binomial(n_judgments, p1)
            count_1_v0 += c1
            total_v0 += n_judgments

        count_1_v1 = 0
        total_v1 = 0
        for item in vertex_1:
            p1 = obs_prob(item.pi, al_true, be_true)  # pi=1 -> p1=beta
            c1 = rng.binomial(n_judgments, p1)
            count_1_v1 += c1
            total_v1 += n_judgments

        alpha_est = count_1_v0 / total_v0 if total_v0 > 0 else 0.0
        beta_est = count_1_v1 / total_v1 if total_v1 > 0 else 1.0
        estimated.append((float(alpha_est), float(beta_est)))

    return estimated


def recover_profiles_from_kernel(
    interior_profiles: List[QueryDoc],
    kernels_true: List[Tuple[float, float]],
    kernels_est: List[Tuple[float, float]],
    n_judgments: int,
    rng: np.random.Generator,
) -> Dict:
    """Recover interior profile values using estimated kernels.

    Given estimated (alpha_f^est, beta_f^est), compute:
        pi_k^est = (s_{f,k}/N - alpha_f^est) / (beta_f^est - alpha_f^est)
    averaged over families.

    Returns dict with 'pi_true', 'pi_est_with', 'pi_est_without', 'rmse_with', 'rmse_without'.
    """
    m = len(kernels_true)
    K = len(interior_profiles)
    pi_true = np.array([p.pi for p in interior_profiles])

    # Generate observations
    obs_1 = np.zeros((m, K), dtype=np.float64)
    for f, (al, be) in enumerate(kernels_true):
        for k, prof in enumerate(interior_profiles):
            p1 = obs_prob(prof.pi, al, be)
            count_1 = rng.binomial(n_judgments, p1)
            obs_1[f, k] = count_1 / n_judgments

    # Recover pi using estimated kernels (structural anchor case)
    pi_est_with = np.zeros(K)
    for k in range(K):
        estimates_k = []
        for f, (al_e, be_e) in enumerate(kernels_est):
            denom = be_e - al_e
            if abs(denom) < 1e-6:
                continue
            pi_hat = (obs_1[f, k] - al_e) / denom
            estimates_k.append(pi_hat)
        pi_est_with[k] = float(np.clip(np.mean(estimates_k), 0.0, 1.0)) if estimates_k else 0.5

    # Without structural anchors: use true kernels as upper bound (oracle case)
    pi_est_oracle = np.zeros(K)
    for k in range(K):
        estimates_k = []
        for f, (al_t, be_t) in enumerate(kernels_true):
            denom = be_t - al_t
            if abs(denom) < 1e-6:
                continue
            pi_hat = (obs_1[f, k] - al_t) / denom
            estimates_k.append(pi_hat)
        pi_est_oracle[k] = float(np.clip(np.mean(estimates_k), 0.0, 1.0)) if estimates_k else 0.5

    # Without anchors: no identification (show gauge ambiguity)
    # Use a "naive" estimator that ignores the kernel (raw fraction = s_{f,k}/N averaged over f)
    pi_est_without = np.zeros(K)
    for k in range(K):
        pi_est_without[k] = float(np.mean(obs_1[:, k]))

    rmse_with = float(np.sqrt(np.mean((pi_est_with - pi_true) ** 2)))
    rmse_oracle = float(np.sqrt(np.mean((pi_est_oracle - pi_true) ** 2)))
    rmse_without = float(np.sqrt(np.mean((pi_est_without - pi_true) ** 2)))

    return {
        "pi_true": pi_true.tolist(),
        "pi_est_with_anchors": pi_est_with.tolist(),
        "pi_est_without_anchors": pi_est_without.tolist(),
        "pi_est_oracle": pi_est_oracle.tolist(),
        "rmse_with_structural_anchors": rmse_with,
        "rmse_oracle": rmse_oracle,
        "rmse_without_anchors": rmse_without,
    }


def measurement_structural_anchor(
    interior_profiles: List[QueryDoc],
    vertex_0: List[QueryDoc],
    vertex_1: List[QueryDoc],
    kernels_true: List[Tuple[float, float]],
    n_vertex_values: List[int],
    n_judgments: int = 100,
    n_rep: int = 20,
    rng: np.random.Generator = None,
) -> Dict:
    """Measure identification accuracy vs number of structural anchors.

    n_vertex_values: number of vertex_0 + vertex_1 items to include.
    For each n_v, uses n_v//2 from each type.

    Returns dict: {n_v: {'mean_rmse_with', 'mean_rmse_without', ...}}
    """
    if rng is None:
        rng = np.random.default_rng(42)

    results: Dict[int, Dict] = {}

    for n_v in n_vertex_values:
        n_each = max(1, n_v // 2)
        n0 = min(n_each, len(vertex_0))
        n1 = min(n_each, len(vertex_1))
        if n0 == 0 or n1 == 0:
            logger.warning("Not enough vertex items for n_v=%d", n_v)
            continue

        rmse_with_all: List[float] = []
        rmse_without_all: List[float] = []

        for rep in range(n_rep):
            idx0 = rng.choice(len(vertex_0), size=n0, replace=False)
            idx1 = rng.choice(len(vertex_1), size=n1, replace=False)
            v0_sample = [vertex_0[i] for i in idx0]
            v1_sample = [vertex_1[i] for i in idx1]

            # Estimate kernels from structural anchors
            kernels_est = estimate_kernel_from_vertex_items(
                v0_sample, v1_sample, kernels_true, n_judgments, rng
            )

            # Recover profiles
            rec = recover_profiles_from_kernel(
                interior_profiles, kernels_true, kernels_est, n_judgments, rng
            )
            rmse_with_all.append(rec["rmse_with_structural_anchors"])
            rmse_without_all.append(rec["rmse_without_anchors"])

        results[n_v] = {
            "mean_rmse_with": float(np.mean(rmse_with_all)),
            "std_rmse_with": float(np.std(rmse_with_all)),
            "mean_rmse_without": float(np.mean(rmse_without_all)),
            "std_rmse_without": float(np.std(rmse_without_all)),
            "n0_used": n0,
            "n1_used": n1,
            "n_rep": n_rep,
        }
        logger.info(
            "M4 n_v=%d: RMSE_with=%.4f ± %.4f | RMSE_without=%.4f ± %.4f",
            n_v,
            results[n_v]["mean_rmse_with"],
            results[n_v]["std_rmse_with"],
            results[n_v]["mean_rmse_without"],
            results[n_v]["std_rmse_without"],
        )

    return results


# ---------------------------------------------------------------------------
# Default kernel set (synthetic; E2 calibration injection point)
# ---------------------------------------------------------------------------


def default_kernels(m: int, rng: np.random.Generator, seed: int = 0) -> List[Tuple[float, float]]:
    """Generate m synthetic evaluator family kernels.

    Each family has distinct (alpha, beta) drawn from plausible ranges.
    This is the injection point for E2-calibrated kernels:
      Replace this function's output with E2-calibrated values when available.

    Plausible ranges for LLM-judge binary relevance assessment:
      alpha (FPR): 0.05–0.35 (false positive rate)
      beta  (TPR): 0.55–0.92 (true positive rate)
    """
    kernels: List[Tuple[float, float]] = []
    # Stratified: spread across the alpha-beta space for diversity
    alpha_base = np.linspace(0.08, 0.32, m)
    beta_base = np.linspace(0.60, 0.90, m)
    alpha_jitter = rng.uniform(-0.02, 0.02, size=m)
    beta_jitter = rng.uniform(-0.02, 0.02, size=m)
    for i in range(m):
        a = float(np.clip(alpha_base[i] + alpha_jitter[i], 0.05, 0.40))
        b = float(np.clip(beta_base[i] + beta_jitter[i], 0.50, 0.95))
        kernels.append((a, b))
    return kernels


def load_external_kernels(path: str) -> Optional[List[Tuple[float, float]]]:
    """Load E2-calibrated kernels from a JSON file.

    Expected JSON format:
      [{"alpha": 0.12, "beta": 0.78}, ...]

    Returns None if file not found.
    """
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as fh:
        data = json.load(fh)
    kernels = [(float(d["alpha"]), float(d["beta"])) for d in data]
    logger.info("Loaded %d external kernels from %s", len(kernels), path)
    return kernels
