#!/usr/bin/env python3
"""
e2_experiment.py  --  E2: Contract-A real-data experiment (WSDM 2027).

All 33 LLMJudge judges are used (full mapping from Ito, 2026-08-04).
Three family definitions are pre-registered:

  Definition A [main]:    GPT (7) / Llama (19) / T5-pipeline (7)  -- 3 families
  Definition B [sens-1]:  GPT+T5 merged (14) / Llama (19)         -- 2 families
                          Hypothesis: T5 shares GPT-3.5 rubric -> same family
  Definition C [sens-2]:  GPT (7) / Llama-8B (18) / Llama-70B (1) / T5 (7)
                          -- 4 families: model-size split

All three definitions are run and reported. Fishing is prohibited.

Measurements
------------
M1: Effective-family-count GoF test (per definition)
    (a) Per-family Kruskal rank via SVD of N x 4 item-label-dist matrix
        Check Kruskal condition:  sum kruskal_rank(C_f) >= 2K+2  for K in {2,3,4}
    (b) GMM BIC/AIC on item-level family-mean feature vectors (k=1..n_fam+2)
    (c) ANOVA + pairwise Welch t-tests on family-mean item scores

M2: Leave-one-family-out (LOFO) over-identification test (per definition)
    Predict held-out family's item scores from remaining families.
    Evaluation: Spearman rho + bootstrap 95% CI + Cohen weighted kappa + RMSE.

M3: Within- vs. cross-family correlation (Definition A main; B/C for comparison)
    Metrics: Spearman rho (0-3 ordinal), Cohen weighted kappa, Pearson phi (binary)
    Reference: arXiv:2605.29800 (within phi~0.437, cross phi~0.389, diff~0.048)
    FT stratification within Llama: FT vs non-FT sub-group analysis.

Constraints
-----------
- CPU only, zero GPU, zero API calls
- Seeds: [42, 123, 456, 789, 1024]  (5 seeds for GMM / bootstrap)
- Existing files: never deleted or overwritten
- Output prefix: e2_  (all files)

Date:   2026-08-04
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, ttest_ind
from sklearn.metrics import cohen_kappa_score
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Matplotlib global style ──────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family":      "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "figure.dpi":       300,
    }
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR    = PROJECT_ROOT / "data"
SUBS_DIR    = DATA_DIR / "llmjudge_submissions"
MAP_FILE    = DATA_DIR / "family_mapping.json"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGS_DIR    = RESULTS_DIR / "e2_figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

SCORES:      List[int]  = [0, 1, 2, 3]
SEEDS:       List[int]  = [42, 123, 456, 789, 1024]
N_BOOTSTRAP: int        = 2000

# ── Color-blind-safe palette (Okabe-Ito) ─────────────────────────────────────
C_BLUE  = "#0072B2"
C_GREEN = "#009E73"
C_RED   = "#D55E00"
C_PURP  = "#CC79A7"
C_GRAY  = "#999999"
C_ORG   = "#E69F00"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Family-Definition Registry (pre-registered before analysis)
# ═══════════════════════════════════════════════════════════════════════════════

def build_definitions(family_json: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """
    Construct three pre-registered family definitions from the full mapping JSON.

    Returns dict: def_label -> {family_label -> [judge_name, ...]}
    All definitions use ALL 33 public runs.
    """
    # Index by judge name
    judge_info = {k: v for k, v in family_json.items() if not k.startswith("_")}

    gpt_judges   = sorted(j for j, v in judge_info.items() if v["family"] == "GPT")
    llama_judges = sorted(j for j, v in judge_info.items() if v["family"] == "Llama")
    t5_judges    = sorted(j for j, v in judge_info.items() if v["family"] == "T5")

    # Llama sub-splits for Definition C
    llama_8b  = sorted(
        j for j in llama_judges
        if "70B" not in judge_info[j].get("model", "")
    )
    llama_70b = sorted(
        j for j in llama_judges
        if "70B" in judge_info[j].get("model", "")
    )

    definitions: Dict[str, Dict[str, List[str]]] = {
        # ── Definition A: Main analysis (3 families) ─────────────────────────
        "A": {
            "GPT":   gpt_judges,    # 7
            "Llama": llama_judges,  # 19
            "T5":    t5_judges,     # 7
        },
        # ── Definition B: Sensitivity-1 (2 families; GPT+T5 merged) ──────────
        # Hypothesis: TREMA pipeline shares GPT-3.5 rubric -> same systematic error
        "B": {
            "GPT_T5": gpt_judges + t5_judges,  # 14
            "Llama":  llama_judges,             # 19
        },
        # ── Definition C: Sensitivity-2 (4 families; Llama size split) ───────
        # Hypothesis: Llama-8B and Llama-70B have distinct confusion matrices
        "C": {
            "GPT":       gpt_judges,   # 7
            "Llama_8B":  llama_8b,     # 18
            "Llama_70B": llama_70b,    # 1
            "T5":        t5_judges,    # 7
        },
    }

    log.info("Family definitions constructed:")
    for d_lbl, d in definitions.items():
        counts = {f: len(js) for f, js in d.items()}
        log.info("  Def %s: %s  (total=%d)", d_lbl, counts, sum(counts.values()))

    return definitions


def build_ft_labels(family_json: Dict[str, Any]) -> Dict[str, bool]:
    """Return {judge_name: fine_tuned (bool)} for all non-meta entries."""
    return {
        k: bool(v.get("fine_tuned", False))
        for k, v in family_json.items()
        if not k.startswith("_")
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_submission(filepath: Path) -> pd.Series:
    """
    Parse one LLMJudge submission file.
    Format: ``query_id  iter  doc_id  score``  (whitespace-separated, 0-3)
    Returns Series with MultiIndex (query_id, doc_id) and name = stem.
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["query_id", "iter", "doc_id", "score"],
        dtype={"query_id": str, "iter": str, "doc_id": str, "score": int},
    )
    idx = pd.MultiIndex.from_arrays(
        [df["query_id"].values, df["doc_id"].values],
        names=["query_id", "doc_id"],
    )
    return pd.Series(df["score"].values, index=idx, name=filepath.stem)


def load_all_judges(definitions: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
    """
    Load all judges appearing in any definition.
    Returns DataFrame: shape (n_items, n_judges_total).
    """
    all_judges: List[str] = sorted(
        set(j for d in definitions.values() for jlist in d.values() for j in jlist)
    )
    series_list = []
    missing = []
    for judge in all_judges:
        fp = SUBS_DIR / f"{judge}.txt"
        if not fp.exists():
            missing.append(judge)
            continue
        series_list.append(load_submission(fp))

    if missing:
        raise FileNotFoundError(
            f"Submission files not found: {missing}. Cannot proceed."
        )

    matrix = pd.concat(series_list, axis=1)
    before = len(matrix)
    matrix = matrix.dropna()
    after  = len(matrix)
    if before != after:
        log.warning(
            "Dropped %d rows with NaN (expected 0: all judges should share items).",
            before - after,
        )

    log.info(
        "All judges loaded: %d items x %d judges",
        matrix.shape[0], matrix.shape[1],
    )
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════

def cohen_kappa_w(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's weighted kappa, linear weights, labels 0-3."""
    return float(
        cohen_kappa_score(
            a.astype(int), b.astype(int), weights="linear", labels=SCORES
        )
    )


def pearson_phi_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson phi on binarized scores: 0->0, {1,2,3}->1."""
    ab = (a > 0).astype(float)
    bb = (b > 0).astype(float)
    rho, _ = stats.pearsonr(ab, bb)
    return float(rho)


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(a, b)
    return float(rho)


def bootstrap_diff_ci(
    wx: np.ndarray,
    cx: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap CI for mean(wx) - mean(cx) by group-resampling."""
    rng   = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    nw, nc = len(wx), len(cx)
    for i in range(n_boot):
        iw = rng.integers(0, nw, size=nw)
        ic = rng.integers(0, nc, size=nc)
        diffs[i] = wx[iw].mean() - cx[ic].mean()
    obs = float(wx.mean() - cx.mean())
    return {
        "observed":  obs,
        "ci_lo": float(np.quantile(diffs, alpha / 2)),
        "ci_hi": float(np.quantile(diffs, 1 - alpha / 2)),
    }


def welch_1sided_greater(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Welch's t-test, H1: mean(a) > mean(b)."""
    t, p2 = ttest_ind(a, b, equal_var=False)
    p1 = float(p2 / 2 if t > 0 else 1 - p2 / 2)
    return {"t": float(t), "p_two_sided": float(p2), "p_one_sided": p1}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def family_mean_features(
    matrix: pd.DataFrame,
    fam_def: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Item-level mean score per family.
    Returns DataFrame (n_items, n_families), columns = family labels.
    """
    out: Dict[str, np.ndarray] = {}
    for fam, judges in fam_def.items():
        avail = [j for j in judges if j in matrix.columns]
        out[fam] = matrix[avail].mean(axis=1).values
    return pd.DataFrame(out, index=matrix.index)


def item_label_dist(
    matrix: pd.DataFrame,
    fam_def: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """
    Per-family empirical label distribution per item.
    Returns {family -> ndarray (n_items, 4)}.
    X[i, l] = fraction of judges in that family giving score l to item i.
    """
    N = len(matrix)
    result: Dict[str, np.ndarray] = {}
    for fam, judges in fam_def.items():
        avail = [j for j in judges if j in matrix.columns]
        X = matrix[avail].values.astype(int)
        dist = np.stack([(X == l).mean(axis=1) for l in SCORES], axis=1)
        result[fam] = dist
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. M1: Effective-Family-Count GoF Test
# ═══════════════════════════════════════════════════════════════════════════════

def run_m1_for_def(
    matrix: pd.DataFrame,
    fam_def: Dict[str, List[str]],
    def_label: str,
) -> Dict[str, Any]:
    """
    M1 for one family definition.

    (a) Per-family Kruskal rank (SVD on N x 4 item-label-dist matrix)
        Practical rank = #{sigma_i / sigma_1 > 0.01}
        Kruskal condition: sum_{f} k_rank(C_f) >= 2K+2  for K in {2,3,4}

    (b) GMM BIC/AIC on item-level family-mean vectors (k=1..n_fam+2)

    (c) ANOVA + pairwise Welch t-tests on family-mean item scores
    """
    log.info("  M1 [Def %s]: running ...", def_label)
    fam_labels = list(fam_def.keys())
    feat_df    = family_mean_features(matrix, fam_def)
    feat       = feat_df.values           # (N, n_fam)
    N          = feat.shape[0]
    label_dist = item_label_dist(matrix, fam_def)

    # ── (a) Kruskal rank ─────────────────────────────────────────────────────
    fam_kruskal:    Dict[str, int]  = {}
    fam_sv_profiles: Dict[str, list] = {}
    fam_sv_ratios:   Dict[str, list] = {}

    for fam in fam_labels:
        X_dist = label_dist[fam]   # (N, 4)
        _, sv, _ = np.linalg.svd(X_dist, full_matrices=False)
        ratio = sv / sv[0]
        prank = int((ratio > 0.01).sum())
        fam_kruskal[fam]      = prank
        fam_sv_profiles[fam]  = sv.tolist()
        fam_sv_ratios[fam]    = ratio.tolist()
        log.info(
            "    [Def %s] Family %s: kruskal_rank=%d, sv_ratios=%s",
            def_label, fam, prank, [f"{r:.3f}" for r in ratio],
        )

    total_kruskal = sum(fam_kruskal.values())
    kruskal_check: Dict[str, Any] = {}
    for K in [2, 3, 4]:
        thr  = 2 * K + 2
        cond = bool(total_kruskal >= thr)
        kruskal_check[f"K={K}"] = {
            "threshold_needed":  thr,
            "total_kruskal_rank": total_kruskal,
            "condition_satisfied": cond,
        }

    # SVD on the combined family-feature matrix
    _, sv_comb, _ = np.linalg.svd(feat, full_matrices=False)
    sv_comb_ratio  = (sv_comb / sv_comb[0]).tolist()

    # ── (b) GMM BIC / AIC ────────────────────────────────────────────────────
    max_k = len(fam_labels) + 2
    gmm_bic: Dict[int, Dict] = {}
    gmm_aic: Dict[int, Dict] = {}

    for k in range(1, max_k + 1):
        bics, aics = [], []
        for seed in SEEDS:
            gm = GaussianMixture(
                n_components=k, random_state=seed, n_init=5, max_iter=500
            )
            gm.fit(feat)
            bics.append(gm.bic(feat))
            aics.append(gm.aic(feat))
        gmm_bic[k] = {
            "mean": float(np.mean(bics)),
            "std":  float(np.std(bics)),
            "values": [float(v) for v in bics],
        }
        gmm_aic[k] = {
            "mean": float(np.mean(aics)),
            "std":  float(np.std(aics)),
        }
        log.info(
            "    [Def %s] GMM k=%d: BIC=%.1f±%.1f  AIC=%.1f±%.1f",
            def_label, k,
            gmm_bic[k]["mean"], gmm_bic[k]["std"],
            gmm_aic[k]["mean"], gmm_aic[k]["std"],
        )

    opt_k_bic = int(min(gmm_bic, key=lambda k: gmm_bic[k]["mean"]))
    opt_k_aic = int(min(gmm_aic, key=lambda k: gmm_aic[k]["mean"]))

    # ── (c) ANOVA + pairwise tests ────────────────────────────────────────────
    fam_arrays = {fam: feat_df[fam].values for fam in fam_labels}
    if len(fam_labels) >= 2:
        f_stat, p_anova = stats.f_oneway(*[fam_arrays[f] for f in fam_labels])
    else:
        f_stat, p_anova = np.nan, np.nan

    pairwise: Dict[str, Any] = {}
    for f1, f2 in combinations(fam_labels, 2):
        a, b = fam_arrays[f1], fam_arrays[f2]
        t, p  = stats.ttest_ind(a, b, equal_var=False)
        d     = float((a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2))
        pairwise[f"{f1}_vs_{f2}"] = {
            "mean_a":   float(a.mean()),
            "mean_b":   float(b.mean()),
            "mean_diff": float(a.mean() - b.mean()),
            "cohens_d": d,
            "t_stat":   float(t),
            "p_value":  float(p),
        }

    result: Dict[str, Any] = {
        "definition":          def_label,
        "n_families":          len(fam_labels),
        "family_labels":       fam_labels,
        "family_sizes": {f: len(js) for f, js in fam_def.items()},
        "n_items":             N,
        "n_judges":            sum(len(js) for js in fam_def.values()),
        # (a) Kruskal
        "per_family_kruskal_rank": fam_kruskal,
        "per_family_sv_profiles":  fam_sv_profiles,
        "per_family_sv_ratios":    fam_sv_ratios,
        "total_kruskal_rank":      total_kruskal,
        "kruskal_condition_check": kruskal_check,
        "combined_feature_svd": {
            "singular_values":         sv_comb.tolist(),
            "explained_variance_ratios": sv_comb_ratio,
        },
        # (b) GMM
        "gmm_bic":    {str(k): v for k, v in gmm_bic.items()},
        "gmm_aic":    {str(k): v for k, v in gmm_aic.items()},
        "optimal_k_bic": opt_k_bic,
        "optimal_k_aic": opt_k_aic,
        # (c) ANOVA
        "anova": {"F": float(f_stat), "p_value": float(p_anova)},
        "pairwise_ttest": pairwise,
    }

    log.info(
        "  M1 [Def %s] done: opt_k_BIC=%d, kruskal_total=%d, ANOVA p=%.3e",
        def_label, opt_k_bic, total_kruskal, p_anova,
    )
    return result


def run_m1_all(
    matrix: pd.DataFrame,
    definitions: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    return {
        d_lbl: run_m1_for_def(matrix, d, d_lbl)
        for d_lbl, d in definitions.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. M2: Leave-One-Family-Out (LOFO) Over-Identification Test
# ═══════════════════════════════════════════════════════════════════════════════

def run_m2_for_def(
    matrix: pd.DataFrame,
    fam_def: Dict[str, List[str]],
    def_label: str,
) -> Dict[str, Any]:
    """
    M2 for one family definition.
    Predict held-out family mean scores from the other families.
    Evaluation: Spearman rho (+ bootstrap CI), Cohen kappa_w, RMSE.
    """
    log.info("  M2 [Def %s]: running ...", def_label)
    fam_labels = list(fam_def.keys())
    feat_df    = family_mean_features(matrix, fam_def)
    lofo_results: Dict[str, Any] = {}

    for fam_out in fam_labels:
        fam_train = [f for f in fam_labels if f != fam_out]
        judges_out = [j for j in fam_def[fam_out] if j in matrix.columns]
        judges_in  = [j for f in fam_train for j in fam_def[f] if j in matrix.columns]

        y_true = matrix[judges_out].mean(axis=1).values.astype(float)

        # Predictor: mean of training-family averages
        X_pred_raw = feat_df[fam_train].values   # (N, len(fam_train))
        y_pred_raw = X_pred_raw.mean(axis=1)

        # OLS calibration
        slope, intercept, r_val, _, _ = stats.linregress(y_pred_raw, y_true)
        y_pred_cal = slope * y_pred_raw + intercept

        rho_raw, p_rho_raw = spearmanr(y_pred_raw, y_true)
        rho_cal, p_rho_cal = spearmanr(y_pred_cal, y_true)

        y_true_int = np.clip(np.round(y_true), 0, 3).astype(int)
        y_cal_int  = np.clip(np.round(y_pred_cal), 0, 3).astype(int)
        kappa = cohen_kappa_w(y_cal_int.astype(float), y_true_int.astype(float))
        rmse  = float(np.sqrt(np.mean((y_pred_cal - y_true) ** 2)))
        mae   = float(np.mean(np.abs(y_pred_cal - y_true)))

        # Bootstrap CI for rho_raw
        N = len(y_true)
        boot_rhos = np.empty(N_BOOTSTRAP)
        rng = np.random.default_rng(42)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, N, size=N)
            r, _ = spearmanr(y_pred_raw[idx], y_true[idx])
            boot_rhos[i] = r

        boot_ci_lo = float(np.quantile(boot_rhos, 0.025))
        boot_ci_hi = float(np.quantile(boot_rhos, 0.975))

        # Fisher z test: H0: rho = 0
        z_stat = float(np.arctanh(rho_raw))
        se_z   = 1.0 / np.sqrt(max(N - 3, 1))
        p_fz   = float(2 * (1 - stats.norm.cdf(abs(z_stat / se_z))))

        lofo_results[fam_out] = {
            "family_held_out":   fam_out,
            "family_training":   fam_train,
            "n_judges_held_out": len(judges_out),
            "n_judges_training": len(judges_in),
            "n_items":           N,
            "spearman_rho_raw": {
                "value":   float(rho_raw),
                "p_value": float(p_rho_raw),
                "p_fisher_z": p_fz,
                "bootstrap_ci_95pct": [boot_ci_lo, boot_ci_hi],
            },
            "spearman_rho_calibrated": {
                "value":   float(rho_cal),
                "p_value": float(p_rho_cal),
            },
            "ols_calibration": {
                "slope":     float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_val ** 2),
            },
            "cohen_kappa_w": float(kappa),
            "rmse":          rmse,
            "mae":           mae,
        }
        log.info(
            "    [Def %s] fam_out=%s: rho_raw=%.3f (p=%.4e), kappa=%.3f, RMSE=%.3f",
            def_label, fam_out, rho_raw, p_rho_raw, kappa, rmse,
        )

    return {
        "definition": def_label,
        "lofo_results": lofo_results,
        "caveats": _m2_caveats(fam_def),
    }


def _m2_caveats(fam_def: Dict[str, List[str]]) -> List[str]:
    """Generate honest caveats about M2 for this definition."""
    caveats = [
        "LOFO prediction uses item-level mean scores (not instance-level) as both "
        "predictor and target. This measures family-level consistency, not "
        "individual-judge generalization.",
        "Calibration (OLS) is fitted and evaluated on the same N=4423 items; "
        "Spearman rho is thus not a held-out estimate.",
    ]
    singletons = [f for f, js in fam_def.items() if len(js) == 1]
    if singletons:
        caveats.append(
            f"Families with 1 judge ({singletons}): the 'held-out family mean' is "
            "a single judge's raw scores—lower reliability than multi-judge averages."
        )
    return caveats


def run_m2_all(
    matrix: pd.DataFrame,
    definitions: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    return {
        d_lbl: run_m2_for_def(matrix, d, d_lbl)
        for d_lbl, d in definitions.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. M3: Within- vs. Cross-Family Correlation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pairwise(
    matrix: pd.DataFrame,
    fam_def: Dict[str, List[str]],
    ft_labels: Dict[str, bool],
) -> pd.DataFrame:
    """
    Compute all pairwise correlations for judges in this definition.
    Returns DataFrame with one row per pair.
    """
    all_judges = [j for jlist in fam_def.values() for j in jlist if j in matrix.columns]
    family_lookup = {j: f for f, jlist in fam_def.items() for j in jlist}

    records = []
    for j1, j2 in combinations(all_judges, 2):
        a = matrix[j1].values.astype(int)
        b = matrix[j2].values.astype(int)

        f1, f2 = family_lookup[j1], family_lookup[j2]
        within = (f1 == f2)

        rho = spearman_rho(a.astype(float), b.astype(float))
        kap = cohen_kappa_w(a.astype(float), b.astype(float))
        phi = pearson_phi_binary(a.astype(float), b.astype(float))

        # FT status for Llama stratification
        ft1 = ft_labels.get(j1, False)
        ft2 = ft_labels.get(j2, False)
        if within and f1 == "Llama":
            if ft1 and ft2:
                llama_subtype = "within_FT"
            elif (not ft1) and (not ft2):
                llama_subtype = "within_nonFT"
            else:
                llama_subtype = "cross_FT_nonFT"
        else:
            llama_subtype = "N/A"

        records.append(
            {
                "judge1":            j1,
                "judge2":            j2,
                "family1":           f1,
                "family2":           f2,
                "within_family":     within,
                "fine_tuned_j1":     ft1,
                "fine_tuned_j2":     ft2,
                "llama_subtype":     llama_subtype,
                "spearman_rho":      rho,
                "cohen_kappa_w":     kap,
                "pearson_phi_binary": phi,
            }
        )
    return pd.DataFrame(records)


def _summarize_group(sub: pd.DataFrame, label: str) -> Dict:
    if len(sub) == 0:
        return {"label": label, "n_pairs": 0}
    return {
        "label":   label,
        "n_pairs": len(sub),
        "spearman_rho": {
            "mean":   float(sub["spearman_rho"].mean()),
            "std":    float(sub["spearman_rho"].std()),
            "median": float(sub["spearman_rho"].median()),
            "min":    float(sub["spearman_rho"].min()),
            "max":    float(sub["spearman_rho"].max()),
        },
        "cohen_kappa_w": {
            "mean": float(sub["cohen_kappa_w"].mean()),
            "std":  float(sub["cohen_kappa_w"].std()),
        },
        "pearson_phi_binary": {
            "mean": float(sub["pearson_phi_binary"].mean()),
            "std":  float(sub["pearson_phi_binary"].std()),
        },
    }


def run_m3_for_def(
    matrix: pd.DataFrame,
    fam_def: Dict[str, List[str]],
    ft_labels: Dict[str, bool],
    def_label: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """M3 for one family definition."""
    log.info("  M3 [Def %s]: computing pairwise correlations ...", def_label)

    pairs_df = compute_pairwise(matrix, fam_def, ft_labels)

    within_df = pairs_df[pairs_df["within_family"]]
    cross_df  = pairs_df[~pairs_df["within_family"]]

    within_sum = _summarize_group(within_df, "within_family")
    cross_sum  = _summarize_group(cross_df,  "cross_family")

    # Statistical tests
    tests: Dict[str, Any] = {}
    for metric in ["spearman_rho", "cohen_kappa_w", "pearson_phi_binary"]:
        wx = within_df[metric].values
        cx = cross_df[metric].values
        if len(wx) == 0 or len(cx) == 0:
            tests[metric] = {"note": "empty group, test skipped"}
            continue
        tests[metric] = {
            "welch_t_test": welch_1sided_greater(wx, cx),
            "bootstrap_diff_ci_95pct": bootstrap_diff_ci(wx, cx, seed=42),
        }

    # Per-family within-group breakdown
    within_by_family: Dict[str, Any] = {}
    for fam in fam_def.keys():
        sub = within_df[within_df["family1"] == fam]
        within_by_family[fam] = _summarize_group(sub, f"within_{fam}")

    # Llama FT stratification (Definition A Llama)
    llama_stratification: Dict[str, Any] = {}
    for subtype in ["within_FT", "within_nonFT", "cross_FT_nonFT"]:
        sub = pairs_df[pairs_df["llama_subtype"] == subtype]
        llama_stratification[subtype] = _summarize_group(sub, subtype)

    # Log key numbers
    w_phi = within_sum.get("pearson_phi_binary", {}).get("mean", float("nan"))
    c_phi = cross_sum.get("pearson_phi_binary", {}).get("mean", float("nan"))
    log.info(
        "  M3 [Def %s]: within/cross rho=%.3f/%.3f, phi=%.3f/%.3f, diff_phi=%.3f",
        def_label,
        within_sum["spearman_rho"]["mean"] if "spearman_rho" in within_sum else float("nan"),
        cross_sum["spearman_rho"]["mean"]  if "spearman_rho" in cross_sum  else float("nan"),
        w_phi, c_phi, w_phi - c_phi,
    )

    result: Dict[str, Any] = {
        "definition":   def_label,
        "n_pairs_total": len(pairs_df),
        "n_pairs_within": len(within_df),
        "n_pairs_cross":  len(cross_df),
        "within_family":  within_sum,
        "cross_family":   cross_sum,
        "differences": {
            "spearman_rho_diff": (
                within_sum["spearman_rho"]["mean"] - cross_sum["spearman_rho"]["mean"]
                if "spearman_rho" in within_sum and "spearman_rho" in cross_sum
                else None
            ),
            "phi_diff": w_phi - c_phi,
        },
        "statistical_tests":      tests,
        "within_by_family":       within_by_family,
        "llama_ft_stratification": llama_stratification,
        "reference_arXiv_2605_29800": {
            "reported_within_phi": 0.437,
            "reported_cross_phi":  0.389,
            "reported_diff":       0.048,
            "note": (
                "We use Pearson phi on binarized scores (0->0, 1-3->1) as the "
                "closest reconstruction of their metric. Spearman rho on the "
                "full 0-3 ordinal scale is also reported."
            ),
        },
        "caveats": _m3_caveats(fam_def),
    }

    return result, pairs_df


def _m3_caveats(fam_def: Dict[str, List[str]]) -> List[str]:
    caveats = []
    singletons = [f for f, js in fam_def.items() if len(js) == 1]
    if singletons:
        caveats.append(
            f"Families with 1 judge ({singletons}): "
            "no within-family pairs exist for these families."
        )
    if "T5" in fam_def:
        caveats.append(
            "T5/TREMA family judges all use the same two-stage pipeline "
            "(ChatGPT-3.5 rubric + FLAN-T5-Large grading). "
            "Their within-family correlation may reflect shared GPT-3.5 rubric "
            "rather than FLAN-T5 model family."
        )
    return caveats


def run_m3_all(
    matrix: pd.DataFrame,
    definitions: Dict[str, Dict[str, List[str]]],
    ft_labels: Dict[str, bool],
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    results = {}
    pairs   = {}
    for d_lbl, d in definitions.items():
        res, pdf = run_m3_for_def(matrix, d, ft_labels, d_lbl)
        results[d_lbl] = res
        pairs[d_lbl]   = pdf
    return results, pairs


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Figures
# ═══════════════════════════════════════════════════════════════════════════════

FAMILY_COLORS = {
    "GPT":       C_BLUE,
    "Llama":     C_GREEN,
    "T5":        C_RED,
    "GPT_T5":    C_PURP,
    "Llama_8B":  C_GREEN,
    "Llama_70B": C_ORG,
}


def plot_m1_bic_all(m1_all: Dict[str, Any], path: Path) -> None:
    """BIC curves for all three definitions, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, (d_lbl, m1) in zip(axes, m1_all.items()):
        bic = m1["gmm_bic"]
        ks  = [int(k) for k in sorted(bic, key=int)]
        means = [bic[str(k)]["mean"] for k in ks]
        stds  = [bic[str(k)]["std"]  for k in ks]
        ax.errorbar(ks, means, yerr=stds, fmt="o-", color=C_BLUE,
                    capsize=4, linewidth=1.5, markersize=5,
                    label="BIC (mean $\\pm$ std)")
        opt = m1["optimal_k_bic"]
        ax.axvline(opt, color=C_RED, linestyle="--", linewidth=1.2,
                   label=f"Opt. $k={opt}$")
        n_fam = m1["n_families"]
        ax.axvline(n_fam, color=C_GRAY, linestyle=":", linewidth=1.0,
                   label=f"Nominal $m={n_fam}$")
        ax.set_xlabel("$k$")
        ax.set_ylabel("BIC" if ax is axes[0] else "")
        ax.set_title(f"Definition {d_lbl} ({n_fam} families)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("M1: GMM BIC vs. Assumed Relevance-Class Count", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_m1_kruskal(m1_all: Dict[str, Any], path: Path) -> None:
    """Per-family Kruskal-rank bar chart for each definition."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (d_lbl, m1) in zip(axes, m1_all.items()):
        fams   = m1["family_labels"]
        kranks = [m1["per_family_kruskal_rank"][f] for f in fams]
        colors = [FAMILY_COLORS.get(f, C_GRAY) for f in fams]
        ax.bar(fams, kranks, color=colors, alpha=0.75, width=0.5)
        # Draw threshold lines
        for K, ls, lbl in [(2, "-", "K=2 (thr=6)"),
                            (3, "--", "K=3 (thr=8)"),
                            (4, ":", "K=4 (thr=10)")]:
            ax.axhline(
                (2*K+2) / len(fams),   # per-family threshold for equal contribution
                color=C_GRAY, linestyle=ls, linewidth=0.8, label=lbl
            )
        ax.set_ylabel("Kruskal rank" if ax is axes[0] else "")
        ax.set_ylim(0, 5)
        ax.set_title(f"Def {d_lbl}: Kruskal Ranks\n(total={sum(kranks)})")
        ax.legend(fontsize=6)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("M1: Per-Family Kruskal Rank (practical, threshold 1%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_m2_lofo_all(m2_all: Dict[str, Any], path: Path) -> None:
    """LOFO Spearman rho bar chart for all definitions."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (d_lbl, m2) in zip(axes, m2_all.items()):
        lofo  = m2["lofo_results"]
        fams  = [f for f in lofo if isinstance(lofo[f], dict)]
        rhos  = [lofo[f]["spearman_rho_raw"]["value"] for f in fams]
        lo    = [lofo[f]["spearman_rho_raw"]["value"] -
                 lofo[f]["spearman_rho_raw"]["bootstrap_ci_95pct"][0]
                 for f in fams]
        hi    = [lofo[f]["spearman_rho_raw"]["bootstrap_ci_95pct"][1] -
                 lofo[f]["spearman_rho_raw"]["value"]
                 for f in fams]
        colors = [FAMILY_COLORS.get(f, C_GRAY) for f in fams]
        x = np.arange(len(fams))
        ax.bar(x, rhos, color=colors, alpha=0.75, width=0.5,
               yerr=np.array([lo, hi]), capsize=5,
               error_kw={"linewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([f"Hold\n{f}" for f in fams], fontsize=7)
        ax.set_ylabel("Spearman $\\rho$" if ax is axes[0] else "")
        ax.set_title(f"Def {d_lbl}")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("M2: LOFO Prediction Spearman $\\rho$ (raw, 95% bootstrap CI)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_m3_boxplot_defA(pairs_df_A: pd.DataFrame, m3_A: Dict, path: Path) -> None:
    """Boxplot for Definition A M3: Spearman rho and Phi, with FT stratification."""
    w_rho = pairs_df_A.loc[pairs_df_A["within_family"],  "spearman_rho"].values
    c_rho = pairs_df_A.loc[~pairs_df_A["within_family"], "spearman_rho"].values
    w_phi = pairs_df_A.loc[pairs_df_A["within_family"],  "pearson_phi_binary"].values
    c_phi = pairs_df_A.loc[~pairs_df_A["within_family"], "pearson_phi_binary"].values

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # Left: Spearman rho
    bp1 = axes[0].boxplot(
        [w_rho, c_rho],
        tick_labels=["Within\nFamily", "Cross\nFamily"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    for patch, clr in zip(bp1["boxes"], [C_BLUE, C_GRAY]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Spearman $\\rho$")
    axes[0].set_title("Spearman $\\rho$ (ordinal 0-3)")
    d_rho = w_rho.mean() - c_rho.mean()
    axes[0].text(0.97, 0.03, f"$\\Delta$ = {d_rho:+.3f}",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=9,
                 color=C_RED)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Right: phi (binary) with reference lines
    bp2 = axes[1].boxplot(
        [w_phi, c_phi],
        tick_labels=["Within\nFamily", "Cross\nFamily"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    for patch, clr in zip(bp2["boxes"], [C_BLUE, C_GRAY]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    axes[1].axhline(0.437, color=C_RED, linestyle="--", linewidth=1.0,
                    label="Ref. within (arXiv:2605.29800)")
    axes[1].axhline(0.389, color=C_GRAY, linestyle="--", linewidth=1.0,
                    label="Ref. cross (arXiv:2605.29800)")
    axes[1].set_ylabel("Pearson $\\phi$ (binary)")
    axes[1].set_title("Pearson $\\phi$ (binary relevant/irrelevant)")
    d_phi = w_phi.mean() - c_phi.mean()
    axes[1].text(0.97, 0.03, f"$\\Delta$ = {d_phi:+.3f}",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=9,
                 color=C_RED)
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "M3 (Definition A): Within- vs. Cross-Family Judge Agreement\n"
        "GPT (7) / Llama (19) / T5 (7)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_m3_llama_ft(pairs_df_A: pd.DataFrame, path: Path) -> None:
    """Boxplot: Llama FT stratification of within-Llama pairs."""
    subtype_labels = {
        "within_FT":     "Within FT",
        "within_nonFT":  "Within\nnon-FT",
        "cross_FT_nonFT": "FT vs\nnon-FT",
    }
    data = []
    labels = []
    for key, lbl in subtype_labels.items():
        sub = pairs_df_A.loc[pairs_df_A["llama_subtype"] == key, "spearman_rho"].values
        if len(sub) > 0:
            data.append(sub)
            labels.append(f"{lbl}\n(n={len(sub)})")

    if not data:
        log.warning("No Llama FT stratification data; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
    )
    colors_ft = [C_GREEN, C_ORG, C_PURP]
    for patch, clr in zip(bp["boxes"], colors_ft):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("M3: Llama Within-Family FT Stratification\n(Definition A)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def plot_m3_phi_comparison(m3_all: Dict[str, Any], path: Path) -> None:
    """Compare within/cross phi across all three definitions."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.array([0, 1, 2])
    width = 0.35
    def_labels = ["A", "B", "C"]
    w_phis = []
    c_phis = []
    for d in def_labels:
        w = m3_all[d]["within_family"]
        c = m3_all[d]["cross_family"]
        w_phis.append(w.get("pearson_phi_binary", {}).get("mean", float("nan")))
        c_phis.append(c.get("pearson_phi_binary", {}).get("mean", float("nan")))

    ax.bar(x - width/2, w_phis, width, color=C_BLUE, alpha=0.75,
           label="Within family")
    ax.bar(x + width/2, c_phis, width, color=C_GRAY, alpha=0.75,
           label="Cross family")
    ax.axhline(0.437, color=C_RED, linestyle="--", linewidth=1.0,
               label="Ref. within")
    ax.axhline(0.389, color=C_GRAY, linestyle=":",  linewidth=1.0,
               label="Ref. cross")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Def A\n(GPT/Llama/T5)",
         "Def B\n(GPT+T5/Llama)",
         "Def C\n(4 families)"]
    )
    ax.set_ylabel("Pearson $\\phi$ (binary)")
    ax.set_title("M3: Within/Cross-Family $\\phi$ Across All Definitions")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Environment Log
# ═══════════════════════════════════════════════════════════════════════════════

def make_env_log(matrix: pd.DataFrame, definitions: Dict) -> Dict:
    """Reproducibility manifest."""
    import hashlib
    sub_files = sorted(SUBS_DIR.glob("*.txt"))
    data_fp = hashlib.md5(
        b"\n".join(f"{f.name}:{f.stat().st_size}".encode() for f in sub_files)
    ).hexdigest()
    pkgs: Dict[str, str] = {}
    for pkg in ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]:
        try:
            pkgs[pkg] = __import__(pkg).__version__
        except Exception:
            pkgs[pkg] = "unknown"

    return {
        "timestamp":       datetime.now().isoformat(),
        "hostname":        socket.gethostname(),
        "platform":        platform.platform(),
        "python_version":  sys.version,
        "packages":        pkgs,
        "seeds":           SEEDS,
        "n_bootstrap":     N_BOOTSTRAP,
        "n_judges":        len(matrix.columns),
        "n_items":         len(matrix),
        "data_fingerprint": data_fp,
        "definitions": {
            d: {f: len(js) for f, js in d_def.items()}
            for d, d_def in definitions.items()
        },
        "gpu_used":   False,
        "api_calls":  0,
        "data_dir":   str(DATA_DIR),
        "results_dir": str(RESULTS_DIR),
        "note": (
            "Full family mapping (33/33 judges) from Ito (2026-08-04). "
            "Three family definitions pre-registered. "
            "CPU-only, zero API calls, zero GPU."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=" * 65)
    log.info("E2 Experiment: Contract-A Real-Data  (WSDM 2027)")
    log.info("=" * 65)

    # Load family mapping JSON
    with open(MAP_FILE, encoding="utf-8") as fh:
        family_json = json.load(fh)
    log.info("Loaded family mapping: %d non-meta entries", len(
        [k for k in family_json if not k.startswith("_")]
    ))

    # Build pre-registered definitions and FT labels
    definitions = build_definitions(family_json)
    ft_labels   = build_ft_labels(family_json)

    # Load all 33 judges into one matrix
    matrix = load_all_judges(definitions)

    # ── M1 ──────────────────────────────────────────────────────────────────
    log.info("Running M1 (all definitions) ...")
    m1_all = run_m1_all(matrix, definitions)

    # ── M2 ──────────────────────────────────────────────────────────────────
    log.info("Running M2 (all definitions) ...")
    m2_all = run_m2_all(matrix, definitions)

    # ── M3 ──────────────────────────────────────────────────────────────────
    log.info("Running M3 (all definitions) ...")
    m3_all, pairs_all = run_m3_all(matrix, definitions, ft_labels)

    # ── Save JSON results ────────────────────────────────────────────────────
    for tag, data in [
        ("e2_m1_gof",         m1_all),
        ("e2_m2_lofo",        m2_all),
        ("e2_m3_correlation", m3_all),
    ]:
        p = RESULTS_DIR / f"{tag}.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        log.info("Saved %s", p)

    # ── Save environment log ─────────────────────────────────────────────────
    env = make_env_log(matrix, definitions)
    with open(RESULTS_DIR / "e2_env_log.json", "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2, ensure_ascii=False)
    log.info("Saved e2_env_log.json")

    # ── Save pair CSVs ───────────────────────────────────────────────────────
    for d_lbl, pdf in pairs_all.items():
        p = RESULTS_DIR / f"e2_m3_pairs_def{d_lbl}.csv"
        pdf.to_csv(p, index=False)
        log.info("Saved %s", p)

    # ── Figures ──────────────────────────────────────────────────────────────
    plot_m1_bic_all(m1_all,    FIGS_DIR / "e2_m1_bic_all.pdf")
    plot_m1_kruskal(m1_all,    FIGS_DIR / "e2_m1_kruskal_all.pdf")
    plot_m2_lofo_all(m2_all,   FIGS_DIR / "e2_m2_lofo_all.pdf")
    plot_m3_boxplot_defA(
        pairs_all["A"], m3_all["A"], FIGS_DIR / "e2_m3_boxplot_defA.pdf"
    )
    plot_m3_llama_ft(pairs_all["A"], FIGS_DIR / "e2_m3_llama_ft.pdf")
    plot_m3_phi_comparison(m3_all,   FIGS_DIR / "e2_m3_phi_comparison.pdf")

    # ── Final summary ────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 65)
    log.info("RESULT SUMMARY")
    log.info("=" * 65)
    for d_lbl in ["A", "B", "C"]:
        m1 = m1_all[d_lbl]
        m2 = m2_all[d_lbl]
        m3 = m3_all[d_lbl]
        log.info("--- Definition %s (%d families) ---", d_lbl, m1["n_families"])
        log.info(
            "  M1: opt_k_BIC=%d | kruskal_total=%d",
            m1["optimal_k_bic"], m1["total_kruskal_rank"],
        )
        for K, v in m1["kruskal_condition_check"].items():
            log.info(
                "      %s: threshold=%d satisfied=%s",
                K, v["threshold_needed"], v["condition_satisfied"],
            )
        lofo_rhos = {
            f: m2["lofo_results"][f]["spearman_rho_raw"]["value"]
            for f in m2["lofo_results"]
            if isinstance(m2["lofo_results"][f], dict)
        }
        log.info("  M2: LOFO rho per family = %s",
                 {f: f"{r:.3f}" for f, r in lofo_rhos.items()})
        w_phi = m3["within_family"].get("pearson_phi_binary", {}).get("mean")
        c_phi = m3["cross_family"].get("pearson_phi_binary", {}).get("mean")
        w_rho = m3["within_family"].get("spearman_rho", {}).get("mean")
        c_rho = m3["cross_family"].get("spearman_rho", {}).get("mean")
        log.info(
            "  M3: rho within/cross = %.3f / %.3f (diff=%.3f) | "
            "phi within/cross = %.3f / %.3f (diff=%.3f) | "
            "[ref phi diff=0.048]",
            w_rho or float("nan"), c_rho or float("nan"),
            (w_rho - c_rho) if (w_rho and c_rho) else float("nan"),
            w_phi or float("nan"), c_phi or float("nan"),
            (w_phi - c_phi) if (w_phi and c_phi) else float("nan"),
        )
    log.info("")
    log.info("Results: %s", RESULTS_DIR)
    log.info("Figures: %s", FIGS_DIR)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
