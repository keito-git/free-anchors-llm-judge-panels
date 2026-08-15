"""Verify the theoretical diagnosis for Corollary 1(ii) refinement.

Kernel parameterization (L=2, column-stochastic):
    C(alpha, beta) = [[1-alpha, 1-beta],
                      [alpha,   beta  ]]
alpha = FPR-like (P(output 1 | truth 0)), beta = TPR-like (P(output 1 | truth 1)).

Claims to verify on e3_convexhull_check.json data:
 1. seed=0, step 3->4: C_3 = M @ C_2 with M column-stochastic, M >= 0
    (Blackwell garbling => constraint unconditionally redundant => G_4 = G_3).
 2. Row-cone (panel domination) criterion in 2D reduces to likelihood-ratio range:
    new kernel NON-dominated  <=>  beta_g/alpha_g > max_f beta_f/alpha_f
                               or  (1-beta_g)/(1-alpha_g) < min_f (1-beta_f)/(1-alpha_f)
    Check this against all 40 steps: dominated should imply diam_change == 0
    (and n_removed == 0 where measured); non-dominated => prototype-free set shrinks.
"""
from pathlib import Path
import json
import numpy as np

PATH = str(Path(__file__).resolve().parents[1] / "results" / "e3_convexhull_check.json")

def C(alpha: float, beta: float) -> np.ndarray:
    return np.array([[1 - alpha, 1 - beta], [alpha, beta]])

with open(PATH) as f:
    data = json.load(f)

# --- Claim 1: garbling certificate for seed 0, step 3->4 ---
k = data["per_seed"]["0"]["kernels"]
C2 = C(k[2]["alpha"], k[2]["beta"])
C3 = C(k[3]["alpha"], k[3]["beta"])
M = C3 @ np.linalg.inv(C2)
print("=== Claim 1: seed0 step 3->4, M = C3 @ C2^{-1} ===")
print("M =\n", np.round(M, 6))
print("M >= 0 entrywise:", bool((M >= -1e-12).all()))
print("column sums of M:", np.round(M.sum(axis=0), 10))
print("reconstruction max err:", np.abs(M @ C2 - C3).max())

# Also: which single existing kernel garbles it? check all f in {0,1,2}
for f in range(3):
    Cf = C(k[f]["alpha"], k[f]["beta"])
    Mf = C3 @ np.linalg.inv(Cf)
    print(f"  garbling of kernel_{f}? M >= 0: {bool((Mf >= -1e-12).all())}  min entry: {Mf.min():.6f}")

# --- Claim 2: LR-range criterion vs experiment over all 40 steps ---
print("\n=== Claim 2: panel-domination (LR-range) vs experimental outcome ===")
print(f"{'seed':>4} {'step':>7} {'r1_new':>8} {'r1_max':>8} {'r0_new':>8} {'r0_min':>8} "
      f"{'dominated':>9} {'diam_chg':>9} {'shrank':>6}")
rows = []
for seed, sd in data["per_seed"].items():
    kernels = sd["kernels"]
    for st in sd["steps"]:
        m_prev = st["m_prev"]
        a_g, b_g = st["new_kernel_alpha"], st["new_kernel_beta"]
        prev = kernels[:m_prev]
        r1_new = b_g / a_g                      # LR of output "1"
        r0_new = (1 - b_g) / (1 - a_g)          # LR of output "0"
        r1_max = max(p["beta"] / p["alpha"] for p in prev)
        r0_min = min((1 - p["beta"]) / (1 - p["alpha"]) for p in prev)
        dominated = (r1_new <= r1_max + 1e-12) and (r0_new >= r0_min - 1e-12)
        rows.append((seed, st["step"], dominated, st["mean_diam_change"], st["diameter_shrank"]))
        print(f"{seed:>4} {st['step']:>7} {r1_new:8.4f} {r1_max:8.4f} {r0_new:8.4f} {r0_min:8.4f} "
              f"{str(dominated):>9} {st['mean_diam_change']:9.5f} {str(st['diameter_shrank']):>6}")

# Cross-tab
print("\n--- cross-tab: dominated x diameter_shrank ---")
for dom in (True, False):
    for shr in (True, False):
        n = sum(1 for r in rows if r[2] == dom and r[4] == shr)
        print(f"dominated={dom!s:>5}  diam_shrank={shr!s:>5}: {n}")

# Key consistency: dominated => diameter must NOT shrink (necessary direction, unconditional)
viol = [r for r in rows if r[2] and r[4]]
print("\nviolations of 'dominated => no shrink':", len(viol), viol)

# Non-dominated but diameter unchanged (allowed: set shrinks w/o diameter change, or
# prototype constraints already exclude the region)
nd_nochange = [r for r in rows if (not r[2]) and (not r[4])]
print("non-dominated but diameter unchanged (layer-2 cases):", nd_nochange)

# Dominated steps (predicted n_removed == 0)
dom_steps = [(r[0], r[1]) for r in rows if r[2]]
print("dominated steps (predict n_removed = 0):", dom_steps)

# --- sanity: convex hull membership implies domination on this data ---
# (all steps are outside hull per the JSON, so nothing to check; just record)
print("\nall steps outside (alpha,beta) hull per JSON:", data["aggregate"]["n_outside_hull"], "/ 40")
