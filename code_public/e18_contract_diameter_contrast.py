"""E18: measure the identification set diameter under both contracts.

E13 reported only descriptive statistics for NTCIR-10 INTENT-2, which left the
distribution contract without a quantitative counterpart to the LLMJudge
measurements.  The obstacle is that NTCIR carries no judge panel, so family
kernels cannot be estimated there and the m >= 1 diameters of E2c have no
analogue.

What can be compared is the part of the identification set that the true-value
profiles alone determine, that is, the gauge set at m = 0, where the only
constraint is that the transformed profiles stay in the simplex.  That
quantity is defined identically for both collections, so it isolates the
effect of the contract on identification: under the point contract profiles sit
at vertices, under the distribution contract they sit in the interior, and the
gauge set responds to that geometry.

Both collections are reduced to the same L before comparison, because the
gauge dimension is L(L-1) and diameters at different L are not comparable.
NTCIR levels are mapped onto four grades by merging its two lowest levels.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_criterion_and_unified_radius import unified_radius
from e6_sensitivity_and_vossler import load, item_profiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e18_contract_diameter_contrast.json")
NTCIR = os.path.join(ROOT, "data", "ntcir_intent2", "INTENT2ResearchPurposeData")
SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 4000
L = 4


def ntcir_L4(lang: str):
    """NTCIR truth profiles for one subtask, reduced from five grades to four."""
    from e13_ntcir_measured_intent import load_iprob, load_dqrels, profiles, SUBTASKS
    ip, dq = SUBTASKS[lang]
    P = load_iprob(os.path.join(NTCIR, ip))
    rel = load_dqrels(os.path.join(NTCIR, dq))
    M, keys, _ = profiles(P, rel, uniform=False)
    M = np.column_stack([M[:, 0] + M[:, 1], M[:, 2], M[:, 3], M[:, 4]])
    return M / M.sum(axis=1, keepdims=True)


def main() -> None:
    # ---- Contract A: LLMJudge prototypes (identical construction to E2c) ----
    judges, runs, gold, fam_of, keys = load()
    P = item_profiles(judges, runs, keys)
    g = np.array([gold[k] for k in keys])
    protos_A = np.vstack([P[g == k].mean(axis=0) for k in range(L)])
    rA, nA = unified_radius([], protos_A, SEEDS, PER)

    # ---- Contract B: NTCIR profiles taken from the E13 result file ----
    e13 = json.load(open(os.path.join(ROOT, "results",
                                      "e13_ntcir_measured_intent.json")))

    out = {
        "env": {
            "script": "e18_contract_diameter_contrast.py",
            "seeds": SEEDS,
            "n_dirs_per_seed": PER,
            "L": L,
            "note": ("m = 0 diameters only; NTCIR carries no judge panel, so "
                     "family kernels and therefore m >= 1 diameters are not "
                     "defined for the distribution-contract collection."),
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "contract_A_llmjudge": {
            "prototypes": "gold-conditioned mean, L=4",
            "eta": float(protos_A.min()),
            "floor_2eta": 2 * float(protos_A.min()),
            "diameter_m0": rA,
            "n_directions": nA,
        },
        "contract_B_ntcir": {
            "source": "results/e13_ntcir_measured_intent.json",
            "interior_fraction": e13["summary"]["interior_fraction_by_subtask"],
        },
    }

    for lang in ("Japanese", "Chinese"):
        M = ntcir_L4(lang)
        # Match the contract-A construction: L prototypes, each the mean profile
        # of the items whose dominant grade is that level.
        dom = M.argmax(axis=1)
        rows, counts = [], {}
        for k in range(L):
            sel = M[dom == k]
            counts[k] = int(sel.shape[0])
            if sel.shape[0] == 0:
                continue
            v = sel.mean(axis=0)
            rows.append(v / v.sum())
        protos_B = np.vstack(rows)
        r, n = unified_radius([], protos_B, SEEDS, PER)
        # Also report the raw-profile version, which uses far more constraints.
        rng = np.random.default_rng(20260814)
        idx = rng.choice(M.shape[0], size=min(400, M.shape[0]), replace=False)
        r_raw, _ = unified_radius([], M[np.sort(idx)], SEEDS, PER)
        out["contract_B_ntcir"][lang] = {
            "n_profiles_total": int(M.shape[0]),
            "prototypes": "mean profile per dominant grade (matches contract A)",
            "n_prototypes": int(protos_B.shape[0]),
            "items_per_dominant_grade": counts,
            "eta": float(protos_B.min()),
            "floor_2eta": 2 * float(protos_B.min()),
            "diameter_m0": r,
            "n_directions": n,
            "diameter_m0_raw_400_profiles": r_raw,
        }
        print(f"[B:{lang}] K={protos_B.shape[0]} eta={protos_B.min():.4f} "
              f"diameter_m0={r:.4f}  (raw 400 profiles: {r_raw:.4f})", flush=True)

    print(f"[A:LLMJudge] eta={protos_A.min():.4f} diameter_m0={rA:.4f}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
