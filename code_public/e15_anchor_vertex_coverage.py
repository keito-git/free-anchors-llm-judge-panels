"""E15: which vertices do the freely available anchors actually cover?

Inspection of the LLMJudge panel suggests that every near-vertex item sits at
the same vertex.  If that holds, assumption i) of Theorem 4, which asks
for a near-vertex item for every label, cannot be met by this collection at
all, and E12 cannot be read as a demonstration of full identification recovery.

This script settles three things.

  A  Vertex coverage of the near-vertex pool, per label, at several thresholds.
  B  The radius under kernel constraints alone, with the prototype constraint
     dropped.  If the anchored radius equals it, the saturation seen in E12 is
     explained: further copies of the same vertex add no new cut.
  C  The E12 sweep recomputed at the direction budget used elsewhere, so the
     reported reduction ratio is comparable across experiments.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e12_structural_anchor_realdata import load, radius, max_step, L4, FAMS, L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e15_anchor_vertex_coverage.json")
SEEDS = [42, 123, 456, 789, 1024, 20260804]
N_DIRS = 3000          # matches E12
FRACTIONS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def kernel_only_radius(kernels, protos, seeds, n_dirs):
    """Radius when only the kernel constraints bind (prototype simplex dropped)."""
    eye = np.eye(L)
    best = 0.0
    for s in seeds:
        rng = np.random.default_rng(s)
        for _ in range(n_dirs):
            D = rng.normal(size=(L, L))
            D -= D.mean(axis=0, keepdims=True)
            nn = np.linalg.norm(D)
            if nn < 1e-12:
                continue
            D /= nn
            lo, hi = 0.0, 2.0
            for _ in range(45):
                t = 0.5 * (lo + hi)
                A = eye + t * D
                try:
                    Ainv = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    hi = t
                    continue
                ok = all((Ck @ Ainv >= -1e-12).all() for Ck in kernels)
                lo, hi = (t, hi) if ok else (lo, t)
            A = eye + lo * D
            best = max(best, float(np.abs(protos @ A.T - protos).sum(axis=1).max()))
    return best


def main() -> None:
    judges, fam_of, P, g = load()
    kern = [np.array(json.load(open(L4))["kernels_from_gold"][f]) for f in FAMS]
    vd = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vd)
    n = len(order)

    res = {"env": {"script": "e15_anchor_vertex_coverage.py", "seeds": SEEDS,
                   "n_dirs_per_seed": N_DIRS, "n_items": int(n),
                   "gpu_used": False, "api_calls": 0, "cost_usd": 0.0}}

    # A: vertex coverage
    cov = {}
    for q in (0.05, 0.10, 0.20):
        idx = order[: int(q * n)]
        lab = P[idx].argmax(axis=1)
        cov[f"top_{int(q*100)}pct"] = {
            "pool_size": int(len(idx)),
            "label_counts": {str(k): int(v) for k, v in sorted(Counter(lab.tolist()).items())},
            "distinct_vertices_covered": int(len(set(lab.tolist()))),
        }
    per_label = {}
    for l in range(L):
        d = 2.0 * (1.0 - P[:, l])
        per_label[str(l)] = {"min_l1_distance_to_vertex": float(d.min()),
                             "n_items_within_0p10": int((d <= 0.10).sum())}
    res["A_vertex_coverage"] = {"pools": cov, "per_label_nearest": per_label,
        "all_L_vertices_available": bool(all(per_label[str(l)]["min_l1_distance_to_vertex"] <= 0.10 for l in range(L))),
        "reading": ("Theorem 4 assumption i) asks for a near-vertex item for every label. "
                    "This block reports whether the collection can supply them.")}

    # B and C: sweep at the E12 budget, plus the kernel-only reference
    rng0 = np.random.default_rng(7)
    interior = order[int(0.50 * n):]
    near = order[: int(0.10 * n)]
    base = rng0.choice(interior, size=min(400, len(interior)), replace=False)
    sweep = []
    for frac in FRACTIONS:
        k = int(round(frac * len(base)))
        protos = P[base] if k == 0 else np.vstack([P[base], P[rng0.choice(near, size=min(k, len(near)), replace=False)]])
        sweep.append({"anchor_fraction": frac, "n_anchor_items": int(k),
                      "radius": radius(kern, protos, SEEDS, N_DIRS)})
        print(f"frac={frac:.3f} n={k:3d} radius={sweep[-1]['radius']:.6f}", flush=True)
    res["C_sweep_at_E12_budget"] = sweep

    ref = kernel_only_radius(kern, P[base], SEEDS, N_DIRS)
    anchored = sweep[-1]["radius"]
    res["B_kernel_only_reference"] = {
        "radius_kernel_constraints_only": ref,
        "radius_with_anchors": anchored,
        "difference": float(anchored - ref),
        "saturation_explained": bool(abs(anchored - ref) < 1e-6),
        "reading": ("If the anchored radius equals the kernel-only radius, the anchors have "
                    "stopped binding and further copies of the same vertex cannot help."),
    }
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nkernel-only radius = {ref:.6f}; anchored = {anchored:.6f}; "
          f"saturation explained = {res['B_kernel_only_reference']['saturation_explained']}")
    print(f"all L vertices available = {res['A_vertex_coverage']['all_L_vertices_available']}")


if __name__ == "__main__":
    main()
