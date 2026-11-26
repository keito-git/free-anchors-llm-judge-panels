"""E36: E32 second-panel sweep with the anchor pool restricted to gold-confirmed vertices.

Companion to E35.  E34 showed that 6 of the 29 consensus-selected pool items of
the second panel (BEIR TREC-COVID) carry a gold label different from their
consensus vertex.  The E32 measurement is repeated with the pool restricted to
items whose gold label equals their consensus vertex.  Kernels, interior
prototypes, estimator and seeds are those of E32 (imported).

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e32_anchor_effect_on_interior import (  # noqa: E402
    N_STARTS, MAXITER, PER, SEEDS, second_panel, make_ops)
from e34_anchor_gold_check import second_panel_with_gold  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e36_gold_confirmed_anchors_second_panel.json")
FRACS = [0.0, 0.02, 0.10, 0.20]
N_BASE = 120


def main() -> None:
    P, kern, L = second_panel()
    P2, g, _ = second_panel_with_gold()
    assert np.allclose(P, P2)
    optimised = make_ops(L)
    vd = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vd)
    near_idx = order[: int(0.10 * len(order))]
    amb_idx = order[int(0.50 * len(order)):]
    rng0 = np.random.default_rng(7)
    base = rng0.choice(amb_idx, size=min(N_BASE, len(amb_idx)), replace=False)
    interior = P[base]
    cons = P[near_idx].argmax(axis=1)
    confirmed = near_idx[cons == g[near_idx]]
    cov = sorted(set(int(c) for c in P[confirmed].argmax(axis=1)))
    print(f"pool {len(near_idx)} -> gold-confirmed {len(confirmed)}; vertices {cov}",
          flush=True)
    rows = []
    for frac in FRACS:
        n_anchor = int(round(frac * len(base)))
        if n_anchor > 0:
            anchors = rng0.choice(confirmed, size=min(n_anchor, len(confirmed)),
                                  replace=False)
            constrain = np.vstack([interior, P[anchors]])
        else:
            constrain = interior
        d = optimised(kern, constrain, interior)
        rows.append({"anchor_fraction": frac, "n_anchor_items": int(n_anchor),
                     "interior_diameter": d})
        print(f"  rate={frac:<6} anchors={n_anchor:<3} interior diameter={d:.6f}",
              flush=True)
    base_d = rows[0]["interior_diameter"]
    out = {
        "env": {"script": "e36_gold_confirmed_anchors_second_panel.py", "seeds": SEEDS,
                "n_dirs_per_seed": PER,
                "estimator": f"optimised, {N_STARTS} starts, maxiter {MAXITER}",
                "pool": "E32 top-10% pool restricted to consensus vertex == gold",
                "gpu_used": False, "api_calls": 0, "cost_usd": 0.0},
        "n_pool_consensus": int(len(near_idx)),
        "n_pool_gold_confirmed": int(len(confirmed)),
        "vertices_covered_by_confirmed": cov,
        "sweep": rows,
        "reduction_pct_largest": 100 * (1 - rows[-1]["interior_diameter"] / base_d),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
