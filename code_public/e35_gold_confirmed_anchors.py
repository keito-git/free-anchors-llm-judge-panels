"""E35: E32 first-panel sweep with the anchor pool restricted to gold-confirmed vertices.

E34 showed that 50 of the 442 consensus-selected pool items of the first panel
carry a gold label other than 0, although every one of them sits at the label-0
vertex by judge consensus.  Here the E32 measurement is repeated with the pool
restricted to items whose gold label equals their consensus vertex, so that
every anchor is a true anchor for this panel.  The interior prototypes, the
kernels, the estimator and the seeds are those of E32 (imported, not
re-implemented).

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e12_structural_anchor_realdata import load as load12  # noqa: E402
from e32_anchor_effect_on_interior import (  # noqa: E402
    N_STARTS, MAXITER, PER, SEEDS, first_panel, make_ops)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e35_gold_confirmed_anchors.json")
FRACS = [0.0, 0.005, 0.05, 0.20]
N_BASE = 400


def main() -> None:
    P, kern, L = first_panel()
    _, _, P12, g = load12()
    assert np.allclose(P, P12)
    optimised = make_ops(L)
    vd = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vd)
    near_idx = order[: int(0.10 * len(order))]
    amb_idx = order[int(0.50 * len(order)):]
    # Same interior prototypes as E32 (first draw of rng seed 7).
    rng0 = np.random.default_rng(7)
    base = rng0.choice(amb_idx, size=min(N_BASE, len(amb_idx)), replace=False)
    interior = P[base]
    cons = P[near_idx].argmax(axis=1)
    confirmed = near_idx[cons == g[near_idx]]
    print(f"pool {len(near_idx)} -> gold-confirmed {len(confirmed)}", flush=True)

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
        "env": {"script": "e35_gold_confirmed_anchors.py", "seeds": SEEDS,
                "n_dirs_per_seed": PER,
                "estimator": f"optimised, {N_STARTS} starts, maxiter {MAXITER}",
                "pool": "E32 top-10% pool restricted to consensus vertex == gold",
                "gpu_used": False, "api_calls": 0, "cost_usd": 0.0},
        "n_pool_consensus": int(len(near_idx)),
        "n_pool_gold_confirmed": int(len(confirmed)),
        "sweep": rows,
        "reduction_pct_largest": 100 * (1 - rows[-1]["interior_diameter"] / base_d),
        "reduction_pct_first_rate": 100 * (1 - rows[1]["interior_diameter"] / base_d),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
