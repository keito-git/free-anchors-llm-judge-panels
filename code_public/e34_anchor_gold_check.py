"""E34: are the consensus-selected anchors true anchors? (camera-ready, reviewer #2)

The structural-anchor pool of E12/E32 is chosen by judge consensus: the items
whose 33-judge label distribution lies closest to a simplex vertex.  Corollary
C1(i) says that a confusion shared by every family is invisible to agreement,
so consensus alone cannot certify that an item's true profile is a vertex.
For the point-contract panels studied here the human gold label is available,
so we can check directly whether the consensus vertex of each pool item matches
its gold label.

Reuses the original loaders and the exact pool / draw construction of E32
(no re-implementation of the selection rule).

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e12_structural_anchor_realdata import load as load12  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e34_anchor_gold_check.json")


def second_panel_with_gold():
    """Same construction as e32.second_panel(), but also returns gold labels."""
    sample_p = os.path.join(ROOT, "data", "second_panel_treccovid", "sample.json")
    panel_p = os.path.join(ROOT, "data", "second_panel_treccovid", "panel_labels.json")
    sample = {i["item_id"]: i for i in json.load(open(sample_p))["items"]}
    panel = json.load(open(panel_p))
    judges = list(panel["labels"])
    keys = sorted(set.intersection(*[set(panel["labels"][j]) for j in judges]))
    gold = np.array([sample[k]["gold"] for k in keys])
    L = 3
    P = np.zeros((len(keys), L))
    for i, k in enumerate(keys):
        c = np.zeros(L)
        for j in judges:
            c[panel["labels"][j][k]] += 1
        P[i] = c / c.sum()
    return P, gold, L


def check(name, P, g, L, n_base, fracs):
    vd = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vd)
    near_idx = order[: int(0.10 * len(order))]
    amb_idx = order[int(0.50 * len(order)):]
    # Replicate the exact random draws of e32.sweep().
    rng0 = np.random.default_rng(7)
    base = rng0.choice(amb_idx, size=min(n_base, len(amb_idx)), replace=False)
    drawn = set()
    for frac in fracs:
        n_anchor = int(round(frac * len(base)))
        if n_anchor > 0:
            a = rng0.choice(near_idx, size=min(n_anchor, len(near_idx)), replace=False)
            drawn.update(int(x) for x in a)
    drawn = np.array(sorted(drawn))

    def summary(idx):
        cons = P[idx].argmax(axis=1)
        return {
            "n_items": int(len(idx)),
            "max_vertex_distance": float(vd[idx].max()),
            "consensus_vertex_counts": {str(l): int((cons == l).sum()) for l in range(L)},
            "gold_label_counts": {str(l): int((g[idx] == l).sum()) for l in range(L)},
            "consensus_equals_gold": int((cons == g[idx]).sum()),
            "consensus_equals_gold_frac": float((cons == g[idx]).mean()),
        }

    res = {"pool_top10pct": summary(near_idx), "drawn_anchors": summary(drawn)}
    print(name, json.dumps(res, indent=1))
    return res


def main() -> None:
    out = {"env": {"script": "e34_anchor_gold_check.py",
                   "pool_rule": "top 10% by l1 distance 2(1-max p) to nearest vertex (as E32)",
                   "draws": "rng seed 7, identical order to e32.sweep()",
                   "gpu_used": False, "api_calls": 0, "cost_usd": 0.0}}
    _, _, P1, g1 = load12()
    out["first_panel"] = check("first_panel", P1, g1, 4, 400, [0.0, 0.005, 0.05, 0.20])
    P2, g2, L2 = second_panel_with_gold()
    out["second_panel"] = check("second_panel", P2, g2, L2, 120, [0.0, 0.02, 0.10, 0.20])
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
