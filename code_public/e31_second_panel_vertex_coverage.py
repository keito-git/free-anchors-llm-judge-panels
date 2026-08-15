"""E31: why the anchor rule did not reproduce on the second panel.

E30 found that injecting near-vertex items into the TREC-COVID pool does not
shrink the identification set and at larger injection rates enlarges it, which
is the opposite of what E12 found on the first panel.  Before treating that as
a property of the collection it is worth checking the precondition the rule
actually needs.

Theorem 4 requires vertex prototypes covering every label.  E15 showed that on
the first panel the freely available near-vertex items all sit at one label, so
the precondition already failed there and E12's reduction was not a
demonstration of full recovery.  This script runs the same measurement on the
second panel: for each label, how close does the near-vertex pool get to that
vertex, and how are the pool items distributed over labels.

If the concentration is worse here, the failure in E30 is explained by the
precondition rather than by the collection, and the design rule can be stated
with the condition attached instead of being weakened outright.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "data", "second_panel_treccovid", "sample.json")
PANEL = os.path.join(ROOT, "data", "second_panel_treccovid", "panel_labels.json")
OUT = os.path.join(ROOT, "results", "e31_second_panel_vertex_coverage.json")

L = 3
SLICES = (0.05, 0.10, 0.20)


def main() -> None:
    sample = {i["item_id"]: i for i in json.load(open(SAMPLE))["items"]}
    panel = json.load(open(PANEL))
    judges = list(panel["labels"])
    keys = sorted(set.intersection(*[set(panel["labels"][j]) for j in judges]))
    gold = np.array([sample[k]["gold"] for k in keys])

    P = np.zeros((len(keys), L))
    for i, k in enumerate(keys):
        c = np.zeros(L)
        for j in judges:
            c[panel["labels"][j][k]] += 1
        P[i] = c / c.sum()

    vertex_dist = 2.0 * (1.0 - P.max(axis=1))
    order = np.argsort(vertex_dist)
    eye = np.eye(L)

    out = {
        "env": {"script": "e31_second_panel_vertex_coverage.py",
                "panel": "TREC-COVID, nine judges, seven families",
                "n_items": len(keys), "L": L,
                "gpu_used": False, "api_calls": 0, "cost_usd": 0.0},
        "distance_to_each_vertex": {},
        "pool_slices": {},
    }

    # distance from each vertex to the nearest item in the whole panel
    for lab in range(L):
        d = np.abs(P - eye[lab]).sum(axis=1)
        out["distance_to_each_vertex"][f"label_{lab}"] = float(d.min())
    print("min l1 distance to each vertex (all items):",
          {k: round(v, 4) for k, v in out["distance_to_each_vertex"].items()},
          flush=True)

    for frac in SLICES:
        idx = order[: int(frac * len(order))]
        pool = P[idx]
        per_vertex = {}
        for lab in range(L):
            d = np.abs(pool - eye[lab]).sum(axis=1)
            per_vertex[f"label_{lab}"] = float(d.min())
        nearest = Counter(int(x) for x in pool.argmax(axis=1))
        out["pool_slices"][f"top_{int(frac*100)}pct"] = {
            "n_items": int(len(idx)),
            "max_l1_distance_to_own_vertex": float(vertex_dist[idx].max()),
            "min_distance_per_vertex": per_vertex,
            "items_per_dominant_label": {str(k): v for k, v in sorted(nearest.items())},
            "gold_per_dominant_label": {
                str(k): int((gold[idx] == k).sum()) for k in range(L)},
        }
        print(f"top {int(frac*100)}%: n={len(idx)} "
              f"dominant-label counts={dict(sorted(nearest.items()))} "
              f"per-vertex min={ {k: round(v,4) for k,v in per_vertex.items()} }",
              flush=True)

    slices = out["pool_slices"]
    covered = {s: sum(1 for v in d["min_distance_per_vertex"].values() if v < 0.10)
               for s, d in slices.items()}
    out["summary"] = {
        "vertices_within_0.10_by_slice": covered,
        "all_vertices_covered_in_any_slice": any(v == L for v in covered.values()),
        "reading": (
            "Theorem 4 needs a near-vertex item for every label; the count above "
            "says how many labels the free pool actually reaches."),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nvertices within 0.10 by slice:", covered)
    print("saved", OUT)


if __name__ == "__main__":
    main()
