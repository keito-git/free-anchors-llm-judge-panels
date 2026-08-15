"""E17: is the flat diameter an artefact of how families were defined?

Every real-data result so far uses one partition of the 33 judges into
families: GPT, Llama and T5 by model lineage.  A reader can reasonably ask
whether that particular partition manufactures the conclusion.  Here the same
measurement is repeated under four partitions built from the metadata already
recorded for each judge, and the question is whether the pattern survives:
the first family binds, and later families leave the diameter unchanged.

Partitions
  A  lineage        GPT / Llama / T5                       (the paper's default)
  C  fine-tuning    GPT / Llama zero-shot / Llama fine-tuned / T5
  D  architecture   decoder-only / encoder-decoder
  E  parameter size proprietary / 70B / 8B / sub-1B

Families are added in the listed order, so the m-th value uses the first m
families.  Prototypes and kernels are estimated exactly as in E2c.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e9_criterion_and_unified_radius import unified_radius
from e6_sensitivity_and_vossler import (
    L,
    VALID,
    SUB_DIR,
    GOLD,
    MAP,
    read_qrel,
    item_profiles,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e17_partition_robustness.json")
SEEDS = [42, 123, 456, 789, 1024, 20260804]
PER = 4000


def assign(meta: dict) -> dict:
    """Map one judge's metadata to a label under each partition."""
    fam = meta.get("family", "unknown")
    model = meta.get("model", "") or ""
    ft = bool(meta.get("fine_tuned", False))

    if fam == "Llama":
        c = "Llama-ft" if ft else "Llama-zs"
    else:
        c = fam

    d = "encoder-decoder" if fam == "T5" else "decoder-only"

    if fam == "GPT":
        e = "proprietary"
    elif "70B" in model:
        e = "70B"
    elif fam == "Llama":
        e = "8B"
    else:
        e = "sub-1B"
    return {"A": fam, "C": c, "D": d, "E": e}


ORDER = {
    "A": ["GPT", "Llama", "T5"],
    "C": ["GPT", "Llama-zs", "Llama-ft", "T5"],
    "D": ["decoder-only", "encoder-decoder"],
    "E": ["proprietary", "8B", "70B", "sub-1B"],
}
DESC = {
    "A": "model lineage (paper default)",
    "C": "lineage with Llama split by fine-tuning",
    "D": "architecture",
    "E": "parameter size band",
}


def kernels_for(judges, runs, gold, labels, keys, fams):
    C = {}
    for f in fams:
        cols = [j for j, ff in zip(judges, labels) if ff == f]
        M = np.zeros((L, L))
        for k in keys:
            g = gold[k]
            for j in cols:
                v = runs[j][k]
                if v in VALID:
                    M[v, g] += 1
        s = M.sum(axis=0, keepdims=True)
        s[s == 0] = 1.0
        C[f] = M / s
    return C


def main() -> None:
    judges = sorted(f[:-4] for f in os.listdir(SUB_DIR) if f.endswith(".txt"))
    runs = {j: read_qrel(os.path.join(SUB_DIR, j + ".txt")) for j in judges}
    gold = read_qrel(GOLD)
    meta = json.load(open(MAP))
    keys = sorted(set(gold) & set.intersection(*[set(runs[j]) for j in judges]))
    keys = [k for k in keys if gold[k] in VALID]

    tags = [assign(meta.get(j) or {}) for j in judges]
    P = item_profiles(judges, runs, keys)
    g = np.array([gold[k] for k in keys])
    protos = np.vstack([P[g == k].mean(axis=0) for k in range(L)])

    out = {
        "env": {
            "script": "e17_partition_robustness.py",
            "seeds": SEEDS,
            "n_dirs_per_seed": PER,
            "prototypes": "gold-conditioned mean (identical to E2c)",
            "gpu_used": False,
            "api_calls": 0,
            "cost_usd": 0.0,
        },
        "partitions": {},
    }

    for key, fams in ORDER.items():
        labels = [t[key] for t in tags]
        counts = {f: sum(1 for x in labels if x == f) for f in fams}
        fams_present = [f for f in fams if counts[f] > 0]
        C = kernels_for(judges, runs, gold, labels, keys, fams_present)
        kern = [C[f] for f in fams_present]

        row = {"description": DESC[key], "families": fams_present,
               "judges_per_family": {f: counts[f] for f in fams_present},
               "diameter": {}}
        for m in range(0, len(fams_present) + 1):
            r, _ = unified_radius(kern[:m], protos, SEEDS, PER)
            row["diameter"][f"m={m}"] = r
        vals = [row["diameter"][f"m={m}"] for m in range(1, len(fams_present) + 1)]
        row["first_family_reduction"] = row["diameter"]["m=0"] - vals[0]
        row["max_spread_after_first"] = max(vals) - min(vals)
        row["flat_after_first_family"] = bool(max(vals) - min(vals) < 1e-6)
        out["partitions"][key] = row
        seq = "  ".join(f"m{m}={row['diameter'][f'm={m}']:.4f}"
                        for m in range(0, len(fams_present) + 1))
        print(f"[{key}] {DESC[key]:44s} {seq}  flat_after_first="
              f"{row['flat_after_first_family']}", flush=True)

    out["summary"] = {
        "flat_after_first_family_in_all_partitions": all(
            v["flat_after_first_family"] for v in out["partitions"].values()
        ),
        "n_partitions": len(out["partitions"]),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nsummary:", json.dumps(out["summary"]))
    print("saved", OUT)


if __name__ == "__main__":
    main()
