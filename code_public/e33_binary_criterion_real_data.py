"""E33: does the arithmetic criterion predict correctly on real panels?

The paper's most practical claim is that for binary labels a practitioner can
decide, before procuring a family, whether it will shrink the identification
set, using two likelihood ratios and four arithmetic operations
(Proposition, binary criterion).  Until now that claim rested on 60 synthetic
configurations, where the kernels were drawn to order.  A reader is entitled to
ask whether it holds on panels that were not constructed for it.

Both panels are binarised to L = 2, relevant against not relevant, at the split
each collection uses.  For every family f the criterion needs

    alpha_f = P(judge says relevant | truth not relevant)
    beta_f  = P(judge says relevant | truth relevant)
    r1(f) = beta_f / alpha_f,     r0(f) = (1 - beta_f) / (1 - alpha_f)

and predicts that adding family g to a set S strictly shrinks the gauge set iff
r1(g) exceeds every r1 in S or r0(g) falls below every r0 in S.  We enumerate
every (S, g) pair with S non-empty, record the prediction, then measure whether
the gauge set actually shrank, and report the agreement.

Measuring set shrinkage rather than diameter shrinkage is deliberate: the
criterion is about the set, and Remark (set vs diameter) says the two differ.
A direction counts as cut when its admissible step falls.

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "e33_binary_criterion_real_data.json")

SEEDS = [42, 123, 456]
PER = 2000
TOL = 1e-9
MAX_PAIRS = 60          # exhaustive below this, seeded sample above


def max_step(kernels, D, protos):
    eye = np.eye(2)
    lo, hi = 0.0, 2.0
    for _ in range(45):
        t = 0.5 * (lo + hi)
        A = eye + t * D
        ok = True
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            ok = False
        if ok:
            ok = all((Ck @ Ainv >= -1e-12).all() for Ck in kernels)
        if ok:
            ok = bool((protos @ A.T >= -1e-12).all())
        lo, hi = (t, hi) if ok else (lo, t)
    return lo


def set_shrinks(kern_S, kern_Sg, protos) -> tuple:
    """Does adding a family cut any admissible direction?"""
    cut = 0
    total = 0
    for s in SEEDS:
        rng = np.random.default_rng(s)
        for _ in range(PER // len(SEEDS)):
            D = rng.normal(size=(2, 2))
            D -= D.mean(axis=0, keepdims=True)
            n = np.linalg.norm(D)
            if n < 1e-12:
                continue
            t0 = max_step(kern_S, D, protos)
            t1 = max_step(kern_Sg, D, protos)
            total += 1
            if t1 < t0 - 1e-6:
                cut += 1
    return cut > 0, cut, total


def analyse(name, labels_by_family, gold_bin, out):
    """labels_by_family: family -> array of binary judgements per item."""
    fams = list(labels_by_family)
    C, ratios = {}, {}
    for f in fams:
        y = labels_by_family[f]
        a = float(((y == 1) & (gold_bin == 0)).sum() / max((gold_bin == 0).sum(), 1))
        b = float(((y == 1) & (gold_bin == 1)).sum() / max((gold_bin == 1).sum(), 1))
        a = min(max(a, 1e-6), 1 - 1e-6)
        b = min(max(b, 1e-6), 1 - 1e-6)
        C[f] = np.array([[1 - a, 1 - b], [a, b]])
        ratios[f] = {"alpha": a, "beta": b, "r1": b / a, "r0": (1 - b) / (1 - a)}

    protos = np.eye(2)          # point contract at L = 2
    pairs = [(S, g) for k in range(1, len(fams))
             for S in itertools.combinations(fams, k)
             for g in fams if g not in S]
    exhaustive = len(pairs) <= MAX_PAIRS
    if not exhaustive:                      # seeded sample, reported as such
        idx = np.random.default_rng(2026).choice(len(pairs), MAX_PAIRS,
                                                 replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
    rows, agree, n = [], 0, 0
    for S, g in pairs:
            if True:
                pred = (ratios[g]["r1"] > max(ratios[f]["r1"] for f in S) + TOL
                        or ratios[g]["r0"] < min(ratios[f]["r0"] for f in S) - TOL)
                obs, cut, tot = set_shrinks([C[f] for f in S],
                                            [C[f] for f in S] + [C[g]], protos)
                rows.append({"S": list(S), "g": g, "predicted": bool(pred),
                             "observed": bool(obs), "directions_cut": cut,
                             "directions": tot, "match": bool(pred == obs)})
                agree += pred == obs
                n += 1
    out[name] = {
        "families": fams, "likelihood_ratios": ratios,
        "enumeration": "exhaustive" if exhaustive else f"seeded sample of {MAX_PAIRS}",
        "n_pairs": n, "n_agree": agree,
        "agreement_pct": 100 * agree / max(n, 1),
        "disagreements": [r for r in rows if not r["match"]],
        "pairs": rows,
    }
    print(f"[{name}] families={len(fams)} pairs={n} agreement="
          f"{100*agree/max(n,1):.1f}%", flush=True)


def first_panel(out):
    from e6_sensitivity_and_vossler import load
    judges, runs, gold, fam_of, keys = load()
    g = np.array([gold[k] for k in keys])
    gold_bin = (g >= 2).astype(int)          # LLMJudge: 2 and 3 are relevant
    by_fam = {}
    for f in ("GPT", "Llama", "T5"):
        cols = [j for j, ff in zip(judges, fam_of) if ff == f]
        # a family votes relevant when its majority does
        v = np.zeros(len(keys))
        for i, k in enumerate(keys):
            vals = [runs[j][k] for j in cols if runs[j][k] in (0, 1, 2, 3)]
            v[i] = 1 if np.mean([x >= 2 for x in vals]) > 0.5 else 0
        by_fam[f] = v.astype(int)
    analyse("first_panel_LLMJudge", by_fam, gold_bin, out)


def second_panel(out):
    SAMPLE = os.path.join(ROOT, "data", "second_panel_treccovid", "sample.json")
    PANEL = os.path.join(ROOT, "data", "second_panel_treccovid", "panel_labels.json")
    sample = {i["item_id"]: i for i in json.load(open(SAMPLE))["items"]}
    panel = json.load(open(PANEL))
    fam_of = {j: v["family"] for j, v in panel["env"]["judges"].items()}
    judges = list(panel["labels"])
    keys = sorted(set.intersection(*[set(panel["labels"][j]) for j in judges]))
    gold_bin = np.array([1 if sample[k]["gold"] >= 2 else 0 for k in keys])
    fams = []
    for j in judges:
        if fam_of[j] not in fams:
            fams.append(fam_of[j])
    by_fam = {}
    for f in fams:
        cols = [j for j in judges if fam_of[j] == f]
        v = np.array([1 if np.mean([panel["labels"][j][k] >= 2 for j in cols]) > 0.5
                      else 0 for k in keys])
        by_fam[f] = v
    analyse("second_panel_TRECCOVID", by_fam, gold_bin, out)


def main() -> None:
    out = {"env": {"script": "e33_binary_criterion_real_data.py",
                   "seeds": SEEDS, "n_dirs_per_seed": PER,
                   "binarisation": "relevant iff gold >= 2 in both collections",
                   "gpu_used": False, "api_calls": 0, "cost_usd": 0.0}}
    first_panel(out)
    second_panel(out)
    tot = sum(out[k]["n_pairs"] for k in out if k != "env")
    ag = sum(out[k]["n_agree"] for k in out if k != "env")
    out["summary"] = {"total_pairs": tot, "total_agreements": ag,
                      "overall_agreement_pct": 100 * ag / max(tot, 1)}
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nsummary:", json.dumps(out["summary"], indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
