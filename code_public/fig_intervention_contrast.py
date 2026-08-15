"""Contrast the three interventions on a common, honest scale.

The paper's two headline measurements are easy to misread side by side.  The
family sweep (E2c) and the anchor sweep (E12) use different prototype sets, so
their absolute diameters are not comparable: 1.1644 and 0.7196 are diameters of
different identification sets.  Plotting them on one axis would suggest a
comparison the data does not support.

What is comparable is how much each intervention moves its own baseline.  This
figure therefore plots the diameter after the intervention divided by the
diameter before it, for the three interventions the paper measures:

  adding 26 judges in two further lineage families        (E19)
  adding one within-lineage family, Llama split on FT     (E20)
  adding two unambiguous items already in the pool        (E24)

All values are the optimised diameters, not the sampled ones, since E19 showed
sampling understates by up to 40% and exaggerates every intervention.

A bar at 1.0 means the intervention changed nothing.

Inputs : results/e16_direction_budget_convergence.json  (E2c at 24,000 dirs)
         results/e17_partition_robustness.json
         results/e12_structural_anchor_realdata.json
Output : paper/figures/intervention_contrast.pdf and .png

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_DIR = ROOT / "paper" / "figures"

C_GRAY = "#999999"
C_BLUE = "#0072B2"
C_RED = "#D55E00"

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "figure.dpi": 300,
    }
)


def main() -> None:
    e19 = json.load(open(RES / "e19_worst_direction_optimisation.json"))
    e20 = json.load(open(RES / "e20_optimised_reruns.json"))
    e24 = json.load(open(RES / "e24_guarded_reverification.json"))

    r = {x["m"]: x["optimised_diameter"] for x in e19["rows"]}
    fam_before, fam_after = r[1], r[3]

    c = e20["E17_partitions"]["C"]["optimised"]
    split_before, split_after = c["m=1"], c["m=2"]

    a = e24["anchor_pair"]
    anc_before = a["no_anchor"]["guarded"]
    anc_after = a["two_anchors"]["guarded"]

    bars = [
        ("+26 judges\n(2 lineage families)", fam_before, fam_after, C_GRAY),
        ("+1 family\n(within-lineage split)", split_before, split_after, C_BLUE),
        ("+2 unambiguous\nitems", anc_before, anc_after, C_RED),
    ]

    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    xs = range(len(bars))
    ratios = [aft / bef for _, bef, aft, _ in bars]
    ax.bar(xs, ratios, width=0.55, color=[c for *_, c in bars], alpha=0.85,
           edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)

    for x, (lab, before, after, _), rr in zip(xs, bars, ratios):
        ax.text(x, rr + 0.03, f"{rr:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(x, 0.04, f"{before:.3f}\u2192{after:.3f}", ha="center",
                va="bottom", fontsize=6.5, color="white")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for lab, *_ in bars], fontsize=7.5)
    ax.set_ylabel("Diameter after / before")
    ax.set_ylim(0, 1.25)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        q = OUT_DIR / f"intervention_contrast.{ext}"
        fig.savefig(q, bbox_inches="tight")
        print(f"Saved: {q}")
    print("ratios:", [f"{x:.4f}" for x in ratios])
    plt.close(fig)


if __name__ == "__main__":
    main()
