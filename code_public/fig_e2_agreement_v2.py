"""Regenerate the E2 within- vs cross-family agreement figure at publication size.

The first version was drawn at 9x4 inches and then scaled down to one column,
which made every label unreadable.  Here the figure is drawn close to its final
printed size, the boxes are narrowed so the axes are not mostly empty space, the
overall title is dropped because it belongs in the caption, and the two panels
are labelled (a) and (b) so the text can refer to them individually.

Input : results/e2_m3_pairs_defA.csv  (written by e2_experiment.py)
Output: paper/figures/e2_m3_boxplot_defA.pdf and .png

CPU only.  No API calls.  No GPU.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "e2_m3_pairs_defA.csv"
OUT_DIR = ROOT / "paper" / "figures"

C_BLUE = "#0072B2"
C_RED = "#D55E00"
C_GRAY = "#999999"

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "figure.dpi": 300,
    }
)


def panel(ax, within, cross, ylabel, title) -> float:
    bp = ax.boxplot(
        [within, cross],
        tick_labels=["Within", "Cross"],
        widths=0.42,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.4},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.35},
    )
    for patch, clr in zip(bp["boxes"], [C_BLUE, C_GRAY]):
        patch.set_facecolor(clr)
        patch.set_alpha(0.7)
    ax.set_xlim(0.45, 2.55)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return within.mean() - cross.mean()


def main() -> None:
    df = pd.read_csv(SRC)
    w = df.loc[df["within_family"]]
    c = df.loc[~df["within_family"]]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.05))

    panel(
        axes[0],
        w["spearman_rho"].values,
        c["spearman_rho"].values,
        "Spearman $\\rho$",
        "(a) Ordinal labels 0-3",
    )
    panel(
        axes[1],
        w["pearson_phi_binary"].values,
        c["pearson_phi_binary"].values,
        "Pearson $\\phi$",
        "(b) Binarised relevance",
    )
    h1 = axes[1].axhline(0.437, color=C_RED, linestyle="--", linewidth=0.8,
                         label="Reference, within family")
    h2 = axes[1].axhline(0.389, color=C_GRAY, linestyle="--", linewidth=0.8,
                         label="Reference, cross family")

    fig.tight_layout(pad=0.3, w_pad=1.0, rect=(0, 0.09, 1, 1))
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False,
               fontsize=7, handlelength=1.6, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.01))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"e2_m3_boxplot_defA.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
