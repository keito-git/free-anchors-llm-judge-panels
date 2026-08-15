#!/usr/bin/env python3
"""
fig_protocol_flow_v2.py
Figure B v2: Diagnostic Protocol Flowchart (Section 5)

CHANGE FROM v1:  Corrected structure after the T4 proof was completed.
  T4 requires m>=3 + Kruskal condition as PREREQUISITES, not alternatives.

Correct flow:
  Kruskal NO  → T2 (human anchors). Structural anchor impossible.
  Kruskal YES → Contract type?
    Point contract  → T1 (no anchor needed; m>=3 satisfied)
    Distribution    → Estimate FCI (2eta) → Structural anchors available?
                        Yes → T4 (vertex items; no human labels; m>=3 already satisfied)
                        No  → T2 (human anchors)

T4 box includes E12 real-data result:
  "2 anchors halve orbit diameter (0.720 -> 0.367); saturates" (see E12)

Usage:
  python fig_protocol_flow_v2.py [--outdir <path>]

Output: fig_protocol_flow_v2.pdf  fig_protocol_flow_v2.png
  (old version fig_protocol_flow.pdf/png is preserved unchanged)

Dependencies: numpy, matplotlib
"""

from pathlib import Path as _FsPath
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

FS_BOX   = 7.2
FS_LABEL = 6.5
FS_HEAD  = 8.5
FS_NOTE  = 6.0   # E12 annotation

# Okabe-Ito safe palette
C_PROC  = "#AED6F1"   # light blue  -- process box
C_DEC   = "#FAD7A0"   # light amber -- decision diamond
C_T1    = "#A9DFBF"   # green       -- T1 terminal
C_T4    = "#D5F5E3"   # pale green  -- T4 terminal
C_T2    = "#FDEBD0"   # pale tan    -- T2 terminal
C_NOTE  = "#FDFEFE"   # near-white  -- L=2 note
C_EDGE  = "#222222"
C_ARROW = "#444444"

# ------------------------------------------------------------------ #
# Layout (axes-fraction coords; axes xlim=ylim=[0,1])
#
# Main horizontal chain (Y=0.82):
#   Input [0.015, 0.105]  cx=0.060
#   Partn [0.135, 0.225]  cx=0.180
#   GoF   [0.265, 0.395]  cx=0.330
#   Kruskal-diamond [0.440, 0.540] cx=0.490
#   T2A   [0.785, 0.895]  cx=0.840  (Kruskal NO → T2)
#
# Contract decision (Y=0.60):
#   Contract-diamond [0.440, 0.540] cx=0.490  (same x as Kruskal)
#   T1    [0.645, 0.755]  cx=0.700  (Point → T1)
#
# FCI + Anchors row (Y=0.38):
#   FCI   [0.435, 0.545]  cx=0.490  (same x as Contract)
#   Anchors-diamond [0.640, 0.740] cx=0.690
#
# Terminals right:
#   T4    [0.815, 0.925]  cx=0.870  Y=0.52
#   T2B   [0.815, 0.925]  cx=0.870  Y=0.24
#
# Verified no overlap (see docstring analysis).
# ------------------------------------------------------------------ #

W_SM = 0.045   # small process boxes
W_MD = 0.055   # medium (FCI)
W_LG = 0.065   # GoF (larger text)
W_T  = 0.055   # terminals
WD   = 0.050   # diamond half-width
H    = 0.090   # box height
HD   = 0.095   # diamond half-height

XIN   = 0.060;  XPAR = 0.180;  XGOF = 0.330;  XKRU = 0.490
XT2A  = 0.840   # T2 Kruskal-No
XCON  = 0.490   # Contract diamond (same x as Kruskal)
XT1   = 0.700   # T1 terminal
XFCI  = 0.490   # FCI (same x as Contract)
XANC  = 0.690   # Anchors diamond
XT4   = 0.870;  XT2B = 0.870

Y_CHAIN = 0.82;  Y_T2A = 0.82
Y_CON   = 0.60;  Y_T1  = 0.60
Y_FCI   = 0.38;  Y_ANC = 0.38
Y_T4    = 0.52;  Y_T2B = 0.24


# ------------------------------------------------------------------ #
# Drawing helpers
# ------------------------------------------------------------------ #

def rect(ax, cx, cy, hw, h, text, fc, ec=C_EDGE, lw=0.9, fs=FS_BOX):
    x0, y0 = cx - hw, cy - h / 2
    p = FancyBboxPatch((x0, y0), 2 * hw, h,
                       boxstyle="round,pad=0.02",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, fontsize=fs, ha="center", va="center",
            color=C_EDGE, multialignment="center", zorder=4)


def diamond(ax, cx, cy, hw, hh, text, fc=C_DEC, ec=C_EDGE, lw=0.9, fs=FS_BOX):
    verts = [(cx, cy + hh), (cx + hw, cy),
             (cx, cy - hh), (cx - hw, cy), (cx, cy + hh)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    patch = mpatches.PathPatch(Path(verts, codes),
                               facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    ax.text(cx, cy, text, fontsize=fs - 0.5, ha="center", va="center",
            color=C_EDGE, multialignment="center", zorder=4)


def arr(ax, x0, y0, x1, y1, lbl="", lx=0, ly=0.018, la="center", lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=lw, mutation_scale=8), zorder=2)
    if lbl:
        mx = (x0 + x1) / 2 + lx
        my = (y0 + y1) / 2 + ly
        ax.text(mx, my, lbl, fontsize=FS_LABEL, ha=la, va="bottom",
                style="italic", color=C_EDGE, zorder=5)


def elbow(ax, pts, lbl="", li=0, lx=0.015, ly=0, lw=1.1):
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=C_ARROW, lw=lw, zorder=2, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=lw, mutation_scale=8), zorder=2)
    if lbl:
        mx = (pts[li][0] + pts[li+1][0]) / 2 + lx
        my = (pts[li][1] + pts[li+1][1]) / 2 + ly
        ax.text(mx, my, lbl, fontsize=FS_LABEL, ha="left", va="center",
                style="italic", color=C_EDGE, zorder=5)


# ------------------------------------------------------------------ #
# Build the corrected flowchart
# ------------------------------------------------------------------ #

def build(ax):
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # ---- Title ----
    ax.text(0.5, 0.97,
            "Diagnostic Protocol (§5): Panel Identification Workflow",
            fontsize=FS_HEAD, ha="center", va="top", weight="bold", color=C_EDGE)

    # ---- Main horizontal chain ----
    rect(ax, XIN,  Y_CHAIN, W_SM, H, "Judgment\nLogs",           fc=C_PROC)
    rect(ax, XPAR, Y_CHAIN, W_SM, H, "Partition\nby Family",     fc=C_PROC)
    rect(ax, XGOF, Y_CHAIN, W_LG, H,
         "GoF Diagnosis\ni) Kruskal rank\nii) GMM BIC\niii) LOFO",
         fc=C_PROC, fs=FS_BOX - 0.4)
    diamond(ax, XKRU, Y_CHAIN, WD, HD, "Kruskal\ncondition\nmet?")

    # ---- Main chain arrows ----
    arr(ax, XIN + W_SM,  Y_CHAIN, XPAR - W_SM,  Y_CHAIN)
    arr(ax, XPAR + W_SM, Y_CHAIN, XGOF - W_LG,  Y_CHAIN)
    arr(ax, XGOF + W_LG, Y_CHAIN, XKRU - WD,    Y_CHAIN)

    # ====== BRANCH A: Kruskal NO → T2 directly ======
    # Kruskal condition not met → T4 impossible (T4 requires Kruskal); T2 only option.
    elbow(ax,
          [(XKRU + WD, Y_CHAIN),
           (XT2A - W_T, Y_CHAIN)],
          lbl="No", li=0, lx=0.01, ly=0.015)
    rect(ax, XT2A, Y_T2A, W_T, H,
         "T2: Human anchors\n$n$ items\nerror $\\propto c/\\sqrt{n}$",
         fc=C_T2, fs=FS_BOX - 0.5)

    # ====== BRANCH B: Kruskal YES → Contract type ======
    arr(ax, XKRU, Y_CHAIN - HD, XCON, Y_CON + HD,
        lbl="Yes", lx=0.015, ly=0)
    diamond(ax, XCON, Y_CON, WD, HD,
            "Contract\ntype?", fs=FS_BOX - 0.5)

    # -- B1: Point contract → T1 --
    arr(ax, XCON + WD, Y_CON, XT1 - W_T, Y_T1,
        lbl="Point", lx=0.01, ly=0.015)
    rect(ax, XT1, Y_T1, W_T, H,
         "T1: Identification\nachieved ($m{\\geq}3$)\nNo anchor needed",
         fc=C_T1, fs=FS_BOX - 0.5)

    # -- B2: Distribution contract → FCI --
    arr(ax, XFCI, Y_CON - HD, XFCI, Y_FCI + H / 2,
        lbl="Distribution", lx=0.015, ly=0)
    rect(ax, XFCI, Y_FCI, W_MD, H,
         "Estimate FCI\n$\\eta = \\min_{k,j}\\hat{p}_j^{(k)}$\n(floor $= 2\\eta$)",
         fc=C_PROC, fs=FS_BOX - 0.6)

    # FCI → Anchors decision
    arr(ax, XFCI + W_MD, Y_FCI, XANC - WD, Y_ANC)

    # ---- Anchors decision diamond ----
    diamond(ax, XANC, Y_ANC, WD, HD,
            "Structural\nanchors\navailable?", fs=FS_BOX - 0.8)

    # -- B2a: Anchors YES → T4 --
    elbow(ax,
          [(XANC, Y_ANC + HD),
           (XANC, Y_T4),
           (XT4 - W_T, Y_T4)],
          lbl="Yes", li=0, lx=0.015, ly=0)
    rect(ax, XT4, Y_T4, W_T, H,
         "T4: Seed vertex items\n($m{\\geq}3$ already met)\nno human labels",
         fc=C_T4, fs=FS_BOX - 0.7)

    # E12 annotation below T4
    ax.text(XT4, Y_T4 - H / 2 - 0.025,
            r"Real data: 2 items → orbit ${\times}0.51$ ($0.720{\to}0.367$); saturates",
            fontsize=FS_NOTE, ha="center", va="top",
            color="#555555", style="italic", zorder=5)

    # -- B2b: Anchors NO → T2B --
    elbow(ax,
          [(XANC, Y_ANC - HD),
           (XANC, Y_T2B),
           (XT2B - W_T, Y_T2B)],
          lbl="No", li=0, lx=0.015, ly=0)
    rect(ax, XT2B, Y_T2B, W_T, H,
         "T2: Human anchors\n$n$ items\nerror $\\propto c/\\sqrt{n}$",
         fc=C_T2, fs=FS_BOX - 0.7)

    # ------------------------------------------------------------------ #
    # L=2 shortcut note (bottom, separated from flow)
    # ------------------------------------------------------------------ #
    nx0, ny0, nw, nh = 0.22, 0.01, 0.50, 0.065
    ax.add_patch(FancyBboxPatch(
        (nx0, ny0), nw, nh,
        boxstyle="round,pad=0.02",
        facecolor=C_NOTE, edgecolor="#AAAAAA", linewidth=0.7, zorder=3))
    ax.text(nx0 + nw / 2, ny0 + nh / 2,
            r"$L{=}2$ shortcut (Prop.): "
            r"family $g$ panel-dominated iff "
            r"$r_1(g){\leq}\max_f r_1(f)$ and $r_0(g){\geq}\min_f r_0(f)$"
            " (four-operation check)",
            fontsize=FS_LABEL + 0.2, ha="center", va="center",
            style="italic", color="#444444", zorder=4,
            multialignment="center")

    # ------------------------------------------------------------------ #
    # Legend (bottom-left, above L=2 note)
    # ------------------------------------------------------------------ #
    lx, ly = 0.02, 0.10
    lw_b, lh_b, lgap = 0.085, 0.036, 0.018
    for fc, lbl in [(C_PROC, "Process"),
                    (C_DEC,  "Decision"),
                    (C_T1,   "T1: no anchor"),
                    (C_T4,   "T4: struct. anchor"),
                    (C_T2,   "T2: human anchor")]:
        ax.add_patch(FancyBboxPatch(
            (lx, ly - lh_b / 2), lw_b, lh_b,
            boxstyle="round,pad=0.01",
            facecolor=fc, edgecolor=C_EDGE, linewidth=0.7, zorder=3))
        ax.text(lx + lw_b / 2, ly, lbl,
                fontsize=FS_LABEL - 0.5, ha="center", va="center",
                color=C_EDGE, zorder=4)
        lx += lw_b + lgap


def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure B v2: Corrected Diagnostic Protocol Flowchart.")
    parser.add_argument("--outdir", default=
        str(_FsPath(__file__).resolve().parents[1] / "paper" / "figures"),
        help="Output directory for PDF and PNG.")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    build(ax)
    plt.tight_layout(pad=0.15)

    pdf_path = f"{args.outdir}/fig_protocol_flow_v2.pdf"
    png_path = f"{args.outdir}/fig_protocol_flow_v2.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
