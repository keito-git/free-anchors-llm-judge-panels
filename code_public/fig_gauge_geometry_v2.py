#!/usr/bin/env python3
"""
fig_gauge_geometry_v2.py
Figure A v2: Geometry of the Gauge Theorem (T3 + C1 + T4) -- Corrected T4 prerequisites

CHANGE FROM v1:
  Panel (c) annotation updated after the T4 proof was completed.
  T4 full identification (conclusion b) requires m>=3 AND Kruskal condition,
  not vertex seeding alone. Previous annotation omitted these prerequisites.

Correct annotations for panel (c):
  Line 1: "All L vertex items, m>=3, Kruskal => G = permutations"
  Line 2: "Interior profiles fully identifiable; no human labels (T4)"

Old version preserved as fig_gauge_geometry.pdf/png.

Usage:
  python fig_gauge_geometry_v2.py [--outdir <path>]

Output: fig_gauge_geometry_v2.pdf  fig_gauge_geometry_v2.png

Design intent (unchanged from v1):
  (a) Point Contract: vertex profiles -> G = permutations only (T1)
  (b) Distribution Contract: interior profiles -> G orbit >= 2eta for any m (Cor. C1-i)
      The figure does NOT depict the retracted "orbit shrinks with m" claim.
  (c) + Structural Anchor (T4): L vertex items seeded, m>=3, Kruskal condition
      -> G collapses to permutations; interior profiles identifiable without human labels.

Dependencies: numpy, matplotlib, scipy
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull

# ---- Lab-standard font config ----
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

FS_TITLE = 9.5   # panel title
FS_ANNOT = 7.8   # annotation text
FS_VERT  = 8.5   # vertex labels

# Okabe-Ito colorblind-safe palette
COL_VERTEX   = "#0072B2"  # blue      -- vertex profiles (point contract)
COL_INTERIOR = "#E69F00"  # orange    -- interior profiles (distribution contract)
COL_ANCHOR   = "#009E73"  # green     -- structural anchors (T4)
COL_ORBIT    = "#E69F00"  # orange    -- gauge orbit shading
COL_FLOOR    = "#D55E00"  # vermillion-- 2eta arrow / floor annotation
COL_EDGE     = "#222222"  # near-black

# ---- Simplex utilities ----
_V1 = np.array([0.0, 0.0])
_V2 = np.array([1.0, 0.0])
_V3 = np.array([0.5, np.sqrt(3) / 2])
_VERTS_CART = np.array([_V1, _V2, _V3])


def bary2cart(p):
    """Barycentric (p1,p2,p3) -> 2D Cartesian."""
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        return p[0] * _V1 + p[1] * _V2 + p[2] * _V3
    # batched
    return p @ _VERTS_CART


def draw_simplex(ax, fc="#F5F5F5", ec=COL_EDGE, lw=1.5):
    """Draw the triangle and return Cartesian vertex positions."""
    tri = MplPolygon(_VERTS_CART, closed=True,
                     facecolor=fc, edgecolor=ec, linewidth=lw, zorder=0)
    ax.add_patch(tri)
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.28, np.sqrt(3) / 2 + 0.10)
    ax.set_aspect("equal")
    ax.axis("off")
    return _VERTS_CART.copy()


def label_vertices(ax, vc, labels=None, fs=FS_VERT, pad=0.07):
    """Add vertex labels e_1, e_2, e_3."""
    if labels is None:
        labels = [r"$e_1$", r"$e_2$", r"$e_3$"]
    offsets = [(-pad, -pad), (pad, -pad), (0, pad * 0.9)]
    ha_list = ["right", "left", "center"]
    va_list = ["top", "top", "bottom"]
    for pt, lab, off, ha, va in zip(vc, labels, offsets, ha_list, va_list):
        ax.text(pt[0] + off[0], pt[1] + off[1], lab,
                fontsize=fs, ha=ha, va=va, color=COL_EDGE, zorder=10)


# ---- Gauge orbit computation (unchanged from v1) ----

def _arm_points(p, j_star, n_steps=120):
    """
    Points along the analytical arm from p towards the j*-extreme.
    Uses S_t = (1-t)I + t e_{j*} 1^T, t in [0, eta_{j*}].
    S_t^{-1} p = (p - t e_{j*}) / (1 - t).
    """
    L = len(p)
    eta = p[j_star]
    pts = []
    for t in np.linspace(0.0, eta, n_steps):
        if t >= 1.0:
            continue
        q = p.copy()
        q[j_star] -= t
        q /= (1.0 - t)
        if np.all(q >= -1e-9) and abs(q.sum() - 1.0) < 1e-9:
            pts.append(np.maximum(q, 0.0))
    return pts


def compute_g0_orbit(p, n_rand=6000, seed=42):
    """
    Numerical approximation of the G^0 gauge orbit:
      {S^{-1} p : S invertible column-stochastic, S^{-1} p in Delta}.
    Returns array of barycentric points (N, L).
    """
    rng = np.random.default_rng(seed)
    L = len(p)
    orbit = [p.copy()]

    for j_star in range(L):
        orbit.extend(_arm_points(p, j_star, n_steps=200))

    alpha_vals = [0.05, 0.15, 0.5, 1.0, 3.0]
    per_alpha = n_rand // len(alpha_vals)
    for alpha in alpha_vals:
        for _ in range(per_alpha):
            S = np.column_stack([rng.dirichlet(np.full(L, alpha)) for _ in range(L)])
            det = np.linalg.det(S)
            if abs(det) < 1e-9:
                continue
            try:
                q = np.linalg.solve(S, p)
            except np.linalg.LinAlgError:
                continue
            if np.all(q >= -1e-9) and abs(q.sum() - 1.0) < 1e-9:
                orbit.append(np.maximum(q, 0.0))

    return np.array(orbit)


# ---- Panel builders ----

def panel_a(ax):
    """(a) Point Contract: vertex profiles -> G = permutations. [UNCHANGED from v1]"""
    vc = draw_simplex(ax)
    label_vertices(ax, vc)

    for pt in vc:
        ax.scatter(pt[0], pt[1], s=75, color=COL_VERTEX, zorder=6,
                   edgecolors="white", linewidths=0.8)

    ax.set_title("(a) Point Contract", fontsize=FS_TITLE, pad=5, color=COL_EDGE)

    ax.text(0.5, -0.16,
            r"$\mathcal{G} =$ permutations only",
            fontsize=FS_ANNOT, ha="center", va="center",
            color=COL_VERTEX,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.text(0.5, -0.24,
            "Identified up to label relabeling (T1)",
            fontsize=FS_ANNOT - 0.8, ha="center", va="center", color=COL_EDGE)


def panel_b(ax):
    """(b) Distribution Contract: interior profiles -> gauge orbit >= 2eta. [UNCHANGED]"""
    vc = draw_simplex(ax)
    label_vertices(ax, vc)

    p = np.array([0.50, 0.30, 0.20])
    eta = float(np.min(p))
    p_cart = bary2cart(p)

    orbit_bary = compute_g0_orbit(p)
    orbit_cart = bary2cart(orbit_bary)

    if len(orbit_cart) >= 3:
        try:
            hull = ConvexHull(orbit_cart)
            hull_pts = orbit_cart[hull.vertices]
            poly = MplPolygon(hull_pts, closed=True,
                              facecolor=COL_ORBIT, alpha=0.22,
                              edgecolor=COL_ORBIT, linewidth=1.3,
                              linestyle="--", zorder=1)
            ax.add_patch(poly)
            cx = hull_pts[:, 0].mean()
            cy = hull_pts[:, 1].mean()
            ax.text(cx - 0.05, cy + 0.05, "gauge\norbit",
                    fontsize=FS_ANNOT - 1.2, ha="center", va="center",
                    color=COL_ORBIT, style="italic", zorder=7)
        except Exception:
            pass

    ax.scatter(p_cart[0], p_cart[1], s=80, color=COL_INTERIOR, zorder=8,
               edgecolors="white", linewidths=0.9, marker="o")
    ax.text(p_cart[0] + 0.04, p_cart[1] + 0.04, r"$p^{(k)}$",
            fontsize=FS_ANNOT, color=COL_INTERIOR, ha="left", va="bottom", zorder=9)

    # 2eta arrow
    p_ext = np.array([0.625, 0.375, 0.0])
    pe_cart = bary2cart(p_ext)
    ax.annotate("",
                xy=pe_cart, xytext=p_cart,
                arrowprops=dict(arrowstyle="<->",
                                color=COL_FLOOR,
                                lw=1.5, mutation_scale=10))
    mid = (p_cart + pe_cart) / 2
    ax.text(mid[0] + 0.06, mid[1] - 0.01,
            r"$\geq 2\eta$",
            fontsize=FS_ANNOT + 0.5, color=COL_FLOOR,
            ha="left", va="center", zorder=9)

    ax.set_title("(b) Distribution Contract", fontsize=FS_TITLE, pad=5, color=COL_EDGE)

    ax.text(0.5, -0.16,
            r"$\mathcal{G}\supset L(L{-}1)$-dim continuum (T3b)",
            fontsize=FS_ANNOT, ha="center", va="center",
            color=COL_INTERIOR,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.text(0.5, -0.24,
            r"Floor $2\eta$ persists for any $m$ (Cor. C1-i)",
            fontsize=FS_ANNOT - 0.8, ha="center", va="center", color=COL_FLOOR)


def panel_c(ax):
    """
    (c) + Structural Anchor (T4): CORRECTED prerequisites after the T4 proof was completed.
    T4 conclusion (b) full identification requires:
      i)  All L vertex prototypes in pool
      ii) pi_k > 0
      iii) m >= 3
      iv) Kruskal rank condition
      v)  joint injectivity (auto-implied when L<=6 and K_tilde >= L+1)
    Only conclusion (a) G=permutations holds from i) alone.
    Annotation now explicitly states m>=3 + Kruskal as prerequisites.
    """
    vc = draw_simplex(ax)
    label_vertices(ax, vc)

    # Structural anchors at all L=3 vertices (diamond markers, green)
    for pt in vc:
        ax.scatter(pt[0], pt[1], s=90, color=COL_ANCHOR, zorder=6,
                   edgecolors="white", linewidths=0.8, marker="D")

    # Interior profile -- identifiable under T4 (no orbit shading)
    p = np.array([0.50, 0.30, 0.20])
    p_cart = bary2cart(p)
    ax.scatter(p_cart[0], p_cart[1], s=80, color=COL_INTERIOR, zorder=8,
               edgecolors="white", linewidths=0.9, marker="o")
    ax.text(p_cart[0] + 0.04, p_cart[1] + 0.04, r"$p^{(k)}$",
            fontsize=FS_ANNOT, color=COL_INTERIOR, ha="left", va="bottom", zorder=9)

    # Small inner ring to signal collapsed orbit (single-point identified set)
    ax.scatter(p_cart[0], p_cart[1], s=8, color="white", zorder=9,
               edgecolors=COL_INTERIOR, linewidths=0.7)

    # "anchors (vertex items)" label near bottom-left vertex
    ax.text(vc[0][0] + 0.14, vc[0][1] + 0.09,
            "anchors\n(vertex items)",
            fontsize=FS_ANNOT - 1.5, ha="left", va="bottom",
            color=COL_ANCHOR, style="italic", zorder=10)

    # Title (unchanged)
    ax.set_title("(c) + Structural Anchor (T4)", fontsize=FS_TITLE, pad=5,
                 color=COL_EDGE)

    # *** CORRECTED ANNOTATIONS: m>=3 + Kruskal now explicit ***
    # Line 1: main claim -- full prerequisites listed
    ax.text(0.5, -0.16,
            r"All $L$ vertex items, $m{\geq}3$, Kruskal $\Rightarrow\,\mathcal{G}=$ perm.",
            fontsize=FS_ANNOT, ha="center", va="center",
            color=COL_ANCHOR,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    # Line 2: conclusion
    ax.text(0.5, -0.24,
            "Interior profiles fully identifiable; no human labels (T4)",
            fontsize=FS_ANNOT - 0.8, ha="center", va="center", color=COL_EDGE)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure A v2: Gauge Theorem geometry (corrected T4 prereqs).")
    parser.add_argument("--outdir", default=
        str(Path(__file__).resolve().parents[1] / "paper" / "figures"),
        help="Output directory for PDF and PNG.")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85))

    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])

    plt.tight_layout(pad=0.5, w_pad=0.8)

    pdf_path = f"{args.outdir}/fig_gauge_geometry_v2.pdf"
    png_path = f"{args.outdir}/fig_gauge_geometry_v2.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
