"""Shared AAAI figure style for the paper figures.

Enforces the AAAI / scientific-visualization checklist (AAAI-26 author kit;
Rougier et al. "Ten Simple Rules for Better Figures"; Wong, Nature Methods 2011):
  - vector PDF export (Overleaf/pdflatex) PLUS a >=300 dpi PNG sibling (HTML preview),
  - embedded TrueType fonts (pdf.fonttype=42, ps.fonttype=42) -- NEVER Type-3,
  - Times New Roman text + STIX math (matches the AAAI body), one consistent set of
    point sizes across every figure,
  - Okabe-Ito colorblind-safe palette + redundant (marker / line-style / hatch) encoding,
  - top/right spines off, light grid (data-ink).

Usage in each make_*.py:
    from fig_style import apply_style, savefig_dual, C_TRALO, C_FIORETTO, C_HOUNIE, \
        C_WASH, BACKBONE_COLOR, OKABE
    apply_style()
    ...
    savefig_dual(fig, OUT, "fig_name")        # writes fig_name.pdf AND fig_name.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (Wong B., Nature Methods 8:441, 2011).
OKABE = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}

# Stable method/role -> color map used identically across ALL figures.
C_TRALO    = OKABE["blue"]        # TraLO (ours)
C_FIORETTO = OKABE["vermillion"]  # Fioretto-LDF (the escalating dual)
C_HOUNIE   = OKABE["green"]       # Hounie-RCL
C_WASH     = "#9a9a9a"            # neutral / wash bars (luminance-distinct from blue)

# Backbone -> color (deployment, octmnist). Blue/green/orange differ in luminance.
BACKBONE_COLOR = {
    "MobileNetV3":  OKABE["blue"],
    "RegNetY400MF": OKABE["green"],
    "ViTB16":       OKABE["orange"],
}


def apply_style():
    """Apply the shared rcParams. Call once at the top of each figure script."""
    plt.rcParams.update({
        # --- font embedding: TrueType / Type 42, never Type 3 (AAAI hard spec) ---
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # --- Times-like serif text + STIX math (matches the AAAI body font) ---
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # --- one consistent set of point sizes (>= 8 pt at final printed size) ---
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # --- data-ink: light grid, no top/right frame, no hairlines ---
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 2.0,
        # --- raster fallback at >= 300 dpi (the PNG used by the HTML preview) ---
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def legend_clear(ax, *args, candidates=None, pad=0.015, verbose=True, **kwargs):
    """Place a legend where it does NOT sit on top of the plotted data.

    matplotlib's loc="best" only avoids overlap approximately and gives up
    silently when it cannot find room, which is how Fig. 4's legend ended up
    with the Fioretto curve running straight through its label. This tries each
    candidate location, counts how many plotted points the legend box would
    cover, and keeps the first location that covers none. If every candidate
    collides it keeps the least-bad one and says so, so the problem surfaces at
    generation time instead of in the PDF.

    Returns the Legend. Extra args/kwargs are forwarded to ax.legend().
    """
    if candidates is None:
        candidates = ["upper left", "upper right", "lower left", "lower right",
                      "center left", "center right", "upper center",
                      "lower center", "center"]

    # Plotted points in axes coordinates (0..1), for lines and marker scatters.
    pts = []
    for ln in ax.get_lines():
        xy = ln.get_xydata()
        if xy is None or len(xy) == 0:
            continue
        pts.append(ax.transAxes.inverted().transform(ax.transData.transform(xy)))
    for coll in ax.collections:
        try:
            off = coll.get_offsets()
        except Exception:
            continue
        if off is not None and len(off):
            pts.append(ax.transAxes.inverted().transform(ax.transData.transform(off)))
    # Bars/patches: sample each rectangle's corners so filled shapes count too.
    for p in ax.patches:
        try:
            bb = p.get_window_extent()
        except Exception:
            continue
        c = ax.transAxes.inverted().transform(
            [[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x0, bb.y1], [bb.x1, bb.y1]])
        pts.append(c)

    if pts:
        import numpy as _np
        allpts = _np.vstack(pts)
        allpts = allpts[_np.isfinite(allpts).all(axis=1)]
    else:
        allpts = None

    best = None
    for loc in candidates:
        leg = ax.legend(*args, loc=loc, **kwargs)
        ax.figure.canvas.draw()
        bb = leg.get_window_extent()
        x0, y0 = ax.transAxes.inverted().transform((bb.x0, bb.y0))
        x1, y1 = ax.transAxes.inverted().transform((bb.x1, bb.y1))
        if allpts is None:
            return leg
        inside = ((allpts[:, 0] > x0 - pad) & (allpts[:, 0] < x1 + pad) &
                  (allpts[:, 1] > y0 - pad) & (allpts[:, 1] < y1 + pad))
        n = int(inside.sum())
        if n == 0:
            return leg
        if best is None or n < best[0]:
            best = (n, loc)
        leg.remove()

    n, loc = best
    if verbose:
        print("  WARNING: legend on %r overlaps data at every candidate location; "
              "using %r (%d points covered). Consider shrinking the legend, "
              "widening the axes, or moving it outside the axes."
              % (ax.get_title() or ax.get_ylabel(), loc, n))
    return ax.legend(*args, loc=loc, **kwargs)


def savefig_dual(fig, out_dir, stem):
    """Save BOTH a vector PDF (for Overleaf/pdflatex) and a >=300 dpi PNG (HTML preview).

    main.tex uses extensionless \\includegraphics{stem}, so pdflatex prefers the PDF;
    scripts/build_paper_html.py resolves the .png sibling for the browser render.
    """
    pdf = os.path.join(out_dir, stem + ".pdf")
    png = os.path.join(out_dir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png)
    return pdf, png
