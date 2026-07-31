"""Shared AAAI figure style for the final_AAAI_PAPER figures.

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
