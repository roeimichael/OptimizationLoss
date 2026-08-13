"""fig_datasets: compact single-column row of DermMNIST skin lesions ordered by clinical
severity, ending at melanoma (the constrained class the count cap restricts). Kept small
so the clinically graphic melanoma does not dominate the page.

Images load via the medmnist package (cached in ~/.medmnist); no training data needed.
Run:  python paper/scripts/make_datasets_fig.py
"""
import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from medmnist import DermaMNIST

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import apply_style, savefig_dual
apply_style()

OUT = str(Path(__file__).resolve().parents[1] / "figures")

# size=224: MedMNIST v2 high-resolution variant of the same test images (same split
# ordering as the 28px benchmark), so the severity picks below are identical lesions,
# just crisp enough for print. The model trains on 28px; this teaser is illustrative.
ds = DermaMNIST(split="test", download=True, size=224)
imgs, labels = ds.imgs, ds.labels.flatten()

# (class idx, sample index within class, short name, risk, colour) benign -> constrained
CELLS = [
    (5, 5,  "nevus",     "benign",      "#1f7a1f"),
    (2, 15, "keratosis", "benign",      "#1f7a1f"),
    (1, 15, "carcinoma", "malignant",   "#a01f1f"),
    (4, 15, "melanoma",  "constrained", "#0b3d66"),
]

# Small single-column row; risk is carried by frame colour + short label, spelled out
# in the caption, so the panel stays compact. Natural size ~= 0.85*columnwidth so the
# labels print near 1:1 (no shrink from LaTeX scaling).
fig, axes = plt.subplots(1, 4, figsize=(2.85, 0.92))
for ax, (cls, k, name, risk, col) in zip(axes, CELLS):
    idx = np.where(labels == cls)[0]
    ax.imshow(imgs[idx[k]], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(col); s.set_linewidth(1.2)
    ax.set_xlabel(name, fontsize=7.4, color=col, fontweight="bold", labelpad=1.5)
fig.tight_layout(pad=0.2, w_pad=0.5)
pdf, png = savefig_dual(fig, OUT, "fig_datasets")
plt.close(fig)
print("WROTE", pdf)
