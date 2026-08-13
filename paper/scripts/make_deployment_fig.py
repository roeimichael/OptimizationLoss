"""Regenerate the AAAI deployment figure (fig_deployment) -- FULL WIDTH (figure*).

Grouped bar chart of MEAN native satisfaction (the `sat` column, 0/1 per cell)
per method, with one bar per backbone inside each method cluster. Per-dataset
means are overlaid as dots so the seed/dataset spread behind each bar is visible.

  Trained-under-constraint optimizers (TraLO, Hounie-RCL, Fioretto-LDF,
  TraLO-bounded) satisfy the count natively; the post-hoc clippers (Heuristic,
  LP-LG) reach it natively only at loose caps.

Source: paper/data/corpus/corpus_final.csv, sweep=='paper_final'.
  columns used: model {MobileNetV3, RegNetY400MF, ViTB16}, method, dataset, sat

Run:  python paper/scripts/make_deployment_fig.py
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_style import apply_style, savefig_dual, BACKBONE_COLOR

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paper", "figures")
SRC = os.path.join(ROOT, "paper", "data", "corpus", "corpus_final.csv")
os.makedirs(OUT, exist_ok=True)

apply_style()

# Method order: trained-with-constraint optimizers first (high native sat),
# then post-hoc clippers (low native sat). Labels match the body text EXACTLY.
METHOD_ORDER = ["tralo", "hounie_rcl", "fioretto_ldf", "tralo_bounded",
                "heuristic", "danits_lp"]
METHOD_LABELS = {
    "tralo": "TraLO",
    "hounie_rcl": "Hounie-RCL",
    "fioretto_ldf": "Fioretto-LDF",
    "tralo_bounded": "TraLO-bounded",
    "heuristic": "Heuristic",
    "danits_lp": "LP-LG",
}
POSTHOC = {"heuristic", "danits_lp"}

BACKBONE_ORDER = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
BACKBONE_LABELS = {
    "MobileNetV3": "MobileNetV3",
    "RegNetY400MF": "RegNetY-400MF",
    "ViTB16": "ViT-B/16",
}
# Redundant encoding (survives grayscale): each backbone gets a distinct hatch.
BACKBONE_HATCH = {"MobileNetV3": "", "RegNetY400MF": "///", "ViTB16": "..."}


def make_deployment():
    df = pd.read_csv(SRC)
    pf = df[df["sweep"] == "paper_final"].copy()

    # Mean native satisfaction per (model, method) and per (model, method, dataset).
    piv = (pf.groupby(["model", "method"])["sat"].mean()
              .unstack().reindex(index=BACKBONE_ORDER, columns=METHOD_ORDER))
    n_per = (pf.groupby(["model", "method"])["sat"].count()
               .unstack().reindex(index=BACKBONE_ORDER, columns=METHOD_ORDER))
    ds_mean = (pf.groupby(["model", "method", "dataset"])["sat"].mean())

    n_methods = len(METHOD_ORDER)
    n_bb = len(BACKBONE_ORDER)
    x = np.arange(n_methods, dtype=float)
    total_w = 0.80
    bar_w = total_w / n_bb

    fig, ax = plt.subplots(figsize=(7.0, 3.0))

    first_posthoc = min(i for i, m in enumerate(METHOD_ORDER) if m in POSTHOC)
    ax.axvspan(first_posthoc - 0.5, n_methods - 0.5,
               color="#bdbdbd", alpha=0.16, zorder=0)

    for j, bb in enumerate(BACKBONE_ORDER):
        offsets = x + (j - (n_bb - 1) / 2.0) * bar_w
        vals = piv.loc[bb, METHOD_ORDER].to_numpy(dtype=float)
        ax.bar(offsets, vals, width=bar_w * 0.90,
               color=BACKBONE_COLOR[bb], edgecolor="white", linewidth=0.5,
               hatch=BACKBONE_HATCH[bb], label=BACKBONE_LABELS[bb], zorder=3)
        # Overlay per-dataset means (3 datasets) so the spread behind each bar shows.
        for i, m in enumerate(METHOD_ORDER):
            try:
                pts = [ds_mean.loc[(bb, m, d)] for d in
                       sorted(pf[(pf.model == bb) & (pf.method == m)]["dataset"].unique())]
            except KeyError:
                pts = []
            if pts:
                jit = np.linspace(-bar_w * 0.18, bar_w * 0.18, len(pts))
                # clip_on=False + white edge: dots at exactly 0.00 (clippers) would
                # otherwise be half-swallowed by the x-axis spine.
                ax.scatter(np.full(len(pts), offsets[i]) + jit, pts,
                           s=7, color="#222222", edgecolor="white", linewidth=0.6,
                           zorder=5, clip_on=False)

    ax.axhline(1.0, color="#444444", lw=0.9, ls="--", alpha=0.7, zorder=2)
    ax.text(n_methods - 0.55, 1.012, "target = 1.00", ha="right", va="bottom",
            fontsize=8, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_ylabel("mean native satisfaction")
    ax.set_ylim(0.0, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.5, n_methods - 0.5)

    ax.text((first_posthoc - 1) / 2.0, 1.075, "trained under constraint",
            ha="center", va="center", fontsize=8.5, color="#333333", fontstyle="italic")
    ax.text((first_posthoc + n_methods - 1) / 2.0, 1.075, "post-hoc clippers",
            ha="center", va="center", fontsize=8.5, color="#333333", fontstyle="italic")

    # Legend in the empty upper area over the short clipper bars; add the dot proxy.
    handles, labels = ax.get_legend_handles_labels()
    dot = plt.Line2D([], [], marker="o", color="#222222", linestyle="none",
                     markersize=4, markeredgecolor="white", label="per-dataset mean")
    ax.legend(handles + [dot], labels + ["per-dataset mean"],
              loc="center right", bbox_to_anchor=(1.0, 0.60), frameon=False,
              ncol=1, title="backbone", title_fontsize=8, fontsize=7.5)

    fig.tight_layout()
    pdf, png = savefig_dual(fig, OUT, "fig_deployment")
    plt.close(fig)

    print("MEAN NATIVE SATISFACTION (paper_final):")
    print(piv.round(4).to_string())
    print("\nN per (backbone, method):")
    print(int(n_per.fillna(0).to_numpy().flatten()[0]), "(typical)")
    print(n_per.astype("Int64").to_string())
    return png, piv


if __name__ == "__main__":
    p, piv = make_deployment()
    print("WROTE", p, os.path.getsize(p))
