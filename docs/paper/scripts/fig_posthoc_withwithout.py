"""Constraint satisfaction WITHOUT vs WITH post-hoc adjustment, per dataset.

Shows what the post-hoc step contributes. Post-hoc guarantees feasibility, so
"with post-hoc" is 100% for every method (right end of each arrow). The filled
dot is "without post-hoc" -- the fraction of seeds whose trained model already
satisfies every budget on its own. The arrow length is therefore exactly the
feasibility post-hoc had to supply:

  * TraLO / Hounie  -> dot already at (near) 100%: post-hoc is a no-op.
  * Fioretto / TraLO-bounded -> short arrow: post-hoc cleans up a few cells.
  * Danits / Heuristic -> full arrow from 0%: post-hoc does ALL the work.

This is a FEASIBILITY comparison. Quality (Macro F1) is reported after post-hoc
and is a separate axis; raw (pre-post-hoc) F1 is not aggregated here because the
raw prediction files live on the experiment server.

Output:  paper/figures/fig_posthoc_withwithout_v2.png
Source:  docs/all_cells_raw.csv (`sat` = raw / pre-post-hoc satisfaction).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
OUT_PATH = PAPER_DIR / "figures" / "fig_posthoc_withwithout_v2.png"

DATASETS = [("tissuemnist", "TissueMNIST"), ("dermmnist", "DermMNIST"),
            ("aider", "AIDER")]
ORDER = ["tralo", "hounie_rcl", "fioretto_ldf", "tralo_bounded",
         "danits_lp", "heuristic"]
LABELS = {"tralo": "TraLO (ours)", "hounie_rcl": "Hounie RCL",
          "fioretto_ldf": "Fioretto LDF", "tralo_bounded": "TraLO-bounded",
          "danits_lp": "Danits LP", "heuristic": "Heuristic"}
COLORS = {"tralo": "#1976D2", "hounie_rcl": "#C62828",
          "fioretto_ldf": "#2E7D32", "tralo_bounded": "#FB8C00",
          "danits_lp": "#9E9E9E", "heuristic": "#616161"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    raw = pd.read_csv(PAPER_DIR.parent / "docs" / "all_cells_raw.csv")
    raw = raw[raw.model == "MobileNetV3"].copy()
    raw["L"] = raw.tight.str.extract(r"L(\d+)").astype(int)
    raw["G"] = raw.tight.str.extract(r"G(\d+)").astype(int)
    sym = raw[raw.L == raw.G]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    y = np.arange(len(ORDER))[::-1]

    for ax, (ds, title) in zip(axes, DATASETS):
        s = sym[sym.ds == ds].groupby("method")["sat"].mean() * 100.0
        for yi, m in zip(y, ORDER):
            wo = float(s.get(m, np.nan))
            c = COLORS[m]
            if wo < 99.5:
                ax.annotate("", xy=(100, yi), xytext=(wo, yi),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2,
                                            shrinkA=0, shrinkB=0, alpha=0.9))
                ax.scatter([100], [yi], s=45, facecolor="white",
                           edgecolor=c, linewidth=1.5, zorder=5)
            ax.scatter([wo], [yi], s=95, color=c, edgecolor="black",
                       linewidth=0.8, zorder=6)
            lab = f"{wo:.0f}%" + ("  (no-op)" if wo >= 99.5 else "")
            ha = "left" if wo < 80 else "right"
            dx = 3 if wo < 80 else -3
            ax.text(wo + dx, yi + 0.28, lab, va="bottom", ha=ha,
                    fontsize=8.3, color=c, fontweight="bold")

        ax.axvline(100, color="0.4", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlim(-6, 116)
        ax.set_ylim(-0.6, len(ORDER) - 0.4)
        ax.set_xlabel("Constraint satisfaction (% of seeds)")
        ax.grid(axis="x", alpha=0.15)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[m] for m in ORDER])

    fig.suptitle("Constraint satisfaction without ($\\bullet$) vs with "
                 "($\\rightarrow$ 100%) post-hoc adjustment\n"
                 "arrow length = feasibility the post-hoc step must supply "
                 "(TraLO/Hounie: already feasible, post-hoc is a no-op)",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
