"""Budget overflow before vs after post-hoc enforcement (normalized).

Makes the flip count tangible WITHOUT the two traps:
  (1) global and local violations count equally -- the metric is the total
      number of over-budget predictions (= flips), not a global-only count;
  (2) raw prediction counts are NOT shown, because a larger raw count is not a
      quality advantage (the over-budget predictions are the model's
      least-confident guesses, which post-hoc discards). Instead we show
      "budget overflow" = over-budget predictions as a PERCENT of the budget K,
      a pure feasibility-distance metric. After post-hoc every method is at 0%.

Quality (Macro F1) is tied across methods and is reported separately; this
figure says nothing about accuracy by design.

Output:  paper/figures/fig_posthoc_beforeafter_v2.png
Source:  docs/all_cells_raw.csv (flips) + budget K from the training logs.
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
OUT_PATH = PAPER_DIR / "figures" / "fig_posthoc_beforeafter_v2.png"

CELL = "L30_G30"
K = 67  # MEL budget at G30 on DermMNIST (global cap; local caps sum to it)

ORDER = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
         "danits_lp", "heuristic"]
LABELS = {"tralo": "TraLO (ours)", "tralo_bounded": "TraLO-bounded",
          "fioretto_ldf": "Fioretto LDF", "hounie_rcl": "Hounie RCL",
          "danits_lp": "Danits LP", "heuristic": "Heuristic"}
COLORS = {"tralo": "#1976D2", "tralo_bounded": "#64B5F6",
          "fioretto_ldf": "#2E7D32", "hounie_rcl": "#C62828",
          "danits_lp": "#9E9E9E", "heuristic": "#616161"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    raw = pd.read_csv(PAPER_DIR.parent / "docs" / "all_cells_raw.csv")
    d = raw[(raw.ds == "dermmnist") & (raw.model == "MobileNetV3")
            & (raw.tight == CELL)]
    flips = d.groupby("method")["flips"].mean()

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(len(ORDER))[::-1]

    for yi, m in zip(y, ORDER):
        f = float(flips[m])
        pct = 100.0 * f / K
        c = COLORS[m]
        ax.barh(yi, pct, height=0.62, color=c, edgecolor="black",
                linewidth=0.7, zorder=3)
        ax.text(pct + 2.5, yi, f"{pct:.0f}%  ({f:.0f} corrections)",
                va="center", ha="left", fontsize=9, fontweight="bold",
                color=c)

    ax.axvline(0, color="black", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in ORDER])
    ax.set_xlabel("Budget overflow before post-hoc "
                  "(over-budget predictions, global + local, as % of budget $K$)")
    ax.set_xlim(0, 100.0 * float(flips.max()) / K * 1.34)
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.grid(axis="x", alpha=0.18, zorder=0)

    # The "after" guarantee, stated once, in the empty upper-right region.
    ax.text(0.985, 0.90,
            "After post-hoc:\nevery method = 0% (feasible).\n"
            "Macro F1 after enforcement\nis tied ($\\approx$0.74).",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#F5F5F5",
                      edgecolor="#BBBBBB"))

    ax.set_title("How far each model's own predictions are from the budget "
                 "(DermMNIST $L30\\_G30$, MEL)\n"
                 "lower = closer to natively feasible; "
                 "bar length = forced post-hoc corrections", fontsize=11)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("overflow %:", {m: round(100 * float(flips[m]) / K, 1) for m in ORDER})
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
