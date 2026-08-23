"""Asymmetric tightness, full four-metric summary (replaces the flips-only bar).

Showing only post-hoc flips read as cherry-picking the one metric TraLO wins.
The honest story is a no-free-lunch one: across the 20 asymmetric (L != G)
configs on DermMNIST, TraLO ties every method on accuracy/Macro F1 and wins
decisively on the two deployability metrics (post-hoc flips and in-training
satisfaction). This figure shows all four side by side: quality metrics on the
top row (tied), deployability metrics on the bottom row (won), so the
"flip reduction at no accuracy cost" claim is visible, not asserted.

Output:  paper/figures/fig_asymmetric_summary_v2.png
Source:  paper/tables/B_asymmetric_tightness/table_B_phase2_asymmetric_derm.csv
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
CSV_PATH = PAPER_DIR / "tables" / "B_asymmetric_tightness" / "table_B_phase2_asymmetric_derm.csv"
OUT_PATH = PAPER_DIR / "figures" / "fig_asymmetric_summary_v2.png"

# Top to bottom; TraLO family in blues, iterative baselines green/red,
# post-hoc allocators in greys.
ORDER = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
         "danits_lp", "heuristic"]
LABELS = {"tralo": "TraLO (ours)", "tralo_bounded": "TraLO-bounded",
          "fioretto_ldf": "Fioretto LDF", "hounie_rcl": "Hounie RCL",
          "danits_lp": "Danits LP", "heuristic": "Heuristic"}
COLORS = {"tralo": "#1976D2", "tralo_bounded": "#64B5F6",
          "fioretto_ldf": "#2E7D32", "hounie_rcl": "#C62828",
          "danits_lp": "#9E9E9E", "heuristic": "#616161"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    g = df.groupby("method")
    agg = pd.DataFrame({
        "F1": g["macro_f1_mean"].mean(), "F1_e": g["macro_f1_std"].mean(),
        "Acc": g["accuracy_mean"].mean(), "Acc_e": g["accuracy_std"].mean(),
        "Flips": g["flips_mean"].mean(), "Flips_e": g["flips_std"].mean(),
        "Sat": g["satisfied_pct"].mean(),
    }).reindex(ORDER)

    # y positions, TraLO at the top.
    y = np.arange(len(ORDER))[::-1]
    cols = [COLORS[m] for m in ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), sharey=True)
    (axF1, axAcc), (axFlip, axSat) = axes

    def barh(ax, vals, errs, fmt, xlabel, title, log=False, xlim=None,
             pad=1.0):
        ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.6,
                height=0.66, zorder=3,
                xerr=errs, error_kw=dict(ecolor="0.3", elinewidth=0.9,
                                         capsize=2.5))
        if log:
            ax.set_xscale("log")
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_title(title, fontsize=10.5, pad=6)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.grid(axis="x", alpha=0.18, zorder=0)
        for yi, v in zip(y, vals):
            if log:
                xtext = v * pad
            else:
                xtext = v + pad
            ax.text(xtext, yi, fmt.format(v), va="center", ha="left",
                    fontsize=8.7, fontweight="bold")

    axF1.set_yticks(y)
    axF1.set_yticklabels([LABELS[m] for m in ORDER], fontsize=9.5)

    barh(axF1, agg["F1"].to_numpy(), agg["F1_e"].to_numpy(), "{:.3f}",
         "Macro F1", "Macro F1  (higher = better)", xlim=(0, 0.92), pad=0.012)
    barh(axAcc, agg["Acc"].to_numpy(), agg["Acc_e"].to_numpy(), "{:.3f}",
         "Accuracy", "Accuracy  (higher = better)", xlim=(0, 1.02), pad=0.012)
    barh(axFlip, agg["Flips"].to_numpy(), agg["Flips_e"].to_numpy(), "{:.1f}",
         "Post-hoc flips (log scale)", "Post-hoc flips  (lower = better)",
         log=True, xlim=(1, 320), pad=1.18)
    barh(axSat, agg["Sat"].to_numpy(), np.zeros(len(ORDER)), "{:.0f}%",
         "In-training satisfaction (%)", "In-training Sat%  (higher = better)",
         xlim=(0, 118), pad=2.0)

    # Row framing so the contrast is unmissable.
    axF1.annotate("quality: tied", xy=(0, 1.18), xycoords="axes fraction",
                  fontsize=10, fontweight="bold", color="#555555")
    axFlip.annotate("deployability: TraLO wins", xy=(0, 1.18),
                    xycoords="axes fraction", fontsize=10, fontweight="bold",
                    color="#1976D2")

    fig.suptitle("Asymmetric tightness ($L\\neq G$) on DermMNIST, mean over "
                 "20 off-diagonal configs:\nTraLO ties on accuracy/F1 and wins "
                 "on flips and satisfaction", fontsize=11.5, y=1.06)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("F1 tie:", agg["F1"].round(3).to_dict())
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
