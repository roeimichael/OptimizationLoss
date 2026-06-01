"""Clean headline figure: Macro F1 vs symmetric tightness, one panel per dataset.

Visual choices:
  - TraLO emphasized: thick saturated blue line, large markers, top z-order.
  - Baselines muted: thinner lines, smaller markers, lower z-order, less-saturated colors.
  - TraLO-bounded dropped from the headline (it is a TraLO ablation and lives
    in Table F / component-ablation figure).
  - Single Y-axis per panel (no twin axis). Flips story is a separate sister
    figure (fig_flips_headline.png).
  - All 3 datasets shown — AIDER is in the paper claim set, no exclusion.

Output: paper/figures/fig_headline_f1.png and fig_headline_flips.png
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
ROOT_DIR = PAPER_DIR.parent
FIG_DIR = PAPER_DIR / "figures"
TABLE_A = ROOT_DIR / "docs" / "table_a_summary.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# TraLO bright/bold; baselines desaturated.
STYLE = {
    "tralo":        {"label": "TraLO (ours)", "color": "#1976D2",
                     "lw": 3.2, "ms": 9.5, "marker": "o", "zorder": 10,
                     "alpha": 1.0, "ls": "-"},
    "fioretto_ldf": {"label": "Fioretto LDF", "color": "#6BAE6F",
                     "lw": 1.6, "ms": 5.5, "marker": "D", "zorder": 5,
                     "alpha": 0.85, "ls": "--"},
    "hounie_rcl":   {"label": "Hounie RCL",   "color": "#E07B7B",
                     "lw": 1.6, "ms": 5.5, "marker": "^", "zorder": 5,
                     "alpha": 0.85, "ls": "--"},
    "danits_lp":    {"label": "Danits LP",    "color": "#A88BC0",
                     "lw": 1.4, "ms": 4.5, "marker": "v", "zorder": 3,
                     "alpha": 0.75, "ls": ":"},
    "heuristic":    {"label": "Heuristic",    "color": "#A89386",
                     "lw": 1.4, "ms": 4.5, "marker": "X", "zorder": 3,
                     "alpha": 0.75, "ls": ":"},
}
METHODS = ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic", "tralo"]
# Note: TraLO last in the loop so it draws on top.

TIGHT_LABELS = ["L20", "L30", "L50", "L70", "L80"]
TIGHT_TO_KEY = {t: f"{t}_G{t[1:]}" for t in TIGHT_LABELS}
TIGHT_X = [20, 30, 50, 70, 80]

DATASETS = [
    ("tissuemnist", "TissueMNIST"),
    ("dermmnist",   "DermMNIST"),
    ("aider",       "AIDER"),
]


def _series(df, ds, method, value_col):
    out = []
    for t in TIGHT_LABELS:
        key = TIGHT_TO_KEY[t]
        row = df[(df["ds"] == ds) & (df["tight"] == key) & (df["method"] == method)]
        out.append(float(row[value_col].iloc[0]) if not row.empty else np.nan)
    return out


def _plot(metric_col, ylabel, suptitle, out_name, log_y=False, log_floor=None):
    df = pd.read_csv(TABLE_A)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (ds, ds_label) in zip(axes, DATASETS):
        for method in METHODS:
            ys = _series(df, ds, method, metric_col)
            if log_floor is not None:
                ys = [max(y, log_floor) if not np.isnan(y) else y for y in ys]
            s = STYLE[method]
            ax.plot(
                TIGHT_X, ys,
                label=s["label"] if ds == DATASETS[0][0] else None,
                color=s["color"], linewidth=s["lw"], linestyle=s["ls"],
                marker=s["marker"], markersize=s["ms"], alpha=s["alpha"],
                zorder=s["zorder"],
                markeredgecolor="black" if method == "tralo" else "none",
                markeredgewidth=0.6 if method == "tralo" else 0,
            )
        ax.set_title(ds_label)
        ax.set_xlabel(r"Symmetric tightness  $L = G$  (% of class size)")
        ax.set_xticks(TIGHT_X)
        ax.set_xticklabels([f"L{x}" for x in TIGHT_X])
        ax.grid(alpha=0.18)
        if log_y:
            ax.set_yscale("log")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(suptitle, y=1.005, fontsize=12)
    fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.28)
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    _plot(
        metric_col="F1m_mean",
        ylabel="Macro F1",
        suptitle=r"Macro F1 vs symmetric tightness (4 seeds, MobileNetV3)",
        out_name="fig_headline_f1.png",
    )
    _plot(
        metric_col="Flips_mean",
        ylabel="Post-hoc flips required (log scale)",
        suptitle=r"Post-hoc flips required vs symmetric tightness "
                 r"(lower is better; 4 seeds, MobileNetV3)",
        out_name="fig_headline_flips.png",
        log_y=True,
        log_floor=0.5,
    )


if __name__ == "__main__":
    main()
