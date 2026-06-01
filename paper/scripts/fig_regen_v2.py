"""Regenerate the three v2 results figures for the paper from current
Blackwell (paperv2_phase1 / paper400_tralofix / paper400_baselines) data.

Outputs (300 dpi, Agg backend):
  paper/figures/fig_convergence_v2.png
  paper/figures/fig_f1_tightness_v2.png
  paper/figures/fig_satisfaction_v2.png

Data sources:
  - Per-epoch training logs cached under paper/data_cache/training_logs/
    (24 logs = 2 datasets x 3 methods x 4 seeds at L30_G30, fetched from
    dsisco02:/home/dsi/michaer8/OptimizationLoss/results/pending_runs/).
    Root assignments per cell (chosen to match the 360-cell Table A sweep):
      tissuemnist / tralo        -> paper400_tralofix/tissuemnist/L30_G30/seed_{N}
      tissuemnist / fioretto_ldf -> paper400_baselines/tissuemnist/L30_G30/fioretto_ldf/seed_{N}
      tissuemnist / hounie_rcl   -> paper400_baselines/tissuemnist/L30_G30/hounie_rcl/seed_{N}
      dermmnist   / *            -> paperv2_phase1/dermmnist/L30_G30/{method}/seed_{N}
  - Aggregate Table A summary: docs/table_a_summary.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
ROOT_DIR = PAPER_DIR.parent
FIG_DIR = PAPER_DIR / "figures"
LOG_CACHE = PAPER_DIR / "data_cache" / "training_logs"
TABLE_A = ROOT_DIR / "docs" / "table_a_summary.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METHOD_COLORS = {
    "tralo":          "#1976D2",
    "tralo_bounded":  "#0D47A1",
    "fioretto_ldf":   "#2E7D32",
    "hounie_rcl":     "#E53935",
    "danits_lp":      "#7B1FA2",
    "heuristic":      "#5D4037",
}
METHOD_LABEL = {
    "tralo":         "TraLO (ours)",
    "tralo_bounded": "TraLO-bounded",
    "fioretto_ldf":  "Fioretto LDF",
    "hounie_rcl":    "Hounie RCL",
    "danits_lp":     "Danits LP",
    "heuristic":     "Heuristic",
}
# B&W-friendly style differentiation. Each method gets a unique
# (linestyle, marker, hatch) tuple so the figures are readable when
# printed in grayscale.
METHOD_LINESTYLE = {
    "tralo":          "-",
    "tralo_bounded":  "--",
    "fioretto_ldf":   "-.",
    "hounie_rcl":     (0, (3, 1, 1, 1)),  # dash-dot-dot
    "danits_lp":      ":",
    "heuristic":      (0, (5, 2, 1, 2)),  # long dash, dot
}
METHOD_MARKER = {
    "tralo":          "o",
    "tralo_bounded":  "s",
    "fioretto_ldf":   "D",
    "hounie_rcl":     "^",
    "danits_lp":      "v",
    "heuristic":      "X",
}
METHOD_HATCH = {
    "tralo":          "",       # solid fill, ours, highest visual weight
    "tralo_bounded":  "///",
    "fioretto_ldf":   "\\\\\\",
    "hounie_rcl":     "xxx",
    "danits_lp":      "...",
    "heuristic":      "++",
}
METHOD_LINEWIDTH = {
    "tralo":          2.2,   # ours: thicker
    "tralo_bounded":  1.5,
    "fioretto_ldf":   1.5,
    "hounie_rcl":     1.5,
    "danits_lp":      1.5,
    "heuristic":      1.5,
}

# --- Headline-style overrides (used by polished figures only) --------------
# Mirrors fig_headline.py: TraLO saturated bright blue, thick line, big
# markers, top z-order, black marker edge; baselines muted/desaturated.
HEADLINE_STYLE = {
    "tralo":        {"label": "TraLO (ours)", "color": "#1976D2",
                     "lw": 3.2, "ms": 9.5, "marker": "o", "zorder": 10,
                     "alpha": 1.0, "ls": "-"},
    "tralo_bounded": {"label": "TraLO-bounded", "color": "#0D47A1",
                     "lw": 1.6, "ms": 5.5, "marker": "s", "zorder": 4,
                     "alpha": 0.85, "ls": "--"},
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


# ---------------------------------------------------------------------------
# Figure 1: convergence (excess vs epoch) ------------------------------------
# ---------------------------------------------------------------------------
# Path map: (dataset, method, seed) -> relative path under LOG_CACHE
CONVERGENCE_METHODS = ["tralo", "fioretto_ldf", "hounie_rcl"]
SEEDS = [1, 2, 3, 4]

# Tissuemnist constrained class index (GE = 4); dermmnist (MEL = 4). Both 4.
CONSTRAINED_CLASS = 4


def _log_path(ds: str, method: str, seed: int) -> Path:
    """Map (ds, method, seed) -> training_log.csv path inside LOG_CACHE."""
    if ds == "tissuemnist":
        if method == "tralo":
            return (LOG_CACHE / "paper400_tralofix" / "tissuemnist"
                    / "L30_G30" / f"seed_{seed}" / "training_log.csv")
        return (LOG_CACHE / "paper400_baselines" / "tissuemnist"
                / "L30_G30" / method / f"seed_{seed}" / "training_log.csv")
    if ds == "dermmnist":
        return (LOG_CACHE / "paperv2_phase1" / "dermmnist"
                / "L30_G30" / method / f"seed_{seed}" / "training_log.csv")
    raise ValueError(f"Unknown dataset {ds!r}")


def _compute_excess_tralo(df: pd.DataFrame, cls: int = CONSTRAINED_CLASS) -> tuple[np.ndarray, np.ndarray]:
    """For TraLO/tralo_bounded logs (wide schema with Hard_Class*/Limit_Class*/Group*).

    Returns (epochs, total_excess) where total_excess sums global excess
    (max(0, Hard_Class{cls} - Limit_Class{cls})) and local excess summed
    across groups (max(0, Group{g}_Hard_Class{cls} - Group{g}_Limit_Class{cls})).
    Per-row, inf limits become 0 excess (cap inactive).
    """
    epochs = df["Epoch"].to_numpy(dtype=float)

    g_hard = df[f"Hard_Class{cls}"].to_numpy(dtype=float)
    g_lim = df[f"Limit_Class{cls}"].to_numpy(dtype=float)
    # inf limit -> no excess
    g_excess = np.where(np.isfinite(g_lim), np.maximum(0.0, g_hard - g_lim), 0.0)

    local_excess = np.zeros_like(g_excess)
    # Find all GroupN_Hard_Class{cls} columns
    group_idx = 0
    while True:
        h_col = f"Group{group_idx}_Hard_Class{cls}"
        l_col = f"Group{group_idx}_Limit_Class{cls}"
        if h_col not in df.columns or l_col not in df.columns:
            break
        h = df[h_col].to_numpy(dtype=float)
        l = df[l_col].to_numpy(dtype=float)
        local_excess = local_excess + np.where(
            np.isfinite(l), np.maximum(0.0, h - l), 0.0
        )
        group_idx += 1

    total = g_excess + local_excess
    return epochs, total


def _compute_excess_baseline(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For fioretto_ldf / hounie_rcl logs (long schema with `total_excess`)."""
    epochs = df["epoch"].to_numpy(dtype=float)
    excess = df["total_excess"].to_numpy(dtype=float)
    return epochs, excess


def _load_curve(ds: str, method: str, seed: int):
    p = _log_path(ds, method, seed)
    if not p.exists():
        return None, None, p
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"  ! failed to read {p}: {e}")
        return None, None, p
    if method == "tralo" or method == "tralo_bounded":
        x, y = _compute_excess_tralo(df)
    else:
        x, y = _compute_excess_baseline(df)
    # Normalize x to start at 0 (TraLO logs start at warmup_epochs).
    if len(x) > 0:
        x = x - x[0]
    return x, y, p


# Distinct line styles so the three methods read apart in B/W too.
# Fioretto sits ABOVE TraLO (higher z) with a bold long-dash so that where the
# two nearly coincide (e.g. TissueMNIST) the green dashes ride visibly on top of
# the solid blue line instead of being hidden under it.
CONV_STYLE = {
    "tralo":        {"color": "#1976D2", "ls": "-",          "lw": 2.8, "z": 10, "ms": 8},
    "fioretto_ldf": {"color": "#2E7D32", "ls": (0, (7, 3)),  "lw": 2.2, "z": 12, "ms": 5},
    "hounie_rcl":   {"color": "#C62828", "ls": (0, (1, 1.4)), "lw": 1.8, "z": 6, "ms": 8},
}


def _seed_band(ds, method):
    """Best-so-far excess for every seed, aligned on a common integer epoch
    grid (forward-filled past each seed's last epoch). Returns grid, median,
    lo, hi across seeds, or None if no seed loaded."""
    curves, xmax = [], 0
    for seed in SEEDS:
        x, y, _ = _load_curve(ds, method, seed)
        if x is None or len(x) == 0:
            continue
        y = np.minimum.accumulate(np.asarray(y, dtype=float))
        curves.append((np.asarray(x, dtype=float), y))
        xmax = max(xmax, x[-1])
    if not curves:
        return None
    grid = np.arange(0, int(xmax) + 1)
    stack = np.vstack([np.interp(grid, x, y, left=y[0], right=y[-1])
                       for x, y in curves])
    return grid, np.median(stack, 0), stack.min(0), stack.max(0)


def figure_convergence():
    datasets = ["tissuemnist", "dermmnist"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    skipped: list[str] = []

    for ax, ds in zip(axes, datasets):
        conv_epochs = []
        # Baselines first, TraLO last so its line sits on top.
        for method in ["fioretto_ldf", "hounie_rcl", "tralo"]:
            # Fioretto on Derm is bimodal (2/4 seeds satisfy, 2/4 stay stuck):
            # a median+band hides that, so draw the individual seeds instead.
            # Two curves dive to 0 (dots); two run flat at excess 2-3 (no dot).
            if ds == "dermmnist" and method == "fioretto_ldf":
                st = CONV_STYLE[method]
                labeled = False
                for seed in SEEDS:
                    x, y, _ = _load_curve(ds, method, seed)
                    if x is None or len(x) == 0:
                        continue
                    x = np.asarray(x, dtype=float)
                    y = np.minimum.accumulate(np.asarray(y, dtype=float))
                    if np.any(y <= 0.0):
                        k = int(np.argmax(y <= 0.0))
                        xs, ys, dot = x[:k + 1], y[:k + 1], x[k]
                        conv_epochs.append(x[k])
                    else:
                        xs, ys, dot = x, y, None
                    ax.plot(xs, ys, color=st["color"], linestyle=st["ls"],
                            linewidth=1.1, alpha=0.75, zorder=st["z"],
                            label=(HEADLINE_STYLE[method]["label"]
                                   if not labeled else None))
                    labeled = True
                    if dot is not None:
                        ax.plot(dot, 0, marker="o", color=st["color"],
                                markersize=6, markeredgecolor="black",
                                markeredgewidth=0.7, zorder=st["z"] + 1)
                continue
            band = _seed_band(ds, method)
            if band is None:
                skipped.append(f"{ds}/{method}")
                continue
            grid, med, lo, hi = band
            st = CONV_STYLE[method]
            # A method "converges" when its median best-so-far reaches 0.
            # Truncate the curve there (with a dot) so we don't drag a flat
            # zero tail; a curve with no dot never reaches feasibility.
            if np.any(med <= 0.0):
                k = int(np.argmax(med <= 0.0))   # first feasible epoch
                end = k + 1
                conv_epochs.append(grid[k])
            else:
                k = None
                end = len(grid)
            g, m, l, h = grid[:end], med[:end], lo[:end], hi[:end]
            ax.fill_between(g, l, h, color=st["color"], alpha=0.12,
                            linewidth=0, zorder=st["z"] - 5)
            ax.plot(g, m, color=st["color"], linestyle=st["ls"],
                    linewidth=st["lw"], zorder=st["z"],
                    label=HEADLINE_STYLE[method]["label"])
            if k is not None:
                ax.plot(grid[k], 0, marker="o", color=st["color"],
                        markersize=st["ms"], markeredgecolor="black",
                        markeredgewidth=0.8, zorder=st["z"] + 1)
        ax.set_title(f"{ds.replace('mnist','MNIST')} (L30, G30)")
        ax.set_xlabel("Constraint-phase epoch")
        ax.set_ylabel("Hard-count excess")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(bottom=-0.3)
        # Show a little past the slowest method that DOES converge.
        xcap = (max(conv_epochs) * 1.18) if conv_epochs else None
        ax.set_xlim(0, xcap)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--",
                   alpha=0.5, zorder=1)
        ax.grid(alpha=0.15, zorder=0)
        # Explain the convergence dot in the legend rather than the title.
        dot_proxy = Line2D([0], [0], marker="o", color="0.35", linestyle="None",
                           markersize=7, markeredgecolor="black",
                           markeredgewidth=0.7, label="reaches feasibility")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [dot_proxy], labels + ["reaches feasibility"],
                  loc="upper right", frameon=False, fontsize=8)

    fig.suptitle("Best-so-far constraint excess during optimization "
                 "(median over 4 seeds)", y=1.0, fontsize=12)
    out = FIG_DIR / "fig_convergence_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    if skipped:
        print("  Skipped curves:", skipped)
    return out, skipped


# ---------------------------------------------------------------------------
# Figure 2: F1 macro + Post-hoc flips vs tightness ---------------------------
# ---------------------------------------------------------------------------
TIGHT_LABELS = ["L20", "L30", "L50", "L70", "L80"]
TIGHT_TO_KEY = {t: f"{t}_G{t[1:]}" for t in TIGHT_LABELS}
TIGHT_X = [20, 30, 50, 70, 80]

F1_FIG_METHODS = ["tralo", "tralo_bounded", "fioretto_ldf",
                  "hounie_rcl", "danits_lp", "heuristic"]


def _metric_vs_tightness(metric_col, ylabel, title, outname, symlog=False):
    """One clean single-axis line plot of `metric_col` vs tightness, two
    panels (TissueMNIST, DermMNIST), one line per method. No dual axis."""
    df = pd.read_csv(TABLE_A)
    datasets = ["tissuemnist", "dermmnist", "aider"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), sharex=True)
    for ax, ds in zip(axes, datasets):
        for method in F1_FIG_METHODS:
            ys = []
            for t in TIGHT_LABELS:
                key = TIGHT_TO_KEY[t]
                row = df[(df["ds"] == ds) & (df["tight"] == key) & (df["method"] == method)]
                ys.append(float(row[metric_col].iloc[0]) if not row.empty else np.nan)
            s = HEADLINE_STYLE[method]
            ax.plot(
                TIGHT_X, ys,
                color=s["color"], linestyle=s["ls"], linewidth=s["lw"],
                marker=s["marker"], markersize=s["ms"], alpha=s["alpha"],
                markeredgecolor="black", markeredgewidth=0.5,
                label=s["label"], zorder=s["zorder"],
            )
        if symlog:
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(bottom=-0.3)
        ax.set_title("AIDER" if ds == "aider" else f"{ds.replace('mnist','MNIST')}")
        ax.set_xlabel("Symmetric tightness $L = G$ (% of class size)")
        ax.grid(alpha=0.15)
        ax.set_xticks(TIGHT_X)
        ax.set_xticklabels([str(x) for x in TIGHT_X])
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, y=0.99, fontsize=11)
    # Reserve the bottom band for the legend so it never overlaps the xlabels.
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = FIG_DIR / outname
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def figure_f1_tightness():
    _metric_vs_tightness(
        "F1m_mean", "Macro F1",
        "Macro F1 vs tightness, by dataset and method",
        "fig_f1_tightness_v2.png")
    _metric_vs_tightness(
        "Flips_mean", "Post-hoc flips required (symmetric-log)",
        "Post-hoc flips vs tightness, by dataset and method",
        "fig_flips_tightness_v2.png", symlog=True)


# ---------------------------------------------------------------------------
# Figure 3: in-training satisfaction --------------------------------------
# ---------------------------------------------------------------------------
# Only the four methods that ATTEMPT in-training feasibility are drawn. The two
# post-hoc allocators (Danits LP, Heuristic) sit at 0% on every cell by
# construction -- as empty bars they read as missing data, so they are stated
# in a note instead. The point of the figure is reliability: TraLO and Hounie
# hold 100% everywhere; Fioretto LDF and TraLO-bounded drop out on some cells.
SAT_METHODS = ["tralo", "hounie_rcl", "fioretto_ldf", "tralo_bounded"]
SAT_COLORS = {"tralo": "#1976D2", "hounie_rcl": "#C62828",
              "fioretto_ldf": "#2E7D32", "tralo_bounded": "#FB8C00"}
SAT_LABELS = {"tralo": "TraLO (ours)", "hounie_rcl": "Hounie RCL",
              "fioretto_ldf": "Fioretto LDF", "tralo_bounded": "TraLO-bounded"}


def figure_satisfaction():
    df = pd.read_csv(TABLE_A)
    datasets = ["tissuemnist", "dermmnist", "aider"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    n_methods = len(SAT_METHODS)
    width = 0.19
    x = np.arange(len(TIGHT_LABELS))
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    for ax, ds in zip(axes, datasets):
        for i, method in enumerate(SAT_METHODS):
            sat = []
            for t in TIGHT_LABELS:
                key = TIGHT_TO_KEY[t]
                row = df[(df["ds"] == ds) & (df["tight"] == key)
                         & (df["method"] == method)]
                sat.append(float(row["Sat%_mean"].iloc[0]) if not row.empty
                           else np.nan)
            is_tralo = method == "tralo"
            ax.bar(
                x + offsets[i], sat, width=width,
                color=SAT_COLORS[method], edgecolor="black",
                linewidth=1.4 if is_tralo else 0.5,
                alpha=1.0 if is_tralo else 0.92,
                label=SAT_LABELS[method], zorder=5 if is_tralo else 3,
            )
        ax.set_title("AIDER" if ds == "aider"
                     else ds.replace("mnist", "MNIST"))
        ax.set_xticks(x); ax.set_xticklabels(TIGHT_LABELS)
        ax.set_xlabel("Symmetric tightness")
        ax.set_ylim(0, 1.08)
        ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle=":",
                   alpha=0.6, zorder=1)
        ax.grid(axis="y", alpha=0.15, zorder=0)
    axes[0].set_ylabel("In-training satisfaction rate\n"
                       r"(fraction of 4 seeds, before post-hoc)")

    fig.suptitle("In-training constraint satisfaction: TraLO and Hounie hold "
                 "100% everywhere; Fioretto and TraLO-bounded drop out on "
                 "some cells", y=1.0, fontsize=11)
    # The post-hoc cluster, stated rather than drawn as empty bars.
    fig.text(0.5, 0.085,
             "Danits LP and Heuristic are omitted: both satisfy 0% of cells "
             "before post-hoc, by construction (they allocate only afterward).",
             ha="center", va="center", fontsize=9, fontstyle="italic",
             color="0.35")
    fig.tight_layout(rect=[0, 0.13, 1, 0.94])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_methods,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    out = FIG_DIR / "fig_satisfaction_v2.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating fig_convergence_v2 ...")
    conv_out, conv_skipped = figure_convergence()
    print("Generating fig_f1_tightness_v2 ...")
    figure_f1_tightness()
    print("Generating fig_satisfaction_v2 ...")
    figure_satisfaction()
    print("\nDone.")
    if conv_skipped:
        print(f"  ({len(conv_skipped)} convergence curves were skipped due to missing logs.)")
