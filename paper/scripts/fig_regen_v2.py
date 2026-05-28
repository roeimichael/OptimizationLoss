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


def figure_convergence():
    datasets = ["tissuemnist", "dermmnist"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    skipped: list[str] = []

    # Plot order: baselines first, TraLO last so it draws on top.
    method_draw_order = [m for m in CONVERGENCE_METHODS if m != "tralo"] + \
                        (["tralo"] if "tralo" in CONVERGENCE_METHODS else [])

    for ax, ds in zip(axes, datasets):
        for method in method_draw_order:
            s = HEADLINE_STYLE[method]
            label_used = False
            for seed in SEEDS:
                x, y, p = _load_curve(ds, method, seed)
                if x is None or len(x) == 0:
                    skipped.append(f"{ds}/{method}/seed_{seed} ({p})")
                    continue
                lbl = s["label"] if not label_used else None
                me = max(1, len(x) // 10)
                ax.plot(
                    x, y,
                    color=s["color"], linewidth=s["lw"], alpha=s["alpha"],
                    linestyle=s["ls"], marker=s["marker"], markersize=s["ms"],
                    markevery=me,
                    markeredgecolor="black" if method == "tralo" else "none",
                    markeredgewidth=0.6 if method == "tralo" else 0,
                    zorder=s["zorder"],
                    label=lbl,
                )
                label_used = True
        ax.set_title(f"{ds.replace('mnist','MNIST')} (L30, G30)")
        ax.set_xlabel("Constraint-phase epoch")
        ax.set_ylabel("Total hard-count excess (lower = closer to feasible)")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--",
                   alpha=0.5, zorder=1)
        ax.grid(alpha=0.15, zorder=0)
        ax.legend(loc="upper right", frameon=False)

    fig.suptitle("Convergence of hard-count excess during constraint optimisation "
                 "(4 seeds per method, L30 / G30 tightness)",
                 y=1.02, fontsize=11)
    out = FIG_DIR / "fig_convergence_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    if skipped:
        print("  Skipped curves:")
        for s in skipped:
            print(f"    - {s}")
    return out, skipped


# ---------------------------------------------------------------------------
# Figure 2: F1 macro + Post-hoc flips vs tightness ---------------------------
# ---------------------------------------------------------------------------
TIGHT_LABELS = ["L20", "L30", "L50", "L70", "L80"]
TIGHT_TO_KEY = {t: f"{t}_G{t[1:]}" for t in TIGHT_LABELS}
TIGHT_X = [20, 30, 50, 70, 80]

F1_FIG_METHODS = ["tralo", "tralo_bounded", "fioretto_ldf",
                  "hounie_rcl", "danits_lp", "heuristic"]


def figure_f1_tightness():
    df = pd.read_csv(TABLE_A)
    datasets = ["tissuemnist", "dermmnist"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax_l, ds in zip(axes, datasets):
        ax_r = ax_l.twinx()
        for method in F1_FIG_METHODS:
            f1s, flips = [], []
            for t in TIGHT_LABELS:
                key = TIGHT_TO_KEY[t]
                row = df[(df["ds"] == ds) & (df["tight"] == key) & (df["method"] == method)]
                if row.empty:
                    f1s.append(np.nan); flips.append(np.nan)
                else:
                    f1s.append(float(row["F1m_mean"].iloc[0]))
                    flips.append(float(row["Flips_mean"].iloc[0]))
            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLE[method]
            mk = METHOD_MARKER[method]
            lw = METHOD_LINEWIDTH[method]
            ax_l.plot(
                TIGHT_X, f1s,
                color=color, linestyle=ls, linewidth=lw,
                marker=mk, markersize=7, markeredgecolor="black",
                markeredgewidth=0.5, label=METHOD_LABEL[method],
            )
            # Replace zero flips with a small floor so log scale works.
            flips_log = [max(f, 0.5) for f in flips]
            # Right axis (flips, dotted) uses the same marker but a fixed
            # dotted linestyle to read as "same method, different metric".
            ax_r.plot(
                TIGHT_X, flips_log,
                color=color, linestyle=":", linewidth=lw * 0.7,
                marker=mk, markersize=5.5, alpha=0.75,
                markeredgecolor="black", markeredgewidth=0.4,
            )
        ax_l.set_title(f"{ds.replace('mnist','MNIST')}")
        ax_l.set_xlabel(r"Symmetric tightness $L = G$ (% of class size)")
        ax_l.set_ylabel("Macro F1 (solid, circles)")
        ax_r.set_yscale("log")
        ax_r.set_ylabel("Post-hoc flips required (dotted, triangles; log scale)")
        ax_l.grid(alpha=0.15)
        ax_l.set_xticks(TIGHT_X)
        ax_l.set_xticklabels([f"L{x}" for x in TIGHT_X])
    # Single legend across panels (place under left subplot).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Macro F1 and post-hoc flips vs tightness on TissueMNIST and "
                 "DermMNIST (AIDER excluded for clarity)", y=1.02, fontsize=11)
    out = FIG_DIR / "fig_f1_tightness_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3: in-training satisfaction bar chart -------------------------------
# ---------------------------------------------------------------------------
# Headline satisfaction figure drops tralo_bounded (an ablation; lives in
# the component-ablation figure only). TraLO is rendered last with a darker
# outline so the bar pops.
SAT_METHODS = ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic", "tralo"]


def figure_satisfaction():
    df = pd.read_csv(TABLE_A)
    datasets = ["tissuemnist", "dermmnist", "aider"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    n_methods = len(SAT_METHODS)
    width = 0.15
    x = np.arange(len(TIGHT_LABELS))
    # Offset bars symmetrically around each tightness tick.
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * width

    # Pre-compute TraLO sat values per dataset so we can decide on a callout.
    tralo_all_100 = True

    for ax, ds in zip(axes, datasets):
        for i, method in enumerate(SAT_METHODS):
            sat = []
            for t in TIGHT_LABELS:
                key = TIGHT_TO_KEY[t]
                row = df[(df["ds"] == ds) & (df["tight"] == key)
                         & (df["method"] == method)]
                if row.empty:
                    sat.append(np.nan)
                else:
                    sat.append(float(row["Sat%_mean"].iloc[0]))

            s = HEADLINE_STYLE[method]
            is_tralo = method == "tralo"
            if is_tralo and np.any(np.array([v for v in sat if not np.isnan(v)]) < 1.0):
                tralo_all_100 = False
            ax.bar(
                x + offsets[i], sat, width=width,
                color=s["color"],
                edgecolor="black",
                linewidth=1.4 if is_tralo else 0.5,
                alpha=1.0 if is_tralo else 0.78,
                hatch=METHOD_HATCH.get(method, ""),
                label=s["label"],
                zorder=5 if is_tralo else 3,
            )
        ax.set_title(ds.replace("mnist", "MNIST").upper() if ds == "aider"
                     else ds.replace("mnist", "MNIST"))
        ax.set_xticks(x); ax.set_xticklabels(TIGHT_LABELS)
        ax.set_xlabel("Symmetric tightness")
        ax.set_ylim(0, 1.12)
        ax.axhline(1.0, color="#1976D2", linewidth=0.8, linestyle=":",
                   alpha=0.55, zorder=1)
        ax.grid(axis="y", alpha=0.15, zorder=0)
    axes[0].set_ylabel("In-training constraint satisfaction rate\n"
                       r"(fraction of 4 seeds)")

    # Headline callout: TraLO reaches 100% at every tightness on all datasets.
    if tralo_all_100:
        for ax in axes:
            ax.text(
                0.02, 0.06,
                "TraLO: 100% satisfaction\nat every tightness",
                transform=ax.transAxes,
                fontsize=8.5, color="#1976D2", fontweight="bold",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#1976D2",
                          linewidth=0.9, alpha=0.9),
                zorder=12,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_methods,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("In-training constraint satisfaction by method, tightness, "
                 "and dataset (before any post-hoc reallocation)",
                 y=1.02, fontsize=11)
    out = FIG_DIR / "fig_satisfaction_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
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
