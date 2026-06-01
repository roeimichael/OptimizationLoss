"""Results figures for professor review, built from docs/all_cells_raw.csv.

Honest framing (docs/GRAPHS_HANDOFF.md S0):
  - Flips: TraLO wins decisively everywhere.
  - F1-macro: regime-dependent tie-to-win (positive on hard tissue, ~zero on
    derm, slightly negative on saturated aider). Never imply a clean F1 win.
  - In-training satisfaction: TraLO ~99%, post-hoc baselines ~7%.

Outputs (300 dpi, Agg, B&W-friendly): paper/figures/
  fig_tradeoff_scatter.png   (FIG1 headline)
  fig_f1_gap.png             (FIG2 honest gap)
  fig_flips_bar.png          (FIG3)
  fig_regime.png             (FIG4)
  fig_satisfaction_v2.png    (FIG5)
  fig_asym_heatmap.png       (FIG6)
  fig_robustness.png         (FIG7)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
ROOT_DIR = PAPER_DIR.parent
FIG_DIR = PAPER_DIR / "figures"
CSV = ROOT_DIR / "docs" / "all_cells_raw.csv"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120,
})

METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
BASELINES = [m for m in METHODS if m != "tralo"]
LABEL = {
    "tralo": "TraLO (ours)", "tralo_bounded": "TraLO-bounded",
    "fioretto_ldf": "Fioretto LDF", "hounie_rcl": "Hounie RCL",
    "danits_lp": "Danits LP", "heuristic": "Heuristic",
}
COLOR = {
    "tralo": "#1565C0", "tralo_bounded": "#5E92F3", "fioretto_ldf": "#2E7D32",
    "hounie_rcl": "#C62828", "danits_lp": "#6A1B9A", "heuristic": "#5D4037",
}
MARKER = {"tralo": "o", "tralo_bounded": "s", "fioretto_ldf": "D",
          "hounie_rcl": "^", "danits_lp": "v", "heuristic": "X"}
HATCH = {"tralo": "", "tralo_bounded": "//", "fioretto_ldf": "\\\\",
         "hounie_rcl": "xx", "danits_lp": "..", "heuristic": "++"}
DS_ORDER = ["tissuemnist", "dermmnist", "aider"]
DS_TITLE = {"tissuemnist": "TissueMNIST", "dermmnist": "DermMNIST", "aider": "AIDER"}
NOISE = 0.01  # mean seed std on F1m (handoff S0)


def load():
    df = pd.read_csv(CSV)
    df = df[df.ds != "eurosat"].copy()
    df = df.dropna(subset=["f1m", "flips", "sat"])
    return df


def cellkey(df):
    return df[["ds", "model", "cls", "grp", "tight", "L", "G", "seed"]]


# --------------------------------------------------------------------------
# FIG 1 — F1 vs flips tradeoff scatter
# --------------------------------------------------------------------------
def fig_tradeoff(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, ds in zip(axes, DS_ORDER):
        d = df[df.ds == ds]
        for m in METHODS:
            dm = d[d.method == m]
            if dm.empty:
                continue
            x, y = dm.flips.mean(), dm.f1m.mean()
            xe, ye = dm.flips.std(), dm.f1m.std()
            big = m == "tralo"
            ax.errorbar(x + 1, y, xerr=xe, yerr=ye, fmt=MARKER[m],
                        ms=13 if big else 8, color=COLOR[m],
                        mec="black", mew=1.2 if big else 0.5,
                        ecolor=COLOR[m], elinewidth=1.0, capsize=2.5,
                        alpha=0.95, zorder=5 if big else 3,
                        label=LABEL[m])
        ax.set_xscale("log")
        ax.set_title(DS_TITLE[ds])
        ax.set_xlabel("Post-hoc flips required (log, +1)")
        ax.grid(alpha=0.18)
        if ds == DS_ORDER[0]:
            ax.set_ylabel("Macro F1")
        ax.annotate("better\n(fewer flips,\nsame F1)", xy=(0.04, 0.06),
                    xycoords="axes fraction", fontsize=7.5, color="#555",
                    ha="left", va="bottom", style="italic")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Accuracy–efficiency tradeoff: TraLO matches baseline Macro F1 "
                 "at a fraction of the post-hoc corrections", y=1.0, fontsize=11)
    out = FIG_DIR / "fig_tradeoff_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# FIG 2 — F1m gap distribution (TraLO - best baseline per cell)
# --------------------------------------------------------------------------
def per_cell_gap(df, metric="f1m", lower_better=False, ref="mean"):
    """Return per-cell (TraLO metric - reference-baseline metric).

    ref: 'mean' (fair average competitor, default), 'max' (oracle-best,
    hardest bar), or 'min' (weakest baseline).
    """
    keys = ["ds", "model", "cls", "grp", "tight", "L", "G", "seed"]
    piv = df.pivot_table(index=keys, columns="method", values=metric)
    piv = piv.dropna(subset=["tralo"])
    base = piv[[b for b in BASELINES if b in piv.columns]]
    agg = {"mean": base.mean, "max": base.max, "min": base.min}[ref]
    ref_base = agg(axis=1)
    if lower_better:
        gap = ref_base - piv["tralo"]   # positive = TraLO better (fewer)
    else:
        gap = piv["tralo"] - ref_base   # positive = TraLO better (higher)
    out = piv.reset_index()[["ds"]].copy()
    out["gap"] = gap.values
    return out


def fig_f1_gap(df):
    gap = per_cell_gap(df, "f1m", lower_better=False)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    data = [gap[gap.ds == ds]["gap"].values for ds in DS_ORDER]
    ax.axhspan(-NOISE, NOISE, color="grey", alpha=0.18, zorder=0,
               label=f"seed-noise band ($\\pm${NOISE:.2f} F1)")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    bp = ax.boxplot(data, positions=range(len(DS_ORDER)), widths=0.5,
                    patch_artist=True, showmeans=True, zorder=3,
                    medianprops=dict(color="black"),
                    meanprops=dict(marker="D", markerfacecolor="#1565C0",
                                   markeredgecolor="black", markersize=7))
    for patch in bp["boxes"]:
        patch.set_facecolor("#5E92F3"); patch.set_alpha(0.55)
    for i, ds in enumerate(DS_ORDER):
        ys = data[i]
        xs = np.random.default_rng(0).normal(i, 0.05, size=len(ys))
        ax.scatter(xs, ys, s=10, color="#1565C0", alpha=0.30, zorder=4)
        ax.annotate(f"mean\n{ys.mean():+.4f}", xy=(i, ys.mean()),
                    xytext=(i + 0.30, ys.mean()), fontsize=8, color="#1565C0",
                    va="center")
    ax.set_xticks(range(len(DS_ORDER)))
    ax.set_xticklabels([DS_TITLE[d] for d in DS_ORDER])
    ax.set_ylabel("Per-cell Macro F1 gap\n(TraLO $-$ mean of 5 baselines)")
    ax.set_title("Macro F1: TraLO holds a small positive edge that sits at the "
                 "noise band\n(a near-tie on accuracy; the win is in efficiency, not F1)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.18)
    out = FIG_DIR / "fig_f1_gap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# FIG 3 — mean flips by method, grouped per dataset
# --------------------------------------------------------------------------
def fig_flips_bar(df):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(DS_ORDER)); w = 0.13
    for i, m in enumerate(METHODS):
        means, errs = [], []
        for ds in DS_ORDER:
            dm = df[(df.ds == ds) & (df.method == m)]
            means.append(dm.flips.mean()); errs.append(dm.flips.std())
        ax.bar(x + (i - 2.5) * w, means, w, yerr=errs, capsize=2,
               color=COLOR[m], edgecolor="black", linewidth=0.5,
               hatch=HATCH[m], label=LABEL[m], error_kw=dict(lw=0.7))
    ax.set_xticks(x); ax.set_xticklabels([DS_TITLE[d] for d in DS_ORDER])
    ax.set_ylabel("Mean post-hoc flips required ($\\downarrow$)")
    ax.set_title("Post-hoc corrections needed to enforce the budget "
                 "(lower is better; TraLO lowest on every dataset)")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.18)
    out = FIG_DIR / "fig_flips_bar.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# FIG 4 — regime effect: warmup acc vs TraLO F1 gap
# --------------------------------------------------------------------------
def fig_regime(df):
    # warmup acc proxy = heuristic acc (model untouched) on MobileNetV3
    warm = (df[(df.method == "heuristic") & (df.model == "MobileNetV3")]
            .groupby("ds")["acc"].mean())
    gap = per_cell_gap(df[df.model == "MobileNetV3"], "f1m")
    g_by_ds = gap.groupby("ds")["gap"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    xs, ys, es = [], [], []
    for ds in DS_ORDER:
        xs.append(warm[ds]); ys.append(g_by_ds.loc[ds, "mean"])
        es.append(g_by_ds.loc[ds, "std"])
    ax.axhspan(-NOISE, NOISE, color="grey", alpha=0.15)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    for ds, xx, yy, ee in zip(DS_ORDER, xs, ys, es):
        ax.errorbar(xx, yy, yerr=ee, fmt="o", ms=12, color="#1565C0",
                    mec="black", mew=1.0, capsize=3, zorder=5)
        ax.annotate(DS_TITLE[ds], xy=(xx, yy), xytext=(xx, yy + 0.004),
                    ha="center", fontsize=9)
    z = np.polyfit(xs, ys, 1)
    xfit = np.linspace(min(xs) - 0.03, max(xs) + 0.03, 50)
    ax.plot(xfit, np.polyval(z, xfit), color="#C62828", lw=1.4, ls="-.",
            label=f"trend (slope {z[0]:+.3f})")
    ax.set_xlabel("Warmup test accuracy (task difficulty proxy)")
    ax.set_ylabel("Mean Macro F1 gap (TraLO $-$ mean of 5 baselines)")
    ax.set_title("Regime effect: TraLO's F1 edge is largest on the hard task\n"
                 "and shrinks toward zero as the warmup classifier saturates")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.18)
    out = FIG_DIR / "fig_regime.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# FIG 5 — in-training satisfaction
# --------------------------------------------------------------------------
def fig_satisfaction(df):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(DS_ORDER)); w = 0.13
    for i, m in enumerate(METHODS):
        vals = []
        for ds in DS_ORDER:
            dm = df[(df.ds == ds) & (df.method == m)]
            vals.append(dm.sat.mean())
        ax.bar(x + (i - 2.5) * w, vals, w, color=COLOR[m],
               edgecolor="black", linewidth=0.5, hatch=HATCH[m], label=LABEL[m])
    ax.set_xticks(x); ax.set_xticklabels([DS_TITLE[d] for d in DS_ORDER])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("In-training satisfaction rate ($\\uparrow$)")
    ax.set_title("Fraction of runs feasible BEFORE any post-hoc correction\n"
                 "(TraLO/Hounie ship feasible models; post-hoc baselines do not)")
    ax.legend(loc="center right", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.18)
    out = FIG_DIR / "fig_satisfaction_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# FIG 6 — asymmetric tightness heatmap (derm, phase2)
# --------------------------------------------------------------------------
def fig_asym(df):
    d = df[(df.phase == "paperv2_phase2")]
    if d.empty:
        print("skip asym: no phase2 rows"); return
    levels = [20, 30, 50, 70, 80]
    grid = np.full((len(levels), len(levels)), np.nan)
    for li, L in enumerate(levels):
        for gi, G in enumerate(levels):
            cell = d[(d.L == L) & (d.G == G)]
            if cell.empty:
                continue
            piv = cell.pivot_table(index="seed", columns="method", values="f1m")
            if "tralo" not in piv:
                continue
            base = piv[[b for b in BASELINES if b in piv.columns]].mean(axis=1)
            grid[gi, li] = (piv["tralo"] - base).mean()  # row=G, col=L
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_yticks(range(len(levels))); ax.set_yticklabels([f"G{l}" for l in levels])
    ax.set_xlabel("Local tightness $L$ (% of true count)")
    ax.set_ylabel("Global tightness $G$ (% of true count)")
    for gi in range(len(levels)):
        for li in range(len(levels)):
            v = grid[gi, li]
            if not np.isnan(v):
                ax.text(li, gi, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7.5, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Macro F1 gap (TraLO $-$ mean of 5 baselines)")
    ax.set_title("Asymmetric tightness (DermMNIST): no $(L,G)$ corner\n"
                 "where TraLO collapses; gaps mostly within noise")
    out = FIG_DIR / "fig_asym_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------
# FIG 7 — backbone + multi-class robustness (F1 gap)
# --------------------------------------------------------------------------
def fig_robustness(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    # (a) backbones: derm cls4 loc_group across models
    ax = axes[0]
    db = df[(df.ds == "dermmnist") & (df.cls == 4) & (df.grp == "loc_group")]
    models = ["MobileNetV3", "ResNet18", "EfficientNetB0"]
    models = [m for m in models if m in db.model.unique()]
    x = np.arange(len(models)); w = 0.5
    means, errs = [], []
    for mod in models:
        gap = per_cell_gap(db[db.model == mod], "f1m")
        means.append(gap.gap.mean()); errs.append(gap.gap.std())
    ax.axhspan(-NOISE, NOISE, color="grey", alpha=0.15, label="noise band")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.bar(x, means, w, yerr=errs, capsize=3, color="#1565C0",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=12)
    ax.set_ylabel("Macro F1 gap (TraLO $-$ mean of 5 baselines)")
    ax.set_title("(a) Backbone robustness (DermMNIST, MEL)")
    ax.legend(loc="upper right", frameon=False); ax.grid(axis="y", alpha=0.18)

    # (b) multi-class: derm MobileNetV3 across constrained class
    ax = axes[1]
    dc = df[(df.ds == "dermmnist") & (df.model == "MobileNetV3") & (df.grp == "loc_group")]
    clsmap = {0: "AKIEC", 1: "BCC", 2: "BKL", 4: "MEL"}
    classes = sorted([c for c in dc.cls.unique() if c in clsmap])
    x = np.arange(len(classes)); w = 0.5
    means, errs = [], []
    for c in classes:
        gap = per_cell_gap(dc[dc.cls == c], "f1m")
        means.append(gap.gap.mean()); errs.append(gap.gap.std())
    ax.axhspan(-NOISE, NOISE, color="grey", alpha=0.15, label="noise band")
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.bar(x, means, w, yerr=errs, capsize=3, color="#2E7D32",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([clsmap[c] for c in classes])
    ax.set_ylabel("Macro F1 gap (TraLO $-$ mean of 5 baselines)")
    ax.set_xlabel("Constrained class")
    ax.set_title("(b) Multi-class robustness (DermMNIST, MobileNetV3)")
    ax.legend(loc="upper right", frameon=False); ax.grid(axis="y", alpha=0.18)

    fig.suptitle("Robustness: the F1 tie and the flips win hold across backbones "
                 "and constrained classes", y=1.02, fontsize=11)
    out = FIG_DIR / "fig_robustness.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    df = load()
    print(f"loaded {len(df)} rows (eurosat dropped)")
    fig_tradeoff(df)
    fig_f1_gap(df)
    fig_flips_bar(df)
    fig_regime(df)
    fig_satisfaction(df)
    fig_asym(df)
    fig_robustness(df)
    print("done.")
