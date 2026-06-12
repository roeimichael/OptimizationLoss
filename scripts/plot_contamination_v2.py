"""Revised contamination plots — y-axis auto-zoom so the method gaps are visible.

Three plot families:
1. Per-(dataset, tightness) — y-axis tightly cropped to the method span,
   makes the 0.01-0.03 F1 gap actually visible.
2. The headline "winning image": dF1 (in-training - post-hoc best) heatmap
   per (dataset, tightness, sigma). Red = TraLO/in-training wins.
3. Stacked dF1 vs sigma per dataset, one line per tightness — shows
   the monotone shift as contamination grows.
"""
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("paper/HANDOFF/figures/contamination_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_COLORS = {
    "tralo":        "#1f77b4",
    "fioretto_ldf": "#ff7f0e",
    "hounie_rcl":   "#d62728",
    "danits_lp":    "#2ca02c",
    "heuristic":    "#8c564b",
}
METHOD_ORDER = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
DATASETS = ["tissuemnist", "dermmnist", "aider"]
TIGHTS = ["L20_G20", "L30_G30", "L50_G50", "L70_G70"]


def read_metrics(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = row["Value"]
    return out


def collect():
    rows = []
    for ds in DATASETS:
        for root in (f"results/pending_runs/contamination_{ds}",
                     "results/pending_runs/contamination_clean"):
            if not os.path.isdir(root): continue
            for cfg_p in glob.glob(f"{root}/**/config.json", recursive=True):
                try:
                    with open(cfg_p) as f: cfg = json.load(f)
                    if cfg["dataset_mode"] != ds: continue
                    m = read_metrics(cfg_p.replace("config.json","evaluation_metrics.csv"))
                    if "contamination_clean" in root:
                        sigma = 0.0
                    else:
                        sigma = int(cfg["experiment_path"].split("/")[-4][5:]) / 100
                    rows.append({
                        "dataset": ds, "sigma": sigma,
                        "tight": cfg["constraint_tag"],
                        "method": cfg["methodology"],
                        "seed": cfg["hyperparams"]["seed"],
                        "f1": float(m["F1 (Macro)"]),
                    })
                except Exception: continue
    return rows


def cell_means(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["dataset"], r["sigma"], r["tight"], r["method"])].append(r["f1"])
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in out.items()}


def plot_per_ds_zoom(cell, dataset):
    """y-axis CROPPED per subplot to the actual method span + margin."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()
    sigmas = [0.0, 0.10, 0.20, 0.30]
    for ax, tight in zip(axes, TIGHTS):
        all_vals = []
        for me in METHOD_ORDER:
            xs, ys, errs = [], [], []
            for sig in sigmas:
                k = (dataset, sig, tight, me)
                if k in cell:
                    m, s, n = cell[k]
                    xs.append(sig); ys.append(m); errs.append(s / np.sqrt(max(n,1)))
                    all_vals.append(m)
            if xs:
                ax.errorbar(xs, ys, yerr=errs, marker="o", label=me,
                            color=METHOD_COLORS[me], capsize=4, lw=2, ms=8)
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            margin = max((hi - lo) * 0.15, 0.005)
            ax.set_ylim(lo - margin, hi + margin)
        ax.set_title(f"tight = {tight}", fontsize=11)
        ax.set_xlabel("contamination sigma"); ax.set_ylabel("Macro F1")
        ax.grid(alpha=0.3); ax.set_xticks(sigmas)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5,
               bbox_to_anchor=(0.99, 1.03), fontsize=10)
    fig.suptitle(f"{dataset.upper()} — F1 vs contamination (y-axis zoomed to method span)",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    p = OUT_DIR / f"contam_{dataset}_zoomed.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_advantage_lines(cell, dataset):
    """dF1 (TraLO - best post-hoc) vs sigma, one line per tightness."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    sigmas = [0.0, 0.10, 0.20, 0.30]
    for ax, kind in zip(axes, ("TraLO vs post-hoc", "in-training cluster vs post-hoc")):
        for tight in TIGHTS:
            xs, ys = [], []
            for sig in sigmas:
                tr = cell.get((dataset, sig, tight, "tralo"))
                fi = cell.get((dataset, sig, tight, "fioretto_ldf"))
                da = cell.get((dataset, sig, tight, "danits_lp"))
                he = cell.get((dataset, sig, tight, "heuristic"))
                ph_best = max(v[0] for v in (da, he) if v)
                if kind == "TraLO vs post-hoc":
                    if tr: xs.append(sig); ys.append(tr[0] - ph_best)
                else:
                    in_best = max(v[0] for v in (tr, fi) if v)
                    xs.append(sig); ys.append(in_best - ph_best)
            if xs:
                ax.plot(xs, ys, marker="o", label=tight, lw=2.5, ms=10)
        ax.axhline(0, color="black", lw=0.8, linestyle=":")
        ax.fill_between([-0.02, 0.32], 0, 0.05, color="green", alpha=0.06,
                         label="in-training wins")
        ax.fill_between([-0.02, 0.32], -0.05, 0, color="red", alpha=0.06,
                         label="post-hoc wins")
        ax.set_title(f"{kind}", fontsize=11)
        ax.set_xlabel("contamination sigma"); ax.set_xticks(sigmas)
        ax.set_xlim(-0.02, 0.32); ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    axes[0].set_ylabel("dF1 (positive = in-training wins)")
    fig.suptitle(f"{dataset.upper()} — advantage shift as contamination grows",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    p = OUT_DIR / f"advantage_{dataset}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_headline_heatmap(cell):
    """Headline winning image: 3 datasets x 4 tightness x 4 sigmas heatmap
    of dF1(in-training best - post-hoc best). Red = in-training wins."""
    sigmas = [0.0, 0.10, 0.20, 0.30]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, ds in zip(axes, DATASETS):
        grid = np.full((len(TIGHTS), len(sigmas)), np.nan)
        annot = [[""]*len(sigmas) for _ in range(len(TIGHTS))]
        for i, tight in enumerate(TIGHTS):
            for j, sig in enumerate(sigmas):
                tr = cell.get((ds, sig, tight, "tralo"))
                fi = cell.get((ds, sig, tight, "fioretto_ldf"))
                da = cell.get((ds, sig, tight, "danits_lp"))
                he = cell.get((ds, sig, tight, "heuristic"))
                ph = [v for v in (da, he) if v]
                tr_fi = [v for v in (tr, fi) if v]
                if ph and tr_fi:
                    d = max(v[0] for v in tr_fi) - max(v[0] for v in ph)
                    grid[i, j] = d
                    annot[i][j] = f"{d:+.3f}"
        vmax = 0.06   # fixed scale across datasets for comparability
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(sigmas))); ax.set_xticklabels([f"{s:.2f}" for s in sigmas])
        ax.set_yticks(range(len(TIGHTS))); ax.set_yticklabels(TIGHTS)
        for i in range(len(TIGHTS)):
            for j in range(len(sigmas)):
                if annot[i][j]:
                    ax.text(j, i, annot[i][j], ha="center", va="center",
                            fontsize=10, fontweight="bold")
        ax.set_xlabel("contamination sigma")
        ax.set_title(f"{ds}", fontsize=12)
    fig.colorbar(im, ax=axes, label="dF1 (in-train best - post-hoc best)  red=in-train wins")
    fig.suptitle("WINNING IMAGE: in-training advantage grows with contamination",
                 fontsize=13, y=1.04)
    p = OUT_DIR / "headline_heatmap.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_tralo_only_heatmap(cell):
    """Same as above but TraLO vs post-hoc-best (not in-training cluster)."""
    sigmas = [0.0, 0.10, 0.20, 0.30]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, ds in zip(axes, DATASETS):
        grid = np.full((len(TIGHTS), len(sigmas)), np.nan)
        annot = [[""]*len(sigmas) for _ in range(len(TIGHTS))]
        for i, tight in enumerate(TIGHTS):
            for j, sig in enumerate(sigmas):
                tr = cell.get((ds, sig, tight, "tralo"))
                da = cell.get((ds, sig, tight, "danits_lp"))
                he = cell.get((ds, sig, tight, "heuristic"))
                ph = [v for v in (da, he) if v]
                if tr and ph:
                    d = tr[0] - max(v[0] for v in ph)
                    grid[i, j] = d
                    annot[i][j] = f"{d:+.3f}"
        vmax = 0.06
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(sigmas))); ax.set_xticklabels([f"{s:.2f}" for s in sigmas])
        ax.set_yticks(range(len(TIGHTS))); ax.set_yticklabels(TIGHTS)
        for i in range(len(TIGHTS)):
            for j in range(len(sigmas)):
                if annot[i][j]:
                    ax.text(j, i, annot[i][j], ha="center", va="center",
                            fontsize=10, fontweight="bold")
        ax.set_xlabel("contamination sigma")
        ax.set_title(f"{ds}", fontsize=12)
    fig.colorbar(im, ax=axes, label="dF1 (TraLO - post-hoc best)  red=TraLO wins")
    fig.suptitle("TraLO-only: how TraLO's advantage shifts with contamination",
                 fontsize=13, y=1.04)
    p = OUT_DIR / "tralo_only_heatmap.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def main():
    rows = collect()
    print(f"Rows: {len(rows)}")
    cell = cell_means(rows)
    print("\nHeadline heatmaps:")
    plot_headline_heatmap(cell)
    plot_tralo_only_heatmap(cell)
    print("\nAdvantage line plots:")
    for ds in DATASETS:
        plot_advantage_lines(cell, ds)
    print("\nZoomed per-dataset F1 plots:")
    for ds in DATASETS:
        plot_per_ds_zoom(cell, ds)
    print("\nDone.")


if __name__ == "__main__":
    main()
