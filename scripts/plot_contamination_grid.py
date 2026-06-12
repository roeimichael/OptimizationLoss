"""Plot contamination grid (the graphs the user explicitly asked for).

For each dataset, produce SEPARATE plots:
  - One subplot per symmetric tightness (L20/L30/L50/L70).
  - x-axis: contamination level (sigma = 0, 0.10, 0.20, 0.30).
  - y-axis: macro-F1.
  - One line per method.

Also emit a summary CSV with every cell mean F1 + flips + sat%.

Reuses sigma=0 (clean) from the headline sweeps (paper_backbones or asym_tissue_aider)
when available.
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

OUT_DIR = Path("paper/HANDOFF/figures/contamination")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("paper/HANDOFF/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

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
    """Collect all per-seed rows from contamination sweeps + clean baselines."""
    rows = []
    # contamination sweeps (sigma=0.10/0.20/0.30)
    for ds in DATASETS:
        root = f"results/pending_runs/contamination_{ds}"
        if not os.path.isdir(root): continue
        for cfg_p in glob.glob(f"{root}/**/config.json", recursive=True):
            try:
                with open(cfg_p) as f: cfg = json.load(f)
                m = read_metrics(cfg_p.replace("config.json","evaluation_metrics.csv"))
                sigma_tag = cfg["experiment_path"].split("/")[-4]
                sigma = int(sigma_tag[5:]) / 100.0
                rows.append({
                    "dataset": ds, "sigma": sigma, "tight": cfg["constraint_tag"],
                    "method": cfg["methodology"], "seed": cfg["hyperparams"]["seed"],
                    "f1": float(m["F1 (Macro)"]),
                    "flips": float(m["Flips Required"]),
                    "sat": 1 if m.get("Raw All Satisfied","0")=="1" else 0,
                    "acc": float(m["Accuracy"]),
                })
            except Exception:
                continue
    # clean baseline (sigma=0) — explicitly generated sweep with same HP as contam grid
    clean_root = "results/pending_runs/contamination_clean"
    if os.path.isdir(clean_root):
        for cfg_p in glob.glob(f"{clean_root}/**/config.json", recursive=True):
            try:
                with open(cfg_p) as f: cfg = json.load(f)
                m = read_metrics(cfg_p.replace("config.json","evaluation_metrics.csv"))
                rows.append({
                    "dataset": cfg["dataset_mode"], "sigma": 0.0,
                    "tight": cfg["constraint_tag"],
                    "method": cfg["methodology"], "seed": cfg["hyperparams"]["seed"],
                    "f1": float(m["F1 (Macro)"]),
                    "flips": float(m["Flips Required"]),
                    "sat": 1 if m.get("Raw All Satisfied","0")=="1" else 0,
                    "acc": float(m["Accuracy"]),
                })
            except Exception:
                continue
    return rows


def cell_means(rows):
    """Aggregate by (dataset, sigma, tight, method)."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["dataset"], r["sigma"], r["tight"], r["method"])
        out[k]["f1"].append(r["f1"])
        out[k]["flips"].append(r["flips"])
        out[k]["sat"].append(r["sat"])
        out[k]["acc"].append(r["acc"])
    return {k: {m: (np.mean(v), np.std(v), len(v)) for m, v in d.items()}
            for k, d in out.items()}


def write_summary(cell, path):
    """Wide CSV: dataset, sigma, tight, method, f1_mean, f1_std, flips_mean,
    sat_mean, acc_mean, n_seeds."""
    fields = ["dataset","sigma","tight","method","n",
              "f1_mean","f1_std","flips_mean","sat_pct","acc_mean"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(fields)
        for (ds, sig, tight, me), d in sorted(cell.items()):
            n = d["f1"][2]
            w.writerow([ds, f"{sig:.2f}", tight, me, n,
                        f"{d['f1'][0]:.4f}", f"{d['f1'][1]:.4f}",
                        f"{d['flips'][0]:.2f}",
                        f"{d['sat'][0]*100:.0f}",
                        f"{d['acc'][0]:.4f}"])
    print(f"  wrote {path}")


def plot_per_dataset(cell, dataset):
    """One figure per dataset, 4 subplots (one per tightness)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    axes = axes.flatten()
    sigmas_sorted = [0.0, 0.10, 0.20, 0.30]
    for ax, tight in zip(axes, TIGHTS):
        for me in METHOD_ORDER:
            xs, ys, errs = [], [], []
            for sig in sigmas_sorted:
                key = (dataset, sig, tight, me)
                if key in cell and "f1" in cell[key]:
                    mean, std, n = cell[key]["f1"]
                    xs.append(sig); ys.append(mean); errs.append(std/np.sqrt(max(n,1)))
            if xs:
                ax.errorbar(xs, ys, yerr=errs, marker="o", label=me,
                            color=METHOD_COLORS[me], capsize=3, lw=1.6, ms=7)
        ax.set_title(f"tight = {tight}")
        ax.set_xlabel("contamination sigma (Gaussian noise on train+test)")
        ax.set_ylabel("Macro F1")
        ax.grid(alpha=0.3)
        ax.set_xticks(sigmas_sorted)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5,
               bbox_to_anchor=(0.99, 1.02), fontsize=9)
    fig.suptitle(f"{dataset.upper()} — F1 vs contamination, MobileNetV3, 5 methods",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    p = OUT_DIR / f"contam_{dataset}_per_tightness.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_advantage(cell, dataset):
    """ΔF1 (TraLO - best post-hoc) and ΔF1 (in-train best - post-hoc best)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    sigmas_sorted = [0.0, 0.10, 0.20, 0.30]
    for ax, kind in zip(axes, ("TraLO-only","in-training cluster")):
        for tight in TIGHTS:
            xs, ys = [], []
            for sig in sigmas_sorted:
                tr_f1 = cell.get((dataset,sig,tight,"tralo"), {}).get("f1")
                fi_f1 = cell.get((dataset,sig,tight,"fioretto_ldf"), {}).get("f1")
                da_f1 = cell.get((dataset,sig,tight,"danits_lp"), {}).get("f1")
                he_f1 = cell.get((dataset,sig,tight,"heuristic"), {}).get("f1")
                if tr_f1 and (da_f1 or he_f1):
                    ph_best = max(v[0] for v in (da_f1, he_f1) if v)
                    if kind == "TraLO-only":
                        xs.append(sig); ys.append(tr_f1[0] - ph_best)
                    else:
                        in_best = max(v[0] for v in (tr_f1, fi_f1) if v)
                        xs.append(sig); ys.append(in_best - ph_best)
            if xs:
                ax.plot(xs, ys, marker="o", label=tight, lw=1.8, ms=7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"dF1 ({kind} - post-hoc best)")
        ax.set_xlabel("contamination sigma")
        ax.set_xticks(sigmas_sorted); ax.grid(alpha=0.3)
        ax.legend(fontsize=9, title="tightness")
    axes[0].set_ylabel("dF1 (positive = TraLO/in-train wins)")
    fig.suptitle(f"{dataset.upper()} — advantage shift as contamination grows",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = OUT_DIR / f"contam_{dataset}_advantage.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_warmup_acc_vs_sigma(rows):
    """Show that increasing sigma actually drives warmup test-acc down."""
    by_ds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] in ("danits_lp","heuristic"):
            by_ds[r["dataset"]][r["sigma"]].append(r["acc"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ds in DATASETS:
        xs = sorted(by_ds[ds].keys())
        ys = [np.mean(by_ds[ds][s]) for s in xs]
        if xs: ax.plot(xs, ys, marker="o", label=ds, lw=2)
    ax.set_xlabel("contamination sigma"); ax.set_ylabel("post-hoc test acc (warmup-quality proxy)")
    ax.set_title("Contamination reliably depresses warmup quality across all 3 datasets")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p = OUT_DIR / "contam_acc_vs_sigma.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  {p}")


def main():
    print("Collecting...")
    rows = collect()
    print(f"Total per-seed rows: {len(rows)}")
    cell = cell_means(rows)
    write_summary(cell, TABLES_DIR / "contamination_summary.csv")
    print("\nPer-dataset plots:")
    for ds in DATASETS:
        plot_per_dataset(cell, ds)
        plot_advantage(cell, ds)
    print("\nWarmup-quality proxy:")
    plot_warmup_acc_vs_sigma(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
