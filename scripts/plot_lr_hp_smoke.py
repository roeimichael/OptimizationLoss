"""LR + HP smoke analysis under contamination sigma=0.20.

Produces:
  - LR-sweep plot: x=LR, y=F1, one line per method, per dataset
  - HP-variant bar chart: TraLO HP knobs on derm sigma=0.20
  - Summary CSV with all LR + HP cells
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
ROOT = "results/pending_runs/lr_hp_smoke"

METHOD_COLORS = {
    "tralo":        "#1f77b4",
    "fioretto_ldf": "#ff7f0e",
    "hounie_rcl":   "#d62728",
    "danits_lp":    "#2ca02c",
    "heuristic":    "#8c564b",
}
METHOD_ORDER = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]


def read_metrics(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = row["Value"]
    return out


def collect():
    lr_rows, hp_rows = [], []
    for cfg_p in glob.glob(f"{ROOT}/**/config.json", recursive=True):
        try:
            with open(cfg_p) as f: cfg = json.load(f)
            m = read_metrics(cfg_p.replace("config.json","evaluation_metrics.csv"))
            f1 = float(m["F1 (Macro)"])
            flips = float(m["Flips Required"])
            sat = 1 if m.get("Raw All Satisfied","0")=="1" else 0
            acc = float(m["Accuracy"])
        except Exception: continue
        parts = cfg["experiment_path"].split("/")
        section = parts[3]
        if section == "lr_sweep":
            lr_rows.append({"dataset": cfg["dataset_mode"],
                            "lr": cfg["hyperparams"]["lr"],
                            "method": cfg["methodology"],
                            "seed": cfg["hyperparams"]["seed"],
                            "f1": f1, "flips": flips, "sat": sat, "acc": acc})
        elif section == "hp_smoke":
            variant = parts[4]
            hp_rows.append({"variant": variant,
                            "seed": cfg["hyperparams"]["seed"],
                            "f1": f1, "flips": flips, "sat": sat, "acc": acc})
    return lr_rows, hp_rows


def plot_lr(lr_rows):
    """One subplot per dataset, x=LR, y=F1, one line per method."""
    by_d = defaultdict(lambda: defaultdict(list))
    for r in lr_rows:
        by_d[r["dataset"]][(r["lr"], r["method"])].append(r["f1"])
    datasets = sorted(by_d)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4.5),
                              sharey=False, squeeze=False)
    axes = axes[0]
    for ax, ds in zip(axes, datasets):
        lrs = sorted({lr for lr,_ in by_d[ds]})
        for me in METHOD_ORDER:
            xs, ys, errs = [], [], []
            for lr in lrs:
                v = by_d[ds].get((lr, me), [])
                if v:
                    xs.append(lr); ys.append(np.mean(v))
                    errs.append(np.std(v)/np.sqrt(len(v)))
            if xs:
                ax.errorbar(xs, ys, yerr=errs, marker="o", label=me,
                            color=METHOD_COLORS[me], capsize=3, lw=1.6, ms=7)
        ax.set_xscale("log"); ax.set_xlabel("LR")
        ax.set_ylabel("Macro F1"); ax.set_title(f"{ds} (sigma=0.20, L30_G30)")
        ax.grid(alpha=0.3, which="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5,
               bbox_to_anchor=(0.99, 1.02), fontsize=9)
    fig.suptitle("LR sweep under contamination (sigma=0.20, L30_G30, MobileNetV3)",
                 y=1.04, fontsize=12)
    fig.tight_layout()
    p = OUT_DIR / "lr_sweep_contam20.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_hp_variants(hp_rows):
    """Bar chart of TraLO variants on derm_sigma20."""
    by_v = defaultdict(lambda: defaultdict(list))
    for r in hp_rows:
        by_v[r["variant"]]["f1"].append(r["f1"])
        by_v[r["variant"]]["flips"].append(r["flips"])
    variants = sorted(by_v.keys(), key=lambda v: (v != "baseline", v))
    f1_mean = [np.mean(by_v[v]["f1"]) for v in variants]
    f1_std  = [np.std(by_v[v]["f1"]) for v in variants]
    flips_mean = [np.mean(by_v[v]["flips"]) for v in variants]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    bars1 = a1.bar(variants, f1_mean, yerr=f1_std, capsize=4, color="#1f77b4")
    a1.axhline(f1_mean[0], color="red", linestyle="--", lw=1, label="baseline")
    a1.set_ylabel("Macro F1"); a1.set_title("TraLO HP variants on derm sigma=0.20, L30_G30")
    a1.tick_params(axis="x", rotation=20); a1.grid(alpha=0.3, axis="y"); a1.legend()
    for b, m, s in zip(bars1, f1_mean, f1_std):
        a1.text(b.get_x()+b.get_width()/2, m+s+0.001, f"{m:.4f}",
                ha="center", fontsize=8)
    a2.bar(variants, flips_mean, color="#ff7f0e")
    a2.axhline(flips_mean[0], color="red", linestyle="--", lw=1)
    a2.set_ylabel("Flips Required"); a2.set_title("Same — flip count")
    a2.tick_params(axis="x", rotation=20); a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT_DIR / "hp_smoke_derm_contam20.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  {p}")


def write_summary(lr_rows, hp_rows):
    path_lr = TABLES_DIR / "lr_sweep_summary.csv"
    by_dlm = defaultdict(list)
    for r in lr_rows: by_dlm[(r["dataset"], r["lr"], r["method"])].append(r)
    with open(path_lr, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset","lr","method","n","f1_mean","f1_std","flips_mean","sat_pct"])
        for (ds, lr, me), rs in sorted(by_dlm.items()):
            f1s = [r["f1"] for r in rs]; fls = [r["flips"] for r in rs]
            sat = [r["sat"] for r in rs]
            w.writerow([ds, lr, me, len(rs),
                        f"{np.mean(f1s):.4f}", f"{np.std(f1s):.4f}",
                        f"{np.mean(fls):.2f}", f"{np.mean(sat)*100:.0f}"])
    print(f"  wrote {path_lr}")
    path_hp = TABLES_DIR / "hp_smoke_summary.csv"
    by_v = defaultdict(list)
    for r in hp_rows: by_v[r["variant"]].append(r)
    with open(path_hp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant","n","f1_mean","f1_std","flips_mean","sat_pct"])
        for v, rs in sorted(by_v.items()):
            f1s = [r["f1"] for r in rs]; fls = [r["flips"] for r in rs]
            sat = [r["sat"] for r in rs]
            w.writerow([v, len(rs), f"{np.mean(f1s):.4f}", f"{np.std(f1s):.4f}",
                        f"{np.mean(fls):.2f}", f"{np.mean(sat)*100:.0f}"])
    print(f"  wrote {path_hp}")


def main():
    lr_rows, hp_rows = collect()
    print(f"LR rows: {len(lr_rows)}  HP rows: {len(hp_rows)}")
    write_summary(lr_rows, hp_rows)
    plot_lr(lr_rows)
    plot_hp_variants(hp_rows)
    print("Done.")


if __name__ == "__main__":
    main()
