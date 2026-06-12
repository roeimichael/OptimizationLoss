"""Forest plot of TraLO d_F1 vs each baseline, backbone × baseline grid.

Reads from results/pending_runs/{aider_cls3_backbones, derm_cls5_backbones}
Computes paired d_F1 across 5 seeds per (backbone × baseline).
Plots horizontal error bars with 95% CI from paired-t.
"""
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOTS = {
    "AIDER cls 3 (74% majority)": "results/pending_runs/aider_cls3_backbones",
    "Derm cls 5 (NV 67% majority)": "results/pending_runs/derm_cls5_backbones",
}
OUT = "scripts/_paper_agg/forest_majority.png"
BASELINES = ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
BASELINE_LABEL = {"fioretto_ldf": "Fioretto", "hounie_rcl": "Hounie",
                  "danits_lp": "Danits-LP", "heuristic": "Heuristic"}
BACKBONES = ["MobileNetV3", "MobileNetV2", "RegNetY400MF", "ShuffleNetV2"]


def read_metrics(path):
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def collect(root):
    by_seed = defaultdict(dict)
    for cell in sorted(glob.glob(f"{root}/*/*/seed_*")):
        parts = cell.split("/")
        backbone, method, seed_str = parts[-3], parts[-2], parts[-1]
        seed = int(seed_str.replace("seed_", ""))
        m = read_metrics(os.path.join(cell, "evaluation_metrics.csv"))
        if not m:
            continue
        by_seed[(backbone, seed)][method] = float(m.get("F1 (Macro)", 0))
    return by_seed


def paired_stats(by_seed, backbone, baseline):
    tr = [v["tralo"] for (b, s), v in by_seed.items()
          if b == backbone and "tralo" in v]
    bl = [(s, v.get(baseline)) for (b, s), v in by_seed.items()
          if b == backbone and baseline in v]
    pairs = []
    for s, blv in bl:
        v = by_seed.get((backbone, s), {})
        if blv is not None and "tralo" in v:
            pairs.append((v["tralo"], blv))
    if len(pairs) < 2:
        return None
    tr_arr = np.array([p[0] for p in pairs])
    bl_arr = np.array([p[1] for p in pairs])
    diff = tr_arr - bl_arr
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0
    if n > 1 and diff.std(ddof=1) > 0:
        t_crit = stats.t.ppf(0.975, df=n-1)
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
        _, p_val = stats.ttest_rel(tr_arr, bl_arr)
    else:
        ci_low = ci_high = mean
        p_val = float("nan")
    return {"d": mean, "ci_low": ci_low, "ci_high": ci_high, "p": p_val, "n": n}


def main():
    fig, axes = plt.subplots(1, len(ROOTS), figsize=(15, 6), sharey=True)
    for ax, (label, root) in zip(axes, ROOTS.items()):
        if not os.path.exists(root):
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(label)
            continue
        by_seed = collect(root)
        y_pos = 0
        y_labels = []
        for bb in BACKBONES:
            for bl in BASELINES:
                stats_d = paired_stats(by_seed, bb, bl)
                if stats_d is None:
                    continue
                color = "#2ca02c" if stats_d["d"] > 0 else "#d62728"
                ax.errorbar(stats_d["d"], y_pos,
                            xerr=[[stats_d["d"] - stats_d["ci_low"]],
                                  [stats_d["ci_high"] - stats_d["d"]]],
                            fmt="o", color=color, capsize=4, markersize=8)
                sig = ""
                if stats_d["p"] < 0.001:
                    sig = "***"
                elif stats_d["p"] < 0.01:
                    sig = "**"
                elif stats_d["p"] < 0.05:
                    sig = "*"
                if sig:
                    ax.text(stats_d["ci_high"] + 0.005, y_pos, sig,
                            va="center", fontsize=10, color=color, fontweight="bold")
                y_labels.append(f"{bb} vs {BASELINE_LABEL[bl]}")
                y_pos += 1
            y_pos += 0.5  # gap between backbones
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("d_F1 = TraLO − baseline (95% CI)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

    fig.suptitle("TraLO vs baselines on majority-class constraints "
                 "(paired-t, n=5 seeds, L30_G30)\n"
                 "Green = TraLO wins, Red = TraLO loses. * p<0.05, ** p<0.01, *** p<0.001",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
