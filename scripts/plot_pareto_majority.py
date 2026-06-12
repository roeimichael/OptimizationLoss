"""Paper figure: F1 vs flips Pareto on AIDER cls 3 + Derm cls 5 majority.

Each point = (method, tightness) → mean F1 + mean flips across 5 seeds.
TraLO sits at the Pareto-optimal corner (high F1, low flips).
"""
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "results/pending_runs/precision_majority"
OUT = "scripts/_paper_agg/pareto_majority.png"

METHOD_STYLE = {
    "tralo":        ("o", "C0", "TraLO"),
    "fioretto_ldf": ("s", "C1", "Fioretto"),
    "hounie_rcl":   ("D", "C2", "Hounie"),
    "danits_lp":    ("^", "C3", "Danits-LP"),
    "heuristic":    ("v", "C4", "Heuristic"),
}
TIGHT_ORDER = ["L30_G30", "L50_G50", "L70_G70"]
DS_LABEL = {"aider": "AIDER cls 3 (74% majority)",
            "dermmnist": "Derm cls 5 NV (67% majority)"}


def read_metrics(path):
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def main():
    by_key = defaultdict(lambda: {"f1": [], "flips": []})
    for cell in sorted(glob.glob(f"{ROOT}/*/*/*/seed_*")):
        parts = cell.split("/")
        ds, cfg, method, seed_str = parts[-4], parts[-3], parts[-2], parts[-1]
        tight = "_".join(cfg.split("_")[-2:])
        m = read_metrics(os.path.join(cell, "evaluation_metrics.csv"))
        if not m:
            continue
        by_key[(ds, tight, method)]["f1"].append(float(m.get("F1 (Macro)", 0)))
        by_key[(ds, tight, method)]["flips"].append(float(m.get("Flips Required", 0)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ds in zip(axes, ["aider", "dermmnist"]):
        for tight in TIGHT_ORDER:
            for method, (marker, color, label) in METHOD_STYLE.items():
                d = by_key.get((ds, tight, method))
                if not d or not d["f1"]:
                    continue
                mf1 = np.mean(d["f1"])
                mfl = np.mean(d["flips"])
                # tightness as marker size: L30 small, L70 large
                size = {"L30_G30": 60, "L50_G50": 120, "L70_G70": 200}.get(tight, 100)
                ax.scatter(mfl, mf1, marker=marker, color=color, s=size,
                           alpha=0.85, edgecolor="black", linewidth=0.7)
                ax.annotate(tight.split("_")[0],
                            (mfl, mf1), textcoords="offset points",
                            xytext=(8, -3), fontsize=7, color=color)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Post-hoc flips required (lower is better, log scale)")
        ax.set_ylabel("F1 (Macro, higher is better)")
        ax.set_title(DS_LABEL[ds])
        ax.grid(True, alpha=0.3)

    # shared legend
    handles = [plt.Line2D([0], [0], marker=m, color="w",
                          markerfacecolor=c, markeredgecolor="black",
                          markersize=10, label=lab)
               for (m, c, lab) in METHOD_STYLE.values()]
    axes[1].legend(handles=handles, loc="lower right", fontsize=9, title="Method")

    fig.suptitle("F1 vs Flips Pareto — TraLO at the Pareto-optimal corner "
                 "(high F1, low flips)\nMarker size = constraint tightness "
                 "(small=L30 strict → large=L70 loose)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
