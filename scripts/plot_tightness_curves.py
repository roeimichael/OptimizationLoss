"""Tightness vs F1 curves — AIDER cls 3 + Derm cls 5 majority.

Shows method F1 as constraint tightness varies. TraLO's curve sits closer to
the LOOSE-constraint ceiling than LP/heuristic, especially as tightness grows.
"""
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "scripts/_paper_agg/tightness_curves.png"

METHOD_STYLE = {
    "tralo":        ("o", "C0", "TraLO", "-"),
    "fioretto_ldf": ("s", "C1", "Fioretto", "--"),
    "hounie_rcl":   ("D", "C2", "Hounie", "--"),
    "danits_lp":    ("^", "C3", "Danits-LP", ":"),
    "heuristic":    ("v", "C4", "Heuristic", ":"),
}
TIGHT_PCT = {"L10_G10": 10, "L20_G20": 20, "L30_G30": 30,
             "L50_G50": 50, "L70_G70": 70}

ROOTS_AIDER = [
    "results/pending_runs/precision_majority/aider",
    "results/pending_runs/aider_cls3_tight/MobileNetV3",
]
ROOTS_DERM = [
    "results/pending_runs/precision_majority/dermmnist",
    "results/pending_runs/derm_cls5_tight/MobileNetV3",
]


def read_metrics(path):
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def collect(roots):
    by_key = defaultdict(lambda: {"f1": [], "flips": []})
    for root in roots:
        for cell in sorted(glob.glob(f"{root}/*/*/seed_*")):
            parts = cell.split("/")
            cfg, method = parts[-3], parts[-2]
            tight = cfg if (cfg.startswith("L") and "_G" in cfg) \
                else "_".join(cfg.split("_")[-2:])
            m = read_metrics(os.path.join(cell, "evaluation_metrics.csv"))
            if not m or tight not in TIGHT_PCT:
                continue
            by_key[(tight, method)]["f1"].append(float(m.get("F1 (Macro)", 0)))
            by_key[(tight, method)]["flips"].append(float(m.get("Flips Required", 0)))
    return by_key


def plot_panel(ax, by_key, title):
    tights = sorted(set(t for t, _ in by_key), key=lambda t: TIGHT_PCT[t])
    x = [TIGHT_PCT[t] for t in tights]
    for method, (marker, color, label, ls) in METHOD_STYLE.items():
        ys, errs = [], []
        for tight in tights:
            d = by_key.get((tight, method))
            if not d or not d["f1"]:
                ys.append(np.nan); errs.append(0)
            else:
                ys.append(np.mean(d["f1"]))
                errs.append(np.std(d["f1"]) / np.sqrt(len(d["f1"])))
        lw = 2.5 if method == "tralo" else 1.5
        ms = 9 if method == "tralo" else 7
        ax.errorbar(x, ys, yerr=errs, label=label, color=color,
                    marker=marker, linestyle=ls, linewidth=lw,
                    markersize=ms, capsize=4)
    ax.set_xlabel("Constraint tightness (% of natural cap)")
    ax.set_ylabel("F1 (Macro) ± SE across 5 seeds")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # tight (L10) on left, loose (L70) on right
    ax.legend(loc="lower right", fontsize=9)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_panel(axes[0], collect(ROOTS_AIDER),
               "AIDER cls 3 (74% majority) — MobileNetV3, 5 seeds")
    plot_panel(axes[1], collect(ROOTS_DERM),
               "Derm cls 5 NV (67% majority) — MobileNetV3, 5 seeds")
    fig.suptitle("F1 vs constraint tightness — TraLO holds F1 better as "
                 "constraint tightens (esp. vs heuristic/Danits-LP)",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
