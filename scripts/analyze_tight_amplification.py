"""Analyze tight-constraint amplification on AIDER cls 3 + Derm cls 5.

Combines L30/L50/L70 (from precision_majority) with L10/L20 (from *_cls*_tight).
Plots TraLO d_F1 vs tightness — does the win amplify at tighter constraint?
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
from scipy import stats


def read_metrics(path):
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", required=True, choices=["aider", "dermmnist"])
    parser.add_argument("--cls", type=int, required=True)
    args = parser.parse_args()

    roots = []
    if args.ds == "aider" and args.cls == 3:
        roots = [
            "results/pending_runs/precision_majority/aider",
            "results/pending_runs/aider_cls3_tight/MobileNetV3",
        ]
    elif args.ds == "dermmnist" and args.cls == 5:
        roots = [
            "results/pending_runs/precision_majority/dermmnist",
            "results/pending_runs/derm_cls5_tight/MobileNetV3",
        ]

    by_seed = defaultdict(dict)  # (tight, seed) -> {method: f1}
    by_seed_flips = defaultdict(dict)
    for root in roots:
        # try both layouts: precision (cfg/method/seed) and tight (tight/method/seed)
        for cell in sorted(glob.glob(f"{root}/*/*/seed_*")):
            parts = cell.split("/")
            cfg, method, seed_str = parts[-3], parts[-2], parts[-1]
            seed = int(seed_str.replace("seed_", ""))
            # determine tight: if cfg starts with L%d_G%d, use it; else extract from cfg
            if cfg.startswith("L") and "_G" in cfg and not cfg.startswith("constrained"):
                tight = cfg
            else:
                tight = "_".join(cfg.split("_")[-2:])
            m = read_metrics(os.path.join(cell, "evaluation_metrics.csv"))
            if not m:
                continue
            by_seed[(tight, seed)][method] = float(m.get("F1 (Macro)", 0))
            by_seed_flips[(tight, seed)][method] = float(m.get("Flips Required", 0))

    print(f"\n### {args.ds} cls {args.cls} — tight amplification ###\n")
    print(f"{'tight':10s} {'method':14s} {'F1':>8s} {'F1_std':>7s} {'flips':>7s} {'n':>3s}")
    print("-" * 60)
    methods = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
    by_tm = defaultdict(list)
    by_tm_flips = defaultdict(list)
    for (tight, seed), vals in by_seed.items():
        for m in methods:
            if m in vals:
                by_tm[(tight, m)].append(vals[m])
                by_tm_flips[(tight, m)].append(by_seed_flips[(tight, seed)].get(m, 0))
    tights = sorted(set(t for t, _ in by_tm))
    for tight in tights:
        for m in methods:
            f1s = by_tm.get((tight, m), [])
            if not f1s:
                continue
            print(f"{tight:10s} {m:14s} {np.mean(f1s):8.4f} {np.std(f1s):7.4f} "
                  f"{np.mean(by_tm_flips[(tight, m)]):7.1f} {len(f1s):>3d}")
        print()

    print(f"=== paired d_F1 per tightness ===")
    print(f"{'tight':10s} {'baseline':14s} {'d_F1':>9s} {'p':>8s} {'n':>3s} {'sig':>4s}")
    print("-" * 60)
    for tight in tights:
        for bl in ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]:
            paired = []
            for (t, s), vals in by_seed.items():
                if t != tight:
                    continue
                if "tralo" in vals and bl in vals:
                    paired.append((vals["tralo"], vals[bl]))
            if len(paired) < 2:
                continue
            tr_arr = np.array([p[0] for p in paired])
            bl_arr = np.array([p[1] for p in paired])
            diff = tr_arr - bl_arr
            d_mean = diff.mean()
            if diff.std() > 0 and len(paired) > 1:
                _, p_val = stats.ttest_rel(tr_arr, bl_arr)
            else:
                p_val = float("nan")
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                                                ("*" if p_val < 0.05 else ""))
            print(f"{tight:10s} {bl:14s} {d_mean:+9.4f} {p_val:8.4f} "
                  f"{len(paired):>3d} {sig:>4s}")


if __name__ == "__main__":
    main()
