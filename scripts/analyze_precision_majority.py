"""Precision-sweep analysis on majority-class TraLO wins.

Loads results/pending_runs/precision_majority/{aider,dermmnist}/...
For each (ds, tightness): mean F1, paired d_F1 vs each baseline, paired-t p-value.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = "results/pending_runs/precision_majority"


def read_metrics(path):
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def main():
    by_seed = defaultdict(dict)  # (ds, tight, seed) -> {method: f1}
    by_seed_flips = defaultdict(dict)
    for cell in sorted(glob.glob(f"{ROOT}/*/*/*/seed_*")):
        parts = cell.split("/")
        ds, cfg, method, seed_str = parts[-4], parts[-3], parts[-2], parts[-1]
        seed = int(seed_str.replace("seed_", ""))
        # extract tight from cfg, format: constrained{N}_{role}_L{x}_G{y}
        tight = "_".join(cfg.split("_")[-2:])
        m = read_metrics(os.path.join(cell, "evaluation_metrics.csv"))
        if not m:
            continue
        by_seed[(ds, tight, seed)][method] = float(m.get("F1 (Macro)", 0))
        by_seed_flips[(ds, tight, seed)][method] = float(m.get("Flips Required", 0))

    methods = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
    print(f"{'ds':10s} {'tight':10s} {'method':14s} {'F1_mean':>8s} "
          f"{'F1_std':>7s} {'flips':>7s} {'n':>3s}")
    print("-" * 70)
    by_method = defaultdict(list)
    by_method_flips = defaultdict(list)
    for key, vals in sorted(by_seed.items()):
        ds, tight, seed = key
        for m in methods:
            if m in vals:
                by_method[(ds, tight, m)].append(vals[m])
                by_method_flips[(ds, tight, m)].append(by_seed_flips[key].get(m, 0))
    for (ds, tight, m), f1s in sorted(by_method.items()):
        flips = by_method_flips[(ds, tight, m)]
        print(f"{ds:10s} {tight:10s} {m:14s} {np.mean(f1s):8.4f} "
              f"{np.std(f1s):7.4f} {np.mean(flips):7.1f} {len(f1s):>3d}")

    print()
    print("=== paired d_F1 + paired-t p ===")
    print(f"{'ds':10s} {'tight':10s} {'baseline':14s} {'d_F1':>9s} "
          f"{'p':>8s} {'n':>3s} {'sig':>4s}")
    print("-" * 70)
    for ds, tight in sorted(set((d, t) for d, t, _ in by_method)):
        tr_vals_by_seed = {}
        for key, vals in by_seed.items():
            if key[0] == ds and key[1] == tight and "tralo" in vals:
                tr_vals_by_seed[key[2]] = vals["tralo"]
        for bl in ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]:
            paired = []
            for seed, tr in tr_vals_by_seed.items():
                bl_v = by_seed.get((ds, tight, seed), {}).get(bl)
                if bl_v is not None:
                    paired.append((tr, bl_v))
            if len(paired) < 2:
                continue
            tr_arr = np.array([p[0] for p in paired])
            bl_arr = np.array([p[1] for p in paired])
            diff = tr_arr - bl_arr
            d_mean = diff.mean()
            if len(paired) >= 2 and diff.std() > 0:
                t_stat, p_val = stats.ttest_rel(tr_arr, bl_arr)
            else:
                p_val = float("nan")
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            print(f"{ds:10s} {tight:10s} {bl:14s} {d_mean:+9.4f} "
                  f"{p_val:8.4f} {len(paired):>3d} {sig:>4s}")


if __name__ == "__main__":
    main()
