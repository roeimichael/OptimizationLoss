"""Backbone-robustness analysis on majority-class TraLO wins.

Reads either:
  results/pending_runs/aider_cls3_backbones/<backbone>/<method>/seed_X/
  results/pending_runs/derm_cls5_backbones/<backbone>/<method>/seed_X/

For each (backbone, method): mean F1 + flips across 5 seeds.
For each backbone: paired d_F1 of TraLO vs each baseline + paired-t p-value.

Use --dir to point at the experiment root.
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
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    root = args.dir.rstrip("/")
    label = os.path.basename(root)
    print(f"\n### {label} ###\n")

    by_seed = defaultdict(dict)
    by_seed_flips = defaultdict(dict)
    for cell in sorted(glob.glob(f"{root}/*/*/seed_*")):
        parts = cell.split("/")
        backbone, method, seed_str = parts[-3], parts[-2], parts[-1]
        seed = int(seed_str.replace("seed_", ""))
        m = read_metrics(os.path.join(cell, "evaluation_metrics.csv"))
        if not m:
            continue
        by_seed[(backbone, seed)][method] = float(m.get("F1 (Macro)", 0))
        by_seed_flips[(backbone, seed)][method] = float(m.get("Flips Required", 0))

    methods = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
    by_method = defaultdict(list)
    by_method_flips = defaultdict(list)
    for (backbone, seed), vals in by_seed.items():
        for m in methods:
            if m in vals:
                by_method[(backbone, m)].append(vals[m])
                by_method_flips[(backbone, m)].append(by_seed_flips[(backbone, seed)].get(m, 0))

    print(f"{'backbone':16s} {'method':14s} {'F1':>8s} {'F1_std':>7s} {'flips':>7s} {'n':>3s}")
    print("-" * 65)
    for (bb, m), f1s in sorted(by_method.items()):
        flips = by_method_flips[(bb, m)]
        print(f"{bb:16s} {m:14s} {np.mean(f1s):8.4f} {np.std(f1s):7.4f} "
              f"{np.mean(flips):7.1f} {len(f1s):>3d}")

    print()
    print(f"=== paired d_F1 + paired-t p ({label}) ===")
    print(f"{'backbone':16s} {'baseline':14s} {'d_F1':>9s} {'p':>8s} {'n':>3s} {'sig':>4s}")
    print("-" * 65)
    backbones = sorted(set(b for b, _ in by_method))
    for bb in backbones:
        tr_by_seed = {s: v["tralo"]
                      for (b, s), v in by_seed.items()
                      if b == bb and "tralo" in v}
        for bl in ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]:
            paired = []
            for seed, tr in tr_by_seed.items():
                bl_v = by_seed.get((bb, seed), {}).get(bl)
                if bl_v is not None:
                    paired.append((tr, bl_v))
            if len(paired) < 2:
                continue
            tr_arr = np.array([p[0] for p in paired])
            bl_arr = np.array([p[1] for p in paired])
            diff = tr_arr - bl_arr
            d_mean = diff.mean()
            if diff.std() > 0:
                _, p_val = stats.ttest_rel(tr_arr, bl_arr)
            else:
                p_val = float("nan")
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
                                                ("*" if p_val < 0.05 else ""))
            print(f"{bb:16s} {bl:14s} {d_mean:+9.4f} {p_val:8.4f} "
                  f"{len(paired):>3d} {sig:>4s}")


if __name__ == "__main__":
    main()
