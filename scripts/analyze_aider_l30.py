"""Quick analysis of AIDER L30_G30 rotation results.

Compares against AIDER L50 to see if the majority-class (cls 3) win persists
at tighter constraint level.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

ROOT = "results/pending_runs/aider_rotation_L30/MobileNetV3"


def read_metrics(path):
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def main():
    by_key = defaultdict(list)
    by_key_flips = defaultdict(list)
    for cell in sorted(glob.glob(f"{ROOT}/*/*/seed_*")):
        parts = cell.split("/")
        cfg, method = parts[-3], parts[-2]
        try:
            cls_idx = int(cfg.split("_")[0].replace("constrained", ""))
        except (ValueError, IndexError):
            continue
        m = read_metrics(os.path.join(cell, "evaluation_metrics.csv"))
        if not m:
            continue
        by_key[(cls_idx, method)].append(float(m.get("F1 (Macro)", 0)))
        by_key_flips[(cls_idx, method)].append(float(m.get("Flips Required", 0)))

    print(f"{'cls':>4s} {'method':14s} {'F1':>8s} {'F1_std':>7s} {'flips':>7s} {'n':>3s}")
    print("-" * 60)
    for (cls, method), f1s in sorted(by_key.items()):
        fl = by_key_flips[(cls, method)]
        print(f"{cls:>4d} {method:14s} {np.mean(f1s):8.4f} {np.std(f1s):7.4f} "
              f"{np.mean(fl):7.1f} {len(f1s):>3d}")

    print()
    print(f"{'cls':>4s} {'TR_F1':>8s} {'vs_LP+heur':>12s} {'vs_fio':>8s} {'vs_hou':>8s} {'TR_flips':>9s} {'LP_flips':>9s}")
    print("-" * 70)
    for cls in sorted(set(c for c, _ in by_key)):
        tr = by_key.get((cls, "tralo"), [])
        if not tr:
            continue
        tr_mean = np.mean(tr)
        tr_flips = np.mean(by_key_flips.get((cls, "tralo"), [0]))
        d = {}
        for bl in ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]:
            v = by_key.get((cls, bl), [])
            if v:
                d[bl] = tr_mean - np.mean(v)
        lp_heur = [d.get("danits_lp"), d.get("heuristic")]
        lp_heur = [v for v in lp_heur if v is not None]
        d_lp = np.mean(lp_heur) if lp_heur else 0
        lp_flips = np.mean(by_key_flips.get((cls, "danits_lp"), [0]))
        print(f"{cls:>4d} {tr_mean:8.4f} {d_lp:+12.4f} "
              f"{d.get('fioretto_ldf', 0):+8.4f} "
              f"{d.get('hounie_rcl', 0):+8.4f} "
              f"{tr_flips:9.1f} {lp_flips:9.1f}")


if __name__ == "__main__":
    main()
