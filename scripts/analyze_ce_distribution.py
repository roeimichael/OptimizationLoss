"""Inspect actual CE distribution across all cells with training_log.

The regime thresholds in analyze_regime.py are guesses (CE_HIGH=0.15,
CE_LOW=0.05). This script reports actual phase-2 CE distributions
so we can pick sane thresholds + understand the data.
"""
import csv
from collections import defaultdict

import numpy as np

CSV_PATH = "scripts/_audit/saturation_audit.csv"


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if not r["mean_ce"]:
                continue
            try:
                r["mean_ce_f"] = float(r["mean_ce"])
                r["min_ce_f"] = float(r["min_ce"])
                r["max_ce_f"] = float(r["max_ce"])
                r["final_ce_f"] = float(r["final_ce"])
                r["f1"] = float(r["f1_macro"]) if r["f1_macro"] else None
            except ValueError:
                continue
            rows.append(r)

    print(f"=== Cells with valid CE: {len(rows)}\n")

    # Distribution of mean_ce
    mces = [r["mean_ce_f"] for r in rows]
    print("=== mean_ce distribution (phase 2) ===")
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    for q in qs:
        print(f"  q{int(q*100):>3d}: {np.quantile(mces, q):.4f}")
    print()

    # By dataset + model
    print("=== mean_ce by (dataset, model, method=tralo only) ===")
    by_dm = defaultdict(list)
    for r in rows:
        if r["method"] != "tralo":
            continue
        by_dm[(r["dataset"], r["model"])].append(r["mean_ce_f"])
    table = []
    for (ds, mdl), vals in by_dm.items():
        if len(vals) < 2:
            continue
        table.append((ds, mdl, len(vals), np.mean(vals), np.median(vals),
                      np.min(vals), np.max(vals)))
    table.sort(key=lambda x: (x[0], -x[3]))
    print(f"{'dataset':14s} {'model':18s} {'n':>4s} {'mean':>7s} {'med':>7s} "
          f"{'min':>7s} {'max':>7s}")
    for ds, mdl, n, m, md, mn, mx in table:
        print(f"{ds:14s} {mdl:18s} {n:>4d} {m:>7.4f} {md:>7.4f} {mn:>7.4f} {mx:>7.4f}")
    print()

    # For TraLO cells: scatter mean_ce vs F1
    print("=== TraLO cells: mean_ce binned vs F1 (only datasets with >=20 cells) ===")
    by_ds = defaultdict(list)
    for r in rows:
        if r["method"] != "tralo" or r["f1"] is None:
            continue
        by_ds[r["dataset"]].append((r["mean_ce_f"], r["f1"], r["model"], r["constraint_tag"]))

    bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 5.0]
    for ds in sorted(by_ds):
        lst = by_ds[ds]
        if len(lst) < 20:
            continue
        print(f"\n  {ds} (n={len(lst)})")
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i+1]
            sub = [(ce, f1) for ce, f1, _, _ in lst if lo <= ce < hi]
            if not sub:
                continue
            mean_f1 = np.mean([f for _, f in sub])
            print(f"    CE [{lo:.2f}, {hi:.2f}):  n={len(sub):>3d}  "
                  f"mean F1={mean_f1:.4f}")


if __name__ == "__main__":
    main()
