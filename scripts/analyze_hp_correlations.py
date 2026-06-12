"""Mine HP correlations against fsat (saturation fraction).

For every (dataset, model) combo with both saturated AND non-saturated TraLO cells,
look at what HP differs and try to identify levers that push cells into push-pull.

Reads sweep configs from local CSV. Needs to read individual config.json for HP
details that aren't in the audit CSV.
"""
import csv
import json
import os
import re
import subprocess
from collections import defaultdict

import numpy as np

CSV_PATH = "scripts/_audit/saturation_audit_v2.csv"


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            try:
                r["f1"] = float(r["f1_macro"]) if r["f1_macro"] else None
            except ValueError:
                r["f1"] = None
            try:
                r["fsat_f"] = (float(r["frac_train_acc_sat"])
                               if r["frac_train_acc_sat"] else None)
            except ValueError:
                r["fsat_f"] = None
            try:
                r["mean_ce_f"] = float(r["mean_ce"]) if r["mean_ce"] else None
            except ValueError:
                r["mean_ce_f"] = None
            try:
                r["warm_int"] = int(r["warmup_epochs"]) if r["warmup_epochs"] else None
            except ValueError:
                r["warm_int"] = None
            try:
                r["constr_int"] = (int(r["constraint_epochs"])
                                   if r["constraint_epochs"] else None)
            except ValueError:
                r["constr_int"] = None
            rows.append(r)
    return rows


def main():
    rows = load()
    tralo = [r for r in rows if r["method"] == "tralo" and r["fsat_f"] is not None]

    # For each (dataset, model), bucket cells by fsat range and show their HP
    print("=== Per (dataset, model): warmup_epochs vs fsat distribution ===\n")
    by_dm = defaultdict(list)
    for r in tralo:
        by_dm[(r["dataset"], r["model"])].append(r)

    print(f"{'dataset':14s} {'model':16s} {'warmup':>6s} {'n':>4s} {'fsat':>7s} "
          f"{'fsat_std':>9s} {'fsat_min':>9s}")
    print("-" * 80)
    interest = ["dermmnist", "octmnist", "aider", "bloodmnist",
                "retinamnist", "tissuemnist"]
    for (ds, mdl), lst in sorted(by_dm.items()):
        if ds not in interest:
            continue
        if mdl == "MobileNetV3":
            continue  # user excluded
        by_warm = defaultdict(list)
        for r in lst:
            by_warm[r["warm_int"]].append(r)
        for w, sub in sorted(by_warm.items(),
                             key=lambda x: (x[0] if x[0] is not None else 999)):
            if len(sub) < 2:
                continue
            fsats = [r["fsat_f"] for r in sub if r["fsat_f"] is not None]
            print(f"{ds:14s} {mdl:16s} {str(w):>6s} {len(sub):>4d} "
                  f"{np.mean(fsats):>7.3f} {np.std(fsats):>9.3f} "
                  f"{np.min(fsats):>9.3f}")

    # For cells with very low fsat in non-tissuemnist, examine HP
    print("\n=== Low-fsat (<0.5) non-MobileNetV3 TraLO cells with HP detail ===\n")
    low_sat = [r for r in tralo if r["fsat_f"] is not None and r["fsat_f"] < 0.5
               and r["model"] != "MobileNetV3" and r["dataset"] in interest]
    print(f"n_cells: {len(low_sat)}\n")
    # Read config.json for each to get full HP
    by_combo = defaultdict(list)
    for r in low_sat:
        key = (r["sweep"], r["dataset"], r["model"], r["constraint_tag"],
               r["constrained_class"])
        by_combo[key].append(r)
    print(f"{'sweep':24s} {'ds':12s} {'model':16s} {'tag':9s} {'cls':>3s} "
          f"{'n':>3s} {'fsat':>6s} {'F1':>6s} {'warm':>5s}")
    for key, lst in sorted(by_combo.items()):
        sweep, ds, mdl, tag, cls = key
        fsats = [r["fsat_f"] for r in lst]
        f1s = [r["f1"] for r in lst if r["f1"] is not None]
        warms = [r["warm_int"] for r in lst if r["warm_int"] is not None]
        print(f"{sweep[:24]:24s} {ds:12s} {mdl:16s} {tag:9s} {cls:>3s} "
              f"{len(lst):>3d} {np.mean(fsats):>6.3f} "
              f"{np.mean(f1s) if f1s else 0:>6.3f} "
              f"{warms[0] if warms else '?':>5}")

    # For dermmnist MobileNetV2 specifically — what HP gives lowest fsat
    print("\n=== dermmnist MobileNetV2 — fsat by warmup_epochs ===")
    derm_mnv2 = [r for r in tralo
                 if r["dataset"] == "dermmnist" and r["model"] == "MobileNetV2"]
    by_warm = defaultdict(list)
    for r in derm_mnv2:
        by_warm[r["warm_int"]].append(r)
    for w, sub in sorted(by_warm.items(),
                         key=lambda x: (x[0] if x[0] is not None else 999)):
        fsats = [r["fsat_f"] for r in sub if r["fsat_f"] is not None]
        f1s = [r["f1"] for r in sub if r["f1"] is not None]
        if not fsats:
            continue
        print(f"  warmup={w:>5s}  n={len(sub):>3d}  "
              f"fsat={np.mean(fsats):.3f}±{np.std(fsats):.3f}  "
              f"F1={np.mean(f1s):.3f}" if f1s else "")

    # Read trainer.py to see if enable_ce_skip exists and how
    print("\n=== Search for enable_ce_skip references ===")
    try:
        r = subprocess.run(
            ["grep", "-rn", "enable_ce_skip", "src/"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines()[:20]:
            print(f"  {line}")
    except Exception as e:
        print(f"  grep failed: {e}")


if __name__ == "__main__":
    main()
