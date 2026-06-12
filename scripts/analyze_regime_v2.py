"""V2: probe push-pull cells with breakdown by method, raw method names checked."""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

CSV_PATH = "scripts/_audit/saturation_audit.csv"


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            f1 = r.get("f1_macro", "")
            try:
                r["f1"] = float(f1) if f1 else None
            except ValueError:
                r["f1"] = None
            try:
                r["seed_int"] = int(r["seed"])
            except (ValueError, TypeError):
                r["seed_int"] = -1
            try:
                r["mean_ce_f"] = float(r["mean_ce"]) if r["mean_ce"] else None
            except ValueError:
                r["mean_ce_f"] = None
            rows.append(r)
    return rows


def main():
    rows = load()
    print(f"Total: {len(rows)}\n")

    # Show unique methods
    methods = sorted(set(r["method"] for r in rows))
    print(f"Distinct methods: {methods}\n")

    # Push-pull cells with method breakdown
    pp = [r for r in rows if r["regime"] == "push_pull"]
    print(f"=== {len(pp)} push_pull cells, method breakdown ===")
    by_meth = defaultdict(int)
    for r in pp:
        by_meth[r["method"]] += 1
    for m, n in sorted(by_meth.items()):
        print(f"  {m}: {n}")
    print()

    # Show each push-pull cell verbatim
    print("=== Every push_pull cell ===")
    print(f"{'method':14s} {'dataset':12s} {'model':16s} {'tag':10s} {'cls':>4s} "
          f"{'seed':>4s} {'F1':>6s} {'mean_ce':>8s} {'first_sat':>10s}")
    for r in pp:
        print(f"{r['method']:14s} {r['dataset']:12s} {r['model']:16s} "
              f"{r['constraint_tag']:10s} {r['constrained_class']:>4s} "
              f"{r['seed']:>4s} {r['f1_macro']:>6s} {r['mean_ce']:>8s} "
              f"{r['first_sat_epoch']:>10s}")

    print("\n=== push_pull_unsat cell breakdown ===")
    ppu = [r for r in rows if r["regime"] == "push_pull_unsat"]
    by_meth_ppu = defaultdict(int)
    for r in ppu:
        by_meth_ppu[r["method"]] += 1
    for m, n in sorted(by_meth_ppu.items()):
        print(f"  {m}: {n}")

    print("\n=== Every push_pull_unsat cell ===")
    for r in ppu:
        print(f"{r['method']:14s} {r['dataset']:12s} {r['model']:16s} "
              f"{r['constraint_tag']:10s} {r['constrained_class']:>4s} "
              f"{r['seed']:>4s} {r['f1_macro']:>6s} {r['mean_ce']:>8s} "
              f"frac_high={r['frac_ce_high']:>5s} frac_low={r['frac_ce_low']:>5s}")

    # CE distribution among TraLO cells
    print("\n=== TraLO mean_ce distribution (phase 2, all datasets) ===")
    tralo_ces = [r["mean_ce_f"] for r in rows
                 if r["method"] == "tralo" and r["mean_ce_f"] is not None]
    print(f"  n={len(tralo_ces)}")
    if tralo_ces:
        for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
            print(f"  q{int(q*100):>3d}: {np.quantile(tralo_ces, q):.4f}")

    # By dataset
    print("\n=== TraLO mean_ce by (dataset, model) ===")
    print(f"{'dataset':14s} {'model':16s} {'n':>4s} {'mean':>7s} {'med':>7s} "
          f"{'min':>7s} {'max':>7s}")
    by_dm = defaultdict(list)
    for r in rows:
        if r["method"] == "tralo" and r["mean_ce_f"] is not None:
            by_dm[(r["dataset"], r["model"])].append(r["mean_ce_f"])
    table = []
    for (ds, mdl), vals in by_dm.items():
        if len(vals) < 2:
            continue
        table.append((ds, mdl, len(vals), float(np.mean(vals)),
                      float(np.median(vals)), float(np.min(vals)),
                      float(np.max(vals))))
    table.sort(key=lambda x: (x[0], x[3]))
    for ds, mdl, n, m, md, mn, mx in table:
        print(f"{ds:14s} {mdl:16s} {n:>4d} {m:>7.4f} {md:>7.4f} "
              f"{mn:>7.4f} {mx:>7.4f}")


if __name__ == "__main__":
    main()
