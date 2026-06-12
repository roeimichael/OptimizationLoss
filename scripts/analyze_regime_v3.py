"""V3: analyze TraLO-focused regimes using v2 audit (Train_Acc directly)."""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

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
                r["seed_int"] = int(r["seed"])
            except (ValueError, TypeError):
                r["seed_int"] = -1
            for k in ["mean_train_acc", "frac_train_acc_sat",
                      "max_train_acc", "final_train_acc",
                      "mean_ce", "min_ce", "frac_ce_high", "frac_ce_dead"]:
                try:
                    r[k + "_f"] = float(r[k]) if r[k] else None
                except ValueError:
                    r[k + "_f"] = None
            rows.append(r)
    return rows


def main():
    rows = load()
    print(f"Total: {len(rows)}\n")

    # ----- TraLO regime distribution -----
    tralo = [r for r in rows if r["method"] == "tralo"]
    print(f"=== TraLO cells: {len(tralo)} ===\n")

    rh = defaultdict(int)
    for r in tralo:
        rh[r["regime"]] += 1
    print("TraLO regime distribution:")
    for k in ["push_pull", "push_pull_unsat", "transition",
              "saturated", "broken", "no_log"]:
        print(f"  {k:18s}: {rh.get(k, 0):>5d}")
    print()

    # TraLO frac_train_acc_sat distribution
    sats = [r["frac_train_acc_sat_f"] for r in tralo
            if r["frac_train_acc_sat_f"] is not None]
    print(f"TraLO frac_train_acc_sat distribution (n={len(sats)}):")
    for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
        print(f"  q{int(q*100):>3d}: {np.quantile(sats, q):.4f}")
    print()

    # By (dataset, model) - TraLO mean_frac_sat
    print("=== TraLO frac_train_acc_sat by (dataset, model) ===")
    by_dm = defaultdict(list)
    for r in tralo:
        if r["frac_train_acc_sat_f"] is not None:
            by_dm[(r["dataset"], r["model"])].append(r["frac_train_acc_sat_f"])
    table = []
    for (ds, mdl), vals in by_dm.items():
        if len(vals) < 2:
            continue
        table.append((ds, mdl, len(vals), float(np.mean(vals)),
                      float(np.median(vals)), float(np.min(vals)),
                      float(np.max(vals))))
    table.sort(key=lambda x: (x[0], x[3]))
    print(f"{'dataset':14s} {'model':18s} {'n':>4s} {'mean':>7s} {'med':>7s} "
          f"{'min':>7s} {'max':>7s}  <-- lower = less saturation = push-pull")
    for ds, mdl, n, m, md, mn, mx in table:
        print(f"{ds:14s} {mdl:18s} {n:>4d} {m:>7.4f} {md:>7.4f} "
              f"{mn:>7.4f} {mx:>7.4f}")
    print()

    # ----- TraLO push-pull cells with paired d_F1 -----
    pp = [r for r in tralo if r["regime"] == "push_pull"]
    print(f"=== TraLO push_pull cells: {len(pp)} ===")
    if pp:
        print(f"{'dataset':12s} {'model':18s} {'tag':10s} {'cls':>4s} "
              f"{'seed':>4s} {'F1':>6s} {'fsat':>6s} {'mean_ta':>8s} "
              f"{'mean_ce':>8s}")
        for r in sorted(pp, key=lambda x: (x["dataset"], x["model"])):
            print(f"{r['dataset']:12s} {r['model']:18s} "
                  f"{r['constraint_tag']:10s} {r['constrained_class']:>4s} "
                  f"{r['seed']:>4s} {r['f1_macro']:>6s} "
                  f"{r['frac_train_acc_sat']:>6s} {r['mean_train_acc']:>8s} "
                  f"{r['mean_ce']:>8s}")
    print()

    # ----- TraLO push_pull_unsat (failed but in correct regime) -----
    ppu = [r for r in tralo if r["regime"] == "push_pull_unsat"]
    print(f"=== TraLO push_pull_unsat cells: {len(ppu)} ===")
    if ppu:
        print(f"{'dataset':12s} {'model':18s} {'tag':10s} {'cls':>4s} "
              f"{'seed':>4s} {'F1':>6s} {'fsat':>6s} {'mean_ta':>8s} "
              f"{'mean_ce':>8s}")
        for r in sorted(ppu, key=lambda x: (x["dataset"], x["model"])):
            print(f"{r['dataset']:12s} {r['model']:18s} "
                  f"{r['constraint_tag']:10s} {r['constrained_class']:>4s} "
                  f"{r['seed']:>4s} {r['f1_macro']:>6s} "
                  f"{r['frac_train_acc_sat']:>6s} {r['mean_train_acc']:>8s} "
                  f"{r['mean_ce']:>8s}")
    print()

    # ----- Transition cells with LOWEST saturation (closest to push-pull) -----
    trans = [r for r in tralo if r["regime"] == "transition"
             and r["frac_train_acc_sat_f"] is not None]
    trans.sort(key=lambda r: r["frac_train_acc_sat_f"])
    print(f"=== 30 LEAST-saturated TraLO transition cells ===")
    print(f"{'dataset':12s} {'model':18s} {'tag':10s} {'cls':>4s} "
          f"{'seed':>4s} {'F1':>6s} {'fsat':>6s} {'mean_ta':>8s}")
    for r in trans[:30]:
        print(f"{r['dataset']:12s} {r['model']:18s} "
              f"{r['constraint_tag']:10s} {r['constrained_class']:>4s} "
              f"{r['seed']:>4s} {r['f1_macro']:>6s} "
              f"{r['frac_train_acc_sat']:>6s} {r['mean_train_acc']:>8s}")


if __name__ == "__main__":
    main()
