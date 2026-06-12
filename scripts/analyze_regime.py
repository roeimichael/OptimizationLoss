"""Regime analysis on saturation_audit.csv.

1. Regime distribution overall + per dataset + per (dataset, model).
2. Among push-pull cells: list them; cross-join with d_F1.
3. Paired-t F1 (TraLO vs each baseline) within push-pull cells only,
   grouped by (dataset, model, constraint_tag, constrained_class).
"""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

CSV_PATH = "scripts/_audit/saturation_audit.csv"


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            try:
                r["f1"] = float(r["f1_macro"]) if r["f1_macro"] else None
            except ValueError:
                r["f1"] = None
            try:
                r["sat"] = int(r["raw_all_satisfied"]) if r["raw_all_satisfied"] else 0
            except ValueError:
                r["sat"] = 0
            try:
                r["seed_int"] = int(r["seed"])
            except (ValueError, TypeError):
                r["seed_int"] = -1
            rows.append(r)
    return rows


def hist(rows, key_fn):
    h = defaultdict(int)
    for r in rows:
        h[key_fn(r)] += 1
    return dict(h)


def main():
    rows = load()
    print(f"=== Total cells: {len(rows)}\n")

    # Regime distribution overall
    print("=== Regime distribution overall ===")
    rh = hist(rows, lambda r: r["regime"])
    for k in ["push_pull", "push_pull_unsat", "transition", "saturated", "broken"]:
        print(f"  {k:18s}: {rh.get(k, 0):>5d}")
    print()

    # By dataset
    print("=== Regime by dataset ===")
    by_ds = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_ds[r["dataset"]][r["regime"]] += 1
    for ds, h in sorted(by_ds.items()):
        total = sum(h.values())
        print(f"  {ds or '(blank)':16s}  n={total:>5d}  "
              f"push_pull={h.get('push_pull',0):>4d}  "
              f"sat={h.get('saturated',0):>4d}  "
              f"trans={h.get('transition',0):>4d}  "
              f"ppu={h.get('push_pull_unsat',0):>4d}  "
              f"brk={h.get('broken',0):>4d}")
    print()

    # By (dataset, model) — only show ones with >0 push_pull
    print("=== Push-pull cells by (dataset, model) — only nonzero ===")
    by_dm = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_dm[(r["dataset"], r["model"])][r["regime"]] += 1
    rows_dm = []
    for (ds, mdl), h in by_dm.items():
        pp = h.get("push_pull", 0)
        ppu = h.get("push_pull_unsat", 0)
        if pp + ppu > 0:
            rows_dm.append((ds, mdl, pp, ppu, sum(h.values())))
    rows_dm.sort(key=lambda x: -(x[2] + x[3]))
    for ds, mdl, pp, ppu, tot in rows_dm:
        print(f"  {ds or '?':16s} {mdl or '?':16s}  "
              f"push_pull={pp:>4d}  ppu={ppu:>4d}  total={tot:>4d}")
    print()

    # ----- TraLO d_F1 vs baselines, paired-t, WITHIN push_pull cells only -----
    print("=== Paired d_F1 (TraLO - baseline) on PUSH-PULL cells only ===")
    print("Push-pull = CE active >50% of phase 2 AND constraint satisfied at least once.")
    print()

    # Group by (dataset, model, constraint_tag, constrained_class) -> method -> {seed: f1}
    groups = defaultdict(lambda: defaultdict(dict))
    # Track regime presence per cell: only include the (cell, method) pair
    # if that method's row was push_pull. But baselines like heuristic/danits_lp
    # have no training_log -> not push_pull -> excluded.
    # SO we must define push-pull at TRALO-row level, and pair against
    # whatever baseline F1 exists for the same (cell, seed).
    cell_tralo_pp = set()  # cells where TraLO row was push_pull
    f1_by_cell_method_seed = defaultdict(dict)
    for r in rows:
        key = (r["dataset"], r["model"], r["constraint_tag"], r["constrained_class"])
        if r["f1"] is None or r["seed_int"] < 0:
            continue
        f1_by_cell_method_seed[(key, r["method"])][r["seed_int"]] = r["f1"]
        if r["method"] == "tralo" and r["regime"] == "push_pull":
            cell_tralo_pp.add(key)

    print(f"Cells where TraLO ran in push_pull regime: {len(cell_tralo_pp)}\n")

    # For each push-pull cell, paired-t TraLO vs each baseline
    if not cell_tralo_pp:
        print("NO PUSH-PULL CELLS for TraLO. Cannot compute push-pull-only stats.\n")
    else:
        baselines = ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
        results = []
        for cell in sorted(cell_tralo_pp):
            tralo_seeds = f1_by_cell_method_seed.get((cell, "tralo"), {})
            for bl in baselines:
                bl_seeds = f1_by_cell_method_seed.get((cell, bl), {})
                common = sorted(set(tralo_seeds) & set(bl_seeds))
                if len(common) < 2:
                    continue
                tr = np.array([tralo_seeds[s] for s in common])
                bla = np.array([bl_seeds[s] for s in common])
                diff = tr - bla
                d_mean = diff.mean()
                if diff.std(ddof=1) > 0:
                    _, p = stats.ttest_rel(tr, bla)
                else:
                    p = float("nan")
                results.append((cell, bl, d_mean, p, len(common)))

        # Print all of them
        print(f"{'dataset':12s} {'model':16s} {'tag':10s} {'cls':>4s} "
              f"{'baseline':14s} {'d_F1':>9s} {'p':>8s} {'n':>3s} {'sig':>4s}")
        print("-" * 90)
        for (ds, mdl, tag, cls), bl, d, p, n in results:
            sig = ""
            if p < 0.001:
                sig = "***"
            elif p < 0.01:
                sig = "**"
            elif p < 0.05:
                sig = "*"
            print(f"{ds:12s} {mdl:16s} {tag:10s} {cls:>4s} "
                  f"{bl:14s} {d:+9.4f} {p:8.4f} {n:>3d} {sig:>4s}")

        # Tally per baseline
        print("\n=== Tally (push-pull cells only) ===")
        tally = defaultdict(lambda: {"sig_win": 0, "ns_win": 0, "ns_loss": 0, "sig_loss": 0})
        for _, bl, d, p, n in results:
            sig = p < 0.05
            if d > 0 and sig:
                tally[bl]["sig_win"] += 1
            elif d > 0:
                tally[bl]["ns_win"] += 1
            elif d < 0 and sig:
                tally[bl]["sig_loss"] += 1
            else:
                tally[bl]["ns_loss"] += 1
        print(f"{'baseline':14s} {'sig_win':>8s} {'ns_win':>8s} {'ns_loss':>8s} {'sig_loss':>9s}")
        for bl, t in sorted(tally.items()):
            print(f"{bl:14s} {t['sig_win']:>8d} {t['ns_win']:>8d} "
                  f"{t['ns_loss']:>8d} {t['sig_loss']:>9d}")
    print()

    # ----- Push-pull cell details: which exact (dataset, model, tag) combos? -----
    print("=== Push-pull cell list (TraLO rows only) ===")
    pp_rows = [r for r in rows if r["method"] == "tralo" and r["regime"] == "push_pull"]
    by_combo = defaultdict(list)
    for r in pp_rows:
        key = (r["dataset"], r["model"], r["constraint_tag"], r["constrained_class"])
        by_combo[key].append(r)
    for key, lst in sorted(by_combo.items(), key=lambda x: (x[0][0], x[0][1])):
        ds, mdl, tag, cls = key
        f1s = [r["f1"] for r in lst if r["f1"] is not None]
        first_sats = [int(r["first_sat_epoch"]) for r in lst if r["first_sat_epoch"]]
        mean_ce = [float(r["mean_ce"]) for r in lst if r["mean_ce"]]
        if not f1s:
            continue
        print(f"  {ds:12s} {mdl:16s} {tag:10s} cls={cls:>3s}  "
              f"n_seeds={len(lst)}  f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
              f"mean_ce={np.mean(mean_ce):.3f}  "
              f"first_sat~{int(np.mean(first_sats)) if first_sats else '-'}")


if __name__ == "__main__":
    main()
