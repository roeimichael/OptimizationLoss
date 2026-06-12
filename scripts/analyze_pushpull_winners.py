"""Cross-join push-pull TraLO cells with their baseline counterparts.

For every (sweep, dataset, model, constraint_tag, constrained_class) where TraLO
ran in push_pull or transition (with low fsat), pull the baseline F1s for the
same seeds and compute paired d_F1.

Output:
  1. Per-config (sweep, ds, model, tag, cls): TraLO regime + d_F1 vs each baseline
  2. The ones where TraLO is BOTH push-pull AND winning
"""
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
            try:
                r["fsat"] = (float(r["frac_train_acc_sat"])
                             if r["frac_train_acc_sat"] else None)
            except ValueError:
                r["fsat"] = None
            rows.append(r)
    return rows


def main():
    rows = load()
    # Group F1 by (sweep, ds, model, tag, cls, method) -> {seed: f1}
    f1_lookup = defaultdict(dict)
    fsat_lookup = defaultdict(dict)
    for r in rows:
        key = (r["sweep"], r["dataset"], r["model"],
               r["constraint_tag"], r["constrained_class"])
        if r["f1"] is None or r["seed_int"] < 0:
            continue
        f1_lookup[(key, r["method"])][r["seed_int"]] = r["f1"]
        if r["method"] == "tralo" and r["fsat"] is not None:
            fsat_lookup[key][r["seed_int"]] = r["fsat"]

    # For each TraLO config: compute mean fsat across seeds, paired d_F1 vs each baseline
    print("=== TraLO cells with mean fsat < 0.85 (genuinely non-saturated) ===")
    print("Computing paired d_F1 vs each baseline.\n")

    results = []
    baselines = ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
    for key, tralo_seeds in f1_lookup.items():
        cfg_key, method = key
        if method != "tralo":
            continue
        sweep, ds, mdl, tag, cls = cfg_key
        fsats = list(fsat_lookup.get(cfg_key, {}).values())
        if not fsats:
            continue
        mean_fsat = float(np.mean(fsats))
        if mean_fsat >= 0.85:
            continue
        tralo_f1 = list(tralo_seeds.values())
        if len(tralo_f1) < 2:
            continue
        per_baseline = {}
        for bl in baselines:
            bl_seeds = f1_lookup.get((cfg_key, bl), {})
            common = sorted(set(tralo_seeds) & set(bl_seeds))
            if len(common) < 2:
                per_baseline[bl] = None
                continue
            tr = np.array([tralo_seeds[s] for s in common])
            bla = np.array([bl_seeds[s] for s in common])
            diff = tr - bla
            d = diff.mean()
            try:
                _, p = stats.ttest_rel(tr, bla)
            except Exception:
                p = float("nan")
            per_baseline[bl] = (d, p, len(common))
        results.append({
            "sweep": sweep, "ds": ds, "model": mdl, "tag": tag, "cls": cls,
            "mean_fsat": mean_fsat,
            "n_tralo": len(tralo_f1),
            "tralo_f1_mean": float(np.mean(tralo_f1)),
            "tralo_f1_std": float(np.std(tralo_f1)),
            "per_baseline": per_baseline,
        })

    # sort by mean_fsat asc, then by tralo_f1 desc
    results.sort(key=lambda x: (x["mean_fsat"], -x["tralo_f1_mean"]))

    # Header
    print(f"{'sweep':28s} {'ds':10s} {'model':16s} {'tag':9s} {'cls':>3s} "
          f"{'fsat':>5s} {'n':>3s} {'F1':>6s} "
          f"{'d_fior':>8s} {'d_hou':>8s} {'d_dan':>8s} {'d_heur':>8s}")
    print("-" * 140)
    for r in results:
        cells = []
        for bl in baselines:
            pb = r["per_baseline"].get(bl)
            if pb is None:
                cells.append("   --   ")
                continue
            d, p, n = pb
            sig = "*" if p < 0.05 else " "
            cells.append(f"{d:+7.3f}{sig}")
        print(f"{r['sweep'][:28]:28s} {r['ds']:10s} {r['model']:16s} "
              f"{r['tag']:9s} {r['cls']:>3s} {r['mean_fsat']:>5.2f} "
              f"{r['n_tralo']:>3d} {r['tralo_f1_mean']:>6.4f} "
              f"{cells[0]} {cells[1]} {cells[2]} {cells[3]}")

    # Filter: TraLO wins against ALL 4 baselines (any sig)
    print("\n\n=== Cells where TraLO BEATS the heuristic baseline (positive d) ===")
    winners = [r for r in results
               if r["per_baseline"].get("heuristic") is not None
               and r["per_baseline"]["heuristic"][0] > 0]
    print(f"Total such cells: {len(winners)}")
    for r in winners:
        d_h, p_h, n_h = r["per_baseline"]["heuristic"]
        print(f"  {r['sweep'][:28]:28s} {r['ds']:10s} {r['model']:16s} "
              f"{r['tag']:9s} cls={r['cls']:>3s}  fsat={r['mean_fsat']:.2f}  "
              f"n={r['n_tralo']:>2d}  TraLO F1={r['tralo_f1_mean']:.4f}  "
              f"d_heur={d_h:+.4f} (p={p_h:.3f})")


if __name__ == "__main__":
    main()
