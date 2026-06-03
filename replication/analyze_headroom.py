"""Aggregate paired TraLO vs CE results on the capacity-limited TinyCNN.

Reads all *_preds.npy in results/. Each file is shape (n_test, 2): col0 y_true,
col1 y_pred. Groups by arm type + K + seed. Computes per-cell class-0 hard
count + macro F1. Reports paired diff (TraLO - CE) for matched seeds.
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


CONSTRAINED_CLASS = 0
DEFAULT_K = 30


def parse(fname):
    """tralo_1030_preds.npy        -> ("tralo", K=30, seed=0)
       tralo_s1_1104_preds.npy     -> ("tralo", K=30, seed=1)
       cebase_s0_1104_preds.npy    -> ("cebase", K=None, seed=0)
       tralo_K15_s0_1128_preds.npy -> ("tralo", K=15, seed=0)"""
    base = os.path.basename(fname).replace("_preds.npy", "")
    K = DEFAULT_K
    seed = 0
    if base.startswith("cebase"):
        kind = "cebase"
    elif base.startswith("tralo"):
        kind = "tralo"
    else:
        return None
    m = re.search(r"K(\d+)", base)
    if m:
        K = int(m.group(1))
    m = re.search(r"_s(\d+)_", base)
    if m:
        seed = int(m.group(1))
    elif base == "tralo_1030":
        seed = 0
    return kind, K, seed


def main():
    results_dir = "results"
    cells = []
    for f in sorted(glob.glob(f"{results_dir}/*_preds.npy")):
        meta = parse(f)
        if meta is None:
            continue
        kind, K, seed = meta
        arr = np.load(f)
        y, p = arr[:, 0], arr[:, 1]
        c0_hard = int((p == CONSTRAINED_CLASS).sum())
        f1 = f1_score(y, p, average="macro", zero_division=0)
        cells.append({
            "kind": kind, "K": K, "seed": seed, "file": os.path.basename(f),
            "c0_hard": c0_hard, "macro_f1": f1,
        })

    print(f"\nFound {len(cells)} cells:\n")
    print(f"{'kind':<8}{'K':>5}{'seed':>5}{'c0_hard':>10}{'macro_F1':>11}  file")
    print("-" * 80)
    for c in cells:
        sat = " SAT" if c["c0_hard"] <= c["K"] else " viol"
        print(f"{c['kind']:<8}{c['K']:>5}{c['seed']:>5}{c['c0_hard']:>10}"
              f"{c['macro_f1']:>11.4f}  {c['file']}{sat}")

    # --- Aggregate by (kind, K) ---
    print("\n=== AGGREGATE BY (kind, K) ===\n")
    print(f"{'kind':<8}{'K':>5}{'n':>4}{'c0_mean':>10}{'F1_mean':>10}{'F1_std':>9}")
    print("-" * 50)
    groups = defaultdict(list)
    for c in cells:
        groups[(c["kind"], c["K"])].append(c)
    for (kind, K), items in sorted(groups.items()):
        c0_mean = np.mean([x["c0_hard"] for x in items])
        f1_mean = np.mean([x["macro_f1"] for x in items])
        f1_std = np.std([x["macro_f1"] for x in items])
        print(f"{kind:<8}{K:>5}{len(items):>4}{c0_mean:>10.1f}{f1_mean:>10.4f}{f1_std:>9.4f}")

    # --- Paired diffs (TraLO K=30 vs CE, matched seeds) ---
    print("\n=== PAIRED: TraLO (K=30) vs CE (same seed) ===\n")
    ce = {c["seed"]: c for c in cells if c["kind"] == "cebase"}
    tr = {c["seed"]: c for c in cells if c["kind"] == "tralo" and c["K"] == 30}
    paired = sorted(set(ce) & set(tr))
    if not paired:
        print("  no paired seeds with both TraLO K=30 + CE")
    else:
        print(f"{'seed':>5}{'CE_c0':>8}{'TR_c0':>8}{'d_c0':>8}"
              f"{'CE_F1':>10}{'TR_F1':>10}{'d_F1':>9}")
        print("-" * 60)
        dF1 = []
        for s in paired:
            dc0 = tr[s]["c0_hard"] - ce[s]["c0_hard"]
            df = tr[s]["macro_f1"] - ce[s]["macro_f1"]
            dF1.append(df)
            print(f"{s:>5}{ce[s]['c0_hard']:>8}{tr[s]['c0_hard']:>8}"
                  f"{dc0:>+8}{ce[s]['macro_f1']:>10.4f}"
                  f"{tr[s]['macro_f1']:>10.4f}{df:>+9.4f}")
        dF1 = np.array(dF1)
        print(f"\n  paired d_F1: mean={dF1.mean():+.4f}  std={dF1.std():.4f}  n={len(dF1)}")
        sat_count = sum(1 for s in paired if tr[s]["c0_hard"] <= 30)
        print(f"  TraLO satisfaction: {sat_count}/{len(paired)} seeds")

    # --- K-sweep ---
    print("\n=== K-SWEEP (TraLO only, seed 0) ===\n")
    tr_all = [c for c in cells if c["kind"] == "tralo" and c["seed"] == 0]
    if len(tr_all) >= 2:
        print(f"{'K':>5}{'c0_hard':>10}{'macro_F1':>11}{'satisfied':>11}")
        print("-" * 45)
        for c in sorted(tr_all, key=lambda x: x["K"]):
            sat = "yes" if c["c0_hard"] <= c["K"] else "no"
            print(f"{c['K']:>5}{c['c0_hard']:>10}{c['macro_f1']:>11.4f}{sat:>11}")


if __name__ == "__main__":
    main()
