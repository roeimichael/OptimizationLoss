"""Aggregate paired TraLO vs Fioretto vs CE on the capacity-limited TinyCNN."""
import glob
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


CONSTRAINED_CLASS = 0
DEFAULT_K = 30


def parse(fname):
    base = os.path.basename(fname).replace("_preds.npy", "")
    K = DEFAULT_K
    seed = 0
    if base.startswith("cebase"):
        kind = "cebase"
    elif base.startswith("fioretto"):
        kind = "fioretto"
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
    cells = []
    for f in sorted(glob.glob("results/*_preds.npy")):
        meta = parse(f)
        if meta is None:
            continue
        kind, K, seed = meta
        arr = np.load(f)
        y, p = arr[:, 0], arr[:, 1]
        cells.append({
            "kind": kind, "K": K, "seed": seed, "file": os.path.basename(f),
            "c0_hard": int((p == CONSTRAINED_CLASS).sum()),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        })

    print(f"\nFound {len(cells)} cells\n")
    print(f"{'kind':<10}{'K':>5}{'seed':>5}{'c0_hard':>10}{'macro_F1':>11}  file")
    print("-" * 80)
    for c in cells:
        sat = " SAT" if c["c0_hard"] <= c["K"] else " viol"
        print(f"{c['kind']:<10}{c['K']:>5}{c['seed']:>5}{c['c0_hard']:>10}"
              f"{c['macro_f1']:>11.4f}  {c['file']}{sat}")

    # --- Aggregate by (kind, K) ---
    print("\n=== AGGREGATE ===\n")
    print(f"{'kind':<10}{'K':>5}{'n':>4}{'c0_mean':>10}{'F1_mean':>10}{'F1_std':>9}{'sat':>6}")
    print("-" * 55)
    groups = defaultdict(list)
    for c in cells:
        groups[(c["kind"], c["K"])].append(c)
    for (kind, K), items in sorted(groups.items()):
        c0m = np.mean([x["c0_hard"] for x in items])
        f1m = np.mean([x["macro_f1"] for x in items])
        f1s = np.std([x["macro_f1"] for x in items])
        sat = sum(1 for x in items if x["c0_hard"] <= K)
        print(f"{kind:<10}{K:>5}{len(items):>4}{c0m:>10.1f}{f1m:>10.4f}{f1s:>9.4f}"
              f"{sat:>3}/{len(items)}")

    # --- 3-way paired (K=30 only) ---
    print("\n=== 3-WAY PAIRED (K=30, same seeds) ===\n")
    ce_d = {c["seed"]: c for c in cells if c["kind"] == "cebase"}
    tr_d = {c["seed"]: c for c in cells if c["kind"] == "tralo" and c["K"] == 30}
    fi_d = {c["seed"]: c for c in cells if c["kind"] == "fioretto"}
    triseeds = sorted(set(ce_d) & set(tr_d) & set(fi_d))
    if triseeds:
        print(f"{'seed':>5}{'CE_c0':>8}{'TR_c0':>8}{'FI_c0':>8}"
              f"{'CE_F1':>10}{'TR_F1':>10}{'FI_F1':>10}"
              f"{'d_TR':>10}{'d_FI':>10}")
        print("-" * 84)
        d_TR, d_FI = [], []
        for s in triseeds:
            dTR = tr_d[s]["macro_f1"] - ce_d[s]["macro_f1"]
            dFI = fi_d[s]["macro_f1"] - ce_d[s]["macro_f1"]
            d_TR.append(dTR)
            d_FI.append(dFI)
            print(f"{s:>5}{ce_d[s]['c0_hard']:>8}{tr_d[s]['c0_hard']:>8}"
                  f"{fi_d[s]['c0_hard']:>8}{ce_d[s]['macro_f1']:>10.4f}"
                  f"{tr_d[s]['macro_f1']:>10.4f}{fi_d[s]['macro_f1']:>10.4f}"
                  f"{dTR:>+10.4f}{dFI:>+10.4f}")
        print(f"\n  paired d_F1 (TraLO vs CE)    : mean={np.mean(d_TR):+.4f}  std={np.std(d_TR):.4f}")
        print(f"  paired d_F1 (Fioretto vs CE) : mean={np.mean(d_FI):+.4f}  std={np.std(d_FI):.4f}")
        sat_TR = sum(1 for s in triseeds if tr_d[s]["c0_hard"] <= 30)
        sat_FI = sum(1 for s in triseeds if fi_d[s]["c0_hard"] <= 30)
        print(f"  satisfaction (c0_hard ≤ 30):  TraLO {sat_TR}/{len(triseeds)}"
              f"   Fioretto {sat_FI}/{len(triseeds)}")
        c0_TR = np.mean([tr_d[s]["c0_hard"] for s in triseeds])
        c0_FI = np.mean([fi_d[s]["c0_hard"] for s in triseeds])
        c0_CE = np.mean([ce_d[s]["c0_hard"] for s in triseeds])
        print(f"  class-0 hard count mean: CE={c0_CE:.1f}  TraLO={c0_TR:.1f}  "
              f"Fioretto={c0_FI:.1f}")

    # --- K-sweep ---
    print("\n=== K-SWEEP (TraLO multi-seed aggregate) ===\n")
    print(f"{'K':>5}{'n':>4}{'c0_mean':>10}{'F1_mean':>10}{'F1_std':>9}{'sat':>6}")
    print("-" * 45)
    for K in (15, 30, 50):
        items = [c for c in cells if c["kind"] == "tralo" and c["K"] == K]
        if not items:
            continue
        c0m = np.mean([x["c0_hard"] for x in items])
        f1m = np.mean([x["macro_f1"] for x in items])
        f1s = np.std([x["macro_f1"] for x in items])
        sat = sum(1 for x in items if x["c0_hard"] <= K)
        print(f"{K:>5}{len(items):>4}{c0m:>10.1f}{f1m:>10.4f}{f1s:>9.4f}{sat:>3}/{len(items)}")


if __name__ == "__main__":
    main()
