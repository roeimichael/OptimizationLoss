"""Compare paper400_tralofix (new) vs paper400 (old TraLO + Fior + Hounie).

For each cell (dataset, tightness, seed), pull metrics from:
  - paper400_tralofix/<dataset>/<tight>/seed_X     (the v3 fix variant)
  - paper400/<dataset>/MobileNetV3/cls_4/<tight>/<method>/seed_X
    for method in tralo, fioretto_ldf, hounie_rcl

Output: per-cell side-by-side and mean-over-seeds summary.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict


FIX_ROOT = Path("results/pending_runs/paper400_tralofix")
OLD_ROOT = Path("results/pending_runs/paper400_baselines")
MODEL = "MobileNetV3"
CLS = 4


def read_eval(d):
    p = d / "evaluation_metrics.csv"
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for row in rdr:
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def _f(d, k, default=0.0):
    v = (d or {}).get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fix_dir(dataset, tight, seed):
    return FIX_ROOT / dataset / tight / f"seed_{seed}"


def old_dir(dataset, tight, seed, method):
    # paper400_baselines layout: <root>/<dataset>/<tight>/<method>/seed_X
    return OLD_ROOT / dataset / tight / method / f"seed_{seed}"


SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
# Auto-discover tightness levels actually present on disk (sorted by L then G)
def _discover_tights():
    found = set()
    for d in (FIX_ROOT.iterdir() if FIX_ROOT.exists() else []):
        if not d.is_dir():
            continue
        for t in d.iterdir():
            if t.is_dir() and t.name.startswith("L"):
                found.add(t.name)
    def sort_key(tag):
        try:
            L = int(tag.split("_")[0][1:])
            G = int(tag.split("_")[1][1:])
        except Exception:
            return (999, 999)
        return (L, G)
    return sorted(found, key=sort_key)


TIGHTS = _discover_tights() or ["L20_G20", "L30_G30", "L50_G50"]
DATASETS = ["tissuemnist", "eurosat"]
METHODS_OLD = ["tralo", "fioretto_ldf", "hounie_rcl"]

# Per-cell aggregation (mean over seeds).
print(f"{'Dataset':<13} {'Tight':<8} {'Method':<14} {'F1_M':>7} {'F1_C4':>7} "
      f"{'FLIPS':>6} {'RAW_EX':>7} {'n':>3}")
print("-" * 78)
summary = []
for dataset in DATASETS:
    for tight in TIGHTS:
        # tralofix
        f1s, f1cs, flips, rexs = [], [], [], []
        for s in SEEDS:
            ev = read_eval(fix_dir(dataset, tight, s))
            if ev is None:
                continue
            f1s.append(_f(ev, "F1 (Macro)"))
            f1cs.append(_f(ev, "F1_Class4"))
            flips.append(_f(ev, "Flips Required"))
            rexs.append(_f(ev, "Raw Total Excess"))
        if f1s:
            row = ("tralo_fix",
                   sum(f1s)/len(f1s), sum(f1cs)/len(f1cs),
                   sum(flips)/len(flips), sum(rexs)/len(rexs), len(f1s))
            summary.append((dataset, tight, *row))
            print(f"{dataset:<13} {tight:<8} {row[0]:<14} {row[1]:>7.4f} "
                  f"{row[2]:>7.4f} {row[3]:>6.1f} {row[4]:>7.1f} {row[5]:>3}")
        # old methods
        for method in METHODS_OLD:
            f1s, f1cs, flips, rexs = [], [], [], []
            for s in SEEDS:
                ev = read_eval(old_dir(dataset, tight, s, method))
                if ev is None:
                    continue
                f1s.append(_f(ev, "F1 (Macro)"))
                f1cs.append(_f(ev, "F1_Class4"))
                flips.append(_f(ev, "Flips Required"))
                rexs.append(_f(ev, "Raw Total Excess"))
            if f1s:
                row = (method, sum(f1s)/len(f1s), sum(f1cs)/len(f1cs),
                       sum(flips)/len(flips), sum(rexs)/len(rexs), len(f1s))
                summary.append((dataset, tight, *row))
                print(f"{dataset:<13} {tight:<8} {row[0]:<14} {row[1]:>7.4f} "
                      f"{row[2]:>7.4f} {row[3]:>6.1f} {row[4]:>7.1f} {row[5]:>3}")
        print()


# Pivoted summary: F1 lift per cell
print("\n=== F1_Macro lift: tralo_fix - tralo (baseline) ===")
by_cell = defaultdict(dict)
for r in summary:
    dataset, tight, method, f1, f1c, flips, rex, n = r
    by_cell[(dataset, tight)][method] = (f1, f1c, flips)

print(f"{'Dataset':<13} {'Tight':<8} "
      f"{'tralo_fix F1':>13} {'tralo F1':>10} {'fior F1':>10} {'hounie F1':>10} "
      f"{'fix-tralo':>10} {'fix-fior':>10}")
print("-" * 100)
for (dataset, tight), m in by_cell.items():
    fix = m.get("tralo_fix", (None,)*3)
    tr = m.get("tralo", (None,)*3)
    fi = m.get("fioretto_ldf", (None,)*3)
    ho = m.get("hounie_rcl", (None,)*3)
    if fix[0] is None:
        continue
    line = (f"{dataset:<13} {tight:<8} {fix[0]:>13.4f} "
            f"{(tr[0] if tr[0] else 0):>10.4f} "
            f"{(fi[0] if fi[0] else 0):>10.4f} "
            f"{(ho[0] if ho[0] else 0):>10.4f}")
    if tr[0] is not None:
        line += f" {fix[0]-tr[0]:>+10.4f}"
    else:
        line += " " * 11
    if fi[0] is not None:
        line += f" {fix[0]-fi[0]:>+10.4f}"
    print(line)
