"""Audit local constraint enforcement across all 4 methods on the 24 cells.

For each (dataset, tight, method, seed): pull global + local satisfaction
%, total excess, and group-level constrained-class counts vs limits.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict


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


def read_train_csv_last(d):
    p = d / "training_log.csv"
    if not p.exists():
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


FIX_ROOT = Path("results/pending_runs/paper400_tralofix")
BASE_ROOT = Path("results/pending_runs/paper400_baselines")
SEEDS = [1, 2, 3, 4]
TIGHTS = ["L20_G20", "L30_G30", "L50_G50"]
DATASETS = ["tissuemnist", "eurosat"]
METHODS = ["tralo_fix", "tralo", "fioretto_ldf", "hounie_rcl"]


def dir_for(method, dataset, tight, seed):
    if method == "tralo_fix":
        return FIX_ROOT / dataset / tight / f"seed_{seed}"
    return BASE_ROOT / dataset / tight / method / f"seed_{seed}"


print(f"{'Dataset':<13} {'Tight':<8} {'Method':<14} "
      f"{'Glb%':>5} {'Loc%':>5} {'Excess':>7} {'G_hard':>7} {'G0_h':>5} {'G0_K':>5} "
      f"{'G1_h':>5} {'G1_K':>5}  n")
print("-" * 100)
for dataset in DATASETS:
    for tight in TIGHTS:
        for method in METHODS:
            gpcts, lpcts, exs = [], [], []
            ghards, g0hs, g0ks, g1hs, g1ks = [], [], [], [], []
            for s in SEEDS:
                d = dir_for(method, dataset, tight, s)
                ev = read_eval(d)
                last = read_train_csv_last(d)
                if ev is None:
                    continue
                gpcts.append(_f(ev, "Raw Global Satisfied %"))
                lpcts.append(_f(ev, "Raw Local Satisfied %"))
                exs.append(_f(ev, "Raw Total Excess"))
                if last is not None:
                    ghards.append(int(float(last.get("Hard_Class4", 0))))
                    g0hs.append(int(float(last.get("Group0_Hard_Class4", 0))))
                    g0ks.append(int(float(last.get("Group0_Limit_Class4", 0))))
                    g1hs.append(int(float(last.get("Group1_Hard_Class4", 0))))
                    g1ks.append(int(float(last.get("Group1_Limit_Class4", 0))))
            if not gpcts:
                continue
            avg = lambda xs: sum(xs)/len(xs)
            print(f"{dataset:<13} {tight:<8} {method:<14} "
                  f"{avg(gpcts)*100:>5.1f} {avg(lpcts)*100:>5.1f} "
                  f"{avg(exs):>7.1f} "
                  f"{(avg(ghards) if ghards else 0):>7.1f} "
                  f"{(avg(g0hs) if g0hs else 0):>5.1f} {(avg(g0ks) if g0ks else 0):>5.1f} "
                  f"{(avg(g1hs) if g1hs else 0):>5.1f} {(avg(g1ks) if g1ks else 0):>5.1f}  "
                  f"{len(gpcts)}")
        print()
