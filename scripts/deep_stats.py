import csv, sys
import numpy as np
from collections import defaultdict

rows = []
with open("paper/HANDOFF/tables/master_all_sweeps.csv") as f:
    for r in csv.DictReader(f):
        for k in ("f1m","f1w","acc","ece","brier","flips","sat","sat_epoch"):
            try:
                r[k] = float(r[k]) if r[k] not in ("","nan") else np.nan
            except Exception:
                r[k] = np.nan
        rows.append(r)

print("Total rows:", len(rows))
print("Sweeps:", sorted(set(r["sweep"] for r in rows)))

METHODS = ("tralo","tralo_bounded","fioretto_ldf","hounie_rcl","danits_lp","heuristic")
SHORT = {"tralo":"tralo","tralo_bounded":"tr_b","fioretto_ldf":"fior",
         "hounie_rcl":"houn","danits_lp":"dan","heuristic":"heu"}

print("\n=== Per-sweep cell counts ===")
by_sw = defaultdict(int)
for r in rows: by_sw[r["sweep"]] += 1
for sw, n in sorted(by_sw.items()): print(f"  {sw:<25}: {n}")

print("\n=== Sat% per method x sweep ===")
header = "  " + "sweep".ljust(25)
for me in METHODS: header += f" {SHORT[me]:>6}"
print(header)
by_sm = defaultdict(list)
for r in rows: by_sm[(r["sweep"], r["method"])].append(r["sat"])
for sw in sorted({s for s,_ in by_sm}):
    line = "  " + sw.ljust(25)
    for me in METHODS:
        vals = by_sm.get((sw,me), [])
        if vals: line += f" {np.mean(vals)*100:>5.0f}%"
        else:    line += "      -"
    print(line)

print("\n=== Mean Flips per method x sweep ===")
print(header)
by_sm = defaultdict(list)
for r in rows: by_sm[(r["sweep"], r["method"])].append(r["flips"])
for sw in sorted({s for s,_ in by_sm}):
    line = "  " + sw.ljust(25)
    for me in METHODS:
        vals = [v for v in by_sm.get((sw,me), []) if not np.isnan(v)]
        if vals: line += f" {np.mean(vals):>6.1f}"
        else:    line += "      -"
    print(line)

print("\n=== ECE / Brier per method (in-dist sweeps only) ===")
INDIST = ("multiclass_tissue","asym_tissue_aider","g1_mobilenetv2","paper_backbones")
ece_by = defaultdict(list); brier_by = defaultdict(list)
for r in rows:
    if r["sweep"] in INDIST:
        if not np.isnan(r["ece"]): ece_by[r["method"]].append(r["ece"])
        if not np.isnan(r["brier"]): brier_by[r["method"]].append(r["brier"])
print(f"  {'method':<16}  ECE_mean ECE_std  Brier_mean Brier_std  n")
for me in METHODS:
    e = ece_by.get(me, []); b = brier_by.get(me, [])
    if e: print(f"  {me:<16}  {np.mean(e):.4f}   {np.std(e):.4f}  {np.mean(b):.4f}      {np.std(b):.4f}    {len(e)}")

print("\n=== Median Satisfaction Epoch (convergence speed) ===")
by_m = defaultdict(list)
for r in rows:
    if r["sweep"] in INDIST and r["sat_epoch"] > 0:
        by_m[r["method"]].append(r["sat_epoch"])
for me in METHODS:
    v = by_m.get(me, [])
    if v: print(f"  {me:<16}: median={np.median(v):.0f} mean={np.mean(v):.1f}  n={len(v)}")

print("\n=== Paired W/L/T per dataset x baseline (TraLO seed-matched) ===")
for ds in ("tissuemnist","dermmnist","aider"):
    print(f"\n  --- {ds} ---")
    for sweep_label, sweep in [("multiclass","multiclass_tissue"),
                                ("asym","asym_tissue_aider")]:
        subset = [r for r in rows if r["dataset"]==ds and r["sweep"]==sweep]
        if not subset: continue
        by_cell_seed = defaultdict(dict)
        for r in subset:
            by_cell_seed[(r["cls"], r["tight"], r["seed"])][r["method"]] = r["f1m"]
        for base in ("fioretto_ldf","hounie_rcl","danits_lp","heuristic"):
            diffs = [v["tralo"] - v[base] for v in by_cell_seed.values()
                     if "tralo" in v and base in v]
            if diffs:
                wins = sum(1 for d in diffs if d > 1e-4)
                losses = sum(1 for d in diffs if d < -1e-4)
                ties = len(diffs) - wins - losses
                print(f"    {sweep_label:<10} vs {base:<14}: W/L/T={wins}/{losses}/{ties}  dF1={np.mean(diffs):+.4f}  n={len(diffs)}")

print("\n=== Headroom regression: warmup-acc proxy vs dF1 advantage ===")
cripple_sw = ("aider_cripple","derm_cripple","derm_backbone_weak")
groups = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["sweep"] in cripple_sw:
        cond = r["experiment_path"].split("/")[3] if "/" in r["experiment_path"] else "?"
        groups[(r["sweep"], cond, r["tight"])][r["method"]].append((r["f1m"], r["acc"]))
xs, ys = [], []
for key, mm in groups.items():
    if "danits_lp" not in mm: continue
    ph_acc = np.mean([a for _,a in mm["danits_lp"]])
    in_f1 = [f for me in ("tralo","fioretto_ldf") for f,_ in mm.get(me,[])]
    ph_f1 = [f for me in ("danits_lp","heuristic") for f,_ in mm.get(me,[])]
    if in_f1 and ph_f1:
        xs.append(ph_acc); ys.append(np.mean(in_f1) - np.mean(ph_f1))
xs, ys = np.array(xs), np.array(ys)
if len(xs) > 3:
    slope, intercept = np.polyfit(xs, ys, 1)
    corr = np.corrcoef(xs, ys)[0,1]
    print(f"  n={len(xs)}  slope={slope:+.3f}  intercept={intercept:+.3f}  corr_r={corr:+.3f}")
    print(f"  Interpretation: ΔF1 advantage shifts {slope*0.1:+.3f} per +0.10 in warmup acc")
EOF
