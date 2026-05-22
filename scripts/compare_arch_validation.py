"""Compare Blackwell (arch_validation) vs Turing (paper400 + paper400_tralofix).

For each (dataset, tightness, method, seed) cell:
  - Turing F1 macro + flips from paper400 or paper400_tralofix
  - Blackwell F1 macro + flips from arch_validation
  - Delta and direction (does the per-cell ranking hold?)

Prints a per-cell table and a directional summary.
"""
import csv
from pathlib import Path
from collections import defaultdict


def read_eval(p):
    out = {}
    if not p.exists():
        return None
    with open(p) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def metric(d, key, default=0.0):
    if d is None:
        return None
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return None


# ---- Collect Blackwell ----
black = {}  # (ds, tight, method, seed) -> (f1, flips)
for p in Path("results/pending_runs/arch_validation").rglob("evaluation_metrics.csv"):
    parts = p.parts
    # arch_validation/<ds>/<tight>/<method>/<seed>/eval...
    ds, tight, method, seed_dir = parts[-5], parts[-4], parts[-3], parts[-2]
    seed = int(seed_dir.split("_")[1])
    m = read_eval(p)
    black[(ds, tight, method, seed)] = (metric(m, "F1 (Macro)"),
                                         metric(m, "Posthoc_Flips"))


# ---- Collect Turing (from paper400 + tralofix dirs) ----
turing = {}
# tralo (breakthrough): paper400_tralofix/<ds>/<tight>/seed_<n>/
for p in Path("results/pending_runs/paper400_tralofix").rglob("evaluation_metrics.csv"):
    parts = p.parts
    if "seed_" not in parts[-2]:
        continue
    ds, tight, seed_dir = parts[-4], parts[-3], parts[-2]
    seed = int(seed_dir.split("_")[1])
    m = read_eval(p)
    turing[(ds, tight, "tralo", seed)] = (metric(m, "F1 (Macro)"),
                                            metric(m, "Posthoc_Flips"))

# vanilla + fior + hounie: paper400_baselines/<ds>/<tight>/<method>/seed_<n>/
for p in Path("results/pending_runs/paper400_baselines").rglob("evaluation_metrics.csv"):
    parts = p.parts
    ds, tight, method, seed_dir = parts[-5], parts[-4], parts[-3], parts[-2]
    seed = int(seed_dir.split("_")[1])
    m = read_eval(p)
    # paper400_baselines used old name "tralo" for vanilla (now called tralo_bounded)
    if method == "tralo":
        method = "tralo_bounded"
    turing[(ds, tight, method, seed)] = (metric(m, "F1 (Macro)"),
                                          metric(m, "Posthoc_Flips"))


# ---- Per-cell diff ----
print("Per-cell Blackwell vs Turing (seeds 1-2):")
print()
print("%-12s %-9s %-15s %-5s %12s %12s %12s %10s" % (
    "Dataset", "Tight", "Method", "Seed",
    "Turing F1", "Black F1", "Turing flips", "Black flips"))
print("-" * 95)

cells = sorted(set(black.keys()) | set(turing.keys()))
for k in cells:
    ds, t, m, s = k
    tu = turing.get(k, (None, None))
    bl = black.get(k, (None, None))
    tu_f, tu_fl = tu
    bl_f, bl_fl = bl
    print("%-12s %-9s %-15s %-5d %12s %12s %12s %10s" % (
        ds, t, m, s,
        f"{tu_f:.4f}" if tu_f is not None else "--",
        f"{bl_f:.4f}" if bl_f is not None else "--",
        f"{tu_fl:.0f}" if tu_fl is not None else "--",
        f"{bl_fl:.0f}" if bl_fl is not None else "--"))


# ---- Direction check (per-cell ranking) ----
print()
print("Per-cell method ranking: does Blackwell preserve Turing's order?")
print()
methods = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl"]
for ds in sorted({k[0] for k in cells}):
    for t in sorted({k[1] for k in cells if k[0] == ds}):
        for s in sorted({k[3] for k in cells if k[0] == ds and k[1] == t}):
            tu_f1 = {m: turing.get((ds, t, m, s), (None, None))[0] for m in methods}
            bl_f1 = {m: black.get((ds, t, m, s), (None, None))[0] for m in methods}
            if all(v is not None for v in tu_f1.values()) and all(v is not None for v in bl_f1.values()):
                tu_rank = sorted(methods, key=lambda m: -tu_f1[m])
                bl_rank = sorted(methods, key=lambda m: -bl_f1[m])
                same = "MATCH" if tu_rank == bl_rank else "DIFFER"
                print(f"  {ds} {t} seed{s}: Turing {tu_rank} | Blackwell {bl_rank} | {same}")
            else:
                print(f"  {ds} {t} seed{s}: INCOMPLETE (Turing or Blackwell missing)")
