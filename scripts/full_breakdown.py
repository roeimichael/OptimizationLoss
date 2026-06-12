"""Honest comprehensive breakdown of overnight sweep results.

Walks ALL relevant pending_runs subdirs, reads every evaluation_metrics.csv,
groups by (sweep, ds, cls, backbone, tightness, method, seed), and reports:

1. Per-cell F1 mean ± std and n
2. Paired d_F1 (TraLO − baseline) with paired-t p-value
3. Win/tie/loss count per baseline per dataset per tightness
4. Aggregate sweep counts

Goal: stop relying on memory; print the actual computed numbers.
"""
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = "results/pending_runs"

SWEEPS = [
    "aider_rotation_full/MobileNetV3",       # 4 classes, L50, 3 seeds
    "aider_rotation_L30/MobileNetV3",        # 4 classes, L30, 3 seeds
    "derm_rotation_full/MobileNetV3",        # 3 classes, L30, 3 seeds
    "tissue_rotation_full/MobileNetV3",      # 4 classes, L30, 3 seeds
    "precision_majority",                    # aider cls3 + derm cls5 × 3 tightness × 5 seeds
    "aider_cls3_backbones",                  # 4 bb × 5 seeds × L30 (aider cls3)
    "derm_cls5_backbones",                   # 4 bb × 5 seeds × L30 (derm cls5)
    "aider_cls3_tight/MobileNetV3",          # L10, L20 × 5 seeds (aider cls3)
    "derm_cls5_tight/MobileNetV3",           # L10, L20 × 5 seeds (derm cls5)
    "tissue_cls0_tight/MobileNetV3",         # L10, L20 × 5 seeds (tissue cls0)
    "aider_cls3_l20_backbones",              # 4 bb × 5 seeds × L20 (aider cls3)
    "derm_cls5_l20_backbones",               # 4 bb × 5 seeds × L20 (derm cls5)
]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]


def read_metrics(path):
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def parse_cls(cfg):
    m = re.match(r"constrained(\d+)", cfg)
    if m:
        return int(m.group(1))
    return None


def parse_tight(s):
    m = re.search(r"L(\d+)_G(\d+)", s)
    if m:
        return f"L{m.group(1)}_G{m.group(2)}"
    return None


def infer_dataset(sweep, cell_path):
    if "tissue" in sweep:
        return "tissue"
    if "derm" in sweep:
        return "derm"
    if "aider" in sweep:
        return "aider"
    if "precision_majority" in sweep:
        # sub-folder is ds name
        parts = cell_path.split("/")
        for p in parts:
            if p in ("aider", "dermmnist", "tissuemnist"):
                return "derm" if p == "dermmnist" else ("tissue" if p == "tissuemnist" else "aider")
    return "?"


def infer_cls(sweep, cell_path):
    # cell_path is like results/pending_runs/<sweep>/<...>/seed_X
    parts = cell_path.split("/")
    # search every part for "constrained<N>"
    for p in parts:
        c = parse_cls(p)
        if c is not None:
            return c
    # fall back to sweep name hints
    if "cls3" in sweep:
        return 3
    if "cls5" in sweep:
        return 5
    if "cls0" in sweep:
        return 0
    return None


def infer_backbone(sweep, cell_path):
    parts = cell_path.split("/")
    backbones = {"MobileNetV3", "MobileNetV2", "RegNetY400MF", "ShuffleNetV2"}
    for p in parts:
        if p in backbones:
            return p
    return None


def infer_tight(sweep, cell_path):
    # try all components
    for p in cell_path.split("/"):
        t = parse_tight(p)
        if t:
            return t
    return None


def walk():
    # Returns list of rows: {sweep, ds, cls, backbone, tight, method, seed, f1, flips}
    rows = []
    for sweep in SWEEPS:
        for cell in sorted(glob.glob(f"{ROOT}/{sweep}/**/seed_*", recursive=True)):
            if not os.path.isdir(cell):
                continue
            mp = os.path.join(cell, "evaluation_metrics.csv")
            m = read_metrics(mp)
            if not m:
                continue
            seed_str = cell.split("/")[-1]
            try:
                seed = int(seed_str.replace("seed_", ""))
            except ValueError:
                continue
            method = cell.split("/")[-2]
            if method not in METHODS:
                continue
            rel = cell[len(ROOT)+1:]
            rows.append({
                "sweep": sweep,
                "ds": infer_dataset(sweep, cell),
                "cls": infer_cls(sweep, cell),
                "backbone": infer_backbone(sweep, cell),
                "tight": infer_tight(sweep, cell),
                "method": method,
                "seed": seed,
                "f1": float(m.get("F1 (Macro)", 0)),
                "flips": float(m.get("Flips Required", 0)),
                "raw_sat": m.get("Raw All Satisfied", "0") == "1",
                "cell": rel,
            })
    return rows


def main():
    rows = walk()
    print(f"Total result cells loaded: {len(rows)}\n")
    print("Sweep-level counts:")
    by_sweep = defaultdict(int)
    for r in rows:
        by_sweep[r["sweep"]] += 1
    for s, n in sorted(by_sweep.items()):
        print(f"  {n:4d}  {s}")
    print()

    # Build per-cell aggregates (ds, cls, backbone, tight, method) -> list of (seed, f1)
    grouped = defaultdict(list)
    for r in rows:
        if None in (r["ds"], r["cls"], r["backbone"], r["tight"]):
            continue
        key = (r["ds"], r["cls"], r["backbone"], r["tight"], r["method"])
        grouped[key].append((r["seed"], r["f1"]))

    # Per-config breakdown
    print("\n" + "="*100)
    print("PER-CONFIG F1 MEANS (ds, cls, backbone, tight, method)")
    print("="*100)
    print(f"{'ds':6s} {'cls':>3s} {'backbone':12s} {'tight':10s} {'method':14s} {'F1':>8s} {'std':>7s} {'n':>3s}")
    print("-" * 80)
    cell_keys = sorted({(d, c, b, t) for (d, c, b, t, _) in grouped})
    for (d, c, b, t) in cell_keys:
        for m in METHODS:
            vals = grouped.get((d, c, b, t, m), [])
            if not vals:
                continue
            f1s = [v for _, v in vals]
            print(f"{d:6s} {c:>3d} {b:12s} {t:10s} {m:14s} {np.mean(f1s):8.4f} "
                  f"{np.std(f1s):7.4f} {len(f1s):>3d}")
        print()

    # Paired d_F1 + paired-t p per cell
    print("\n" + "="*100)
    print("PAIRED d_F1 (TraLO − baseline) WITH PAIRED-t p-VALUE")
    print("="*100)
    print(f"{'ds':6s} {'cls':>3s} {'backbone':12s} {'tight':10s} {'baseline':14s} "
          f"{'d_F1':>9s} {'p':>8s} {'n':>3s} {'sig':>4s}")
    print("-" * 90)
    # win/tie/loss tally by (baseline, ds)
    tally = defaultdict(lambda: {"win_sig": 0, "win_ns": 0, "tie_zero": 0,
                                  "loss_ns": 0, "loss_sig": 0})
    for (d, c, b, t) in cell_keys:
        tr = {s: f for s, f in grouped.get((d, c, b, t, "tralo"), [])}
        for bl in ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]:
            blv = {s: f for s, f in grouped.get((d, c, b, t, bl), [])}
            common = sorted(set(tr) & set(blv))
            if len(common) < 2:
                continue
            tr_arr = np.array([tr[s] for s in common])
            bl_arr = np.array([blv[s] for s in common])
            diff = tr_arr - bl_arr
            d_mean = diff.mean()
            if diff.std(ddof=1) > 0:
                _, p_val = stats.ttest_rel(tr_arr, bl_arr)
            else:
                p_val = float("nan")
            sig = ""
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            print(f"{d:6s} {c:>3d} {b:12s} {t:10s} {bl:14s} "
                  f"{d_mean:+9.4f} {p_val:8.4f} {len(common):>3d} {sig:>4s}")
            # tally
            if d_mean > 0 and sig:
                tally[(bl, d)]["win_sig"] += 1
            elif d_mean > 0:
                tally[(bl, d)]["win_ns"] += 1
            elif d_mean == 0:
                tally[(bl, d)]["tie_zero"] += 1
            elif d_mean < 0 and sig:
                tally[(bl, d)]["loss_sig"] += 1
            else:
                tally[(bl, d)]["loss_ns"] += 1
        print()

    # Win/tie/loss tally
    print("\n" + "="*100)
    print("WIN / TIE / LOSS TALLY (TraLO vs baseline) PER (BASELINE × DATASET)")
    print("="*100)
    print(f"{'ds':10s} {'baseline':14s} {'sig_win':>8s} {'ns_win':>8s} {'tie':>5s} "
          f"{'ns_loss':>8s} {'sig_loss':>9s} {'total':>6s}")
    print("-" * 80)
    for (bl, d), t in sorted(tally.items()):
        total = sum(t.values())
        print(f"{d:10s} {bl:14s} {t['win_sig']:>8d} {t['win_ns']:>8d} {t['tie_zero']:>5d} "
              f"{t['loss_ns']:>8d} {t['loss_sig']:>9d} {total:>6d}")

    # Per-baseline overall
    print("\n" + "="*100)
    print("WIN / TIE / LOSS TALLY (TraLO vs baseline) — OVERALL ACROSS ALL CELLS")
    print("="*100)
    print(f"{'baseline':14s} {'sig_win':>8s} {'ns_win':>8s} {'tie':>5s} "
          f"{'ns_loss':>8s} {'sig_loss':>9s} {'total':>6s}")
    print("-" * 70)
    bl_totals = defaultdict(lambda: {"win_sig": 0, "win_ns": 0, "tie_zero": 0,
                                     "loss_ns": 0, "loss_sig": 0})
    for (bl, d), t in tally.items():
        for k, v in t.items():
            bl_totals[bl][k] += v
    for bl, t in sorted(bl_totals.items()):
        total = sum(t.values())
        print(f"{bl:14s} {t['win_sig']:>8d} {t['win_ns']:>8d} {t['tie_zero']:>5d} "
              f"{t['loss_ns']:>8d} {t['loss_sig']:>9d} {total:>6d}")


if __name__ == "__main__":
    main()
