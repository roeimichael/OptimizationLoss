"""Honest read of OctMNIST smoke + expansion results. No claims, just paired
TraLO-vs-baseline tables for whatever cells are on disk."""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def collect(root):
    rows = []
    for cfg_path in glob.glob(f"{root}/**/config.json", recursive=True):
        cell = os.path.dirname(cfg_path)
        m_path = os.path.join(cell, "evaluation_metrics.csv")
        if not os.path.exists(m_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        m = {}
        with open(m_path) as f:
            for r in csv.DictReader(f):
                m[r["Metric"]] = r["Value"]
        try:
            rows.append({
                "method": cfg["methodology"],
                "model": cfg["model_name"],
                "tight": cfg["constraint_tag"],
                "seed": cfg["hyperparams"]["seed"],
                "cclass": cfg["dataset_config"].get("constrained_class"),
                "macro_f1": float(m["F1 (Macro)"]),
                "acc": float(m["Accuracy"]),
                "flips": float(m.get("Flips Required", "nan")),
                "sat": int(m.get("Raw All Satisfied", "0") == "1"),
            })
        except (KeyError, ValueError):
            continue
    return rows


def paired(rows, baseline, key_fn):
    by = defaultdict(dict)
    for r in rows:
        by[key_fn(r)][r["method"]] = r
    out = []
    for k, by_m in by.items():
        if "tralo" not in by_m or baseline not in by_m:
            continue
        out.append({
            "key": k,
            "tralo_f1": by_m["tralo"]["macro_f1"],
            "bl_f1":    by_m[baseline]["macro_f1"],
            "d_f1":     by_m["tralo"]["macro_f1"] - by_m[baseline]["macro_f1"],
            "tralo_flips": by_m["tralo"]["flips"],
            "bl_flips":    by_m[baseline]["flips"],
            "d_flips":     by_m["tralo"]["flips"] - by_m[baseline]["flips"],
            "tralo_sat":   by_m["tralo"]["sat"],
            "bl_sat":      by_m[baseline]["sat"],
        })
    return out


def summarize(deltas, label):
    if not deltas:
        print(f"  {label}: no paired data")
        return
    d_f1 = np.array([x["d_f1"] for x in deltas])
    d_fl = np.array([x["d_flips"] for x in deltas])
    n = len(d_f1)
    w = int((d_f1 > 1e-4).sum())
    l = int((d_f1 < -1e-4).sum())
    t = n - w - l
    sat_t = sum(x["tralo_sat"] for x in deltas)
    sat_b = sum(x["bl_sat"] for x in deltas)
    print(f"  {label:25s}  n={n:3d}  d_f1 mean={d_f1.mean():+.4f} (median {np.median(d_f1):+.4f})  "
          f"W/T/L = {w}/{t}/{l}   d_flips mean={d_fl.mean():+.2f}   sat T/B = {sat_t}/{sat_b}")


def report(root, label):
    print(f"\n{'=' * 80}\n  {label}   root={root}\n{'=' * 80}")
    rows = collect(root)
    if not rows:
        print("  NO DATA")
        return
    print(f"  cells found: {len(rows)} "
          f"(methods: {sorted(set(r['method'] for r in rows))})")

    def key(r):
        return (r["model"], r["tight"], r["seed"], r["cclass"])

    print("\n  TraLO vs Hounie:")
    summarize(paired(rows, "hounie_rcl", key), "ALL")
    for tight in sorted(set(r["tight"] for r in rows)):
        sub = [r for r in rows if r["tight"] == tight]
        summarize(paired(sub, "hounie_rcl", key), f"tight={tight}")

    print("\n  TraLO vs Fioretto:")
    summarize(paired(rows, "fioretto_ldf", key), "ALL")
    for tight in sorted(set(r["tight"] for r in rows)):
        sub = [r for r in rows if r["tight"] == tight]
        summarize(paired(sub, "fioretto_ldf", key), f"tight={tight}")

    print("\n  TraLO vs heuristic + danits_lp (post-hoc baselines):")
    for bl in ("heuristic", "danits_lp"):
        summarize(paired(rows, bl, key), f"vs {bl}")

    # Per-method raw means (sanity)
    print("\n  Per-method raw means (all cells):")
    for m in sorted(set(r["method"] for r in rows)):
        sub = [r for r in rows if r["method"] == m]
        f = np.mean([r["macro_f1"] for r in sub])
        s = np.mean([r["sat"] for r in sub])
        print(f"    {m:15s}  n={len(sub):3d}  macro_f1={f:.4f}  sat_rate={s:.2f}")


if __name__ == "__main__":
    report("results/pending_runs/octmnist_smoke",     "OctMNIST SMOKE (12-cell)")
    report("results/pending_runs/octmnist_expansion", "OctMNIST EXPANSION (60-cell)")
