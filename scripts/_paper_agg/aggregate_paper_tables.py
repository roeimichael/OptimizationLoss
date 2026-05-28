#!/usr/bin/env python
"""Aggregate paperv2 phase2/4/5 results into per-table mean+/-std CSVs.

Writes:
  paper/tables/table_B_phase2_asymmetric_derm.csv
  paper/tables/table_D_phase4_multiclass_derm.csv
  paper/tables/table_E_phase5_sexgroup_derm.csv
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.path.expanduser("~/OptimizationLoss"))
OUT_DIR = ROOT / "paper" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map evaluation_metrics.csv "Metric" labels to internal short keys we care about
METRIC_KEYS = {
    "F1 (Macro)": "macro_f1",
    "Accuracy": "accuracy",
    "Flips Required": "flips",
    "Raw All Satisfied": "satisfied",
}

METHOD_ORDER = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]


def parse_eval_metrics(path: Path) -> dict:
    out = {}
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        next(rdr, None)  # header
        for row in rdr:
            if len(row) < 2:
                continue
            metric, val = row[0].strip(), row[1].strip()
            if metric in METRIC_KEYS:
                try:
                    out[METRIC_KEYS[metric]] = float(val)
                except ValueError:
                    pass
    return out


def load_cell(eval_csv: Path):
    """Return (config_dict, metrics_dict) or None on failure."""
    cfg_path = eval_csv.parent / "config.json"
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return None
    metrics = parse_eval_metrics(eval_csv)
    if not metrics:
        return None
    return cfg, metrics


def collect(phase_dir: Path):
    """Yield (cfg, metrics) for every evaluation_metrics.csv under phase_dir."""
    pattern = str(phase_dir / "**" / "evaluation_metrics.csv")
    for p in glob.iglob(pattern, recursive=True):
        loaded = load_cell(Path(p))
        if loaded is not None:
            yield loaded


def mean_std(vals):
    if not vals:
        return float("nan"), float("nan"), 0
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)  # sample std
    return m, math.sqrt(var), n


def aggregate(rows_by_key: dict):
    """rows_by_key: { key_tuple: { method: [ {metrics...} ] } }
    Returns flat list of dicts with key fields + aggregated metric columns.
    """
    out = []
    for key, by_method in rows_by_key.items():
        for method in METHOD_ORDER:
            seeds = by_method.get(method, [])
            f1s = [s["macro_f1"] for s in seeds if "macro_f1" in s]
            flips = [s["flips"] for s in seeds if "flips" in s]
            sats = [s["satisfied"] for s in seeds if "satisfied" in s]
            accs = [s["accuracy"] for s in seeds if "accuracy" in s]
            f1_m, f1_s, n_f1 = mean_std(f1s)
            fl_m, fl_s, n_fl = mean_std(flips)
            acc_m, acc_s, _ = mean_std(accs)
            sat_pct = (sum(sats) / len(sats) * 100.0) if sats else float("nan")
            out.append({
                **{f"k_{i}": v for i, v in enumerate(key)},
                "method": method,
                "n_seeds": n_f1,
                "macro_f1_mean": f1_m,
                "macro_f1_std": f1_s,
                "flips_mean": fl_m,
                "flips_std": fl_s,
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "satisfied_pct": sat_pct,
            })
    return out


def write_csv(rows, key_columns, out_path: Path):
    if not rows:
        print(f"[warn] no rows for {out_path}")
        return
    # Rename k_0, k_1, ... to key_columns
    header = list(key_columns) + [
        "method", "n_seeds",
        "macro_f1_mean", "macro_f1_std",
        "flips_mean", "flips_std",
        "accuracy_mean", "accuracy_std",
        "satisfied_pct",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            row = [r.get(f"k_{i}", "") for i in range(len(key_columns))]
            row += [
                r["method"], r["n_seeds"],
                f"{r['macro_f1_mean']:.6f}" if not math.isnan(r["macro_f1_mean"]) else "",
                f"{r['macro_f1_std']:.6f}" if not math.isnan(r["macro_f1_std"]) else "",
                f"{r['flips_mean']:.4f}" if not math.isnan(r["flips_mean"]) else "",
                f"{r['flips_std']:.4f}" if not math.isnan(r["flips_std"]) else "",
                f"{r['accuracy_mean']:.6f}" if not math.isnan(r["accuracy_mean"]) else "",
                f"{r['accuracy_std']:.6f}" if not math.isnan(r["accuracy_std"]) else "",
                f"{r['satisfied_pct']:.2f}" if not math.isnan(r["satisfied_pct"]) else "",
            ]
            w.writerow(row)
    print(f"[ok] wrote {out_path} ({len(rows)} rows)")


def sort_constraint_tag(tag: str):
    # L20_G30 -> (20, 30)
    try:
        parts = tag.split("_")
        return (int(parts[0][1:]), int(parts[1][1:]))
    except Exception:
        return (999, 999)


# ---------- Table B: phase2 asymmetric ----------
def table_B():
    rows_by_key = defaultdict(lambda: defaultdict(list))
    anomalies = []
    n_cells = 0
    for cfg, metrics in collect(ROOT / "results/pending_runs/paperv2_phase2"):
        tag = cfg.get("constraint_tag", "?")
        method = cfg.get("methodology", "?")
        rows_by_key[(tag,)][method].append(metrics)
        n_cells += 1
    print(f"[B] found {n_cells} eval files")
    # sort tag
    sorted_rows = aggregate(rows_by_key)
    # sort by tag then by method
    method_index = {m: i for i, m in enumerate(METHOD_ORDER)}
    sorted_rows.sort(key=lambda r: (sort_constraint_tag(r["k_0"]), method_index.get(r["method"], 99)))
    write_csv(sorted_rows, ["constraint_tag"], OUT_DIR / "table_B_phase2_asymmetric_derm.csv")
    return sorted_rows


# ---------- Table D: phase4 multiclass ----------
def table_D():
    rows_by_key = defaultdict(lambda: defaultdict(list))
    n_cells = 0
    for cfg, metrics in collect(ROOT / "results/pending_runs/paperv2_phase4"):
        cls = cfg.get("dataset_config", {}).get("constrained_class")
        tag = cfg.get("constraint_tag", "?")
        method = cfg.get("methodology", "?")
        rows_by_key[(cls, tag)][method].append(metrics)
        n_cells += 1
    print(f"[D] found {n_cells} eval files")
    sorted_rows = aggregate(rows_by_key)
    method_index = {m: i for i, m in enumerate(METHOD_ORDER)}
    sorted_rows.sort(key=lambda r: (r["k_0"] if r["k_0"] is not None else 999,
                                     sort_constraint_tag(r["k_1"]),
                                     method_index.get(r["method"], 99)))
    write_csv(sorted_rows, ["constrained_class", "constraint_tag"], OUT_DIR / "table_D_phase4_multiclass_derm.csv")
    return sorted_rows


# ---------- Table E: phase5 sex-group ----------
def table_E():
    rows_by_key = defaultdict(lambda: defaultdict(list))
    n_cells = 0
    for cfg, metrics in collect(ROOT / "results/pending_runs/paperv2_phase5"):
        tag = cfg.get("constraint_tag", "?")
        method = cfg.get("methodology", "?")
        rows_by_key[(tag,)][method].append(metrics)
        n_cells += 1
    print(f"[E] found {n_cells} eval files")
    sorted_rows = aggregate(rows_by_key)
    method_index = {m: i for i, m in enumerate(METHOD_ORDER)}
    sorted_rows.sort(key=lambda r: (sort_constraint_tag(r["k_0"]), method_index.get(r["method"], 99)))
    write_csv(sorted_rows, ["constraint_tag"], OUT_DIR / "table_E_phase5_sexgroup_derm.csv")
    return sorted_rows


# ---------- Reporting / win-rate ----------
def winrate_report(rows, key_cols, label):
    """For each unique key (a 'condition'), compare TraLO vs Fioretto and Hounie on macro_f1 and flips."""
    # group rows by key tuple
    by_cond = defaultdict(dict)
    for r in rows:
        key = tuple(r.get(f"k_{i}") for i in range(len(key_cols)))
        by_cond[key][r["method"]] = r

    n_conds = 0
    f1_vs_fior = 0
    f1_vs_houn = 0
    flips_vs_fior = 0
    flips_vs_houn = 0
    f1_vs_both = 0
    flips_vs_both = 0
    missing = []

    for key, methods in by_cond.items():
        tralo = methods.get("tralo")
        fior = methods.get("fioretto_ldf")
        houn = methods.get("hounie_rcl")
        if tralo is None or tralo["n_seeds"] == 0:
            missing.append((key, "tralo"))
            continue
        if fior is None or fior["n_seeds"] == 0:
            missing.append((key, "fioretto_ldf"))
            continue
        if houn is None or houn["n_seeds"] == 0:
            missing.append((key, "hounie_rcl"))
            continue
        n_conds += 1
        if tralo["macro_f1_mean"] > fior["macro_f1_mean"]:
            f1_vs_fior += 1
        if tralo["macro_f1_mean"] > houn["macro_f1_mean"]:
            f1_vs_houn += 1
        # lower flips is better
        if tralo["flips_mean"] < fior["flips_mean"]:
            flips_vs_fior += 1
        if tralo["flips_mean"] < houn["flips_mean"]:
            flips_vs_houn += 1
        if tralo["macro_f1_mean"] > fior["macro_f1_mean"] and tralo["macro_f1_mean"] > houn["macro_f1_mean"]:
            f1_vs_both += 1
        if tralo["flips_mean"] < fior["flips_mean"] and tralo["flips_mean"] < houn["flips_mean"]:
            flips_vs_both += 1

    print(f"\n=== {label} win-rate (n_conditions = {n_conds}) ===")
    if n_conds:
        print(f"  macro_f1: TraLO > Fioretto : {f1_vs_fior}/{n_conds}  ({100*f1_vs_fior/n_conds:.0f}%)")
        print(f"  macro_f1: TraLO > Hounie   : {f1_vs_houn}/{n_conds}  ({100*f1_vs_houn/n_conds:.0f}%)")
        print(f"  macro_f1: TraLO > BOTH     : {f1_vs_both}/{n_conds}  ({100*f1_vs_both/n_conds:.0f}%)")
        print(f"  flips   : TraLO < Fioretto : {flips_vs_fior}/{n_conds}  ({100*flips_vs_fior/n_conds:.0f}%)")
        print(f"  flips   : TraLO < Hounie   : {flips_vs_houn}/{n_conds}  ({100*flips_vs_houn/n_conds:.0f}%)")
        print(f"  flips   : TraLO < BOTH     : {flips_vs_both}/{n_conds}  ({100*flips_vs_both/n_conds:.0f}%)")
    if missing:
        print(f"  [missing]: {len(missing)} conditions missing a method-cell")
        for m in missing[:5]:
            print(f"    - {m}")

    # also count NaN seeds anomalies
    nan_rows = [r for r in rows if r["n_seeds"] == 0]
    if nan_rows:
        print(f"  [empty-method-cells]: {len(nan_rows)}")
        for r in nan_rows[:5]:
            kkey = {f"k_{i}": r.get(f"k_{i}") for i in range(len(key_cols))}
            print(f"    - {kkey} method={r['method']}")


if __name__ == "__main__":
    rb = table_B()
    rd = table_D()
    re = table_E()
    winrate_report(rb, ["constraint_tag"], "Table B (phase2 asymmetric)")
    winrate_report(rd, ["constrained_class", "constraint_tag"], "Table D (phase4 multiclass)")
    winrate_report(re, ["constraint_tag"], "Table E (phase5 sex-group)")
