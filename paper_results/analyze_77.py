"""Comprehensive analysis of the 77 completed paper_rerun cells.

Aggregates per (dataset, model, constrained-class, tightness, method) over seeds:
  - F1 macro, F1 of constrained class, accuracy
  - K (constraint limit), final pred count, raw pred count
  - posthoc flips, raw satisfaction rate, sat epoch
  - convergence check (TraLO/Fioretto/Hounie)

Output: human-readable tables to stdout + CSV at paper_results/analysis_77.csv.
"""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("results/pending_runs/paper_rerun")
OUT_CSV = Path("paper_results/analysis_77.csv")


def parse_eval(path):
    """Return dict of {Metric: float}. Some values are blank."""
    out = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0] == "Metric":
                continue
            k, v = row[0], row[1]
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                out[k] = v
    return out


def parse_log_last(path):
    """Return last row of training_log.csv as dict."""
    with open(path) as f:
        reader = csv.DictReader(f)
        last = None
        for row in reader:
            last = row
        return last or {}


def count_preds(pred_csv, cls):
    """Count rows where Predicted_Label == cls."""
    if not os.path.exists(pred_csv):
        return None
    n = 0
    with open(pred_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["Predicted_Label"]) == cls:
                n += 1
    return n


def collect():
    """Walk paper_rerun, return list of per-run records."""
    rows = []
    for cfg_path in ROOT.rglob("config.json"):
        d = cfg_path.parent
        eval_path = d / "evaluation_metrics.csv"
        if not eval_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        parts = d.parts
        # .../paper_rerun/{dataset}/{model}/cls_X/{tight}/{method}/seed_N
        dataset, model, cls_str, tight, method, seed = parts[-6:]
        cls = int(cls_str.replace("cls_", ""))
        ev = parse_eval(eval_path)
        log = parse_log_last(d / "training_log.csv") if (d / "training_log.csv").exists() else {}
        K = log.get(f"Limit_Class{cls}", "")
        try:
            K = int(float(K))
        except (ValueError, TypeError):
            K = None
        final_count = count_preds(d / "final_predictions.csv", cls)
        raw_count = count_preds(d / "final_predictions_raw.csv", cls)
        rows.append({
            "dataset": dataset, "model": model, "cls": cls,
            "tight": tight, "method": method, "seed": seed,
            "acc": ev.get("Accuracy"),
            "f1m": ev.get("F1 (Macro)"),
            "f1c": ev.get(f"F1_Class{cls}"),
            "K": K,
            "final_count": final_count,
            "raw_count": raw_count,
            "flips": ev.get("Flips Required"),
            "raw_excess": ev.get("Raw Total Excess"),
            "raw_satisfied": ev.get("Raw All Satisfied"),
            "sat_epoch": ev.get("Satisfaction Epoch"),
            "group_col": cfg.get("dataset_config", {}).get("group_column"),
        })
    return rows


def fmt(x, w=6, p=4):
    if x is None or x == "":
        return f"{'—':>{w}}"
    if isinstance(x, str):
        return f"{x:>{w}}"
    return f"{x:>{w}.{p}f}"


def aggregate(rows):
    """Group by (dataset, model, cls, tight, method); compute mean+std.

    K is identical for all methods of the same cell — propagate from any
    method that recorded it (typically tralo) to the others."""
    cell_K = {}
    for r in rows:
        cell = (r["dataset"], r["model"], r["cls"], r["tight"])
        if r["K"] is not None and cell not in cell_K:
            cell_K[cell] = r["K"]
    for r in rows:
        cell = (r["dataset"], r["model"], r["cls"], r["tight"])
        if r["K"] is None and cell in cell_K:
            r["K"] = cell_K[cell]
    g = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["model"], r["cls"], r["tight"], r["method"])
        g[key].append(r)
    agg = []
    for key, runs in sorted(g.items()):
        dataset, model, cls, tight, method = key

        def mu(field):
            xs = [r[field] for r in runs if r[field] not in (None, "")]
            return mean(xs) if xs else None

        def sd(field):
            xs = [r[field] for r in runs if r[field] not in (None, "")]
            return stdev(xs) if len(xs) >= 2 else 0.0

        # raw_satisfied is 0/1 -> sat rate
        sat = [r["raw_satisfied"] for r in runs
               if r["raw_satisfied"] not in (None, "")]
        sat_rate = (sum(sat) / len(sat)) if sat else None

        agg.append({
            "dataset": dataset, "model": model, "cls": cls,
            "tight": tight, "method": method, "n": len(runs),
            "K": runs[0]["K"],
            "group_col": runs[0]["group_col"],
            "acc": mu("acc"),
            "f1m_mean": mu("f1m"), "f1m_std": sd("f1m"),
            "f1c_mean": mu("f1c"), "f1c_std": sd("f1c"),
            "final_count": mu("final_count"),
            "raw_count": mu("raw_count"),
            "flips": mu("flips"),
            "raw_excess": mu("raw_excess"),
            "sat_rate": sat_rate,
            "sat_epoch": mu("sat_epoch"),
        })
    return agg


def print_main_table(agg):
    """One row per (dataset, model, cls, tight, method)."""
    hdr = (f"{'dataset':<11} {'model':<14} cls {'tight':<7} {'method':<13} "
           f"N {'K':>4} {'raw_cnt':>7} {'fin_cnt':>7} {'flips':>5} "
           f"{'sat%':>5} {'raw_exc':>7} {'F1m':>7} {'F1c':>7} {'acc':>7}")
    print(hdr)
    print("-" * len(hdr))
    last_combo = None
    for r in agg:
        combo = (r["dataset"], r["model"], r["cls"], r["tight"])
        if last_combo and combo != last_combo:
            print()
        last_combo = combo
        sat_pct = (f"{r['sat_rate']*100:.0f}"
                   if r["sat_rate"] is not None else "—")
        K = r["K"] if r["K"] is not None else "—"
        print(f"{r['dataset']:<11} {r['model']:<14} {r['cls']:<3} {r['tight']:<7} "
              f"{r['method']:<13} {r['n']} {str(K):>4} "
              f"{fmt(r['raw_count'], 7, 1)} {fmt(r['final_count'], 7, 1)} "
              f"{fmt(r['flips'], 5, 1)} "
              f"{sat_pct:>5} {fmt(r['raw_excess'], 7, 1)} "
              f"{fmt(r['f1m_mean'], 7)} {fmt(r['f1c_mean'], 7)} "
              f"{fmt(r['acc'], 7)}")


def tralo_convergence_check(rows):
    """Per-cell TraLO satisfaction summary."""
    print("\n=== TraLO convergence audit ===")
    print("Looking for any TraLO run with raw_satisfied=False or large excess")
    bad = [r for r in rows
           if r["method"] == "tralo" and r["raw_satisfied"] in (0, 0.0)]
    if not bad:
        print("✓ All TraLO runs satisfy in-training (raw_satisfied=True)")
    else:
        print(f"⚠ {len(bad)} TraLO runs failed to satisfy in-training:")
        for r in bad:
            print(f"  {r['dataset']}/{r['model']}/cls_{r['cls']}/{r['tight']}/"
                  f"seed_{r['seed']}: excess={r['raw_excess']} flips={r['flips']}")


def head_to_head_per_cell(agg):
    """For each (dataset, model, cls, tight) compute TraLO vs Fioretto vs Hounie."""
    print("\n=== Head-to-head F1m by (dataset/model/cls/tight) ===")
    by_cell = defaultdict(dict)
    for r in agg:
        cell = (r["dataset"], r["model"], r["cls"], r["tight"])
        by_cell[cell][r["method"]] = r
    print(f"{'cell':<45} {'TraLO_F1m':<12} {'Fioretto_F1m':<14} {'Hounie_F1m':<12} "
          f"{'TraLO_sat':<10} {'Hounie_sat':<10}")
    for cell, methods in sorted(by_cell.items()):
        d, m, c, t = cell
        cell_str = f"{d}/{m}/cls_{c}/{t}"
        tr = methods.get("tralo", {})
        fi = methods.get("fioretto_ldf", {})
        ho = methods.get("hounie_rcl", {})
        print(f"{cell_str:<45} "
              f"{fmt(tr.get('f1m_mean'), 12)} {fmt(fi.get('f1m_mean'), 14)} "
              f"{fmt(ho.get('f1m_mean'), 12)} "
              f"{fmt(tr.get('sat_rate'), 10, 2)} {fmt(ho.get('sat_rate'), 10, 2)}")


def global_vs_local_summary(agg):
    """Compare datasets in terms of local-constraint impact and TraLO performance."""
    print("\n=== Dataset comparison (global+local constraints) ===")
    by_ds = defaultdict(lambda: {"tralo_sat_rates": [], "tralo_f1m": [],
                                 "tralo_flips": [], "fioretto_f1m": [],
                                 "hounie_f1m": [], "group_col": ""})
    for r in agg:
        ds = r["dataset"]
        by_ds[ds]["group_col"] = r["group_col"]
        if r["method"] == "tralo" and r["sat_rate"] is not None:
            by_ds[ds]["tralo_sat_rates"].append(r["sat_rate"])
            if r["f1m_mean"] is not None:
                by_ds[ds]["tralo_f1m"].append(r["f1m_mean"])
            if r["flips"] is not None:
                by_ds[ds]["tralo_flips"].append(r["flips"])
        if r["method"] == "fioretto_ldf" and r["f1m_mean"] is not None:
            by_ds[ds]["fioretto_f1m"].append(r["f1m_mean"])
        if r["method"] == "hounie_rcl" and r["f1m_mean"] is not None:
            by_ds[ds]["hounie_f1m"].append(r["f1m_mean"])
    print(f"{'dataset':<12} {'group_col':<14} "
          f"{'TraLO_sat%':<11} {'TraLO_F1m':<11} {'TraLO_flips':<12} "
          f"{'Fioretto_F1m':<14} {'Hounie_F1m':<11}")
    for ds, s in sorted(by_ds.items()):
        sat = mean(s["tralo_sat_rates"])*100 if s["tralo_sat_rates"] else None
        trf = mean(s["tralo_f1m"]) if s["tralo_f1m"] else None
        trflip = mean(s["tralo_flips"]) if s["tralo_flips"] else None
        fif = mean(s["fioretto_f1m"]) if s["fioretto_f1m"] else None
        hof = mean(s["hounie_f1m"]) if s["hounie_f1m"] else None
        print(f"{ds:<12} {(s['group_col'] or '—'):<14} "
              f"{fmt(sat, 11, 1)} {fmt(trf, 11)} {fmt(trflip, 12, 2)} "
              f"{fmt(fif, 14)} {fmt(hof, 11)}")


def write_csv(agg, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "model", "cls", "tight", "method", "n", "K", "group_col",
              "acc", "f1m_mean", "f1m_std", "f1c_mean", "f1c_std",
              "final_count", "raw_count", "flips", "raw_excess",
              "sat_rate", "sat_epoch"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in agg:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nWrote {path}")


def main():
    rows = collect()
    print(f"Collected {len(rows)} runs across all cells\n")
    agg = aggregate(rows)
    print_main_table(agg)
    tralo_convergence_check(rows)
    head_to_head_per_cell(agg)
    global_vs_local_summary(agg)
    write_csv(agg, OUT_CSV)


if __name__ == "__main__":
    main()
