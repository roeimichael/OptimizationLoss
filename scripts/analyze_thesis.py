"""Aggregate thesis sweep results: mean ± std over seeds.

Output: markdown table per axis (penalty ablation, headline, tightness).

Usage:
    python scripts/analyze_thesis.py [--root results/pending_runs/thesis]
"""
import argparse
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_metrics(metrics_csv):
    out = {}
    if not metrics_csv.exists():
        return out
    with open(metrics_csv) as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def _flt(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def collect(root):
    rows = []
    for cfg_path in Path(root).rglob("config.json"):
        cfg = json.load(open(cfg_path))
        if cfg.get("status") != "completed":
            continue
        m = load_metrics(cfg_path.parent / "evaluation_metrics.csv")
        rel = cfg_path.parent.relative_to(root)
        parts = rel.parts
        axis = parts[0] if parts else ""
        scenario = parts[1] if len(parts) > 1 else ""
        results = cfg.get("results", {})
        ds = cfg.get("dataset_config", {})
        cc = ds.get("constrained_class")
        cc_list = cc if isinstance(cc, list) else [cc] if cc is not None else []
        # Per-class F1 of the constrained class(es) — average if multi-class
        per_class_f1 = []
        for c in cc_list:
            v = _flt(m.get(f"F1_Class{c}", ""))
            if v is not None:
                per_class_f1.append(v)
        rows.append({
            "axis": axis,
            "scenario": scenario,
            "rel_path": str(rel),
            "methodology": cfg.get("methodology"),
            "model": cfg.get("model_name"),
            "constraint_tag": cfg.get("constraint_tag"),
            "seed": cfg.get("hyperparams", {}).get("seed"),
            "penalty_mode": cfg.get("hyperparams", {}).get("penalty_mode"),
            "accuracy": _flt(results.get("accuracy")),
            "f1_macro": _flt(results.get("f1_macro")),
            "f1_constrained": (sum(per_class_f1) / len(per_class_f1)
                               if per_class_f1 else None),
            "samples_adjusted": int(results.get("samples_adjusted", -1)),
            "raw_excess": _int(m.get("Raw Total Excess")),
            "min_total_excess": _int(m.get("Min Total Excess")),
            "raw_all_satisfied": _int(m.get("Raw All Satisfied")),
            "ece": _flt(m.get("ECE")),
            "brier": _flt(m.get("Brier Score")),
            "mean_conf": _flt(m.get("Mean Confidence")),
            "mean_entropy": _flt(m.get("Mean Entropy")),
        })
    return rows


def fmt_mean_std(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals: return "-"
    if len(vals) == 1: return f"{vals[0]:.4f}"
    return f"{stats.mean(vals):.4f} ± {stats.stdev(vals):.4f}"


def fmt_int_mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return "-"
    if len(vals) == 1: return str(vals[0])
    return f"{stats.mean(vals):.1f}±{stats.stdev(vals):.1f}"


def aggregate_table(rows, group_keys, label):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        groups[key].append(r)
    print(f"\n## {label}")
    cols = ["F1_macro", "F1_constrained", "acc", "ECE", "Brier", "conf",
            "adj", "raw_exc", "raw_sat%"]
    print(f"\n| {' | '.join(group_keys)} | n | {' | '.join(cols)} |")
    sep = "|".join(["---"] * (len(group_keys) + 1 + len(cols)))
    print(f"|{sep}|")
    for key in sorted(groups.keys(), key=lambda x: tuple(str(v) for v in x)):
        runs = groups[key]
        f1m = fmt_mean_std([r["f1_macro"] for r in runs])
        f1c = fmt_mean_std([r["f1_constrained"] for r in runs])
        acc = fmt_mean_std([r["accuracy"] for r in runs])
        ece = fmt_mean_std([r["ece"] for r in runs])
        brier = fmt_mean_std([r["brier"] for r in runs])
        conf = fmt_mean_std([r["mean_conf"] for r in runs])
        adj = fmt_int_mean([r["samples_adjusted"] for r in runs])
        rexc = fmt_int_mean([r["raw_excess"] for r in runs if r["raw_excess"] is not None])
        sats = [r["raw_all_satisfied"] for r in runs if r["raw_all_satisfied"] is not None]
        rsat = f"{100 * sum(sats) / len(sats):.0f}%" if sats else "-"
        print(f"| {' | '.join(str(k) for k in key)} | {len(runs)} | "
              f"{f1m} | {f1c} | {acc} | {ece} | {brier} | {conf} | {adj} | {rexc} | {rsat} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/pending_runs/thesis")
    args = ap.parse_args()
    rows = collect(args.root)
    if not rows:
        print(f"No completed runs under {args.root}")
        return
    print(f"# Thesis sweep aggregation — {len(rows)} completed runs")

    # Phase A — penalty ablation
    a_rows = [r for r in rows if r["axis"] == "ablation_penalty"]
    if a_rows:
        aggregate_table(a_rows, ("penalty_mode", "model"),
                        "Phase A — Penalty form ablation (TraLO, L50_G50 class 4)")

    # Phase B — headline
    b_rows = [r for r in rows if r["axis"] == "headline"]
    if b_rows:
        aggregate_table(b_rows, ("model", "methodology"),
                        "Phase B — Headline 5-method benchmark (L50_G50 class 4)")

    # Phase C — tightness
    c_rows = [r for r in rows if r["axis"] == "tightness"]
    if c_rows:
        aggregate_table(c_rows, ("constraint_tag", "methodology"),
                        "Phase C — Tightness sweep (MobileNetV3, class 4)")

    # Phase D — extended tightness
    d_rows = [r for r in rows if r["axis"] == "tightness_ext"]
    if d_rows:
        aggregate_table(d_rows, ("constraint_tag", "methodology"),
                        "Phase D — Extended tightness (MobileNetV3, class 4)")

    # Phase E — asymmetric
    e_rows = [r for r in rows if r["axis"] == "asymmetric_ext"]
    if e_rows:
        aggregate_table(e_rows, ("constraint_tag", "methodology"),
                        "Phase E — Asymmetric (MobileNetV3, class 4)")

    # Phase F — multiclass ext
    f_rows = [r for r in rows if r["axis"] == "multiclass_ext"]
    if f_rows:
        aggregate_table(f_rows, ("scenario", "methodology"),
                        "Phase F — Multi-class (MobileNetV3, L50_G50)")


if __name__ == "__main__":
    main()
