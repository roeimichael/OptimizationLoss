"""Quick analysis of completed sweep experiments.

Aggregates evaluation_metrics.csv + config.json across a results tree and
produces a markdown table grouped by axis -> scenario -> methodology.

Usage:
    python scripts/analyze_sweep.py [--root results/pending_runs/overnight_sweep]
"""
import argparse
import csv
import json
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


def collect(root):
    rows = []
    for cfg_path in Path(root).rglob("config.json"):
        cfg = json.load(open(cfg_path))
        if cfg.get("status") != "completed":
            continue
        m = load_metrics(cfg_path.parent / "evaluation_metrics.csv")
        try:
            rel = cfg_path.parent.relative_to(root)
            parts = rel.parts
            axis = parts[0] if len(parts) > 0 else ""
            scenario = parts[1] if len(parts) > 1 else ""
            methodology = parts[2] if len(parts) > 2 else parts[-1]
        except ValueError:
            axis, scenario, methodology = "", "", str(cfg_path.parent)
        results = cfg.get("results", {})
        rows.append({
            "axis": axis,
            "scenario": scenario,
            "methodology": cfg.get("methodology", methodology),
            "model": cfg.get("model_name"),
            "constraint_tag": cfg.get("constraint_tag"),
            "constrained_classes": cfg.get("dataset_config", {}).get("constrained_class"),
            "accuracy": float(results.get("accuracy", float("nan"))),
            "f1_macro": float(results.get("f1_macro", float("nan"))),
            "samples_adjusted": int(results.get("samples_adjusted", -1)),
            "satisfaction_epoch": m.get("satisfaction_epoch", "-"),
            "best_sat_epoch": m.get("best_sat_epoch", "-"),
            "restored_from_epoch": m.get("restored_from_epoch", "-"),
            "raw_satisfied": m.get("raw_all_satisfied", "-"),
            "raw_excess": m.get("raw_total_excess", "-"),
            "ece": float(m.get("ECE", "nan")) if "ECE" in m else float("nan"),
            "training_time_s": float(results.get("training_time", 0.0)),
            "path": str(cfg_path.parent),
        })
    return rows


def fmt_table(rows, group_axes=("axis", "scenario")):
    rows = sorted(rows, key=lambda r: (r["axis"], r["scenario"], r["methodology"]))
    cur = None
    out = []
    for r in rows:
        key = tuple(r[a] for a in group_axes)
        if key != cur:
            cur = key
            out.append("")
            out.append(f"### {' / '.join(str(k) for k in key)}  "
                       f"({r['constraint_tag']}  classes={r['constrained_classes']})")
            out.append("| methodology | acc | F1 | adj | raw_sat | excess | restored | sat_ep | best_sat | t(s) |")
            out.append("|---|---|---|---|---|---|---|---|---|---|")
        out.append(
            f"| {r['methodology']} "
            f"| {r['accuracy']:.4f} "
            f"| {r['f1_macro']:.4f} "
            f"| {r['samples_adjusted']} "
            f"| {r['raw_satisfied']} "
            f"| {r['raw_excess']} "
            f"| {r['restored_from_epoch']} "
            f"| {r['satisfaction_epoch']} "
            f"| {r['best_sat_epoch']} "
            f"| {r['training_time_s']:.0f} |"
        )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/pending_runs/overnight_sweep")
    args = ap.parse_args()
    rows = collect(args.root)
    if not rows:
        print(f"No completed experiments under {args.root}")
        return
    print(f"# Sweep results: {len(rows)} completed runs from {args.root}\n")
    print(fmt_table(rows))


if __name__ == "__main__":
    main()
