"""Check whether TraLO has better calibration (ECE / Brier) than benchmarks.

Reads evaluation_metrics.csv from the completed thesis sweep and aggregates ECE
and Brier across seeds per (model, methodology).

Usage:
    python scripts/calibration_check.py [--root results/pending_runs/thesis]
"""
import argparse
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/pending_runs/thesis")
    args = ap.parse_args()

    bucket = defaultdict(list)
    for cfg_path in Path(args.root).rglob("config.json"):
        cfg = json.load(open(cfg_path))
        if cfg.get("status") != "completed":
            continue
        rel = cfg_path.parent.relative_to(args.root)
        if rel.parts[0] != "headline":
            continue
        m = dict(csv.reader(open(cfg_path.parent / "evaluation_metrics.csv")))
        ece = float(m.get("ECE", "nan"))
        brier = float(m.get("Brier Score", "nan"))
        ent = float(m.get("Mean Entropy", "nan"))
        conf = float(m.get("Mean Confidence", "nan"))
        key = (cfg.get("model_name"), cfg.get("methodology"))
        bucket[key].append({"ece": ece, "brier": brier,
                            "entropy": ent, "confidence": conf})

    print("model | methodology | n | ECE | Brier | mean_entropy | mean_conf")
    print("---|---|---|---|---|---|---")
    for key in sorted(bucket.keys()):
        runs = bucket[key]
        eces = [r["ece"] for r in runs if r["ece"] == r["ece"]]
        briers = [r["brier"] for r in runs if r["brier"] == r["brier"]]
        ents = [r["entropy"] for r in runs if r["entropy"] == r["entropy"]]
        confs = [r["confidence"] for r in runs if r["confidence"] == r["confidence"]]

        def fmt(vals):
            if not vals:
                return "-"
            if len(vals) == 1:
                return f"{vals[0]:.4f}"
            return f"{stats.mean(vals):.4f} ± {stats.stdev(vals):.4f}"

        print(f"{key[0]} | {key[1]} | {len(runs)} | {fmt(eces)} | "
              f"{fmt(briers)} | {fmt(ents)} | {fmt(confs)}")


if __name__ == "__main__":
    main()
