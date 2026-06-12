"""Saturation audit — for every cell with training_log.csv, classify regime.

For each cell:
  - dedupe training_log by epoch, sort
  - phase 2 = epoch >= 1 (warmup-end logged as epoch 0)
  - CE-activity = how much CE loss persists through phase 2
  - regime classification: saturated | push_pull | unsatisfied | collapsed | broken

Reads:
  results/pending_runs/<sweep>/.../seed_*/training_log.csv
  results/pending_runs/<sweep>/.../seed_*/config.json
  results/pending_runs/<sweep>/.../seed_*/evaluation_metrics.csv

Writes:
  /tmp/saturation_audit.csv  (one row per cell)
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = "results/pending_runs"
OUT = "/tmp/saturation_audit.csv"

# CE-activity thresholds for regime classification.
# CE < 0.05 means model has memorized; that's the saturation criterion.
CE_HIGH = 0.15   # "alive" — phase 2 CE genuinely shaping gradients
CE_LOW = 0.05    # "dead" — model has memorized


def read_eval_metrics(path):
    if not os.path.exists(path):
        return {}
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def read_config(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def parse_log(path):
    """Return list of dicts sorted by epoch, deduped (first occurrence wins)."""
    if not os.path.exists(path):
        return []
    seen = set()
    rows = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                ep = int(r["epoch"])
            except (KeyError, ValueError):
                continue
            if ep in seen:
                continue
            seen.add(ep)
            try:
                rows.append({
                    "epoch": ep,
                    "ce_loss": float(r.get("ce_loss", "nan")),
                    "constraint_loss": float(r.get("constraint_loss", "0")),
                    "all_satisfied": int(r.get("all_satisfied", "0")),
                    "total_excess": float(r.get("total_excess", "nan")),
                })
            except ValueError:
                continue
    rows.sort(key=lambda r: r["epoch"])
    return rows


def compute_metrics(log_rows):
    """Phase 2 = epoch >= 1. Returns dict of summary numbers."""
    phase2 = [r for r in log_rows if r["epoch"] >= 1]
    if not phase2:
        return None
    ces = [r["ce_loss"] for r in phase2]
    n = len(ces)
    sat_epochs = [r["epoch"] for r in phase2 if r["all_satisfied"] == 1]
    return {
        "phase2_n": n,
        "mean_ce": sum(ces) / n,
        "min_ce": min(ces),
        "max_ce": max(ces),
        "final_ce": ces[-1],
        "frac_ce_high": sum(1 for c in ces if c >= CE_HIGH) / n,
        "frac_ce_low": sum(1 for c in ces if c <= CE_LOW) / n,
        "mean_constraint": sum(r["constraint_loss"] for r in phase2) / n,
        "any_satisfied": 1 if sat_epochs else 0,
        "first_sat_epoch": sat_epochs[0] if sat_epochs else "",
        "frac_satisfied": sum(1 for r in phase2 if r["all_satisfied"] == 1) / n,
        "warmup_end_ce": next(
            (r["ce_loss"] for r in log_rows if r["epoch"] == 0), ""
        ),
    }


def classify_regime(m, ev):
    """
    saturated  : CE dead through most of phase 2 (frac_ce_low > 0.7)
    push_pull  : CE stays high through most of phase 2 (frac_ce_high > 0.5)
                 AND constraint eventually satisfies
    push_pull_unsat: CE stays high but never satisfies
    transition : in between
    broken     : missing metrics
    """
    if m is None:
        return "broken"
    if m["frac_ce_low"] > 0.7:
        return "saturated"
    if m["frac_ce_high"] > 0.5:
        return "push_pull" if m["any_satisfied"] else "push_pull_unsat"
    return "transition"


def main():
    root = ROOT
    cells = sorted(glob.glob(f"{root}/**/seed_*", recursive=True))
    print(f"Found {len(cells)} candidate cells", file=sys.stderr)

    fields = [
        "sweep", "rel_path",
        "dataset", "model", "method", "constraint_tag", "constrained_class",
        "warmup_epochs", "constraint_epochs", "pretrained", "seed",
        "phase2_n", "mean_ce", "min_ce", "max_ce", "final_ce",
        "frac_ce_high", "frac_ce_low", "mean_constraint",
        "any_satisfied", "first_sat_epoch", "frac_satisfied",
        "warmup_end_ce",
        "f1_macro", "accuracy", "raw_all_satisfied", "flips_required",
        "satisfaction_epoch", "regime",
    ]

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        n_written = 0
        for cell in cells:
            if not os.path.isdir(cell):
                continue
            rel = cell[len(root) + 1:]
            parts = rel.split(os.sep)
            sweep = parts[0]
            seed_str = parts[-1]
            try:
                seed = int(seed_str.replace("seed_", ""))
            except ValueError:
                continue
            method = parts[-2] if len(parts) >= 2 else ""

            cfg = read_config(os.path.join(cell, "config.json"))
            ev = read_eval_metrics(os.path.join(cell, "evaluation_metrics.csv"))
            log = parse_log(os.path.join(cell, "training_log.csv"))
            mm = compute_metrics(log)

            hp = cfg.get("hyperparams", {})
            dsc = cfg.get("dataset_config", {})

            row = {
                "sweep": sweep,
                "rel_path": rel,
                "dataset": cfg.get("dataset_mode", ""),
                "model": cfg.get("model_name", ""),
                "method": cfg.get("methodology", method),
                "constraint_tag": cfg.get("constraint_tag", ""),
                "constrained_class": dsc.get("constrained_class", ""),
                "warmup_epochs": hp.get("warmup_epochs", ""),
                "constraint_epochs": hp.get("constraint_epochs", ""),
                "pretrained": hp.get("pretrained", ""),
                "seed": hp.get("seed", seed),
                "f1_macro": ev.get("F1 (Macro)", ""),
                "accuracy": ev.get("Accuracy", ""),
                "raw_all_satisfied": ev.get("Raw All Satisfied", ""),
                "flips_required": ev.get("Flips Required", ""),
                "satisfaction_epoch": ev.get("Satisfaction Epoch", ""),
            }
            if mm:
                row.update({
                    "phase2_n": mm["phase2_n"],
                    "mean_ce": f"{mm['mean_ce']:.5f}",
                    "min_ce": f"{mm['min_ce']:.5f}",
                    "max_ce": f"{mm['max_ce']:.5f}",
                    "final_ce": f"{mm['final_ce']:.5f}",
                    "frac_ce_high": f"{mm['frac_ce_high']:.4f}",
                    "frac_ce_low": f"{mm['frac_ce_low']:.4f}",
                    "mean_constraint": f"{mm['mean_constraint']:.5f}",
                    "any_satisfied": mm["any_satisfied"],
                    "first_sat_epoch": mm["first_sat_epoch"],
                    "frac_satisfied": f"{mm['frac_satisfied']:.4f}",
                    "warmup_end_ce": (
                        f"{mm['warmup_end_ce']:.5f}"
                        if isinstance(mm["warmup_end_ce"], float)
                        else ""
                    ),
                })
            row["regime"] = classify_regime(mm, ev)
            w.writerow(row)
            n_written += 1
            if n_written % 500 == 0:
                print(f"  wrote {n_written} rows...", file=sys.stderr)

    print(f"Done. {n_written} rows -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
