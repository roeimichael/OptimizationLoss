"""Apply UNIFORM force_exact_count clamp to every run's raw predictions and
write a fair_evaluation_metrics.csv next to evaluation_metrics.csv.

Why: evaluation_metrics.csv uses targeted_correction (asymmetric — drops
over-limit, doesn't fill under-limit). This advantages methods that
over-predict (Hounie) over methods that satisfy in-training at count < K
(TraLO). force_exact_count clamps every method to count == K for fair
F1/acc comparison.

Walks the POST_FIX sweep set used by build_paper_artifacts.py.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.constants import UNLIMITED

# Import the force_exact_count from reposthoc_analysis (same algorithm).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reposthoc_analysis import force_exact_count, parse_run, get_limits_from_log

ROOT = Path("results/pending_runs")

# Sweep roots PER method (matches build_paper_artifacts.POST_FIX)
POST_FIX = {
    "tralo": {"fix_ce_skip", "fix1_validation", "kl_sweep",
              "overnight_2026_05_14", "paper_rerun"},
    "fioretto_ldf": {"convergence_validation_300", "fix1_validation",
                     "overnight_2026_05_14", "paper_rerun"},
    "hounie_rcl": {"hounie_rerun", "convergence_validation_300",
                   "fix1_validation", "overnight_2026_05_14", "paper_rerun"},
    "heuristic": {"convergence_validation_300", "overnight_2026_05_14",
                  "thesis_ext", "thesis", "overnight_sweep", "paper_rerun"},
    "danits_lp": {"convergence_validation_300", "overnight_2026_05_14",
                  "thesis_ext", "thesis", "overnight_sweep", "paper_rerun"},
}


def compute_fair_metrics(y_true, y_pred, y_proba, n_classes):
    """Return dict of metrics matching evaluation_metrics.csv schema."""
    m = {}
    m["Accuracy"] = accuracy_score(y_true, y_pred)
    m["Precision (Macro)"] = precision_score(y_true, y_pred,
                                              average="macro", zero_division=0)
    m["Recall (Macro)"] = recall_score(y_true, y_pred,
                                        average="macro", zero_division=0)
    m["F1 (Macro)"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["Precision (Weighted)"] = precision_score(y_true, y_pred,
                                                 average="weighted",
                                                 zero_division=0)
    m["Recall (Weighted)"] = recall_score(y_true, y_pred,
                                           average="weighted",
                                           zero_division=0)
    m["F1 (Weighted)"] = f1_score(y_true, y_pred, average="weighted",
                                   zero_division=0)
    p_per = precision_score(y_true, y_pred, average=None, zero_division=0,
                            labels=list(range(n_classes)))
    r_per = recall_score(y_true, y_pred, average=None, zero_division=0,
                         labels=list(range(n_classes)))
    f_per = f1_score(y_true, y_pred, average=None, zero_division=0,
                     labels=list(range(n_classes)))
    for c in range(n_classes):
        m[f"Precision_Class{c}"] = float(p_per[c])
        m[f"Recall_Class{c}"] = float(r_per[c])
        m[f"F1_Class{c}"] = float(f_per[c])
        m[f"Support_Class{c}"] = int((y_true == c).sum())
    return m


def process(run_dir, global_con, local_con, constrained):
    parsed = parse_run(run_dir)
    if parsed is None:
        return None
    y_true, y_raw, y_proba, group_ids, n_classes = parsed
    if y_proba.shape[1] != len(global_con):
        return None
    y_new, flips = force_exact_count(y_proba, group_ids,
                                     global_con, local_con, constrained)
    metrics = compute_fair_metrics(y_true, y_new, y_proba, n_classes)
    metrics["Flips Required (uniform)"] = int(flips)
    metrics["Raw Count Constrained"] = int(
        sum((y_raw == c).sum() for c in constrained))
    metrics["New Count Constrained"] = int(
        sum((y_new == c).sum() for c in constrained))
    metrics["K Total"] = int(sum(int(global_con[c]) for c in constrained
                                  if global_con[c] < UNLIMITED))
    return metrics


def gather_runs(method_to_sweeps):
    """Return list of (method, run_dir) tuples for runs matching the sweep set."""
    out = []
    for method, sweeps in method_to_sweeps.items():
        for sweep in sweeps:
            base = ROOT / sweep
            if not base.exists():
                continue
            for cfg_path in base.rglob("config.json"):
                d = cfg_path.parent
                try:
                    cfg = json.load(open(cfg_path))
                except Exception:
                    continue
                if cfg.get("methodology") != method:
                    continue
                if not (d / "final_predictions_raw.csv").exists():
                    continue
                out.append((method, d, cfg))
    return out


def resolve_limits_for_cell(cell_runs):
    """Pick the limits from a run that has them in training_log; share within cell."""
    for d, cfg in cell_runs:
        gd, ld = get_limits_from_log(d)
        if gd is None:
            continue
        # Build full arrays
        n_classes = cfg["dataset_config"]["num_classes"]
        global_con = np.full(n_classes, UNLIMITED, dtype=np.float64)
        for c, v in gd.items():
            global_con[c] = v
        local_con = {}
        for gid, cls_map in ld.items():
            arr = np.full(n_classes, UNLIMITED, dtype=np.float64)
            for c, v in cls_map.items():
                arr[c] = v
            local_con[gid] = arr
        constrained = [c for c, v in gd.items() if v < UNLIMITED]
        return global_con, local_con, constrained
    # Fall back: use config.json's constrained_class + constraint pair.
    d0, cfg0 = cell_runs[0]
    cc = cfg0["dataset_config"].get("constrained_class", [])
    if not isinstance(cc, list):
        cc = [cc]
    n_classes = cfg0["dataset_config"]["num_classes"]
    # Compute K from natural counts × tightness — need the raw preds for that.
    # Cheaper: derive from training_log Limit cols if present (already tried).
    # If still missing, fall back to constraint_tag math.
    pair = cfg0.get("constraint", [None, None])
    tag = cfg0.get("constraint_tag", "")
    # If we have natural test count we could compute it; otherwise skip.
    return None, None, None


def cell_key(d, cfg):
    return (cfg.get("dataset_mode"), cfg.get("model_name"),
            tuple(sorted(cc if isinstance(
                (cc := cfg["dataset_config"].get("constrained_class")), list)
                else [cc])),
            cfg.get("constraint_tag"))


def main():
    runs = gather_runs(POST_FIX)
    print(f"Found {len(runs)} candidate runs", file=sys.stderr)

    # Group by cell so we can share limits
    cells = {}
    for method, d, cfg in runs:
        cells.setdefault(cell_key(d, cfg), []).append((d, cfg))

    written, skipped = 0, 0
    for ck, cell_runs in cells.items():
        gcon, lcon, constrained = resolve_limits_for_cell(cell_runs)
        if gcon is None:
            skipped += len(cell_runs)
            continue
        for d, cfg in cell_runs:
            m = process(d, gcon, lcon, constrained)
            if m is None:
                skipped += 1
                continue
            out_csv = d / "fair_evaluation_metrics.csv"
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Metric", "Value"])
                for k, v in m.items():
                    if isinstance(v, float):
                        w.writerow([k, f"{v:.4f}"])
                    else:
                        w.writerow([k, v])
            written += 1
    print(f"Wrote {written} fair_evaluation_metrics.csv; skipped {skipped}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
