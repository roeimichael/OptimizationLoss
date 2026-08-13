"""Inventory every experiment run, from every results root, into one manifest.

The paper's numbers come from several campaign trees that grew at different
times (the frozen paper_final grid, the Track-B campaigns, the review controls,
the ALM expansion). Nothing previously listed them together, so "what have we
actually run?" could only be answered by globbing -- which is how the ALM arm
ended up absent from the corpus without anyone noticing.

This walks every results root and emits ONE row per run:

    campaign, root, dataset, model, method, cap, seed, warmup, status,
    cc_f1, f1_macro, acc, ece, sat, flips, constrained_class, config_path

`config_path` is the key deliverable: every row points back at the exact
config.json that produced it, so any number in the paper can be traced to a
reproducible run.

Run ON THE SERVER from the repo root:
    python paper/scripts/build_experiment_manifest.py [-o out.csv]

Then pull the CSV to paper/data/manifest/experiments.csv.
"""

import argparse
import csv
import json
import os
import sys

RESULT_ROOTS = [
    "results/pending_runs",
    "results/track_b",
    "results/review_controls2_2026-07",
    "results/review_controls3_2026-07",
    "results/review_graft_2026-07",
    "results/baselines",
]

# Superset of the corpus_final.csv schema on purpose: build_corpus.py projects
# these columns straight down to the corpus the figure/table generators read, so
# there is exactly one place where run data enters the paper.
FIELDS = ["campaign", "root", "dataset", "model", "method", "cap", "seed",
          "warmup", "status", "cc_f1", "cc_prec", "cc_rec", "f1_macro", "acc",
          "ece", "sat", "flips", "constrained_class", "group_column",
          "sweep_tag", "config_path"]


def read_metrics(run_dir):
    """evaluation_metrics.csv is long-format (Metric,Value). Return a dict."""
    path = os.path.join(run_dir, "evaluation_metrics.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("Metric") or "").strip()
                try:
                    out[key] = float(row.get("Value"))
                except (TypeError, ValueError):
                    pass
    except Exception:
        return {}
    return out


def campaign_of(config_path, root):
    """First path segment below the root, skipping lane_gpuN / laneN wrappers."""
    rel = os.path.relpath(config_path, root).replace("\\", "/").split("/")
    for part in rel:
        low = part.lower()
        if low.startswith("lane"):
            continue
        return part
    return os.path.basename(root)


def row_for(config_path, root):
    try:
        with open(config_path) as f:
            c = json.load(f)
    except Exception:
        return None
    hp = c.get("hyperparams", {}) or {}
    dc = c.get("dataset_config", {}) or {}
    run_dir = os.path.dirname(config_path)
    m = read_metrics(run_dir)

    cls = dc.get("constrained_class")
    cc_f1 = cc_prec = cc_rec = None
    if cls is not None:
        cc_f1 = m.get("F1_Class%s" % cls)
        cc_prec = m.get("Precision_Class%s" % cls)
        cc_rec = m.get("Recall_Class%s" % cls)

    res = c.get("results", {}) or {}
    # Satisfaction / flips live in the config's results block, not the metrics
    # csv; different campaigns spelled them differently.
    sat = res.get("constraints_satisfied_natively", res.get("satisfied_natively"))
    if isinstance(sat, bool):
        sat = int(sat)
    flips = res.get("samples_adjusted")

    return {
        "campaign": campaign_of(config_path, root),
        "root": root,
        "dataset": c.get("dataset_mode"),
        "model": c.get("model_name"),
        "method": c.get("methodology"),
        "cap": c.get("constraint_tag"),
        "seed": hp.get("seed"),
        "warmup": hp.get("warmup_epochs"),
        "status": c.get("status"),
        "cc_f1": cc_f1,
        "cc_prec": cc_prec,
        "cc_rec": cc_rec,
        "f1_macro": m.get("F1 (Macro)", res.get("f1_macro")),
        "acc": m.get("Accuracy", res.get("accuracy")),
        "ece": m.get("ECE"),
        "sat": sat,
        "flips": flips,
        "constrained_class": cls,
        "group_column": dc.get("group_column"),
        "sweep_tag": c.get("sweep_tag"),
        "config_path": config_path.replace("\\", "/"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="experiments_manifest.csv")
    ap.add_argument("--roots", nargs="*", default=None)
    args = ap.parse_args()

    roots = args.roots or RESULT_ROOTS
    rows = []
    for root in roots:
        if not os.path.isdir(root):
            print("  (skip, not present) %s" % root, file=sys.stderr)
            continue
        n0 = len(rows)
        for dirpath, _dirnames, filenames in os.walk(root):
            if "config.json" not in filenames:
                continue
            r = row_for(os.path.join(dirpath, "config.json"), root)
            if r:
                rows.append(r)
        print("  %-42s %6d runs" % (root, len(rows) - n0), file=sys.stderr)

    rows.sort(key=lambda r: (str(r["campaign"]), str(r["dataset"]), str(r["model"]),
                             str(r["cap"]), str(r["method"]), r["seed"] or 0))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    done = sum(1 for r in rows if r["status"] == "completed")
    withm = sum(1 for r in rows if r["cc_f1"] is not None)
    print("\nTOTAL %d runs  (completed %d, with cc-F1 %d)  -> %s"
          % (len(rows), done, withm, args.out), file=sys.stderr)


if __name__ == "__main__":
    main()
