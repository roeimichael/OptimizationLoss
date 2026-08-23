"""Extract the ALM arm into one row per run. RUN ON THE SERVER.

The ALM campaigns live outside `results/pending_runs` (the frozen corpus root is
never written to), so they are not in `corpus_final.csv` and the ordinary corpus
build does not see them. This walks their result roots directly.

    python paper/scripts/extract_alm_results.py -o /tmp/alm_results.csv

then copy to paper/data/corpus/alm_results.csv. Consumed by analyze_alm.py and
make_alm_regime_table.py.

Roots:
  results/track_b/b3        the original 24-run tight-cap probe
  results/track_b/b3_full   the 300-run expansion to the full grid
  results/track_b/b3_mnv2   ALM on MobileNetV2
  results/track_b/r1_almrh  ALM + reset + hinge (review round 1 graft)
"""
import argparse
import csv
import glob
import json
import os

ROOTS = ["results/track_b/b3", "results/track_b/b3_full",
         "results/track_b/b3_mnv2", "results/track_b/r1_almrh"]

FIELDS = ["root", "src_sweep", "dataset", "model", "method", "constraint_tag",
          "seed", "cc_f1", "f1_macro", "acc", "flips", "sat", "sat_epoch",
          "config_path"]

# Full names as written by src/training/logging.py. The abbreviations this
# script asked for ("Flips", "Satisfied") match no row, so alm_results.csv came
# back with an empty satisfaction column -- which is why the ALM arm could not
# be placed in the deployment figure.
K_FLIPS = "Flips Required"
K_SAT = "Raw All Satisfied"
K_SAT_EPOCH = "Satisfaction Epoch"


def source_sweep(cfg):
    """The sweep the run was cloned from -- the ONLY corpus rows it may pair with.

    Every ALM config is a clone of a frozen run with the dual rule swapped, so it
    shares that run's warmup cache and settings. Pairing it against a different
    sweep's TraLO run instead would inject the cross-campaign drift measured at
    0.025 cc-F1, five times the +/-0.005 band the comparison is adjudicated at.
    MobileNetV2 makes this concrete: its ALM runs were cloned from seven
    different sweeps.
    """
    src = cfg.get("cloned_from", "")
    parts = src.split("/")
    # results/<root>/<sweep>/... -- e.g. results/pending_runs/paper_final/...
    return parts[2] if len(parts) > 2 else ""


def metric(m, key):
    try:
        return float(m.get(key, ""))
    except (TypeError, ValueError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    rows = []
    for root in ROOTS:
        if not os.path.isdir(root):
            print("skip (absent): %s" % root)
            continue
        n = 0
        for cfg in glob.glob(root + "/**/config.json", recursive=True):
            ev = os.path.join(os.path.dirname(cfg), "evaluation_metrics.csv")
            if not os.path.exists(ev):
                continue          # still pending or failed; not a completed run
            c = json.load(open(cfg))
            m = {r["Metric"]: r["Value"] for r in csv.DictReader(open(ev))}
            # The constrained class is per-dataset, so the cc-F1 key varies.
            cc = c["dataset_config"].get("constrained_class")
            rows.append({
                "root": os.path.basename(root),
                "src_sweep": source_sweep(c),
                "dataset": c["dataset_mode"],
                "model": c["model_name"],
                "method": c["methodology"],
                "constraint_tag": c["constraint_tag"],
                "seed": c["hyperparams"]["seed"],
                "cc_f1": metric(m, "F1_Class%d" % cc),
                "f1_macro": metric(m, "F1 (Macro)"),
                "acc": metric(m, "Accuracy"),
                "flips": metric(m, K_FLIPS),
                "sat": metric(m, K_SAT),
                "sat_epoch": metric(m, K_SAT_EPOCH),
                "config_path": cfg,
            })
            n += 1
        print("%-28s %4d completed runs" % (root, n))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
