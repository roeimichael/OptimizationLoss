"""Extract any track_b campaign into one row per completed run. RUN ON THE SERVER.

Campaigns outside `results/pending_runs` are invisible to the corpus build, so
each needs pulling explicitly. This is the general form of
extract_alm_results.py, which stays separate because it also resolves the ALM
arm's clone provenance.

    python paper/scripts/extract_campaign.py results/track_b/r2_seeds10 -o /tmp/r2.csv
    python paper/scripts/extract_campaign.py results/track_b/r3_rerunvar -o /tmp/r3.csv

Emits `rep` (from a rep_NN directory) so the rerun-variance campaign, whose ten
runs share one seed and differ only by repeat, stays distinguishable.
"""
import argparse
import csv
import glob
import json
import os
import re

FIELDS = ["campaign", "dataset", "model", "method", "constraint_tag", "seed",
          "rep", "cc_f1", "f1_macro", "acc", "flips", "sat", "sat_epoch",
          "config_path"]

# The metrics csv spells these out in full (src/training/logging.py); the short
# names this script used to ask for match nothing, which is why every campaign
# pulled with it arrived with empty flips/sat columns.
K_FLIPS = "Flips Required"
K_SAT = "Raw All Satisfied"
K_SAT_EPOCH = "Satisfaction Epoch"


def num(m, key):
    try:
        return float(m.get(key, ""))
    except (TypeError, ValueError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    rows, skipped = [], 0
    for cfg in glob.glob(args.root + "/**/config.json", recursive=True):
        ev = os.path.join(os.path.dirname(cfg), "evaluation_metrics.csv")
        if not os.path.exists(ev):
            skipped += 1
            continue
        c = json.load(open(cfg))
        m = {r["Metric"]: r["Value"] for r in csv.DictReader(open(ev))}
        cc = c["dataset_config"].get("constrained_class")
        rep = re.search(r"rep_(\d+)", cfg)
        rows.append({
            "campaign": os.path.basename(args.root.rstrip("/")),
            "dataset": c["dataset_mode"],
            "model": c["model_name"],
            "method": c["methodology"],
            "constraint_tag": c["constraint_tag"],
            "seed": c["hyperparams"]["seed"],
            "rep": int(rep.group(1)) if rep else "",
            "cc_f1": num(m, "F1_Class%d" % cc),
            "f1_macro": num(m, "F1 (Macro)"),
            "acc": num(m, "Accuracy"),
            "flips": num(m, K_FLIPS),
            "sat": num(m, K_SAT),
            "sat_epoch": num(m, K_SAT_EPOCH),
            "config_path": cfg,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("%s: %d completed, %d still pending -> %s"
          % (args.root, len(rows), skipped, args.out))


if __name__ == "__main__":
    main()
