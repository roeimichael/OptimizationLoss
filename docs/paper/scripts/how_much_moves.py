"""How much does the constraint phase actually change the model?

The warm-up cache key (`compute_base_model_id`) hashes the model, data, lr,
dropout, batch size, warm-up length and seed -- but NOT the methodology. So
every method in a given (dataset, model, cap, seed) starts the constraint phase
from a bit-identical checkpoint. Whatever their final raw predictions disagree
about is therefore the entire effect of the constraint phase, isolated exactly,
with no extra runs needed.

That makes this the direct test of the ~30-step finding. If the constraint phase
is thirty unit-norm steps on a saturated model, the methods should end up
predicting almost the same thing, and the results table being a wash stops being
a puzzle.

Raw predictions only (`final_predictions_raw.csv`) -- the post-hoc adjusted file
is forced toward the cap by construction and would understate disagreement.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

CELL = ["dataset", "model", "cap", "seed"]


def load(root, warmup_min):
    runs = {}
    for cfg_path in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        try:
            cfg = json.load(open(cfg_path))
        except ValueError:
            continue
        hp = cfg.get("hyperparams") or {}
        if hp.get("warmup_epochs", 0) < warmup_min:
            continue
        key = (cfg.get("dataset_mode"), cfg.get("model_name"),
               cfg.get("constraint_tag"), hp.get("seed"))
        cc = cfg.get("dataset_config", {}).get("constrained_class")
        cc = int(cc[0] if isinstance(cc, (list, tuple)) else cc)
        runs.setdefault(key, {})[cfg.get("methodology")] = (raw, cc)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/track_b")
    ap.add_argument("--warmup-min", type=int, default=50)
    ap.add_argument("--methods", nargs="+",
                    default=["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp"])
    args = ap.parse_args()

    runs = load(args.root, args.warmup_min)
    rows = []
    for key, bym in runs.items():
        have = [m for m in args.methods if m in bym]
        if len(have) < 2:
            continue
        cache = {}
        for m in have:
            path, cc = bym[m]
            try:
                t = pd.read_csv(path)
            except Exception:
                continue
            col = "Prob_Class_%d" % cc
            if "Predicted_Label" not in t.columns or col not in t.columns:
                continue
            cache[m] = (t["Predicted_Label"].to_numpy(int), t[col].to_numpy(float), cc)
        for a, b in itertools.combinations(sorted(cache), 2):
            pa, qa, cc = cache[a]
            pb, qb, _ = cache[b]
            if len(pa) != len(pb):
                continue
            rows.append({
                "dataset": key[0], "model": key[1], "cap": key[2], "seed": key[3],
                "pair": "%s vs %s" % (a, b),
                "disagree": float((pa != pb).mean()),
                # disagreement restricted to the class the constraint acts on
                "disagree_cc": float((((pa == cc) != (pb == cc))).mean()),
                "prob_corr": float(pd.Series(qa).corr(pd.Series(qb), method="pearson")),
                "prob_mad": float(np.abs(qa - qb).mean()),
            })
    if not rows:
        print("no cells with two or more comparable methods under", args.root)
        return 1
    d = pd.DataFrame(rows)
    print("%d method-pair comparisons over %d cells, warm-up >= %d"
          % (len(d), d.groupby(CELL).ngroups, args.warmup_min))
    print("(identical warm-up checkpoint on both sides of every pair)")

    print()
    print("=" * 96)
    print("HOW FAR APART DO THE METHODS END UP?  raw predictions, before post-hoc")
    print("=" * 96)
    t = d.groupby("pair").agg(
        n=("disagree", "size"),
        disagree=("disagree", "median"),
        disagree_cc=("disagree_cc", "median"),
        prob_corr=("prob_corr", "median"),
        prob_mad=("prob_mad", "median")).sort_values("disagree")
    print(t.round(4).to_string())
    print()
    print("  disagree     = fraction of the pool given a different argmax label")
    print("  disagree_cc  = fraction where the two disagree about the CONSTRAINED class")
    print("  prob_corr    = correlation of the constrained-class probability")

    print()
    print("=" * 96)
    print("BY DATASET  (median argmax disagreement)")
    print("=" * 96)
    print(d.pivot_table(index="dataset", columns="pair", values="disagree",
                        aggfunc="median").round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
