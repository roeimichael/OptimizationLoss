"""Adversarial check on the interchangeability result.

`how_much_moves.py` reports the methods disagreeing on 2-4% of the pool and
concludes they are near-interchangeable. But the constrained class is only about
7% of the pool, so a pool-denominator percentage flatters agreement badly: two
methods could disagree about a THIRD of the constrained class and still show
"2%" by that measure.

The fair denominator is the class the constraint actually acts on. This reports
the Jaccard overlap of the constrained-class prediction SETS -- of the samples
either method assigns to the class, what fraction do both? If that stays high,
interchangeability survives. If it collapses, the pool-level number was an
artifact of class rarity and the claim must be withdrawn.

Same exact-comparison design: every pair shares one bit-identical warm-up
checkpoint, and danits_lp never trains past warm-up so it is the zero point.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/track_b")
    ap.add_argument("--warmup-min", type=int, default=50)
    ap.add_argument("--methods", nargs="+",
                    default=["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp"])
    args = ap.parse_args()

    cells = {}
    for cfg_path in glob.glob(os.path.join(args.root, "**", "config.json"), recursive=True):
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        try:
            cfg = json.load(open(cfg_path))
        except ValueError:
            continue
        hp = cfg.get("hyperparams") or {}
        if hp.get("warmup_epochs", 0) < args.warmup_min:
            continue
        m = cfg.get("methodology")
        if m not in args.methods:
            continue
        cc = cfg.get("dataset_config", {}).get("constrained_class")
        cc = int(cc[0] if isinstance(cc, (list, tuple)) else cc)
        key = (cfg["dataset_mode"], cfg["model_name"], cfg["constraint_tag"], hp.get("seed"))
        cells.setdefault(key, {})[m] = (raw, cc)

    rows, prev = [], []
    for key, bym in cells.items():
        loaded = {}
        for m, (path, cc) in bym.items():
            try:
                t = pd.read_csv(path)
            except Exception:
                continue
            if "Predicted_Label" not in t.columns:
                continue
            p = t["Predicted_Label"].to_numpy(int)
            loaded[m] = (p == cc)
            if "True_Label" in t.columns:
                prev.append(float((t["True_Label"].to_numpy(int) == cc).mean()))
        for a, b in itertools.combinations(sorted(loaded), 2):
            A, B = loaded[a], loaded[b]
            if len(A) != len(B):
                continue
            union = int((A | B).sum())
            inter = int((A & B).sum())
            rows.append({
                "dataset": key[0], "pair": "%s vs %s" % (a, b),
                "jaccard": inter / union if union else np.nan,
                "n_a": int(A.sum()), "n_b": int(B.sum()),
                "only_a": int((A & ~B).sum()), "only_b": int((~A & B).sum()),
            })
    if not rows:
        print("nothing comparable")
        return 1
    d = pd.DataFrame(rows)
    print("%d pair comparisons; constrained class is %.1f%% of the pool on average"
          % (len(d), 100 * np.mean(prev) if prev else float("nan")))

    print()
    print("=" * 92)
    print("OVERLAP OF THE CONSTRAINED-CLASS PREDICTION SETS  (Jaccard, higher = more alike)")
    print("=" * 92)
    t = d.groupby("pair").agg(
        n=("jaccard", "size"), jaccard=("jaccard", "median"),
        size_a=("n_a", "median"), size_b=("n_b", "median"),
        only_a=("only_a", "median"), only_b=("only_b", "median")
    ).sort_values("jaccard", ascending=False)
    print(t.round(3).to_string())
    print()
    print("  jaccard 1.00 = identical sets. 0.50 = they share half of what either one selects.")

    print()
    print("=" * 92)
    print("BY DATASET")
    print("=" * 92)
    print(d.pivot_table(index="dataset", columns="pair", values="jaccard",
                        aggfunc="median").round(3).to_string())

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    trained = t[~t.index.str.contains("danits_lp")]
    vs_warm = t[t.index.str.contains("danits_lp")]
    if len(trained) and len(vs_warm):
        lo, hi = trained.jaccard.min(), trained.jaccard.max()
        w = vs_warm.jaccard.median()
        print("  among the trained methods : jaccard %.3f - %.3f" % (lo, hi))
        print("  against the warm-up model : jaccard %.3f (median)" % w)
        if lo > 0.80:
            print("  -> interchangeability SURVIVES on the fair denominator")
        elif lo > 0.60:
            print("  -> WEAKENED: they agree far more with each other than with the")
            print("     warm-up model, but they are not interchangeable")
        else:
            print("  -> REFUTED: the pool-level 2-4%% was class rarity, not agreement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
