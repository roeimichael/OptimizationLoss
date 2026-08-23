"""Test the manuscript's ACTUAL headline claim, which is macro-F1 -- not cc-F1.

The paper claims constraint-time training beats post-hoc clipping by
+1.6 to +5.3 percentage points of macro-F1, on nearly all of the grid. Tonight's
capstone measured cc-F1 (the paper's and the budget-equalized version) and AP.
It did NOT measure macro-F1, so it does not bear on the headline. This does.

Same exact design as the capstone: every method in a cell starts from a
bit-identical warm-up checkpoint, and `danits_lp` never trains past warm-up, so
it is the post-hoc clipper the claim is made against.

Both variants are reported, because the distinction is what decided the cc-F1
version:
  * macro-F1 on `final_predictions.csv` -- the post-hoc adjusted predictions,
    which is what the manuscript reports.
  * macroEq -- budget-equalized, so quota-filling cannot manufacture it.

If the margin survives equalization the headline stands. If it collapses the way
cc-F1 did, it was budget usage.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

CELL = ["dataset", "model", "cap", "seed"]


def adjusted_macro(path):
    """macro-F1 on the post-hoc adjusted predictions -- the paper's metric."""
    try:
        raw = pd.read_csv(os.path.join(path, "final_predictions_raw.csv"))
        fin = pd.read_csv(os.path.join(path, "final_predictions.csv"))
    except Exception:
        return np.nan
    if "True_Label" not in raw.columns or "Predicted_Label" not in fin.columns:
        return np.nan
    y, p = raw["True_Label"].to_numpy(int), fin["Predicted_Label"].to_numpy(int)
    if len(y) != len(p):
        return np.nan
    return f1_score(y, p, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/track_b")
    ap.add_argument("--warmup-min", type=int, default=50)
    ap.add_argument("--zero", default="danits_lp",
                    help="the post-hoc clipper the claim is made against")
    args = ap.parse_args()

    d = A.rows_for(args.root)
    d = d[(d.warmup >= args.warmup_min)
          & d.method.isin(["tralo", "fioretto_ldf", "hounie_rcl",
                           "danits_lp", "heuristic"])].copy()
    if d.empty:
        print("no runs")
        return 1
    d["macro_paper"] = [adjusted_macro(p) for p in d.path]
    d["fill"] = d["count"] / d.K

    print("%d runs, warm-up >= %d" % (len(d), args.warmup_min))
    print()
    print("=" * 92)
    print("PER METHOD")
    print("=" * 92)
    print(d.groupby("method").agg(
        n=("macro_paper", "size"), fill=("fill", "median"),
        macro_paper=("macro_paper", "median"), macroEq=("macroEq", "median"),
        ccF1eq=("ccF1eq", "median")).round(4).to_string())

    print()
    print("=" * 92)
    print("THE HEADLINE: paired macro-F1 vs %s, the post-hoc clipper" % args.zero)
    print("  manuscript claims +1.6 to +5.3 percentage points on nearly all of the grid")
    print("=" * 92)
    for metric, label in [("macro_paper", "macro-F1 as the paper reports it"),
                          ("macroEq", "macro-F1, budget-equalized")]:
        piv = d.pivot_table(index=CELL, columns="method", values=metric)
        if args.zero not in piv.columns:
            print("  no %s runs" % args.zero)
            break
        print()
        print("  --- %s ---" % label)
        for m in ["tralo", "fioretto_ldf", "hounie_rcl"]:
            if m not in piv.columns:
                continue
            s = piv[[m, args.zero]].dropna()
            if s.empty:
                continue
            diff = (s[m] - s[args.zero]) * 100      # percentage points
            print("    %-14s n=%-4d median %+0.2f pp   mean %+0.2f pp   wins %d/%d (%.0f%%)"
                  % (m, len(s), diff.median(), diff.mean(),
                     (diff > 0).sum(), len(s), 100 * (diff > 0).mean()))

    print()
    print("=" * 92)
    print("BY DATASET, TraLO vs %s, macro-F1 (pp)" % args.zero)
    print("=" * 92)
    for metric in ["macro_paper", "macroEq"]:
        piv = d.pivot_table(index=CELL + [], columns="method", values=metric)
        if "tralo" not in piv.columns or args.zero not in piv.columns:
            continue
        s = piv[["tralo", args.zero]].dropna()
        s = s.reset_index()
        s["delta_pp"] = (s["tralo"] - s[args.zero]) * 100
        print("  %s:" % metric)
        print(s.groupby("dataset").delta_pp.agg(
            ["size", "median", "mean"]).round(2).to_string())
        print()

    print("=" * 92)
    print("DOES BUDGET USAGE EXPLAIN IT?  (within cell, across methods)")
    print("=" * 92)
    for m in ["macro_paper", "macroEq"]:
        d["d_" + m] = d[m] - d.groupby(["dataset", "model", "cap"])[m].transform("mean")
    d["d_fill"] = d.fill - d.groupby(["dataset", "model", "cap"]).fill.transform("mean")
    s = d[["d_fill", "d_macro_paper", "d_macroEq"]].dropna()
    if len(s) > 20:
        print("  spearman(fill, paper's macro-F1) = %+0.3f"
              % s[["d_fill", "d_macro_paper"]].corr(method="spearman").iloc[0, 1])
        print("  spearman(fill, equalized       ) = %+0.3f   n=%d"
              % (s[["d_fill", "d_macroEq"]].corr(method="spearman").iloc[0, 1], len(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
