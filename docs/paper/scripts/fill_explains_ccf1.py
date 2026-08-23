"""Does cut depth explain the cc-F1 differences between methods?

The nesting result says every method selects a subset of the warm-up model's
constrained-class predictions, in the same order, differing only in how deep it
cuts. If that is right it forces a prediction about the paper's headline metric:

  - The cc-F1 gaps BETWEEN methods should track how much of the budget each one
    fills, not anything about the objective they optimise.
  - Under budget equalization -- everyone allocated exactly the same number of
    positives -- those gaps should largely disappear.

That is the quota-fill finding from 2026-07-31 restated as a mechanism, so this
is a check on whether the mechanism actually produces the artifact it should.

Both metrics come off the same run: cc-F1 on `final_predictions.csv` is what the
paper reports (post-hoc adjusted), ccF1eq is the budget-equalized version.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

CELL = ["dataset", "model", "cap", "seed"]


def adjusted_ccf1(path, cc):
    """cc-F1 on the post-hoc adjusted predictions -- the paper's headline."""
    try:
        raw = pd.read_csv(os.path.join(path, "final_predictions_raw.csv"))
        fin = pd.read_csv(os.path.join(path, "final_predictions.csv"))
    except Exception:
        return np.nan
    if "True_Label" not in raw.columns or "Predicted_Label" not in fin.columns:
        return np.nan
    y = raw["True_Label"].to_numpy(int)
    p = fin["Predicted_Label"].to_numpy(int)
    if len(y) != len(p):
        return np.nan
    return f1_score(y, p, labels=[cc], average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/track_b")
    ap.add_argument("--warmup-min", type=int, default=50)
    args = ap.parse_args()

    d = A.rows_for(args.root)
    d = d[(d.warmup >= args.warmup_min)
          & d.method.isin(["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp"])].copy()
    if d.empty:
        print("no runs")
        return 1

    import json
    ccs = []
    for p in d.path:
        try:
            cfg = json.load(open(os.path.join(p, "config.json")))
            c = cfg["dataset_config"]["constrained_class"]
            ccs.append(int(c[0] if isinstance(c, (list, tuple)) else c))
        except Exception:
            ccs.append(np.nan)
    d["cc"] = ccs
    d["ccF1_paper"] = [adjusted_ccf1(p, c) if c == c else np.nan
                       for p, c in zip(d.path, d.cc)]
    d["fill"] = d["count"] / d.K            # post-adjustment budget usage

    print("%d runs, warm-up >= %d, under %s" % (len(d), args.warmup_min, args.root))

    print()
    print("=" * 90)
    print("BUDGET USAGE AND BOTH METRICS, per method")
    print("=" * 90)
    print(d.groupby("method").agg(
        n=("ccF1_paper", "size"),
        count=("count", "median"), K=("K", "median"), fill=("fill", "median"),
        ccF1_paper=("ccF1_paper", "median"), ccF1eq=("ccF1eq", "median")
    ).round(4).to_string())

    print()
    print("=" * 90)
    print("PAIRED DELTAS vs the warm-up model, on BOTH metrics")
    print("=" * 90)
    print("  if cut depth explains the gap, the paper's metric moves and the")
    print("  equalized one does not")
    print()
    for metric in ["ccF1_paper", "ccF1eq"]:
        piv = d.pivot_table(index=CELL, columns="method", values=metric)
        if "danits_lp" not in piv.columns:
            continue
        print("  --- %s ---" % metric)
        for m in ["tralo", "fioretto_ldf", "hounie_rcl"]:
            if m not in piv.columns:
                continue
            s = piv[[m, "danits_lp"]].dropna()
            if s.empty:
                continue
            diff = s[m] - s["danits_lp"]
            print("    %-14s n=%-4d median %+0.4f   wins %d/%d"
                  % (m, len(s), diff.median(), (diff > 0).sum(), len(s)))
        print()

    print("=" * 90)
    print("DOES FILL PREDICT THE PAPER'S METRIC?  (within cell, across methods)")
    print("=" * 90)
    for m in ["ccF1_paper", "ccF1eq"]:
        d["d_" + m] = d[m] - d.groupby(["dataset", "model", "cap"])[m].transform("mean")
    d["d_fill"] = d.fill - d.groupby(["dataset", "model", "cap"]).fill.transform("mean")
    s = d[["d_fill", "d_ccF1_paper", "d_ccF1eq"]].dropna()
    if len(s) > 20:
        r1 = s[["d_fill", "d_ccF1_paper"]].corr(method="spearman").iloc[0, 1]
        r2 = s[["d_fill", "d_ccF1eq"]].corr(method="spearman").iloc[0, 1]
        print("  spearman(fill, paper's cc-F1) = %+0.3f" % r1)
        print("  spearman(fill, equalized    ) = %+0.3f    n=%d" % (r2, len(s)))
        print()
        if abs(r1) - abs(r2) > 0.15:
            print("  -> budget usage drives the paper's metric and largely washes out")
            print("     of the equalized one. The cut-depth mechanism produces the artifact.")
        else:
            print("  -> fill does not preferentially explain the paper's metric;")
            print("     the mechanism does NOT reduce to budget usage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
