"""Independent check of the fill-floor claim.

The forensics workflow's headline is that ONE quantity predicts whether a run
ends well: the model's own raw constrained-class count divided by the cap. It
reports a floor near 0.6*K -- below it the representation is damaged, above it
nothing matters, and over-predicting 2x costs nothing.

That claim now underpins everything else, so it gets re-derived here from the
raw files rather than trusted. Two things are done differently on purpose:

  1. Scores are centered WITHIN CELL (dataset, backbone, cap) before any
     aggregation. Without that, "fill predicts score" could just be method
     identity or dataset identity leaking in -- cells differ hugely in absolute
     ccF1eq (0.24 to 0.58), so a pooled correlation would find structure even if
     fill were irrelevant.
  2. The correlation is reported both raw and within-cell. If the within-cell
     version collapses, the effect is confounding, not mechanism.

Run against any campaign root; defaults to the one the workflow used.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "model", "cap"]
BINS = [0, 0.33, 0.60, 0.85, 1.10, 1.25, 1e9]
LABELS = ["<=0.33", "0.33-0.60", "0.60-0.85", "0.85-1.10", "1.10-1.25", ">1.25"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained",
                    default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    args = ap.parse_args()

    d = A.rows_for(args.trained)
    d = d[d.method.isin(["tralo", "fioretto_ldf", "hounie_rcl"])].copy()
    if d.empty:
        print("no runs")
        return 1
    d["fill"] = d.count_raw / d.K

    # centre within cell so neither dataset nor cap identity can carry the effect
    for m in ["ccF1eq", "AP", "macroEq"]:
        d["d_" + m] = d[m] - d.groupby(CELL)[m].transform("mean")

    d["bin"] = pd.cut(d.fill, BINS, labels=LABELS, right=False)
    print("=" * 78)
    print("FILL vs OUTCOME, deviation from the run's OWN cell mean   (%s)"
          % args.trained)
    print("=" * 78)
    t = d.groupby("bin", observed=True).agg(
        n=("fill", "size"), fill=("fill", "mean"),
        d_ccF1eq=("d_ccF1eq", "mean"), d_AP=("d_AP", "mean"),
        d_macroEq=("d_macroEq", "mean"))
    print(t.round(4).to_string())

    print()
    print("=" * 78)
    print("IS IT A FLOOR OR A TREND?")
    print("=" * 78)
    lo = d[d.fill < 0.60]
    hi = d[d.fill >= 0.60]
    print("  below 0.60*K : n=%3d   d_ccF1eq %+0.4f   d_AP %+0.4f"
          % (len(lo), lo.d_ccF1eq.mean(), lo.d_AP.mean()))
    print("  above 0.60*K : n=%3d   d_ccF1eq %+0.4f   d_AP %+0.4f"
          % (len(hi), hi.d_ccF1eq.mean(), hi.d_AP.mean()))
    if len(hi) > 8:
        r = hi[["fill", "d_ccF1eq"]].corr(method="spearman").iloc[0, 1]
        rap = hi[["fill", "d_AP"]].corr(method="spearman").iloc[0, 1]
        print("  ABOVE the floor, does more fill still help?"
              "  spearman(fill, d_ccF1eq)=%+0.3f  d_AP=%+0.3f" % (r, rap))
        print("  -> a floor predicts ~0 here; a trend predicts a clear positive")

    print()
    print("=" * 78)
    print("CONFOUNDING CHECK -- raw vs within-cell correlation")
    print("=" * 78)
    # NOTE Spearman is the WRONG statistic for a floor and will understate it:
    # a floor is flat above the knee, so half the range contributes no rank
    # signal at all. It is reported only to show that centering does not destroy
    # the effect. The binned table above is the statistic that matches the shape.
    for m in ["ccF1eq", "AP"]:
        raw = d[["fill", m]].corr(method="spearman").iloc[0, 1]
        cen = d[["fill", "d_" + m]].corr(method="spearman").iloc[0, 1]
        note = ("sign FLIPS under centering -- the raw value was cell identity"
                if raw * cen < 0 else
                "same sign before and after centering")
        print("  %-8s raw spearman %+0.3f   within-cell %+0.3f   %s"
              % (m, raw, cen, note))

    print()
    print("=" * 78)
    print("WHO LANDS BELOW THE FLOOR?  (this is the mechanism, not the metric)")
    print("=" * 78)
    d["below"] = d.fill < 0.60
    x = d.pivot_table(index="method", columns="dataset", values="below",
                      aggfunc=lambda s: "%d/%d" % (int(s.sum()), len(s)))
    print(x.to_string())
    print()
    print("mean fill per method x dataset")
    print(d.pivot_table(index="method", columns="dataset", values="fill")
          .round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
