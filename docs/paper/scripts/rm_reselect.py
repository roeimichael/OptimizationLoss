"""Is 'budget usage is near-identical, so there is no fill asymmetry for the
metric to expose' a valid inference?

It would be valid only if ccF1eq differed from ccF1adj ONLY through how many
items get the budget. It does not. ccF1eq also throws away each method's own
decision about WHICH items get it and re-picks the top-K by score. So two runs
that spend exactly the same budget can still score differently, and the size of
that re-selection effect is what actually moves the comparison.

Test: restrict to runs where count_adj == count_eq (identical budget spent,
by construction no fill asymmetry at all) and ask whether the two metrics
still differ, and whether they differ by different amounts per method.

    python paper/scripts/rm_reselect.py
"""
import sys

import numpy as np
import pandas as pd

D = "paper/scripts/"
M5 = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
CELL = ["dataset", "model", "cap"]


def main():
    b30 = pd.read_csv(D + "rm_headroom_b30.csv")
    nce = pd.read_csv(D + "rm_headroom_b30_lrc0.0001_noceskip.csv")
    d = pd.concat([nce[nce.method != "x"], b30[b30.method.isin(
        ["heuristic", "danits_lp"])]], ignore_index=True)

    print("=" * 108)
    print("A. reproduce the quoted pooled fill numbers")
    print("=" * 108)
    for lbl, x in [("headroom_b30 (all 5 arms)", b30),
                   ("noceskip trained + b30 clippers", d)]:
        g = x.groupby("method").agg(count_adj=("count_adj", "mean"),
                                    K=("K", "mean"),
                                    exactK=("deficit", lambda s: float((s == 0).mean())))
        print("\n  %s" % lbl)
        print(g.reindex([m for m in M5 if m in g.index]).round(4).to_string())
        print("    quoted range: count_adj 85.1-85.8, exactly-K 0.771-0.833 -> "
              "observed %.1f-%.1f and %.3f-%.3f"
              % (g.count_adj.min(), g.count_adj.max(),
                 g.exactK.min(), g.exactK.max()))

    print()
    print("=" * 108)
    print("B. RUNS THAT SPEND EXACTLY THE SAME BUDGET  (count_adj == count_eq):")
    print("   if the metric only equalized budget, ccF1eq would equal ccF1adj "
          "on every one of them")
    print("=" * 108)
    for lbl, x in [("headroom_b30", b30), ("noceskip trained + b30 clip", d)]:
        s = x[x.count_adj == x.count_eq].copy()
        s["gap"] = s.ccF1eq - s.ccF1adj
        print("\n  %s : %d of %d runs spend an identical budget" %
              (lbl, len(s), len(x)))
        print("    of those, ccF1eq != ccF1adj in %d (%.0f%%)"
              % (int((s.gap.abs() > 1e-12).sum()),
                 100.0 * (s.gap.abs() > 1e-12).mean()))
        g = s.groupby("method")["gap"].agg(["mean", "std", "min", "max", "count"])
        print(g.reindex([m for m in M5 if m in g.index]).round(4).to_string())
        gm = g["mean"]
        print("    per-method re-selection gain spans %+0.4f .. %+0.4f  "
              "(spread %0.4f) -- this is the asymmetry the claim says is absent"
              % (gm.min(), gm.max(), gm.max() - gm.min()))

    print()
    print("=" * 108)
    print("C. WHERE THE BIG CELL-LEVEL REVERSAL COMES FROM")
    print("   octmnist / MobileNetV3 / L30_G30, TraLO minus best clipper:")
    print("   ccF1adj -0.0600 -> ccF1eq +0.0015 (a 0.0615 swing that reverses "
          "the verdict)")
    print("=" * 108)
    q = d[(d.dataset == "octmnist") & (d.model == "MobileNetV3")
          & (d.cap == "L30_G30")]
    cols = ["method", "seed", "K", "count_raw", "count_adj", "count_eq",
            "deficit", "ccF1adj", "ccF1eq", "AP"]
    print(q[cols].sort_values(["method", "seed"])
          .to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\n  per-method means in that one cell:")
    print(q.groupby("method")[["count_raw", "count_adj", "count_eq",
                               "ccF1adj", "ccF1eq", "AP"]]
          .mean().reindex([m for m in M5 if m in set(q.method)])
          .round(4).to_string())
    print("\n  budget deficit is at most %d counts out of K=%d; the metric gap "
          "is %.4f F1. The gap is a RE-SELECTION effect, not a fill effect."
          % (q.deficit.max(), q.K.iloc[0],
             (q.groupby('method').ccF1eq.mean() - q.groupby('method').ccF1adj.mean()).abs().max()))

    print()
    print("=" * 108)
    print("D. duplicate-key contamination in the file set the claim scores")
    print("=" * 108)
    for f in ["rm_extra_robustness.csv"]:
        x = pd.read_csv(D + f)
        for m in ["fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp", "tralo"]:
            s = x[x.method == m]
            if not len(s):
                continue
            dup = s.groupby(CELL + ["seed"]).size()
            print("  %-22s %-14s rows=%3d  keys=%3d  max rows/key=%d"
                  % (f, m, len(s), len(dup), dup.max()))
    print("\n  final_decomp.cells() pivots with the default aggfunc='mean', so "
          "any key with >1 row is silently averaged.")
    print("  It also appends a row for EVERY (dataset,model,cap) group, even "
          "groups with no clipper at all, so its printed")
    print("  'cells=' is a group count, not a count of comparable cells "
          "(extra_robustness: prints 16, only 8 are comparable).")
    x = pd.read_csv(D + "rm_extra_robustness.csv")
    ncomp = 0
    for k, g in x.groupby(CELL):
        if set(g.method) & {"heuristic", "danits_lp"} and set(g.method) & {
                "fioretto_ldf", "hounie_rcl"}:
            ncomp += 1
    print("  verified: extra_robustness has %d (dataset,model,cap) groups, of "
          "which %d contain both a dual and a clipper."
          % (x.groupby(CELL).ngroups, ncomp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
