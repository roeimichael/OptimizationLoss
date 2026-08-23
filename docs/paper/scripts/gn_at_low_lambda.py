"""Does the clip actually bind at the lambda values the PAPER's regime reaches?

The mechanism claim -- the unit-norm clip divides lambda out -- was measured in
the `steps` arm, which uses lambda_step=0.05 and ratchets lambda to 1.4. The
warm-up-50 regime the paper reports never gets lambda above about 0.11. Whether
the clip still binds down there was left as an extrapolation.

It does not have to be. The steps arm ratchets THROUGH the low range on its way
up, logging the pre-clip norm at every epoch, so the question can be answered
from data already on disk: condition on lambda <= 0.15 and look at what
fraction of those steps exceed the bound.

⚠️ Not a perfect substitute for a warm-up-50 probe. In this arm low lambda occurs
only early, when the model is barely trained and the constraint is far from
satisfied; at warm-up 50 low lambda occurs against a saturated model sitting
near the cap. The penalty magnitude differs, so this bounds the answer rather
than settling it. But if the clip does not bind at low lambda even HERE -- where
the constraint is maximally violated and the penalty is at its largest -- it
will not bind at warm-up 50 either.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

LINE = re.compile(r"Epoch (\d+) \[Constraint\].*?lam_T=([\d.]+).*?gn=([\d.]+) clipped=([Yn])")
RUN = re.compile(r"Running (\S+/config\.json)")
BINS = [0, 0.08, 0.12, 0.2, 0.4, 0.8, 1e9]
LABELS = ["<=0.08", "0.08-0.12", "0.12-0.20", "0.20-0.40", "0.40-0.80", ">0.80"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+",
                    default=sorted(glob.glob(os.path.expanduser("~/steps*_l*.log"))))
    args = ap.parse_args()

    rows, cur = [], None
    for lg in args.logs:
        try:
            fh = open(lg, errors="replace")
        except IOError:
            continue
        with fh:
            for ln in fh:
                m = RUN.search(ln)
                if m:
                    parts = m.group(1).split("/")
                    cur = {"dataset": parts[-5], "arm": parts[-3]} if len(parts) >= 6 else None
                    continue
                g = LINE.search(ln)
                if g and cur:
                    rows.append({**cur, "lam": float(g.group(2)),
                                 "gn": float(g.group(3)), "clipped": g.group(4) == "Y"})
    if not rows:
        print("no gn lines parsed")
        return 1
    d = pd.DataFrame(rows)
    d = d[(d.gn > 0) & (d.arm == "steps_clip1")]      # incumbent bound only
    if d.empty:
        print("no steps_clip1 rows with a constraint step")
        return 1
    print("%d logged constraint steps at the incumbent clip bound of 1.0" % len(d))

    print()
    print("=" * 88)
    print("GRADIENT NORM AS A FUNCTION OF LAMBDA")
    print("=" * 88)
    d["bin"] = pd.cut(d.lam, BINS, labels=LABELS, right=False)
    t = d.groupby("bin", observed=True).agg(
        n=("gn", "size"), lam=("lam", "mean"), gn_median=("gn", "median"),
        gn_p90=("gn", lambda s: s.quantile(.9)), frac_clipped=("clipped", "mean"))
    print(t.round(4).to_string())

    print()
    print("=" * 88)
    print("THE QUESTION: at the paper's lambda (0.07-0.11), does the clip bind?")
    print("=" * 88)
    low = d[d.lam <= 0.15]
    if len(low) < 20:
        print("  only %d steps at lambda <= 0.15 -- too few to answer" % len(low))
    else:
        frac = float(low.clipped.mean())
        print("  n = %d steps with lambda <= 0.15" % len(low))
        print("  median gn   = %.4f   (bound is 1.0)" % low.gn.median())
        print("  p90 gn      = %.4f" % low.gn.quantile(.9))
        print("  %% clipped   = %.1f%%" % (100 * frac))
        print()
        if frac < 0.2:
            print("  -> the clip does NOT bind at the paper's lambda.")
            print("     'lambda is divided out' is a HIGH-lambda statement and does not")
            print("     transfer to warm-up 50. There lambda acts normally -- it is just tiny.")
        elif frac > 0.6:
            print("  -> the clip DOES bind even at low lambda; the mechanism transfers.")
        else:
            print("  -> mixed (%.0f%%); neither reading is safe." % (100 * frac))

    print()
    print("=" * 88)
    print("BY DATASET, restricted to lambda <= 0.15")
    print("=" * 88)
    if len(low):
        print(low.groupby("dataset").agg(
            n=("gn", "size"), gn_median=("gn", "median"),
            frac_clipped=("clipped", "mean")).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
