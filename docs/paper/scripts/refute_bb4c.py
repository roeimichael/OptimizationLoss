"""Part 3. Direction of the finer test, and the confound the contrast carries."""
import os
import sys
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402


def sect(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def main():
    sect("A  DIRECTION of the seed-level test (12 cells, counted not averaged)")
    n, k = 12, 8
    pv = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    print("  POSITIVE overshoot->edge slope in 8 of 12 cells (NEGATIVE in 4).")
    print("  P(>= 8 of 12 | fair coin) = %.3f" % pv)
    print("  The one cell reaching p<0.05, octmnist/RegNetY400MF/L30_G30 r=+0.976,")
    print("  is POSITIVE. So the finer-grained version of the same test, on the")
    print("  same runs, leans TOWARD the hypothesis the claim says is contradicted.")

    sect("B  THE CONTRAST IS CONFOUNDED. Swapping backbone changes far more than\n"
         "   overshoot, and on the 4 CONTRADICTS rows it changes them by more.")
    cl = A.rows_for("results/headroom/headroom_b30")
    cl = cl[cl.method == "heuristic"]
    b = cl.groupby(["dataset", "model"]).agg(
        AP=("AP", "mean"), macroEq=("macroEq", "mean"),
        ccF1eq=("ccF1eq", "mean"), raw=("count_raw", "mean")).reset_index()
    print(b.to_string(index=False, float_format=lambda x: "%.4f" % x))
    print()
    print("  %-12s %14s %14s %14s %12s" % ("dataset", "overshoot gap",
                                           "base AP gap", "base macroF1 gap", "verdict"))
    for ds, g in b.groupby("dataset"):
        g = g.set_index("model")
        og = (g.loc["MobileNetV3", "raw"] - g.loc["RegNetY400MF", "raw"]) / g.loc["RegNetY400MF", "raw"]
        ag = (g.loc["MobileNetV3", "AP"] - g.loc["RegNetY400MF", "AP"]) / g.loc["RegNetY400MF", "AP"]
        mg = (g.loc["MobileNetV3", "macroEq"] - g.loc["RegNetY400MF", "macroEq"]) / g.loc["RegNetY400MF", "macroEq"]
        v = "AGREES" if ds == "tissuemnist" else "CONTRADICTS"
        print("  %-12s %+13.1f%% %+13.1f%% %+13.1f%%  %12s"
              % (ds, 100 * og, 100 * ag, 100 * mg, v))
    print()
    print("  On octmnist the backbones differ by +3.7% in overshoot but +7.7% in")
    print("  base average precision -- the ranking quality that ccF1eq is built")
    print("  from -- so the contrast attributes to overshoot a gap that base")
    print("  capacity explains twice over. On tissuemnist, the one dataset the")
    print("  claim calls AGREES, overshoot moves +37.6% while AP moves +2.7%:")
    print("  that is the ONLY dataset where the contrast isolates overshoot at all.")

    sect("C  SUMMARY OF WHAT THE 6 ROWS ACTUALLY CONTAIN")
    rows = [
        ("dermmnist", "L30_G30", "CONTRADICTS", 3.5, 0.451, 0.633, "yes"),
        ("dermmnist", "L50_G50", "CONTRADICTS", 3.5, 0.575, 0.609, "yes"),
        ("octmnist", "L30_G30", "CONTRADICTS", 3.7, 0.300, 0.601, "no"),
        ("octmnist", "L50_G50", "CONTRADICTS", 3.7, 0.738, 0.553, "yes"),
        ("tissuemnist", "L30_G30", "AGREES", 37.6, 0.066, 0.015, "no"),
        ("tissuemnist", "L50_G50", "AGREES", 37.6, 0.380, 0.161, "no"),
    ]
    print("  %-12s %-8s %-12s %10s %9s %14s %8s"
          % ("dataset", "cap", "verdict", "overshoot", "edge p", "P(contra|boot)", "flips?"))
    for r in rows:
        print("  %-12s %-8s %-12s %9.1f%% %9.3f %14.3f %8s" % r)
    print()
    print("  Every row the claim calls CONTRADICTS sits on an overshoot gap of")
    print("  ~3.5%, an edge gap that fails its own t-test at p>=0.30, and a")
    print("  bootstrap verdict within 0.13 of a coin flip. The two AGREES rows")
    print("  sit on the only 37.6% overshoot gap in the study.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
