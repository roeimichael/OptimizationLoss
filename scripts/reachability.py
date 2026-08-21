"""Can a count penalty even REACH this cell's operating point? Ask before running.

A transductive count penalty differentiates the soft count `sum_i p_ic`, so an
item's share of the gradient scales with `dp/dlogit`, which is `p(1-p)` for the
class being pushed. That is near zero for an item the model is already certain
about. So the penalty can only move the cut when the K-th ranked item -- the one
that decides who is in the budget and who is out -- sits somewhere the
derivative is not vanishing.

MEASURED 2026-08-21, dermmnist x ViTB16, 4 seeds per cell, penalty vs its own
lambda=0 control:

    cell       class  p at K   p(1-p)    d capF1
    L30_G20      2    0.9730   0.0258    -0.012   (0 of 4 seeds positive)
    L30_G20      4    0.9768   0.0226
    L50_G30      2    0.9389   0.0550    +0.008   (4 of 4 seeds positive)
    L50_G30      4    0.9484   0.0484

A 2.1x difference in `p(1-p)` at the boundary separates "no signal at any shape
or dose" from "consistent signal at every seed". It also explains the archived
headroom result -- headroom grows with the cap because a looser budget puts the
cut where the gradient can still act, not merely because more items are in play.

*** MEASURE IT EARLY -- THE WINDOW CLOSES AS CE CONVERGES. Reachability is not a
property of the cap alone; it is a property of the cap AND how converged the
model is. The same L50_G30 cell measured on a 4-epoch model has its boundary at
p = 0.939, and measured on the 30-epoch model at p = 0.9990 -- p(1-p) falls from
0.055 to 0.0009, a factor of 60. Run this on a model at the START of the
constraint phase. Run it on a converged model and it will say OUT OF REACH for
every cell, because by then it is.

KEY: THIS IS WHAT "CE SATURATES" MEANS, MECHANICALLY. The oldest rule in the
framework is never to run warm-up 50, because CE saturates and every method
becomes identical. Saturation IS the boundary probability approaching 1: at
p = 0.999 the penalty's per-item gradient at the cut is 0.001, so the count term
has nothing left to push with exactly where the cap reads. Warm-up 1 is not an
arbitrary choice -- it is the setting that leaves the largest reachable window.

USE IT AS A PRE-FLIGHT CHECK on an early-training run: a cell can be screened
before GPU time goes into it. A boundary at p > 0.97 will not respond, and the
honest move is to pick a different cap rather than tune a shape against a
vanishing gradient.

AND IT DOSES `cut_window_items`. `soft_count_mode: margin` weights an item by
`sigma'(m/T)` at margin `m = p_ic - max_{c' != c} p_ic'` instead of by
`p(1-p)`, with T derived so the window holds a fixed number of ITEMS. Whether
that is an improvement is measurable without a GPU, and the answer is not
"always": measured over 160 (class, cell) points on the stored evidence, the
share of gradient reaching the 20 items nearest the decision boundary is 29.4%
for `sum`, 96.1% for margin at a 2-item window, and 3.7% at 40 -- so a WIDE
window is worse-targeted than the count it replaces. The crossover sits between
10 and 20 items on converged models, and higher at warm-up 1. The second table
below re-derives it on whatever runs you point it at.

!! `sum` IS NOT BLIND TO THE BOUNDARY, whatever the older reasoning says. 29.4%
of the weight on 2% of the items is 15x uniform. `p(1-p)` was measured at the
K-th RANKED item, which is not the decision boundary: with a hard count of 300
against K = 44 the boundary is at item 300 and rank 44 is buried inside the
class. Items flip at `m = 0`, where `p_ic` is near the runner-up and `p(1-p)`
is near maximal.

    python -m scripts.reachability <run-dir-or-campaign> [--cut-temp 0.02]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Below this, the boundary item is one the model is already certain about and
# the penalty's per-item gradient there is negligible. Calibrated on the two
# cells above, so treat it as a flag, not a law.
REACHABLE = 0.040


def budgets(run_dir):
    try:
        t = pd.read_csv(run_dir / "training_log.csv")
    except Exception:
        return {}
    out = {}
    for col in t.columns:
        if col.startswith("Limit_Class"):
            v = pd.to_numeric(t[col], errors="coerce")
            v = v[v < 1e9]
            if len(v):
                out[int(col[len("Limit_Class"):])] = int(v.iloc[-1])
    return out


def concentration(r, c, widths):
    """Share of each count's gradient on the 20 items nearest the boundary.

    Returns (sum_share, {width: margin_share}). Only items at `m ~ 0` can
    change their prediction, so this is the count's useful fraction -- and it
    is the whole claim the margin count makes, measurable before any GPU time.

    The plain count weights an item by `p(1-p)` = dp/dlogit; the margin count
    by `sigma'(m/T)` with T derived to hold `width` items.
    """
    cols = sorted((int(x[len("Prob_Class_"):]), x) for x in r.columns
                  if x.startswith("Prob_Class_"))
    P = r[[x for _, x in cols]].to_numpy(dtype=float)
    other = np.delete(P, c, axis=1).max(axis=1)
    m = P[:, c] - other
    near = np.argsort(np.abs(m))[:20]

    def share(w):
        tot = float(w.sum())
        return float(w[near].sum()) / tot if tot > 0 else 0.0

    w_sum = P[:, c] * (1.0 - P[:, c])
    out = {}
    am = np.sort(np.abs(m))
    for n in widths:
        T = max(float(am[min(max(int(n), 1), len(am)) - 1]), 1e-12)
        sig = 1.0 / (1.0 + np.exp(-np.clip(m / T, -30.0, 30.0)))
        out[n] = share(sig * (1.0 - sig))
    return share(w_sum), out


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("root")
    a.add_argument("--widths", type=int, nargs="+",
                   default=[2, 5, 10, 20, 40],
                   help="cut_window_items values to compare against `sum`. "
                        "The crossover -- the largest width still better "
                        "targeted than the plain count -- is the ceiling for "
                        "the dose.")
    args = a.parse_args()

    root = Path(args.root)
    runs = sorted(f.parent for f in root.rglob("final_predictions_raw.csv"))
    if not runs:
        print("no runs with predictions under %s" % root)
        return 1

    print("REACHABILITY -- p(1-p) at the K-th ranked item, per cell\n")
    print("%-44s %6s %5s %8s %9s %s"
          % ("run", "class", "K", "p at K", "p(1-p)", "verdict"))
    print("-" * 92)
    seen = {}
    margin_rows = []
    for d in runs:
        K = budgets(d)
        if not K:
            continue
        r = pd.read_csv(d / "final_predictions_raw.csv")
        for c, k in sorted(K.items()):
            col = "Prob_Class_%d" % c
            if col not in r.columns or k > len(r):
                continue
            p = float(np.sort(r[col].to_numpy())[::-1][k - 1])
            slope = p * (1.0 - p)
            verdict = "live at K" if slope >= REACHABLE else "flat at K"
            name = "/".join(d.relative_to(root).parts[-3:]) or d.name
            print("%-44s %6d %5d %8.4f %9.4f %s"
                  % (name[-44:], c, k, p, slope, verdict))
            seen.setdefault(verdict, 0)
            seen[verdict] += 1
            margin_rows.append(concentration(r, c, args.widths))
    print()
    if margin_rows:
        base = float(np.mean([x[0] for x in margin_rows]))
        print("GRADIENT ON THE BOUNDARY -- dosing `cut_window_items`"
              "   (%d class-cell points)" % len(margin_rows))
        print()
        print("Share of each count's total per-item gradient landing on the 20")
        print("items nearest the DECISION BOUNDARY -- the only items that can flip.")
        print()
        print("  %-24s %8s %9s" % ("count", "share", "vs sum"))
        print("  " + "-" * 44)
        print("  %-24s %7.1f%% %9s" % ("sum (the manuscript's)", 100 * base, "1.00x"))
        cross = 0
        for n in sorted(args.widths):
            mu = float(np.mean([x[1][n] for x in margin_rows]))
            print("  %-24s %7.1f%% %8.2fx" % ("margin, %d items" % n,
                                              100 * mu, mu / base if base else 0))
            if mu > base:
                cross = max(cross, n)
        print()
        if cross:
            print("CROSSOVER: margin is better targeted than `sum` at width <= %d."
                  % cross)
            print("A WIDER window is WORSE than the count it replaces -- T grows")
            print("until the sigmoid is flat over the whole margin range and the")
            print("weighting is nearly uniform. Set cut_window_items below this.")
        else:
            print("NO CROSSOVER at any width tried: `sum` is better targeted")
            print("than every margin window here. Try narrower widths, or take")
            print("the arm off the table for this cell.")
        print()
        print("Narrow is not free: at 2 items the step direction is set by two")
        print("items and `normalize` scales that to unit norm. Concentration")
        print("against variance is the trade-off, and it is a DOSE to sweep.")
        print()
        print("MEASURED ON THESE RUNS AS THEY ARE. If they are converged, `sum`'s")
        print("share is at its highest and the crossover is at its lowest --")
        print("re-measure at the START of the constraint phase before trusting it.")
        print()
    n_bad = seen.get("flat at K", 0)
    if n_bad == sum(seen.values()) and sum(seen.values()) > 0:
        print("The slope is flat at EVERY cut. If these runs are converged that")
        print("is expected and says nothing about the cap -- re-measure on a")
        print("model at the START of the constraint phase.")
        print()
    if n_bad:
        print("%d of %d (class, cell) CUTS sit where p(1-p) has gone flat."
              % (n_bad, sum(seen.values())))
        print()
        print("*** DO NOT READ THAT AS 'the penalty cannot act'. The cut is the")
        print("K-th RANKED item; predictions change at the DECISION BOUNDARY, which")
        print("is a different item whenever the hard count exceeds K -- and there")
        print("p(1-p) is near its MAXIMUM. Measured: `sum` puts 29.4% of its total")
        print("gradient on the 20 items nearest the boundary, 15x uniform. A flat")
        print("slope at the cut means the budget rank is buried inside the class,")
        print("which bears on how much HEADROOM the cap leaves, not on whether the")
        print("penalty has anywhere to push.")
    else:
        print("the slope is live at every cut.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
