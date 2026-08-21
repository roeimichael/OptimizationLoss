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

USE IT AS A PRE-FLIGHT CHECK. This runs on predictions that already exist, so a
cell can be screened before any GPU time goes into it. A cell whose boundary
sits at p > 0.97 will not respond, and the honest thing is to pick a different
cap rather than to tune a shape against a vanishing gradient.

    python -m scripts.reachability <run-dir-or-campaign> [--caps L30_G20 L50_G30]
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


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("root")
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
            verdict = "reachable" if slope >= REACHABLE else "OUT OF REACH"
            name = "/".join(d.relative_to(root).parts[-3:]) or d.name
            print("%-44s %6d %5d %8.4f %9.4f %s"
                  % (name[-44:], c, k, p, slope, verdict))
            seen.setdefault(verdict, 0)
            seen[verdict] += 1
    print()
    n_bad = seen.get("OUT OF REACH", 0)
    if n_bad:
        print("%d of %d (class, cell) boundaries sit where the penalty's own"
              % (n_bad, sum(seen.values())))
        print("gradient vanishes. No shape and no dose reaches those -- the")
        print("lever is the CAP, not the loss.")
    else:
        print("every boundary is reachable; shape and dose are live levers here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
