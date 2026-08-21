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

🛑 MEASURE IT EARLY -- THE WINDOW CLOSES AS CE CONVERGES. Reachability is not a
property of the cap alone; it is a property of the cap AND how converged the
model is. The same L50_G30 cell measured on a 4-epoch model has its boundary at
p = 0.939, and measured on the 30-epoch model at p = 0.9990 -- p(1-p) falls from
0.055 to 0.0009, a factor of 60. Run this on a model at the START of the
constraint phase. Run it on a converged model and it will say OUT OF REACH for
every cell, because by then it is.

🔑 THIS IS WHAT "CE SATURATES" MEANS, MECHANICALLY. The oldest rule in the
framework is never to run warm-up 50, because CE saturates and every method
becomes identical. Saturation IS the boundary probability approaching 1: at
p = 0.999 the penalty's per-item gradient at the cut is 0.001, so the count term
has nothing left to push with exactly where the cap reads. Warm-up 1 is not an
arbitrary choice -- it is the setting that leaves the largest reachable window.

USE IT AS A PRE-FLIGHT CHECK on an early-training run: a cell can be screened
before GPU time goes into it. A boundary at p > 0.97 will not respond, and the
honest move is to pick a different cap rather than tune a shape against a
vanishing gradient.

AND IT DOSES `cut_temp`. `soft_count_mode: margin` replaces the count with the
softened argmax, whose per-item weight is `sigma'(m/T)/T` at the margin
`m = p_ic - max_{c' != c} p_ic'`, instead of `p(1-p)`. That only helps if T is
matched to the cell's actual margin scale: too wide and the window spans
everything, the sigmoid flattens, and p(1-p) dominates again -- it degrades
silently back to `sum`. Too narrow and no item is inside it and the step is
zero. The second table prints how many items land in the window at a given T,
and how much of each count's total gradient reaches the items that can flip.

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


def margin_stats(r, c, k, temp):
    """Margin scale for one capped class.

    Returns (margin at the K-th ranked item, items inside the window, share of
    the plain count's gradient on the 20 items nearest the boundary, the same
    share for the margin count, a T that would put ~20 items in the window).

    The two shares are what the arm trades one for the other: the plain count
    weights an item by `p(1-p)` = dp/dlogit, the margin count by `sigma'(m/T)`.
    Only items near margin 0 can change their prediction, so the share landing
    there is the count's useful fraction.
    """
    cols = sorted((int(x[len("Prob_Class_"):]), x) for x in r.columns
                  if x.startswith("Prob_Class_"))
    P = r[[x for _, x in cols]].to_numpy(dtype=float)
    others = np.delete(P, c, axis=1).max(axis=1)
    m = P[:, c] - others
    near = np.argsort(np.abs(m))[:20]

    w_sum = P[:, c] * (1.0 - P[:, c])
    z = np.clip(m / max(temp, 1e-12), -30.0, 30.0)
    sig = 1.0 / (1.0 + np.exp(-z))
    w_mar = sig * (1.0 - sig)

    def share(w):
        tot = float(w.sum())
        return float(w[near].sum()) / tot if tot > 0 else 0.0

    kth = int(np.argsort(P[:, c])[::-1][min(k, len(m)) - 1])
    t_sug = float(np.sort(np.abs(m))[min(19, len(m) - 1)])
    return float(m[kth]), int((np.abs(m) < temp).sum()),         share(w_sum), share(w_mar), t_sug


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("root")
    a.add_argument("--cut-temp", type=float, default=0.02,
                   help="T for the margin window, to be dosed against the "
                        "margin scale this prints. Default matches "
                        "configs/protocol.yml.")
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
            verdict = "reachable" if slope >= REACHABLE else "OUT OF REACH"
            name = "/".join(d.relative_to(root).parts[-3:]) or d.name
            print("%-44s %6d %5d %8.4f %9.4f %s"
                  % (name[-44:], c, k, p, slope, verdict))
            seen.setdefault(verdict, 0)
            seen[verdict] += 1
            margin_rows.append((name[-44:], c)
                               + margin_stats(r, c, k, args.cut_temp))
    print()
    if margin_rows:
        print("MARGIN SCALE -- dosing `cut_temp` for soft_count_mode: margin"
              "  (T = %g)" % args.cut_temp)
        print()
        print("%-44s %6s %8s %7s %9s %9s"
              % ("run", "class", "m at K", "in win", "sum@20", "margin@20"))
        print("-" * 92)
        for name, c, m_at_k, n_in, sh_sum, sh_mar, t_sug in margin_rows:
            print("%-44s %6d %8.4f %7d %8.1f%% %8.1f%%"
                  % (name, c, m_at_k, n_in, 100 * sh_sum, 100 * sh_mar))
        print()
        print("`m at K` = the margin of the K-th ranked item. It is NOT the")
        print("boundary unless the cap already binds exactly: if the hard count")
        print("is 300 against K=44, the decision boundary sits at item 300 and")
        print("the 256 items between them are what has to peel. A large `m at K`")
        print("means the budget cut and the boundary are far apart, and the")
        print("margin count will work inward from the boundary, not from K.")
        print()
        print("`in win` = items with |margin| < T: how many the penalty can")
        print("actually reach. 0 means the arm takes a zero step. The whole")
        print("test set means the window is not a window and it has degraded")
        print("back to `sum`.")
        print()
        print("`sum@20` / `margin@20` = share of each count's total per-item")
        print("gradient landing on the 20 items nearest the decision boundary")
        print("-- the only items whose prediction can flip. That gap IS the")
        print("arm's entire claim, measured on this cell rather than argued.")
        n_in_all = [r[3] for r in margin_rows]
        t_all = [r[6] for r in margin_rows]
        print()
        print("suggested T for ~20 items in window: %.3g .. %.3g (median %.3g)"
              % (min(t_all), max(t_all), float(np.median(t_all))))
        if max(n_in_all) == 0:
            print("  T IS TOO NARROW ON EVERY CELL HERE: nothing is inside the")
            print("  window, so the constraint would contribute no gradient.")
        elif min(n_in_all) > 500:
            print("  T IS TOO WIDE: the window holds hundreds of items, which")
            print("  is the flat-sigmoid regime that reverts to `sum`.")
    n_bad = seen.get("OUT OF REACH", 0)
    if n_bad == sum(seen.values()) and sum(seen.values()) > 0:
        print("EVERY boundary is out of reach. If these runs are converged that")
        print("is expected and says nothing about the cap -- re-measure on a")
        print("model at the START of the constraint phase.")
        print()
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
