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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.constants import UNLIMITED                     # noqa: E402
from src.losses.transductive_loss import uniform_grad_count   # noqa: E402

# Below this, the boundary item is one the model is already certain about and
# the penalty's per-item gradient there is negligible. Calibrated on the two
# cells above, so treat it as a flag, not a law.
REACHABLE = 0.040


def slope_at(p_col, k, mode):
    """Per-item gradient of the configured count at the K-th ranked item.

    WHY THIS TAKES `mode`. `p(1-p)` is the slope of `sum` and of NOTHING else,
    and `soft_count_mode` has had three legal values since 2026-08-24. This
    file hardcoded `p(1-p)` and printed it as THE reachability verdict, so a
    `uniform` campaign -- whose whole purpose is that the per-item slope is a
    population CONSTANT -- would have been priced with the slope of the arm it
    replaces, and priced `flat at K` exactly where it is designed to be live.

    `uniform`'s weight comes from the shipped function rather than being
    restated: `uniform_grad_count` returns `p + w*(u - u.detach())` with
    `du_c/dz_c = 1`, so the slope in the class logit is `w` for EVERY item.
    """
    import torch
    srt = np.sort(p_col)[::-1]
    p = float(srt[k - 1])
    if mode == "sum":
        return p * (1.0 - p), p
    if mode == "uniform":
        # The slope must be taken w.r.t. the class LOGIT, not w.r.t. p: the
        # whole construction rests on `du_c/dz_c = 1`, and differentiating
        # against p instead returns `w / (p(1-p))` -- item-dependent, which is
        # the exact property `uniform` exists to remove. So push logits through
        # a softmax, as the trainer does. Collapsing the other classes into one
        # column is exact here, because `u_c = z_c - logsumexp(k != c)` already
        # treats them as a single competitor.
        q = np.clip(p_col, 1e-12, 1.0 - 1e-12)
        z = np.stack([np.log(q), np.log1p(-q)], axis=1)
        t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
        uniform_grad_count(torch.softmax(t, dim=1))[:, 0].sum().backward()
        return float(t.grad[:, 0].mean()), p
    if mode == "cut":
        # The window is CENTRED on rank K by construction, so the slope at the
        # K-th ranked item is sech^2(0) = 1.0 exactly -- its maximum, and the
        # largest value any count in this project takes there. That is the
        # whole design and it needs no data to state.
        #
        # ⚠️ DO NOT READ THIS AS A DOSE. Under `constraint_grad_mode:
        # normalize` the delivered step is rescaled to `constraint_grad_clip`
        # regardless, so a slope of 1.0 against `sum`'s 0.026 is a statement
        # about WHERE the gradient sits in the ranking, not about how hard it
        # pushes. FRAMEWORK 2(z12) and the magnitude-is-void result.
        return 1.0, p
    raise SystemExit(
        "soft_count_mode %r has no reachability slope here. `margin`'s slope "
        "is the sigmoid's and depends on the DERIVED window temperature, "
        "which the second table below already prices per width -- read that "
        "instead of a single number at K." % mode)


def count_mode(run_dir):
    try:
        cfg = json.load(open(run_dir / "config.json"))
    except Exception:
        return "sum"
    return str((cfg.get("hyperparams") or {}).get("soft_count_mode", "sum"))


def budgets(run_dir):
    """The integer budget per capped class, for ANY arm.

    The training log is the direct source, but ONLY for trained arms. A
    post-hoc arm runs `constraint_epochs: 0`, so every row it writes carries
    the hardcoded `Limit_Class = inf` default from log_progress_to_csv's
    signature and this returned {} for the clipper -- the exact bar every
    comparison is scored against. A caller that skips a run with no budget
    then drops the control silently and reports the treatment alone.

    So fall back to the config, where the cap is a FRACTION of the class's
    true count (`constraint: [local, global]` x `capped_classes`). Verified
    against the trained arms' own logs on mcbar/dermmnist: 0.3 x {103, 220,
    223} -> {31, 66, 67}, the integers their logs record.
    """
    try:
        t = pd.read_csv(run_dir / "training_log.csv")
    except Exception:
        t = None
    out = {}
    if t is not None:
        for col in t.columns:
            if col.startswith("Limit_Class"):
                v = pd.to_numeric(t[col], errors="coerce")
                v = v[v < UNLIMITED]
                if len(v):
                    out[int(col[len("Limit_Class"):])] = int(v.iloc[-1])
    if out:
        return out

    try:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        frac = float(cfg["constraint"][0])
        classes = cfg.get("capped_classes") or cfg["dataset_config"]["constrained_class"]
        y = pd.read_csv(run_dir / "final_predictions_raw.csv")["True_Label"]
    except Exception:
        return {}
    for c in classes:
        n = int((y == int(c)).sum())
        if n:
            out[int(c)] = int(round(frac * n))
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

    print("SLOPE AT THE BUDGET CUT -- p(1-p) at the K-th RANKED item")
    print("This is the CUT, not the decision boundary. See the second")
    print("table: predictions change at the boundary, and there the slope")
    print("is near its MAXIMUM.")
    print()
    print("%-44s %6s %5s %8s %9s %s"
          % ("run", "class", "K", "p at K", "d count", "verdict"))
    print("`d count` is the slope of the run's OWN soft_count_mode, not always"
          " p(1-p).")
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
            slope, p = slope_at(r[col].to_numpy(), k, count_mode(d))
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
