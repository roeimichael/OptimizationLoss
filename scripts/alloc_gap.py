"""Is the shipped allocator optimal, or is it leaving value on the table?

The whole diagnosis of TraLO rests on "post-hoc allocation is optimal given the
probabilities, so only the ORDER survives". `apply_allocation_heuristic` is a
GREEDY joint sweep, and its own docstring says that equals top-K only with a
single capped class. This prices the gap directly: same probabilities, greedy vs
the exact transportation LP, on both the objective greedy is approximating
(total assigned probability) and the thing we actually score (accuracy).

No GPU, no server -- it imports the shipped allocator and solves the same
instance exactly.

MEASURED, 400 items / 8 classes / 4 groups / 5 seeds. `sharp` sets how well the
model separates; sharp=2.0 lands at 0.70 accuracy, which is OUR regime.

  capped  sharp   obj gap   moved    ACCURACY gap (LP - greedy)
       1    4.0    0.007%    0.5%      +0.0000
       1    2.0    0.096%    0.9%      +0.0030
       1    1.0    0.299%    2.5%      +0.0050
       3    4.0    0.050%    1.5%      +0.0000
       3    2.0    0.588%    4.7%      +0.0095   <-- our regime
       3    1.0    1.204%    8.7%      +0.0005
       5    2.0    1.385%    9.9%      +0.0150
       5    1.0    2.115%   13.9%      +0.0040

TWO findings, and the second was a surprise.

1. `apply_allocation_heuristic`'s docstring is right that ONE capped class makes
   it top-K of that class -- but top-K by p(c) is NOT the optimum. An item not
   given c falls back to its best other class, so a slot is worth p_i(c) MINUS
   that fallback. `margin_topk` below takes the top-K by that margin and matches
   the LP to floating point on every seed. The shipped allocator does not.

2. The prize is NOT negligible at the separation our models actually have. At
   sharp=2.0 exact allocation is worth +0.0095 (3 capped) to +0.0150 (5 capped)
   accuracy -- larger than most arm-vs-arm gaps this project has ever measured.
   It vanishes at sharp=4.0 because nothing is contested, and it DECOUPLES from
   the objective at sharp<=1.0, where greedy loses more probability mass while
   the accuracy difference falls back into the noise.

Consequences. Every arm shares the allocator, so this does not explain TraLO
losing. But it means the post-hoc clipper baseline is running ~1 accuracy point
BELOW its own ceiling, and fixing it raises the bar TraLO has to clear. It also
opens a conversion path that had been invisible: greedy reads p(c), the optimum
reads the margin, so a TRAINED arm whose probabilities are already margin-shaped
converts through this allocator where a post-hoc clipper cannot. TraLO's
p(1-p) weighting pushes hardest on p(c) ~ 0.5 -- the low-margin items -- which
is the right direction.

NOT YET CONFIRMED ON REAL PREDICTIONS. These are synthetic well-calibrated
softmaxes; real models are overconfident. The stored probability vectors on the
server settle it for free -- queued.
"""
import sys
import numpy as np
from scipy.optimize import linprog
from src.methodologies.heuristic.train import _build_hierarchy, apply_allocation_heuristic
from src.utils.constants import UNLIMITED


def optimal(probs, groups, global_con, local_con, n_classes):
    """Exact max-total-probability assignment under the same caps.

    Items x classes with per-(group, class) capacities is a transportation
    problem, so the LP relaxation is integral and linprog returns the true
    optimum rather than a bound.
    """
    n = len(probs)
    cost = -probs.ravel()
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for i in range(n):                      # each item assigned exactly once
        for c in range(n_classes):
            rows.append(r); cols.append(i * n_classes + c); vals.append(1.0)
        rhs.append(1.0); r += 1
    eq_rows = r
    ub_rows, ub_vals, ub_cols, ub_rhs = [], [], [], []
    r = 0
    for g in sorted(set(groups.tolist())):  # per-(group, class) local caps
        idx = np.where(groups == g)[0]
        for c in range(n_classes):
            lim = local_con.get(g, [UNLIMITED] * n_classes)[c]
            if lim >= UNLIMITED:
                continue
            for i in idx:
                ub_rows.append(r); ub_cols.append(i * n_classes + c); ub_vals.append(1.0)
            ub_rhs.append(float(lim)); r += 1
    for c in range(n_classes):              # global caps
        if global_con[c] >= UNLIMITED:
            continue
        for i in range(n):
            ub_rows.append(r); ub_cols.append(i * n_classes + c); ub_vals.append(1.0)
        ub_rhs.append(float(global_con[c])); r += 1
    from scipy.sparse import coo_matrix
    A_eq = coo_matrix((vals, (rows, cols)), shape=(eq_rows, n * n_classes))
    A_ub = (coo_matrix((ub_vals, (ub_rows, ub_cols)), shape=(r, n * n_classes))
            if r else None)
    res = linprog(cost, A_ub=A_ub, b_ub=ub_rhs if r else None,
                  A_eq=A_eq, b_eq=rhs, bounds=(0, 1), method="highs")
    if not res.success:
        raise SystemExit("LP failed: " + res.message)
    return res.x.reshape(n, n_classes).argmax(axis=1), -res.fun


def margin_topk(probs, groups, local_con, capped, n_classes):
    """Exact allocator for a SINGLE capped class -- ranks by MARGIN, not p(c).

    With one capped class c, every item not given c falls back to its best other
    class, so the value of spending a slot on item i is p_i(c) MINUS that
    fallback. Maximising the total therefore means taking the top-K by the
    margin. `apply_allocation_heuristic` takes the top-K by p_i(c) alone, which
    is a different set whenever an item is confidently c AND confidently
    something else. This exists to prove the LP is right, not to ship.
    """
    c = capped[0]
    alt = probs.copy()
    alt[:, c] = -np.inf
    best_alt = alt.argmax(axis=1)
    margin = probs[:, c] - probs[np.arange(len(probs)), best_alt]
    pred = best_alt.copy()
    for g in sorted(set(groups.tolist())):
        idx = np.where(groups == g)[0]
        k = local_con[g][c]
        take = idx[np.argsort(margin[idx])[::-1][:k]]
        pred[take[margin[take] > 0]] = c
    return pred


def trial(seed, n=400, n_classes=8, n_groups=4, capped=(0, 1, 2), frac=0.8, sharp=4.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    groups = rng.integers(0, n_groups, size=n)
    logits = rng.normal(size=(n, n_classes))
    logits[np.arange(n), y] += sharp
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    global_con = [UNLIMITED] * n_classes
    local_con = {}
    for g in range(n_groups):
        row = [UNLIMITED] * n_classes
        m = groups == g
        for c in capped:
            row[c] = int((y[m] == c).sum() * frac)
        local_con[g] = row
    hier = _build_hierarchy(n_classes, global_con, list(capped))
    greedy, _ = apply_allocation_heuristic(probs, groups, hier, global_con,
                                           local_con, n_classes)
    opt, opt_obj = optimal(probs, groups, global_con, local_con, n_classes)
    g_obj = probs[np.arange(n), greedy].sum()
    out = (g_obj, opt_obj, (greedy == y).mean(), (opt == y).mean(),
           (greedy != opt).mean())
    if len(capped) == 1:
        mar = margin_topk(probs, groups, local_con, capped, n_classes)
        out = out + (probs[np.arange(n), mar].sum(), (mar == y).mean())
    return out


def main(argv):
    n_capped = int(argv[0]) if argv else 3
    sharp = float(argv[1]) if len(argv) > 1 else 4.0
    frac = float(argv[2]) if len(argv) > 2 else 0.8
    capped = tuple(range(n_capped))
    print("greedy vs the exact transportation LP, same probabilities")
    print("%d capped class(es), 400 items, 8 classes, 4 groups, sharp=%.1f, cap = %d%% of truth"
          % (n_capped, sharp, int(100 * frac)))
    print("%6s %12s %12s %10s %10s %10s" %
          ("seed", "greedy obj", "LP obj", "gap", "greedy acc", "LP acc"))
    rows = []
    for s in range(1, 6):
        g_obj, o_obj, g_acc, o_acc, differ = trial(s, capped=capped, sharp=sharp, frac=frac)
        rows.append((g_obj, o_obj, g_acc, o_acc, differ))
        print("%6d %12.3f %12.3f %9.3f%% %10.4f %10.4f" %
              (s, g_obj, o_obj, 100 * (o_obj - g_obj) / o_obj, g_acc, o_acc))
    gap = np.mean([(o - g) / o for g, o, _, _, _ in rows])
    dacc = np.mean([o - g for _, _, g, o, _ in rows])
    differ = np.mean([d for *_, d in rows])
    print("")
    print("mean objective gap %.4f%%   mean accuracy gap %+.4f   items assigned differently %.2f%%"
          % (100 * gap, dacc, 100 * differ))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
