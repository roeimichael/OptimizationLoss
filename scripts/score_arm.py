"""Single-class budget equalization, imported by scripts/full_panel.py.

This file used to carry a CLI as well. It was deleted: `score_run` truncated a
multi-class cap list with `cls[0]`, so only the first capped class was measured
and the "budget-equalized" metrics violated caps 2..n -- while FRAMEWORK claimed
"no [0] truncation anywhere". Its `--build-reference` also filtered on
warmup_epochs == 50, the regime the framework forbids headlining, and its
defaults pointed at `newdirections/`, `paper/scripts/` and `results/headroom`
(the LR-trap campaign), none of which exist.

`full_panel.py` is THE scorer. `equalize` here is used only on its single-class
path; `equalize_multi` in full_panel handles several capped classes.
"""
import numpy as np

# The reference is the COMPUTE-MATCHED short-warm-up campaign, not the frozen
# warm-up-50 grid. Two reasons, and both are load-bearing:
#
#   Regime. At warm-up 50 the CE-saturation gate has already fired, so nothing
#   is learning during the constraint phase and every method can only
#   re-threshold a frozen score vector. Optimal re-thresholding IS the post-hoc
#   clipper, so that regime is unwinnable by construction and tells us nothing
#   about a new loss.
#
#   Fairness. The post-hoc arms do no constraint-phase training at all -- they
#   train `warmup_epochs` and allocate -- so at short warm-up an unmatched
#   comparison pits a ~26-epoch model against a 1-epoch model. The headroom
#   campaign pins every arm to the same total optimizer epochs (post-hoc arms
#   warmup=B; trained arms warmup=1 + constraint_epochs=B-1), which is the only
#   comparison that isolates the objective from the compute.


def equalize(y_proba, gids, glob_c, loc, cls):
    """Fill to exactly K: the K highest-scoring samples get the constrained
    class subject to each group's cap, everything else takes its best remaining
    class. Same rule the post-hoc clipper follows, applied to every arm, which
    is what makes the budget a constant instead of a free variable."""
    K = int(glob_c[cls])
    order = np.argsort(-y_proba[:, cls])
    room = {int(g): int(l[cls]) for g, l in loc.items()} if (gids is not None and loc) else {}
    chosen = np.zeros(len(y_proba), dtype=bool)
    taken = 0
    for i in order:
        if taken >= K:
            break
        if room:
            g = int(gids[i])
            if room.get(g, 0) <= 0:
                continue
            room[g] -= 1
        chosen[i] = True
        taken += 1
    other = y_proba.copy()
    other[:, cls] = -np.inf
    y = np.argmax(other, axis=1)
    y[chosen] = cls
    return y


