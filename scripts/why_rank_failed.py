"""WHY did the budgeted ranking loss fail? Measure the loss itself, on real scores.

The loss was built to supply the ONE thing a count penalty cannot: the "which".
LEDGER PART 2.1 proves a count is a function of the MULTISET of probabilities
while the allocator is a function of the RANKS, so no count term can prefer a
good ordering over a bad one with the same multiset. The budgeted ranking loss
anchors a hinge at the K-th order statistic -- the exact place the allocator
cuts -- so that items COMPETE for the K slots.

It did not work. This measures the loss function's own behaviour on REAL score
distributions (the stored probability columns of a completed run) to find out
why, testing three specific mechanical claims:

  1. IRREDUCIBLE FLOOR. The hinge is `softplus(margin + t - s)` on SOFTMAX
     probabilities, which live in [0, 1]. So the argument is bounded in
     [margin-1, margin+1] and softplus NEVER reaches zero. If the floor is large
     relative to the signal, most of the loss is a constant the optimiser cannot
     remove, and the gradient is a small perturbation on top of it.

  2. BUILT-IN UNSATISFIABILITY. K = round(n_pos * cap_fraction) with
     cap_fraction < 1, so by construction n_pos - K true positives sit BELOW the
     cut and are penalised forever. Worse, the cut is not detached: pushing one
     positive up raises t for every other positive. A treadmill.

  3. IS IT ACTUALLY CUTOFF-ANCHORED? The design claim is that items far from the
     cut contribute ~0. In LOGIT space that can be true for a reason that has
     nothing to do with the cut: the softmax Jacobian p(1-p) vanishes as p -> 0
     or 1. If the logit-space gradient tracks p(1-p) rather than distance from
     the cut, the loss is not a cutoff-anchored ranking term at all -- it is an
     uncertainty weighting, and the "which" it supplies is not the allocator's.
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/dsi/michaer8/optloss-rank")
from src.training.rank_loss import budgeted_rank_loss  # noqa: E402

MARGIN = 0.05


def main(run_dir, cap_fraction=0.9):
    cfg = json.load(open(run_dir + "/config.json", encoding="utf-8"))
    classes = cfg["dataset_config"]["constrained_class"]
    df = pd.read_csv(run_dir + "/final_predictions_raw.csv")
    y = df["True_Label"].to_numpy(int)
    groups = df["Group_ID"].to_numpy()

    print("=" * 74)
    print("1. THE IRREDUCIBLE FLOOR -- softplus on probabilities in [0, 1]")
    print("=" * 74)
    print("   per-term loss at the extremes, margin = %.2f:" % MARGIN)
    for d, what in [(-1.0, "perfectly ranked (s - t = 1.0)"),
                    (-0.5, "comfortably ranked"),
                    (0.0, "exactly at the cut"),
                    (+0.5, "badly ranked")]:
        print("      %-32s softplus(%+.2f) = %.4f"
              % (what, MARGIN + d, float(F.softplus(torch.tensor(MARGIN + d)))))
    floor = float(F.softplus(torch.tensor(MARGIN - 1.0)))
    ceil_ = float(F.softplus(torch.tensor(MARGIN + 1.0)))
    print("")
    print("   FLOOR  (best case, every item perfectly ranked) = %.4f per term" % floor)
    print("   CEILING(worst case)                            = %.4f per term" % ceil_)
    print("   -> the optimiser can move at most %.0f%% of the loss; the rest is"
          % (100.0 * (ceil_ - floor) / ceil_))
    print("      a constant floor it cannot remove.")

    print("")
    print("=" * 74)
    print("2. BUILT-IN UNSATISFIABILITY -- how many true positives sit below the cut")
    print("=" * 74)
    tot_pos = tot_below = 0
    for c in classes:
        for g in np.unique(groups):
            m = (groups == g)
            n_pos = int((y[m] == c).sum())
            if n_pos == 0:
                continue
            K = max(1, min(int(round(n_pos * cap_fraction)), int(m.sum()) - 1))
            tot_pos += n_pos
            tot_below += max(0, n_pos - K)
    print("   cap_fraction = %.2f" % cap_fraction)
    print("   true positives in capped classes      : %d" % tot_pos)
    print("   FORCED below the cut by construction  : %d  (%.1f%%)"
          % (tot_below, 100.0 * tot_below / max(tot_pos, 1)))
    print("   -> these are penalised at every step, forever, and cannot be fixed.")
    print("      And the cut is not detached: lifting one positive raises t for")
    print("      all the others. The term grinds against itself.")

    print("")
    print("=" * 74)
    print("3. IS IT CUTOFF-ANCHORED, OR JUST UNCERTAINTY WEIGHTING?")
    print("=" * 74)
    # Build logits whose softmax reproduces a realistic probability column, then
    # read the per-item gradient the loss actually delivers in LOGIT space.
    c = classes[0]
    g0 = np.unique(groups)[0]
    m = groups == g0
    p = df["Prob_Class_%d" % c].to_numpy()[m]
    lab = (y[m] == c).astype(int)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = torch.tensor(np.stack([np.zeros_like(p), np.log(p / (1 - p))], 1),
                         dtype=torch.float32, requires_grad=True)
    t_ = torch.tensor(lab, dtype=torch.long)
    gr = torch.zeros(len(p), dtype=torch.long)
    budgeted_rank_loss(logit, t_, gr, [1], cap_fraction, margin=MARGIN,
                       min_group=2).backward()
    gmag = logit.grad[:, 1].abs().numpy()

    proba = torch.softmax(logit.detach(), 1)[:, 1].numpy()
    k = max(1, int(round(lab.sum() * cap_fraction)))
    cut = np.sort(proba)[::-1][k - 1]
    dist = np.abs(proba - cut)
    jac = proba * (1 - proba)

    def corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print("   group %s, class %d: %d items, cut at p = %.4f" % (g0, c, len(p), cut))
    print("")
    print("   correlation of |gradient| with ...")
    print("      distance from the CUT        : %+.3f   (design claim: strongly NEGATIVE)"
          % corr(gmag, dist))
    print("      softmax Jacobian p*(1-p)     : %+.3f   (confound: how UNCERTAIN the item is)"
          % corr(gmag, jac))
    print("")
    near = gmag[dist <= np.quantile(dist, 0.25)].mean()
    far = gmag[dist >= np.quantile(dist, 0.75)].mean()
    unc = gmag[jac >= np.quantile(jac, 0.75)].mean()
    cert = gmag[jac <= np.quantile(jac, 0.25)].mean()
    print("   mean |gradient|:")
    print("      nearest quartile to the cut  : %.3e" % near)
    print("      farthest quartile from cut   : %.3e   (ratio %.1fx)" % (far, near / max(far, 1e-30)))
    print("      most UNCERTAIN quartile      : %.3e" % unc)
    print("      most CERTAIN quartile        : %.3e   (ratio %.1fx)" % (cert, unc / max(cert, 1e-30)))
    print("")
    print("   READING: if the Jacobian correlation dominates the cut correlation,")
    print("   the loss is weighting UNCERTAIN items, not items at the allocator's")
    print("   cut. The 'which' it supplies is then not the allocator's 'which',")
    print("   and LEDGER PART 2.1's gap is not actually closed by it.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.9))
