"""The budgeted ranking loss: train the CUT, not the COUNT.

WHY THIS IS THE ONLY REMAINING DIRECTION.

LEDGER PART 2.1 proves the constraint cannot re-order: the loss is a function of
the MULTISET of probabilities while the allocator is a function of the RANKS, so
any procedure reading only values of `L` cannot prefer a correct ordering over
the worst ordering with the same multiset. Every count penalty we have run --
weighted (`tralo_stab`, verified live at weight cv 0.83-0.88) or not -- is a
multiset function, and all of them failed.

The constrained-classification literature says the same thing from the other
side, and says where the one crack is. For selection-rate constraints of exactly
our form, the Bayes-optimal constrained classifier IS a group-wise thresholding
rule on the posterior (Zeng, Cheng & Dobriban 2024, via Neyman-Pearson;
Alabdulmohsin 2020), and post-processing attains the optimum **whenever the
score is Bayes-optimal** (Xian, Yin & Zhao, ICML; Fukuchi, ICML 2025). Our
greedy top-K allocator is the plug-in version of that optimum, which is why it
keeps winning. But Woodworth, Gunasekar, Ohannessian & Srebro (COLT 2017) show
post-processing a FIXED, NON-Bayes predictor can be strictly suboptimal, and
that in-processing is justified precisely through hypothesis-class restriction.

So a training-time win is permitted by the theory only by MAKING THE SCORE
BETTER at the place the allocator reads it. That is what this loss does.

HOW IT ESCAPES THE PROOF. For each (group, class) we take the K-th largest score
as the cut `t`, exactly where the deployed allocator will cut, and push true
positives above it and negatives below it:

    t      = K-th largest s_i(c) within the group
    L      = sum_{i: y=c}   softplus(margin + t - s_i)
           + sum_{i: y!=c}  softplus(margin + s_i - t)

`t` is a function of the items, so raising a negative raises the cut and thereby
penalises the positives: the items COMPETE for the K slots. dL/ds_i depends on
i's position RELATIVE to the other items in its group, which is a rank
dependence, not a multiset one. Items far from the cut contribute ~0.

WHY IT IS ANCHORED AT THE K-th ORDER STATISTIC AND NOT AT ALL PAIRS. A plain
pairwise or AP-style surrogate is NOT cutoff-sensitive -- it sharpens the whole
ordering, which is the ordering the score already induces, and would reduce to
"train the same classifier harder". Only an exact-K anchor targets the decision
the allocator actually makes (Petersen et al., ICML 2022; Xie et al., NeurIPS
2020; and the non-cutoff caveat is why Smooth-AP-style losses are the wrong tool
here). The anchor is the whole point of this file.

🛑 THIS USES TRAINING LABELS ONLY. It is ordinary supervised learning on the
train split with the deployment budget SIMULATED from train labels. No test
label enters any gradient -- FRAMEWORK forbids that, and the transductive budget
remains the only test-side information any arm receives. The test-time count
constraint is untouched by this file.
"""
import torch
import torch.nn.functional as F

DEFAULT_MARGIN = 0.05
DEFAULT_WEIGHT = 1.0
DEFAULT_MIN_GROUP = 8


def read_rank_config(hp):
    """The budgeted-ranking knob, validated. Absent means the loss is off."""
    weight = float(hp.get("rank_weight", 0.0))
    if weight < 0:
        raise ValueError("rank_weight must be >= 0, got %r" % weight)
    margin = float(hp.get("rank_margin", DEFAULT_MARGIN))
    if weight > 0 and not (0.0 < margin <= 1.0):
        raise ValueError("rank_margin must be in (0, 1], got %r" % margin)
    min_group = int(hp.get("rank_min_group", DEFAULT_MIN_GROUP))
    if weight > 0 and min_group < 2:
        raise ValueError("rank_min_group must be >= 2, got %r" % min_group)
    return {"weight": weight, "margin": margin, "min_group": min_group}


def budgeted_rank_loss(logits, targets, groups, constrained_classes, cap_fraction,
                       *, margin=DEFAULT_MARGIN, min_group=DEFAULT_MIN_GROUP):
    """Hinge around the per-(group, class) budget cut. Returns a scalar.

    `cap_fraction` mirrors the deployment cap: the budget for a (group, class)
    is round(fraction x number of true positives of that class in the group),
    which is how the campaign's own quotas are built.

    Returns a zero scalar (still attached to the graph) when no group in the
    batch is large enough to have a meaningful cut, so the caller can add it
    unconditionally.
    """
    proba = F.softmax(logits.float(), dim=1)
    total = logits.sum() * 0.0
    terms = 0
    for gid in torch.unique(groups):
        in_group = groups == gid
        if int(in_group.sum()) < min_group:
            continue
        idx = torch.nonzero(in_group, as_tuple=True)[0]
        for c in constrained_classes:
            scores = proba[idx, c]
            positive = targets[idx] == c
            n_pos = int(positive.sum())
            # A cut needs something to keep AND something to exclude.
            if n_pos == 0 or n_pos == int(in_group.sum()):
                continue
            k = max(1, min(int(round(n_pos * cap_fraction)), scores.numel() - 1))
            # The K-th largest score IS the deployed cut. Keeping it in the
            # graph is what makes the items compete: raising a negative lifts
            # the cut and so penalises every positive below it.
            t = torch.topk(scores, k).values[-1]
            pos = scores[positive]
            neg = scores[~positive]
            total = total + F.softplus(margin + t - pos).mean()
            total = total + F.softplus(margin + neg - t).mean()
            terms += 1
    return total / terms if terms else total
