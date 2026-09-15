"""The budgeted ranking loss must do the one thing every count penalty cannot.

LEDGER PART 2.1 proves a multiset function cannot prefer a good ordering over a
bad one. The whole justification for this loss is that its gradient depends on
an item's position RELATIVE to the others in its group, so the items COMPETE for
the K slots. These tests pin that property directly rather than trusting the
derivation, because "it should re-rank" is exactly the kind of claim this
project has been wrong about five times.

The second property pinned here is that the loss is anchored at the K-th order
statistic. A surrogate that pushes on every pair equally is not cutoff-sensitive
and would just sharpen the ordering the score already induces -- the failure
mode the literature warns about for AP-style losses. An item far from the cut
must contribute essentially nothing.
"""
import pytest
import torch

from src.training.rank_loss import budgeted_rank_loss, read_rank_config


def _logits(scores_for_class_1):
    """Logits whose class-1 probability is monotone in the given values."""
    s = torch.tensor(scores_for_class_1, dtype=torch.float32)
    return torch.stack([torch.zeros_like(s), s], dim=1)


def test_positives_are_pushed_up_and_intruding_negatives_down():
    scores = [3.0, 2.5, 2.0, 2.8, 0.1, 0.2]      # index 3 is a high negative
    targets = torch.tensor([1, 1, 1, 0, 0, 0])
    groups = torch.zeros(6, dtype=torch.long)
    logits = _logits(scores).requires_grad_(True)

    loss = budgeted_rank_loss(logits, targets, groups, [1], 1.0, min_group=2)
    loss.backward()
    g = logits.grad[:, 1]

    assert g[3] > 0, "the intruding negative must be pushed DOWN"
    assert (g[:3] < 0).all(), "the true positives must be pushed UP"


def _detached_cut_reference(logits, targets, groups, classes, frac, margin=0.05):
    """The same loss with the cut treated as a CONSTANT.

    This is the null hypothesis for the competition claim. Every count penalty
    this project has run is uniform-direction; the whole argument for the
    budgeted ranking loss is that the cut MOVES with the items, so raising a
    negative lifts the threshold and thereby penalises the positives. That
    coupling is the only thing separating the real loss from this reference,
    and it is invisible to a test that merely checks gradient signs -- both
    versions push positives up and negatives down.
    """
    import torch.nn.functional as F
    proba = F.softmax(logits.float(), dim=1)
    total = logits.sum() * 0.0
    terms = 0
    for gid in torch.unique(groups):
        idx = torch.nonzero(groups == gid, as_tuple=True)[0]
        for c in classes:
            scores = proba[idx, c]
            positive = targets[idx] == c
            n_pos = int(positive.sum())
            if n_pos == 0 or n_pos == idx.numel():
                continue
            k = max(1, min(int(round(n_pos * frac)), scores.numel() - 1))
            t = torch.topk(scores, k).values[-1].detach()        # the difference
            total = total + F.softplus(margin + t - scores[positive]).mean()
            total = total + F.softplus(margin + scores[~positive] - t).mean()
            terms += 1
    return total / terms if terms else total


def test_the_cut_MOVES_with_the_items_so_they_actually_compete():
    # THE mechanism test, stated as a difference from the detached-cut null.
    # An earlier version of this test asserted only that positives go up and
    # negatives go down, which is ALSO true when the cut is frozen -- it passed
    # under a mutation that removed the coupling entirely. Compare against the
    # frozen-cut reference instead, which is the only thing that distinguishes
    # competition from two independent one-sided pushes.
    scores = [3.0, 2.5, 2.0, 2.8, 0.1, 0.2]
    targets = torch.tensor([1, 1, 1, 0, 0, 0])
    groups = torch.zeros(6, dtype=torch.long)

    a = _logits(scores).requires_grad_(True)
    budgeted_rank_loss(a, targets, groups, [1], 1.0, min_group=2).backward()

    b = _logits(scores).requires_grad_(True)
    _detached_cut_reference(b, targets, groups, [1], 1.0).backward()

    assert not torch.allclose(a.grad, b.grad, atol=1e-9), (
        "the cut is not coupled to the items: this loss is two independent "
        "one-sided pushes, not a competition for the K slots")


def test_the_gradient_is_not_the_same_for_every_item():
    # A count penalty's per-item gradient is identical up to the softmax
    # Jacobian. If this loss ever produced that, it would be the same family
    # and LEDGER PART 2.1 would apply to it too.
    torch.manual_seed(0)
    scores = [2.0, 1.0, 0.5, 1.8, 0.2, -1.0, -2.0, 0.9]
    targets = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0])
    groups = torch.zeros(8, dtype=torch.long)
    logits = _logits(scores).requires_grad_(True)

    budgeted_rank_loss(logits, targets, groups, [1], 1.0, min_group=2).backward()
    g = logits.grad[:, 1]

    assert g.abs().max() > 0
    assert g.std() > 1e-6, "per-item gradients must differ; a constant push is the bug"


def test_an_item_far_from_the_cut_contributes_almost_nothing():
    # Cutoff sensitivity. An AP-style surrogate would push hard on a confident,
    # already-correct item; this must not, because the allocator's decision
    # does not depend on it.
    scores = [20.0, 3.0, 2.5, 2.4, -20.0]
    targets = torch.tensor([1, 1, 0, 0, 0])
    groups = torch.zeros(5, dtype=torch.long)
    logits = _logits(scores).requires_grad_(True)

    budgeted_rank_loss(logits, targets, groups, [1], 1.0, min_group=2).backward()
    g = logits.grad[:, 1].abs()

    near = g[1:4].max()
    assert g[0] < near / 10, "a far-above-cut positive must be nearly free"
    assert g[4] < near / 10, "a far-below-cut negative must be nearly free"


def test_groups_are_scored_independently():
    # The deployed cap is PER GROUP. A cut computed across groups would be the
    # global top-K, which is a defect this project has already shipped once.
    scores = [5.0, 4.0, 0.0, -5.0, -4.0, -9.0]
    targets = torch.tensor([1, 0, 0, 1, 0, 0])
    both = torch.tensor([0, 0, 0, 1, 1, 1])
    logits_a = _logits(scores).requires_grad_(True)
    budgeted_rank_loss(logits_a, targets, both, [1], 1.0, min_group=2).backward()

    # Group 1's items sit far below group 0's. If the cut were global, group 1
    # would be entirely below it and its positive would get a large push.
    g = logits_a.grad[:, 1]
    assert g[3] < 0, "group 1's positive is pushed up by ITS OWN group's cut"
    assert abs(float(g[3])) == pytest.approx(abs(float(g[0])), rel=0.5), (
        "a group's gradient must not depend on another group's score scale")


def test_a_group_with_no_positive_or_no_negative_is_skipped():
    # No cut is definable, and forcing one would invent a target.
    scores = [1.0, 2.0, 3.0]
    logits = _logits(scores).requires_grad_(True)
    groups = torch.zeros(3, dtype=torch.long)

    all_pos = budgeted_rank_loss(logits, torch.tensor([1, 1, 1]), groups, [1],
                                 1.0, min_group=2)
    assert float(all_pos) == 0.0

    all_neg = budgeted_rank_loss(logits, torch.tensor([0, 0, 0]), groups, [1],
                                 1.0, min_group=2)
    assert float(all_neg) == 0.0


def test_a_small_group_is_skipped_and_the_result_stays_differentiable():
    logits = _logits([1.0, 2.0]).requires_grad_(True)
    loss = budgeted_rank_loss(logits, torch.tensor([1, 0]),
                              torch.zeros(2, dtype=torch.long), [1], 1.0,
                              min_group=8)
    assert float(loss) == 0.0
    loss.backward()                      # must not raise: still attached
    assert logits.grad is not None


def test_the_budget_fraction_moves_the_cut():
    # The cut is the K-th order statistic, so a tighter cap must change which
    # items are penalised. If the fraction did nothing, the loss would not be
    # budgeted at all.
    scores = [3.0, 2.0, 1.0, 2.5, 1.5, 0.5]
    targets = torch.tensor([1, 1, 1, 0, 0, 0])
    groups = torch.zeros(6, dtype=torch.long)

    grads = {}
    for frac in (0.34, 1.0):
        logits = _logits(scores).requires_grad_(True)
        budgeted_rank_loss(logits, targets, groups, [1], frac,
                           min_group=2).backward()
        grads[frac] = logits.grad[:, 1].clone()

    assert not torch.allclose(grads[0.34], grads[1.0]), (
        "the budget fraction must change the cut, or the loss is not budgeted")


@pytest.mark.parametrize("hp", [
    {"rank_weight": -1.0},
    {"rank_weight": 1.0, "rank_margin": 0.0},
    {"rank_weight": 1.0, "rank_margin": 2.0},
    {"rank_weight": 1.0, "rank_min_group": 1},
])
def test_the_config_rejects_values_that_would_silently_do_nothing(hp):
    with pytest.raises(ValueError):
        read_rank_config(hp)


def test_the_loss_is_off_by_default():
    # It must be opt-in: every existing arm has to keep reproducing.
    assert read_rank_config({})["weight"] == 0.0


def test_the_ranking_arms_get_a_DIFFERENT_warm_up_cache_than_their_controls():
    """The inert-flag trap, closed at its actual cause.

    The budgeted ranking loss acts during the WARM-UP, and `run_warmup` loads
    from a cache keyed by `base_model_id`. If `rank_weight` were not a warm-up
    identity key, `rank_clip` would load `clip`'s cached warm-up and come out
    byte-identical to it -- an arm that runs, logs, and measures exactly
    nothing. Five flags in this project have failed in that shape.
    """
    from configs.gen_campaign import (
        build_hyperparams, compute_base_model_id, load_protocol)

    P = load_protocol()
    keys = P["warmup_identity_keys"]
    assert "rank_weight" in keys and "rank_margin" in keys

    dc = P["datasets"]["fmow2"]
    ids = {}
    for arm in ("clip", "rank_clip", "aug_clip", "aug_rank_clip"):
        hp = build_hyperparams(P, P["arms"][arm], 1)
        ids[arm] = compute_base_model_id(P, "MobileNetV3", hp, "fmow2", dc)

    assert ids["clip"] != ids["rank_clip"], (
        "rank_clip would reuse clip's cached warm-up and measure nothing")
    assert ids["aug_clip"] != ids["aug_rank_clip"]
    # and the plain arms must be untouched by the new keys, or every stored
    # warm-up in the corpus is invalidated
    assert ids["clip"] != ids["aug_clip"]
