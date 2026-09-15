"""The escape from the harm lemma, demonstrated on the gradient itself.

LEDGER PART 2.1 proves the reference constraint cannot re-order, and the proof
is short enough to check here rather than trust. The soft count is

    S_c = sum_i p_i(c)

so for any bounded penalty psi,

    dL / dp_i(c) = psi'(S_c)

which does not depend on i AT ALL. Every item in the scope is pushed by exactly
the same amount, and the only thing that breaks the tie is the softmax Jacobian
p_i(1 - p_i), which is itself a function of p -- the quantity the post-hoc
allocator already ranks by. That is why the constraint cannot tell a confident
correct item from a confident wrong one.

With per-item weights the count becomes S_c = sum_i w_i p_i(c), so

    dL / dp_i(c) = w_i psi'(S_c)

and the push is differentiated by w. This file pins both halves: that the
unweighted gradient really is constant across items (so the problem is real and
not a misreading of the source), and that the weighted one really is not (so the
proposed fix addresses that exact quantity). If the first test ever fails, the
bottleneck has moved and the argument for `tralo_stab` needs rewriting.
"""
import pytest
import torch

from src.training.item_weights import item_weights, read_weight_config


def _penalty(counts, bound):
    """A bounded penalty of the same shape as the one in the loss: psi(S - bound)."""
    return torch.nn.functional.softplus(counts - bound).sum()


def _grad_wrt_proba(w, proba, bound):
    """dL/dp for a weighted soft count, item by item."""
    p = proba.clone().requires_grad_(True)
    counts = (w[:, None] * p).sum(dim=0)
    _penalty(counts, bound).backward()
    return p.grad


def test_the_unweighted_gradient_is_identical_for_every_item():
    # This is the bottleneck, checked rather than assumed.
    torch.manual_seed(0)
    proba = torch.softmax(torch.randn(32, 4), dim=1)
    w = torch.ones(32)

    grad = _grad_wrt_proba(w, proba, bound=5.0)

    for c in range(4):
        column = grad[:, c]
        assert torch.allclose(column, column[0].expand_as(column), atol=1e-7), (
            "the reference constraint pushes every item in a class equally; "
            "if this fails, LEDGER PART 2.1 no longer describes the code")


def test_weighting_makes_the_push_differ_item_by_item():
    torch.manual_seed(0)
    proba = torch.softmax(torch.randn(32, 4), dim=1)
    # Two clusters, one infiltrator: exactly the structure the weight targets.
    feats = torch.cat([torch.randn(16, 5) * 0.01,
                       torch.randn(16, 5) * 0.01 + 10.0])
    preds = torch.cat([torch.zeros(16), torch.ones(16)]).long()
    preds[0] = 1
    gids = torch.zeros(32, dtype=torch.long)
    w = item_weights(read_weight_config(
        {"constraint_weight": "knn_disagree", "constraint_weight_k": 3}),
        feats, preds, gids)

    grad = _grad_wrt_proba(w, proba, bound=5.0)

    for c in range(4):
        column = grad[:, c]
        assert column.std() > 0, "a weighted count must differentiate the push"
    # and the differentiation is the weight, exactly: grad_i = w_i * psi'(S_c)
    ratio = grad[:, 0] / w
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-6), (
        "the per-item push must be exactly w_i times the shared psi'(S_c)")


def test_the_infiltrator_is_pushed_hardest():
    # The point of the whole exercise: the item whose neighbourhood says it does
    # not belong should feel the largest share of the constraint's push.
    torch.manual_seed(0)
    proba = torch.softmax(torch.randn(32, 4), dim=1)
    feats = torch.cat([torch.randn(16, 5) * 0.01,
                       torch.randn(16, 5) * 0.01 + 10.0])
    preds = torch.cat([torch.zeros(16), torch.ones(16)]).long()
    preds[0] = 1
    gids = torch.zeros(32, dtype=torch.long)
    w = item_weights(read_weight_config(
        {"constraint_weight": "knn_disagree", "constraint_weight_k": 3}),
        feats, preds, gids)

    grad = _grad_wrt_proba(w, proba, bound=5.0)

    assert grad[0, 0] == pytest.approx(grad[:, 0].max().item()), (
        "the infiltrator must receive the largest push")
