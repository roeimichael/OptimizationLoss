"""The weighted soft count: does it reduce to the reference, and does it BITE?

Two failure modes are being pinned, and they pull in opposite directions.

  1. The reference arm must be untouched. `uniform` has to be exactly ones, so
     `w[:, None] * p` is bit-identical to `p` and every existing tralo result
     stays reproducible. A weighting feature that silently perturbs the control
     would invalidate the whole corpus.
  2. The weighting arm must not be INERT. Five flags in this project turned out
     to do nothing while looking healthy in the logs, so `weight_spread` has to
     be provably nonzero on a case where neighbourhoods really do disagree, and
     the weight has to land on the RIGHT items -- the ones surrounded by
     neighbours that predict something else.
"""
import numpy as np
import pytest
import torch

from src.training.item_weights import (
    item_weights,
    read_weight_config,
    weight_spread,
)


def _cfg(**kw):
    hp = {"constraint_weight": "knn_disagree", "constraint_weight_k": 3}
    hp.update(kw)
    return read_weight_config(hp)


def test_uniform_is_exactly_ones_and_leaves_the_count_bit_identical():
    feats = torch.randn(64, 8)
    preds = torch.randint(0, 4, (64,))
    gids = torch.zeros(64, dtype=torch.long)
    w = item_weights(read_weight_config({}), feats, preds, gids)

    assert torch.equal(w, torch.ones(64))
    assert weight_spread(w) == 0.0

    proba = torch.softmax(torch.randn(64, 4), dim=1)
    # bit-identical, not merely close: the reference arm must be reproducible.
    assert torch.equal(w[:, None] * proba, proba)


def test_the_weight_lands_on_the_item_whose_neighbours_disagree():
    # Two tight, far-apart clusters. Everyone predicts their cluster's class
    # except one infiltrator sitting inside cluster 0 while predicting class 1.
    torch.manual_seed(0)
    a = torch.randn(20, 6) * 0.01 + torch.tensor([10.0, 0, 0, 0, 0, 0])
    b = torch.randn(20, 6) * 0.01 + torch.tensor([0.0, 10, 0, 0, 0, 0])
    feats = torch.cat([a, b])
    preds = torch.cat([torch.zeros(20), torch.ones(20)]).long()
    infiltrator = 0
    preds[infiltrator] = 1
    gids = torch.zeros(40, dtype=torch.long)

    w = item_weights(_cfg(), feats, preds, gids)

    conformists = torch.cat([w[1:20], w[20:]])
    assert w[infiltrator] > conformists.max(), (
        "the item whose neighbours disagree must carry the largest weight")
    assert weight_spread(w) > 0.0, "a real disagreement pattern cannot be inert"


def test_weights_have_mean_one_inside_every_group():
    torch.manual_seed(1)
    feats = torch.randn(60, 6)
    preds = torch.randint(0, 3, (60,))
    gids = torch.tensor([0] * 25 + [1] * 35)

    w = item_weights(_cfg(), feats, preds, gids)

    for gid in (0, 1):
        assert w[gids == gid].mean().item() == pytest.approx(1.0, abs=1e-5)


def test_a_fully_agreeing_group_falls_back_to_the_reference_sum():
    # Every item agrees with every neighbour, so there is nothing to
    # differentiate and the arm must behave exactly like the unweighted sum.
    feats = torch.randn(30, 5)
    preds = torch.zeros(30, dtype=torch.long)
    gids = torch.zeros(30, dtype=torch.long)

    w = item_weights(_cfg(), feats, preds, gids)

    assert torch.allclose(w, torch.ones(30), atol=1e-6)
    assert weight_spread(w) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("hp", [
    {"constraint_weight": "knn"},                                  # unknown mode
    {"constraint_weight": "knn_disagree", "constraint_weight_k": 0},
    {"constraint_weight": "knn_disagree", "constraint_weight_floor": 0.0},
    {"constraint_weight": "knn_disagree", "constraint_weight_floor": 2.0},
])
def test_the_config_rejects_values_that_would_silently_do_nothing(hp):
    with pytest.raises(ValueError):
        read_weight_config(hp)


def test_k_is_clamped_when_the_group_is_smaller_than_k():
    feats = torch.randn(3, 4)
    preds = torch.tensor([0, 1, 1])
    gids = torch.zeros(3, dtype=torch.long)

    w = item_weights(_cfg(constraint_weight_k=50), feats, preds, gids)

    assert torch.isfinite(w).all()
    assert w.mean().item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_applying_uniform_weights_preserves_dtype_and_value_exactly(dtype):
    # float32 w times a half-precision proba PROMOTES in torch, which would
    # change the reference arm's accumulation precision and break bit-exact
    # reproduction of every stored tralo result. The cast in `apply` prevents it.
    from src.training.item_weights import apply

    proba = torch.softmax(torch.randn(16, 5), dim=1).to(dtype)
    w = torch.ones(16, dtype=torch.float32)

    out = apply(w, proba, 0, 16)

    assert out.dtype == proba.dtype
    assert torch.equal(out, proba)
