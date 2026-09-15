"""The ranking arms must get a DIFFERENT warm-up per CAP, not just per weight.

WHY THIS IS A SEPARATE TRAP FROM `rank_weight`.

`tests/test_rank_loss.py` already pins that `rank_clip` cannot reuse `clip`'s
cached warm-up, because `rank_weight` is a warm-up identity key. That test
passed while a second, independent version of the same bug was live.

The budgeted ranking loss cuts at the K-th order statistic, and K comes from the
CAP -- `warmup.py` reads `config["constraint"][0]`, so L80 and L90 train
genuinely different models. But the cap lives in `config["constraint"]`, NOT in
`hp`, and `compute_base_model_id` only hashes identity keys found in `hp`. Both
caps therefore hashed to ONE `base_model_id`. The first cap to run trained and
cached; the second silently loaded that cache. Every L90 rank cell was a model
trained for an 0.8 cut, then scored against an 0.9 allocation -- precisely the
train/deploy mismatch the loss exists to remove.

Caught 2026-09-15 at 2/40 runs in, by counting distinct warm-up identities in a
generated campaign rather than by reading the code.

The generalisable rule this file encodes: **an identity key must cover every
input that changes the warm-up, including inputs that do not live in `hp`.**
"""
import pytest

from configs.gen_campaign import (
    build_hyperparams, cap_pair, compute_base_model_id, load_protocol,
)


def _ids_by_cap(arm, caps=("L80_G95", "L90_G95"), model="MobileNetV3", ds="fmow2"):
    P = load_protocol()
    dc = P["datasets"][ds]
    out = {}
    for tag in caps:
        hp = build_hyperparams(P, P["arms"][arm], 1)
        # Mirror gen_campaign's stamping step exactly.
        if float(hp.get("rank_weight", 0.0)) > 0:
            hp["rank_cap_fraction"] = cap_pair(tag)[0]
        out[tag] = compute_base_model_id(P, model, hp, ds, dc)
    return out


@pytest.mark.parametrize("arm", ["rank_clip", "aug_rank_clip"])
def test_a_ranking_arm_trains_a_DIFFERENT_warmup_at_each_cap(arm):
    ids = _ids_by_cap(arm)
    assert len(set(ids.values())) == 2, (
        "%s hashes to ONE base_model_id across L80 and L90, so the second cap "
        "will load the first cap's cached warm-up -- a model trained for the "
        "wrong cut. ids=%s" % (arm, ids))


@pytest.mark.parametrize("arm", ["clip", "focal_clip", "aug_clip"])
def test_a_NON_ranking_arm_is_still_cap_invariant(arm):
    """The control, and the reason the fix is safe for the existing corpus.

    An arm that takes no constraint step trains one model and lets the cap act
    only in the post-hoc allocator. Its warm-up MUST stay shared across caps --
    if this started splitting, every stored warm-up in the cache would be
    invalidated and the corpus would become incomparable.
    """
    ids = _ids_by_cap(arm)
    assert len(set(ids.values())) == 1, (
        "%s split its warm-up by cap; that invalidates the cached corpus" % arm)


def test_rank_cap_fraction_is_declared_as_an_identity_key():
    P = load_protocol()
    keys = P["warmup_identity_keys"]
    assert "rank_cap_fraction" in keys
    assert "rank_weight" in keys and "rank_margin" in keys


def test_the_new_key_does_not_change_any_EXISTING_digest():
    """`compute_base_model_id` hashes only keys PRESENT in hp.

    Non-ranking arms never carry `rank_cap_fraction`, so adding it to the
    identity list must leave their digests bit-identical to what the stored
    cache was written under.
    """
    P = load_protocol()
    dc = P["datasets"]["fmow2"]
    for arm in ("clip", "focal_clip", "aug_clip"):
        hp = build_hyperparams(P, P["arms"][arm], 1)
        assert "rank_cap_fraction" not in hp
        with_key = compute_base_model_id(P, "MobileNetV3", hp, "fmow2", dc)
        # Recompute against an identity list that lacks the new key entirely.
        stripped = dict(P)
        stripped["warmup_identity_keys"] = [
            k for k in P["warmup_identity_keys"] if k != "rank_cap_fraction"]
        without_key = compute_base_model_id(stripped, "MobileNetV3", hp, "fmow2", dc)
        assert with_key == without_key, (
            "%s's digest moved; every cached warm-up for it is now unreachable" % arm)


def test_the_warmup_REFUSES_a_stamp_that_disagrees_with_the_cap_it_trains():
    """The drift guard, executed rather than read.

    The stamp and the trained cut come from two places. If they ever disagree,
    the cache is keyed on a cut the run does not use -- the original bug, back
    through a different door.
    """
    from src.pipeline.warmup import run_warmup
    import torch

    X = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 3, (16,))
    groups = torch.arange(16) % 2
    config = {
        "model_name": "SmallCNN",
        "base_model_id": "test-rank-cap-drift",
        "constraint": [0.9, 0.95],                  # trains at 0.9
        "dataset_config": {"constrained_class": [1]},
        "hyperparams": {
            "lr": 0.01, "dropout": 0.0, "batch_size": 8, "warmup_epochs": 1,
            "pretrained": False, "seed": 1,
            "rank_weight": 1.0, "rank_margin": 0.05, "rank_min_group": 4,
            "rank_cap_fraction": 0.8,               # ...but was HASHED at 0.8
        },
    }
    with pytest.raises(ValueError, match="rank_cap_fraction"):
        run_warmup(config, 3, X, y, torch.device("cpu"), groups_train=groups)
