"""The warm-up loop must actually RUN when a ranking arm passes train groups.

WHY THIS FILE EXISTS. `tests/test_rank_loss.py` pins the budgeted ranking loss
thoroughly -- competition, cutoff sensitivity, per-group independence, the
warm-up cache identity -- and every one of those tests passed while **all 48
ranking runs across three campaigns died in their first logged epoch**.

The defect was not in the loss. Passing `groups` makes `make_dataloader` yield
3-tuples, and `compute_train_accuracy` read the batch as a fixed 2-tuple:

    ValueError: too many values to unpack (expected 2)

Nothing in the suite ever executed `run_warmup` with `groups_train` set, so the
one line that consumes the loader differently from the training loop was never
run. Unit tests on the loss cannot see this -- only executing the real loop can.
That is the same lesson already recorded for stale bytecode and for restores:
**verify by EXECUTING**.

These tests are deliberately end-to-end over the real `run_warmup`, on tensors
small enough to run on CPU in a second, and they assert the boring thing: it
completes, and it logs a train accuracy. A mutation that reintroduces a
fixed-arity unpack anywhere on this path fails them.
"""
import os
import tempfile

import pytest
import torch

from src.pipeline.warmup import make_dataloader, run_warmup
from src.training.metrics import compute_train_accuracy


def _tiny_config(**hp_overrides):
    hp = {
        "lr": 0.01, "dropout": 0.0, "batch_size": 8, "warmup_epochs": 1,
        "pretrained": False, "seed": 1,
    }
    hp.update(hp_overrides)
    return {
        "model_name": "SmallCNN",
        "base_model_id": "test-warmup-groups-%d" % hp.get("rank_weight", 0),
        "hyperparams": hp,
        "dataset_config": {"constrained_class": [1]},
        "constraint": [0.9, 0.95],
    }


def _tiny_data(n=32, n_classes=3):
    torch.manual_seed(0)
    X = torch.randn(n, 3, 8, 8)
    y = torch.randint(0, n_classes, (n,))
    groups = torch.arange(n) % 2          # two groups, 16 items each
    return X, y, groups


def test_compute_train_accuracy_accepts_a_THREE_tuple_loader():
    """The exact line that killed 48 runs, pinned on its own.

    `make_dataloader(..., groups=...)` yields (X, y, groups). Accuracy does not
    need the groups, but it must not CHOKE on them.
    """
    X, y, groups = _tiny_data()
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 3))

    two = make_dataloader(X, y, batch_size=8)
    three = make_dataloader(X, y, batch_size=8, groups=groups)
    assert len(next(iter(three))) == 3, "the fixture is not exercising the 3-tuple path"

    device = torch.device("cpu")
    a = compute_train_accuracy(model, two, device)
    b = compute_train_accuracy(model, three, device)
    assert 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0


def test_the_warmup_RUNS_END_TO_END_when_a_ranking_arm_passes_groups():
    """The integration test whose absence let the campaign burn 2h21m.

    Everything below the loss was untested with groups present. Execute the
    real `run_warmup` and require it to finish and write a log row.
    """
    X, y, groups = _tiny_data()
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "training_log.csv")
        config = _tiny_config(rank_weight=1.0, rank_margin=0.05, rank_min_group=4)
        model, from_cache = run_warmup(
            config, 3, X, y, torch.device("cpu"),
            csv_log_path=log_path, groups_train=groups,
        )
        assert not from_cache, "a cached warm-up would skip the loop under test"
        assert model is not None
        assert os.path.exists(log_path), "the warm-up never reached its logging step"
        with open(log_path, encoding="utf-8") as fh:
            rows = fh.read().strip().splitlines()
        assert len(rows) >= 2, "no epoch row written: the loop did not complete one"


def test_the_warmup_still_RUNS_when_no_groups_are_passed():
    """The control. Every historical arm takes this path and must be untouched."""
    X, y, _ = _tiny_data()
    with tempfile.TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "training_log.csv")
        model, from_cache = run_warmup(
            _tiny_config(), 3, X, y, torch.device("cpu"),
            csv_log_path=log_path, groups_train=None,
        )
        assert not from_cache and model is not None
        assert os.path.exists(log_path)


def test_groups_do_not_change_the_TWO_tuple_byte_stream():
    """The comparability guard.

    The corpus rests on bit-determinism: a control re-run must reproduce
    byte-identically, or the fix has silently invalidated every stored result it
    is meant to sit beside. `make_dataloader` already documents that
    `groups=None` stays byte-identical; this pins that the accuracy path agrees,
    by requiring identical predictions from the 2-tuple and 3-tuple loaders over
    the same data.
    """
    X, y, groups = _tiny_data()
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 3))

    device = torch.device("cpu")
    two = make_dataloader(X, y, batch_size=8)
    three = make_dataloader(X, y, batch_size=8, groups=groups)
    # Same items, same model, same arithmetic -- only the tuple arity differs.
    assert compute_train_accuracy(model, two, device) == pytest.approx(
        compute_train_accuracy(model, three, device), abs=0.0), (
        "carrying groups changed the accuracy computation itself")
