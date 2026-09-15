"""The per-epoch snapshot must be invisible to the run it observes.

Training is bit-deterministic given (config, seed), and that is load-bearing:
it is how we established that two campaigns which looked like a replication
were one campaign re-run. An observer that advances an RNG stream, or leaves a
model in a different mode than it found it, breaks that silently -- the run
still completes, still looks healthy, and simply stops matching every result
already on disk. Nobody would notice for weeks.

The other property pinned here is structural: the snapshot writes
PROBABILITIES and never touches a label. An earlier version scored inside the
training loop, which required handing the test labels to `TrainInputs`; two
standing gates rejected it and were right to, because "it only observes" is a
promise about intent that no gate can check. These tests hold the module to the
shape that makes the promise unnecessary.
"""
import csv
import os

import numpy as np
import torch

from src.training import epoch_trace


def _fixture(training=True):
    """Model and inputs, built BEFORE any RNG snapshot a test takes.

    Constructing a Linear and a random tensor consumes the torch RNG. Doing it
    inside the measured region would charge those draws to the snapshot and
    make the test fail for a reason that has nothing to do with it.
    """
    model = torch.nn.Linear(3, 2)
    model.train(training)
    return model, torch.randn(6, 3)


def test_the_next_draw_is_the_one_the_run_would_have_seen(tmp_path):
    # The property that actually matters: what training draws next must not
    # depend on whether the snapshot ran between two epochs.
    model, X_test = _fixture()
    trace = epoch_trace.writer(str(tmp_path))

    torch.manual_seed(11)
    expected = torch.rand(4)

    torch.manual_seed(11)
    epoch_trace.record(trace, epoch_1based=1, phase="constraint",
                       train_acc=0.9, model=model, X_test=X_test)
    got = torch.rand(4)

    assert torch.equal(expected, got), (
        "the snapshot advanced the RNG; training after it would diverge")


def test_the_model_mode_is_restored_in_both_directions(tmp_path):
    trace = epoch_trace.writer(str(tmp_path))
    for training in (True, False):
        model, X_test = _fixture(training)
        epoch_trace.record(trace, epoch_1based=1, phase="constraint",
                           train_acc=0.9, model=model, X_test=X_test)
        assert model.training is training


def test_it_writes_one_probability_file_per_epoch_and_indexes_them(tmp_path):
    model, X_test = _fixture()
    trace = epoch_trace.writer(str(tmp_path))

    for e in (1, 2, 3):
        epoch_trace.record(trace, epoch_1based=e, phase="constraint",
                           train_acc=0.5 + e / 10.0, model=model, X_test=X_test)

    rows = list(csv.DictReader(
        open(os.path.join(str(tmp_path), epoch_trace.INDEX), encoding="utf-8")))
    assert [r["epoch_absolute_1based"] for r in rows] == ["1", "2", "3"]
    assert all(not r["error"] for r in rows)

    for r in rows:
        proba = np.load(os.path.join(str(tmp_path), r["probs_file"]))
        assert proba.shape == (6, 2)
        # Real probabilities, so the offline scorer can allocate on them.
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    # A partial run must leave usable snapshots: one file per epoch, nothing
    # rewritten, so a campaign that dies at epoch 40 still has 39 of them.
    assert len(os.listdir(os.path.join(str(tmp_path), epoch_trace.DIRNAME))) == 3


def test_a_snapshot_failure_is_recorded_and_not_raised(tmp_path):
    class Diverged(torch.nn.Module):
        def forward(self, _x):
            raise RuntimeError("CUDA out of memory")

    model = Diverged()
    model.train(True)
    trace = epoch_trace.writer(str(tmp_path))

    epoch_trace.record(trace, epoch_1based=7, phase="constraint",
                       train_acc=0.9, model=model, X_test=torch.randn(4, 3))

    rows = list(csv.DictReader(
        open(os.path.join(str(tmp_path), epoch_trace.INDEX), encoding="utf-8")))
    assert "out of memory" in rows[0]["error"]
    assert not rows[0]["probs_file"]
    assert model.training is True, "the mode must be restored even on failure"


def test_the_module_never_mentions_a_test_label():
    # Structural, not stylistic. The gates that rejected the first design read
    # the training modules for label access; this keeps the helper they call
    # honest too, so the property cannot drift back in behind a rename.
    import inspect
    source = inspect.getsource(epoch_trace)
    body = source.split('"""', 2)[-1]               # drop the module docstring
    body = "\n".join(line for line in body.splitlines()
                     if not line.strip().startswith("#"))
    for forbidden in ("y_test", "y_true", "true_label"):
        assert forbidden not in body.lower(), (
            "%s appears in the snapshot code; labels must stay out of the "
            "training path entirely" % forbidden)
