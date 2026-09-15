"""Per-epoch probability snapshots: WHEN does the constraint reshape, and when does it damage?

THE QUESTION THIS EXISTS TO ANSWER. Every campaign so far has fixed a budget,
run it to the end, and scored the final model. That reports a SUM over epochs
and cannot separate the epochs where the constraint improved the boundary from
the epochs where it wrecked one cross-entropy was still building. The working
suspicion is that the sign flips at CE saturation: while CE is still moving the
boundary the constraint rides along, and once CE has memorised the training set
the constraint is kicking a dead surface. A single end-of-run number can never
show that, because both halves are already added together.

So every epoch's test probabilities are written to disk. Scoring happens
OFFLINE, in `scripts/epoch_curve.py`.

WHY IT IS SPLIT THAT WAY, AND WHY THAT IS NOT AN INCONVENIENCE. The first
version of this module scored inside the training loop, which meant handing the
test labels to `TrainInputs`. Two standing gates rejected it --
`test_no_methodology_reads_the_test_LABELS_except_to_count_them` and
`test_train_inputs_do_not_expose_held_out_labels` -- and they were right to.
The intent was observational, but intent is not a property a gate can check:
once the labels are in scope inside a training module, the only thing standing
between an observation and a leak is that nobody edits the file. Storing
probabilities keeps the labels out of the training path entirely, so the
property is structural rather than promised. It also means the curve can be
re-scored under any metric later without re-running a single epoch.

🛑 AN EPOCH CHOSEN ON THIS CURVE IS AN ORACLE, NOT A METHOD. The curve is scored
against the test set, so "stop at the best epoch" reads the answer off the
evaluation data and would report an upper bound, not a result. That is a
legitimate and useful thing to measure -- it bounds what any stopping rule could
win -- but it is not a stopping rule. A real one needs a held-out split that is
not the test set, which is the open question recorded in docs/MISSION.md.

The snapshot is pure inference under no_grad, and it restores the model's mode
and every RNG stream it touches, so a traced run stays bit-identical to an
untraced one. Training here is bit-deterministic given (config, seed) and that
is load-bearing -- it is how we established that two campaigns which looked like
a replication were one campaign re-run.
"""
import csv
import logging
import os

import numpy as np
import torch

log = logging.getLogger(__name__)

DIRNAME = "epoch_probs"
INDEX = "epoch_trace.csv"
FIELDS = ["epoch_absolute_1based", "phase", "train_acc", "probs_file", "error"]


def writer(experiment_path):
    """Create the snapshot directory and index, and return a recorder.

    Returns None-safe: callers hold the result and pass it to `record`, which
    does nothing when it is None.
    """
    out_dir = os.path.join(experiment_path, DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(experiment_path, INDEX)
    with open(index_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, FIELDS).writeheader()

    def append(row):
        with open(index_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, FIELDS).writerow(
                {k: row.get(k) for k in FIELDS})

    return {"dir": out_dir, "append": append}


def _probabilities(model, X_test, chunk):
    out = []
    with torch.no_grad():
        for i in range(0, len(X_test), chunk):
            logits = model(X_test[i:i + chunk]).float()
            out.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def record(trace, *, epoch_1based, phase, train_acc, model, X_test, chunk=256):
    """Snapshot this epoch's test probabilities. Never raises, never perturbs.

    One file per epoch rather than one growing archive: a run that dies at
    epoch 40 still leaves 39 usable snapshots, and nothing is rewritten.
    """
    if trace is None:
        return None
    row = {"epoch_absolute_1based": epoch_1based, "phase": phase,
           "train_acc": train_acc}

    was_training = model.training
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_rng = np.random.get_state()
    try:
        model.eval()
        proba = _probabilities(model, X_test, chunk)
        name = "epoch_%04d.npy" % epoch_1based
        np.save(os.path.join(trace["dir"], name), proba)
        row["probs_file"] = os.path.join(DIRNAME, name)
    except Exception as exc:                      # a snapshot is never fatal
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        log.warning("epoch snapshot failed at epoch %d: %s", epoch_1based, exc)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(np_rng)
        model.train(was_training)

    trace["append"](row)
    return row
