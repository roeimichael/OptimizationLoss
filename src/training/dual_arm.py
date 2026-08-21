"""The dual arms' shared scaffolding: hard counts, excess, and the end-of-run tail.

WHY IT IS SHARED. `fioretto_ldf.train` and `fioretto_alm.train` were
AST-IDENTICAL -- 49 statements, byte-for-byte after normalising the log strings
-- and `hounie_rcl.train` differed only in passing one fewer (unused) argument
and in hoisting a dict read out of a `no_grad` block. Three copies of the same
104-line tail is how `reordering_report` came to exist in one arm only, how the
checkpoint restore was gated in two arms and unconditional in the third, and
how the constraint step drifted 20x between arms before
`src/training/constraint_step.py` collapsed it. The arms are being COMPARED to
each other, so a divergence in this code is a difference in the MEASUREMENT.

Nothing here is method-specific. Each arm supplies its own `_train_constraints`
-- the dual rule, which is the only thing the comparison is about -- and gets
the same warm-up-baseline capture, final excess accounting, checkpoint restore
and reordering report as the other two.
"""

import logging

import torch
import torch.nn.functional as F

from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.training.reordering import capped_scores, reordering_report
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def hard_predictions(model, X_test_dev, chunk_size):
    """Chunked argmax over the whole test set, as a numpy array."""
    with torch.no_grad():
        all_hard = []
        for i in range(0, len(X_test_dev), chunk_size):
            chunk_logits = model(X_test_dev[i:i + chunk_size])
            all_hard.append(chunk_logits.argmax(dim=1))
        return torch.cat(all_hard).cpu().numpy()


def count_excess(hard_preds, groups_np, constrained_classes, global_con, local_con):
    """Predictions above budget, summed over BOTH scopes. Integer arithmetic.

    Deriving satisfaction from the global scope alone reports a run as
    satisfied while a local group is over its ceiling, which is the whole
    difference the local caps exist to make.
    """
    excess = sum(
        max(0, int((hard_preds == c).sum()) - int(global_con[c]))
        for c in constrained_classes if global_con[c] < UNLIMITED
    )
    if local_con:
        for g_id, bounds in local_con.items():
            for c in constrained_classes:
                if bounds[c] < UNLIMITED:
                    gc = int(((hard_preds == c) & (groups_np == g_id)).sum())
                    excess += max(0, gc - int(bounds[c]))
    return excess


def ce_epoch(model, train_loader, optimizer, criterion_ce, device,
             amp_dtype, use_amp, scaler):
    """One epoch of cross-entropy on the TRAIN split; returns the batch losses.

    CE keeps running through every constraint epoch, which is what makes the
    trained arms equal-compute with the post-hoc ones: 30 optimizer epochs on
    both sides, no arm getting more opportunity to memorize than another.
    """
    model.train()
    ce_losses = []
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits_ce = model(batch_X)
            ce_loss = criterion_ce(logits_ce, batch_y)
        ce_losses.append(ce_loss.item())
        if scaler:
            scaler.scale(ce_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            ce_loss.backward()
            optimizer.step()
    return ce_losses


def transductive_counts(model, X_test_dev, groups_np, unique_groups,
                        num_classes, chunk_size, device):
    """Pass 1: soft (probability-sum) and hard counts over the test set, no grad.

    `model.eval()` first, and it is load-bearing: dropout off, and BN running
    stats NOT updated from test data -- that would be a leakage source that
    flips a few borderline samples and corrupts the dual update (AUDIT C1).

    Returns `(total_soft, group_soft, hard_preds)`, with `group_soft` keyed by
    plain ints so a `numpy.int64` group id and an `int` one cannot address two
    different buckets.
    """
    model.eval()
    total_soft = torch.zeros(num_classes, device=device)
    group_soft = {int(g): torch.zeros(num_classes, device=device)
                  for g in unique_groups}
    all_hard = []
    with torch.no_grad():
        for i in range(0, len(X_test_dev), chunk_size):
            chunk_logits = model(X_test_dev[i:i + chunk_size])
            chunk_proba = F.softmax(chunk_logits, dim=1)
            total_soft += chunk_proba.sum(dim=0)
            all_hard.append(chunk_logits.argmax(dim=1))
            chunk_groups = groups_np[i:i + chunk_size]
            for g in unique_groups:
                mask = (chunk_groups == g)
                if mask.any():
                    group_soft[int(g)] += chunk_proba[mask].sum(dim=0)
        hard_preds = torch.cat(all_hard).cpu().numpy()
    return total_soft, group_soft, hard_preds


def run_dual_arm(inputs: TrainInputs, train_constraints, tag) -> TrainOutputs:
    """Warm-up baseline -> the arm's dual loop -> excess -> restore -> report.

    `train_constraints(model, inputs, device)` returns the six-tuple every dual
    produces. `tag` only names the arm in the log lines.
    """
    hp = inputs.hyperparams
    allow_restore = _required(hp, "enable_checkpoint_restore", bool)
    model = inputs.model
    device = inputs.device

    # Baseline for the reordering diagnostic, captured BEFORE a single
    # constraint step. This used to exist only in tralo/train.py, so for these
    # three arms "did the constraint phase reorder anything, or only shift a
    # bias the scorer cannot see" was unanswerable -- the same asymmetry that
    # made the CE-skip flag a 0.22 cc-F1 artifact.
    chunk_size = _required(hp, "constraint_chunk_size", int)
    warmup_scores = capped_scores(model, inputs.X_test,
                                  inputs.constrained_classes, chunk_size)

    (satisfaction_epoch, best_sat_state, best_sat_epoch,
     min_excess_state, min_excess_epoch, min_total_excess
     ) = train_constraints(model, inputs, device)

    # Apples-to-apples checkpoint restore (mirrors TraLO). The selection rule is
    # deterministic on the constraint-excess axis -- NOT on F1. Picking the
    # checkpoint by F1 over several candidates is double-dipping on the
    # evaluation metric.
    X_test_dev = inputs.X_test.to(device)
    model.eval()
    hard_preds = hard_predictions(model, X_test_dev, chunk_size)
    final_total_excess = count_excess(
        hard_preds, inputs.group_ids, inputs.constrained_classes,
        inputs.global_con, inputs.local_con)
    final_violates = final_total_excess > 0

    restored_from_epoch = None
    restore_kind = None
    # PARITY with tralo, which gates this and whose campaigns set it False.
    # Unconditional restore here meant that in a head-to-head only tralo kept
    # its trained model, while the others were swapped for a checkpoint chosen
    # on constraint satisfaction -- measured at -0.0351 AP within-run. Any
    # tralo win over the duals would have carried that advantage for free.
    # Default True: runs predating the flag keep their behaviour bit for bit.
    if not allow_restore:
        log.info("%s: enable_checkpoint_restore=False, keeping the trained model", tag)
    if allow_restore and best_sat_state is not None and final_violates:
        log.info("%s: final violates; restoring best-satisfied checkpoint from epoch %d",
                 tag, best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
    elif (allow_restore and min_excess_state is not None
          and final_total_excess > min_total_excess):
        log.info("%s: final excess=%d > min seen excess=%d (epoch %d); "
                 "restoring lowest-excess checkpoint",
                 tag, int(final_total_excess), int(min_total_excess), min_excess_epoch)
        model.load_state_dict({k: v.to(device) for k, v in min_excess_state.items()})
        restored_from_epoch = min_excess_epoch
        restore_kind = "min_excess"

    # AFTER the restore: the restored model is the one whose
    # predictions the scorer reads.
    reorder = reordering_report(model, inputs.X_test, warmup_scores,
                                inputs.constrained_classes, chunk_size)

    return TrainOutputs(
        model=model,
        summary={
            "satisfaction_epoch": satisfaction_epoch,
            "best_sat_epoch": best_sat_epoch,
            "min_excess_epoch": min_excess_epoch,
            "min_total_excess": (None if min_total_excess == float("inf")
                                 else int(min_total_excess)),
            "restored_from_epoch": restored_from_epoch,
            "restore_kind": restore_kind,
            # tau near 1.0 with a large bias_shift = the count moved and
            # the RANKING did not, which 9 of 13 scored metrics cannot see.
            "reordering": reorder,
        },
    )
