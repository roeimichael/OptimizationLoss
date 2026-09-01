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

import csv
import logging

import torch
import torch.nn.functional as F

from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.pipeline.warmup import (make_ce_criterion, make_dataloader,
                                 make_optimizer)
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


def count_fields(constrained_classes):
    """Per-class count columns for the dual arms, named as `build_csv_header`.

    WHY THIS EXISTS. `tralo` logs a per-class Limit/Hard/Soft triple every
    epoch; the three dual arms logged only `total_excess`, a single summed
    scalar. So the project's CENTRAL quantity -- what the count did, per capped
    class, over the constraint phase -- was readable for one of four trained
    arms and `n/a (schema)` for the other three, and every cross-arm count
    comparison had to be reconstructed by hand from the stored predictions.

    It also hid a real finding for longer than it should have: the constraint
    moves one capped class at ~4x the noise floor and the other at or below it
    (section 2 of docs/FRAMEWORK.md), which a SUMMED excess cannot show by
    construction -- one class going down and the other up subtract inside it.

    The counts are already computed. `transductive_counts` returns
    `total_soft` and `hard_preds` every epoch and `count_excess` walks them;
    only the writing was missing, so this costs no forward pass, no extra
    device sync, and -- the part that matters -- draws nothing from the global
    RNG, so it cannot move a result the way an extra shuffled-loader pass
    would.

    The names match `src/training/logging.py` exactly so `scripts/log_health`,
    and anything else reading a training log, treats all four trained arms
    identically instead of branching on the arm.
    """
    return [f"{p}_Class{c}" for c in sorted(constrained_classes)
            for p in ("Limit", "Hard", "Soft")]


def count_row(hard_preds, total_soft, constrained_classes, global_con):
    """One epoch's per-class counts. Hard from argmax, soft from the sum."""
    row = {}
    soft = total_soft.detach().cpu().numpy()
    for c in sorted(constrained_classes):
        lim = global_con[c]
        row[f"Limit_Class{c}"] = (int(lim) if lim < UNLIMITED else UNLIMITED)
        row[f"Hard_Class{c}"] = int((hard_preds == c).sum())
        row[f"Soft_Class{c}"] = float(soft[c])
    return row


class Checkpoints:
    """Best-satisfied / lowest-excess model snapshots, on the EXCESS axis only.

    Selecting a checkpoint by F1 over several candidates is double-dipping on
    the evaluation metric, so the rule here reads the constraint excess and
    nothing else. Each arm kept these five values as loose locals and handed
    them back as a six-tuple, which is how the restore came to be gated in two
    arms and unconditional in a third.

    `allow_restore` gates the CLONE, not merely the restore: every config the
    generator emits sets it false, and a `state_dict()` copied to CPU each
    epoch is ~344 MB on ViTB16 for a checkpoint nothing ever reads.
    """

    def __init__(self, allow_restore, tag):
        self.allow_restore = allow_restore
        self.tag = tag
        self.satisfaction_epoch = None
        self.best_sat_state = None
        self.best_sat_epoch = None
        self.min_excess_state = None
        self.min_excess_epoch = None
        self.min_total_excess = float("inf")
        # HOW MANY CONSTRAINT STEPS ACTUALLY LANDED. finish_constraint_step
        # returns `applied`, and every arm used to bind it to `_applied` and
        # drop it -- so an epoch whose constraint gradient came back NaN or inf
        # silently took no step while the run still wrote `status: completed`.
        # Fioretto lost 10 of its 29 that way, and two arms in one campaign can
        # therefore differ by a third of their dose with nothing reading it.
        self.steps_applied = 0
        self.steps_attempted = 0

    def snapshot(self, model, satisfied, excess):
        """Clone BEFORE the constraint step, so the state matches these counts.

        Saving post-step state would mismatch the counts the epoch reported.
        """
        if self.allow_restore and (satisfied or excess < self.min_total_excess):
            return {k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()}
        return None

    def record_step(self, applied):
        """Count one attempted constraint step and whether it landed.

        Called on every epoch that reached `finish_constraint_step`, so the
        denominator is "epochs that formed a constraint gradient", not
        `constraint_epochs` -- an arm that skipped the backward entirely never
        attempted the step and must not be scored as having lost one.
        """
        self.steps_attempted += 1
        self.steps_applied += 1 if applied else 0

    def record(self, state, satisfied, excess, epoch):
        """`epoch` is 0-based; stored epochs are +1 so cross-method tables
        report the SAME epoch number for the same training step."""
        if satisfied and self.satisfaction_epoch is None:
            self.satisfaction_epoch = epoch + 1
            log.info("%s: first satisfaction at epoch %d", self.tag, epoch + 1)
        if satisfied and state is not None:
            self.best_sat_state = state
            self.best_sat_epoch = epoch + 1
        if excess < self.min_total_excess and state is not None:
            self.min_total_excess = excess
            self.min_excess_state = state
            self.min_excess_epoch = epoch + 1


# The raw gradient norm BEFORE the unit clip belongs in every arm's log. It is
# the whole dose question: the clip delivers exactly 1.000 against raw norms in
# the thousands, which makes the lambda ratchet a no-op. tralo logged it and
# the three duals discarded it, so the comparison was one arm wide.
def open_epoch_log(experiment_path, fields):
    """Write `training_log.csv`'s header now; return a one-row appender.

    Every epoch, in every arm. Logging every fifth epoch once made a 2-epoch
    diagnostic write a single row from BEFORE the treatment had differentiated,
    so four configs with four different outputs wrote byte-identical logs.
    """
    path = experiment_path / "training_log.csv"
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    def append(row):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
    return append


def read_step_config(hp):
    """The six knobs the constraint step reads, resolved ONCE for every arm.

    Each arm read these itself, which is the same duplication that let the
    delivered dose differ 20x between arms while every config said
    `constraint_grad_clip: 1.0`. A default that drifts between arms is
    invisible to `audit_config` -- the key HAS a reader in each of them -- so
    the only durable fix is one reader.

    Defaults reproduce the pre-2026-08-20 behaviour EXACTLY, so a config
    generated before these keys existed still runs and still gives the same
    numbers. That is why `_required` is not used for them: the danger it guards
    against is a default that silently disagrees with the protocol, and
    "clip"/False is the protocol's own historical value.

    Splat it into `finish_constraint_step(model, optimizer, scaler, **cfg)`.
    """
    mode = str(hp.get("constraint_grad_mode", "clip"))
    step_rule = str(hp.get("constraint_step_rule", "shared"))
    # 🛑 SAME GUARD AS `penalty_shape` AND `soft_count_mode`, FOR THE SAME
    # REASON. `finish_constraint_step` compares each of these against exactly
    # ONE literal ("normalize", "sgd"), so any other spelling -- `normalise`,
    # `Normalize`, `SGD` -- falls through to the DEFAULT behaviour and the arm
    # runs the default under a different arm name, then ties the default
    # because it IS the default. `audit_config` stays green (the key has a
    # reader) and `check_parity` stays green (one value across arms); only an
    # md5 across arms would catch it, and only if someone runs one.
    if mode not in ("clip", "normalize"):
        raise ValueError(
            "constraint_grad_mode must be clip / normalize, got %r. An "
            "unrecognised value silently runs `clip`." % mode)
    if step_rule not in ("shared", "sgd"):
        raise ValueError(
            "constraint_step_rule must be shared / sgd, got %r. An "
            "unrecognised value silently runs `shared`." % step_rule)
    return {
        "clip": _required(hp, "constraint_grad_clip"),   # the treatment dose
        "mode": mode,
        "fp32": bool(hp.get("constraint_fp32", False)),
        "step_rule": step_rule,
        "lr": _required(hp, "lr_constraint"),
        "random_direction": bool(hp.get("constraint_random_direction", False)),
    }


def dual_setup(model, inputs, device, lr, batch_size):
    """Optimizer, CE criterion and train loader -- built the same way in every arm.

    `criterion_ce` comes from the CONFIG, exactly as tralo builds it.
    Constructing `nn.CrossEntropyLoss()` directly in a trainer silently ignores
    `class_weighted_ce`, so three of the four trained arms would have run a
    different CE from the fourth the moment that key was turned on -- and no
    gate could see it, because the key IS read (by tralo) and IS emitted.
    """
    return (make_optimizer(model.parameters(), lr, device),
            make_ce_criterion(inputs.config, inputs.y_train,
                              inputs.num_classes, device),
            make_dataloader(inputs.X_train, inputs.y_train, batch_size))


def ce_epoch(model, train_loader, optimizer, criterion_ce, device,
             amp_dtype, use_amp, scaler):
    """One CE epoch on the TRAIN split; returns (batch losses, train accuracy).

    CE keeps running through every constraint epoch, which is what makes the
    trained arms equal-compute with the post-hoc ones: 30 optimizer epochs on
    both sides, no arm getting more opportunity to memorize than another.

    The accuracy is not decoration. The pipeline keeps the FINAL epoch
    unconditionally, so a run that falls off its own trajectory on the last
    epoch is the model that gets scored -- three such collapses were found in
    64 runs, one of them a CONTROL, and that single run reversed the sign of a
    4-seed headline. The detector for it reads train accuracy, and the dual
    arms did not record any, so a collapsed dual run was invisible.
    """
    model.train()
    ce_losses = []
    correct = seen = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits_ce = model(batch_X)
            ce_loss = criterion_ce(logits_ce, batch_y)
        ce_losses.append(ce_loss.item())
        with torch.no_grad():
            correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
            seen += batch_y.size(0)
        if scaler:
            scaler.scale(ce_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            ce_loss.backward()
            optimizer.step()
    return ce_losses, (correct / seen if seen else 0.0)


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

    `train_constraints(model, inputs, device)` runs the arm's own dual rule --
    the only thing the comparison is about -- and returns its `Checkpoints`.
    `tag` only names the arm in the log lines.
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

    ck = train_constraints(model, inputs, device)

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
    if allow_restore and ck.best_sat_state is not None and final_violates:
        log.info("%s: final violates; restoring best-satisfied checkpoint from epoch %d",
                 tag, ck.best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in ck.best_sat_state.items()})
        restored_from_epoch = ck.best_sat_epoch
        restore_kind = "fully_satisfied"
    elif (allow_restore and ck.min_excess_state is not None
          and final_total_excess > ck.min_total_excess):
        log.info("%s: final excess=%d > min seen excess=%d (epoch %d); "
                 "restoring lowest-excess checkpoint",
                 tag, int(final_total_excess), int(ck.min_total_excess), ck.min_excess_epoch)
        model.load_state_dict({k: v.to(device) for k, v in ck.min_excess_state.items()})
        restored_from_epoch = ck.min_excess_epoch
        restore_kind = "min_excess"

    # AFTER the restore: the restored model is the one whose
    # predictions the scorer reads.
    reorder = reordering_report(model, inputs.X_test, warmup_scores,
                                inputs.constrained_classes, chunk_size)

    return TrainOutputs(
        model=model,
        summary={
            "satisfaction_epoch": ck.satisfaction_epoch,
            "best_sat_epoch": ck.best_sat_epoch,
            "min_excess_epoch": ck.min_excess_epoch,
            "min_total_excess": (None if ck.min_total_excess == float("inf")
                                 else int(ck.min_total_excess)),
            "restored_from_epoch": restored_from_epoch,
            "restore_kind": restore_kind,
            # THE DOSE THAT ACTUALLY LANDED. applied < attempted means
            # non-finite constraint gradients dropped that epoch's step in
            # silence; two arms at 29 and 19 are not at equal dose, and until
            # this reached the run summary nothing could say so.
            "constraint_steps_applied": int(ck.steps_applied),
            "constraint_steps_attempted": int(ck.steps_attempted),
            # tau near 1.0 with a large bias_shift = the count moved and
            # the RANKING did not, which 9 of 13 scored metrics cannot see.
            "reordering": reorder,
        },
    )
