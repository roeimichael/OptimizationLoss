"""fioretto_ldf methodology: linear penalty + per-constraint subgradient ascent.

Lifted from the prior fioretto_research/run_fioretto.py module. Checkpoint
restoration mirrors TraLO (best_sat / min_excess on the excess axis, NOT F1)
so the checkpoint selector cannot double-dip on the F1 evaluation metric.
"""

import csv
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.pipeline.setup import setup_runtime
from src.pipeline.warmup import (make_ce_criterion, make_dataloader,
                                 make_optimizer)
from src.training.constraint_step import (
    constraint_autocast, constraint_backward, finish_constraint_step)
from src.training.dual_arm import (ce_epoch, count_excess, run_dual_arm,
                                   transductive_counts)
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _train_constraints(model, inputs, device):
    """Fioretto Algorithm 1/2: linear penalty + per-constraint subgradient dual ascent."""
    hp = inputs.hyperparams
    CLIP = _required(hp, "constraint_grad_clip")   # the treatment dose
    # Defaults reproduce the pre-2026-08-20 behaviour EXACTLY, so a config
    # generated before these keys existed still runs and still gives the same
    # numbers. That is why _required is not used here: the danger it guards
    # against is a default that silently disagrees with the protocol, and
    # "clip"/False is the protocol's own historical value.
    CONSTRAINT_GRAD_MODE = str(hp.get("constraint_grad_mode", "clip"))
    CONSTRAINT_FP32 = bool(hp.get("constraint_fp32", False))
    CONSTRAINT_STEP_RULE = str(hp.get("constraint_step_rule", "shared"))
    CONSTRAINT_RANDOM_DIR = bool(hp.get("constraint_random_direction", False))
    LR_CONSTRAINT = _required(hp, "lr_constraint")
    # Hoisted: the per-epoch snapshot clone is gated on this, and a
    # state_dict() copied to CPU each epoch for a checkpoint nothing
    # reads is ~344 MB per epoch on ViTB16.
    allow_restore = _required(hp, "enable_checkpoint_restore", bool)
    constraint_epochs = _required(hp, "constraint_epochs", int)
    # Apples-to-apples: same early-stop policy as TraLO (5 consecutive
    # satisfied epochs). Without this Fioretto runs the full epoch budget
    # while TraLO exits at ~100 — 3x gradient-budget asymmetry skews F1
    # comparisons. Default matches TraLO.
    stable_count_threshold = _required(hp, "stable_count_threshold", int)
    lr_c = _required(hp, "lr_constraint", float)
    if "fioretto_step_size" not in hp:
        raise ValueError(
            "fioretto_step_size is required in hyperparams. The runner used "
            "to default to 0.01 while the multi-methodology generator "
            "defaulted to 0.005, producing inconsistent baselines silently. "
            "Set it explicitly in your config (typical sweep: 0.001/0.005/0.01).")
    step_size = float(hp["fioretto_step_size"])
    batch_size = hp.get("batch_size", 64)
    # protocol.yml carries this in BOTH the constraint_phase and chunked
    # blocks, so the 256 inline default could only ever fire on a
    # hand-written config -- exactly what _required exists to refuse.
    chunk_size = _required(hp, "constraint_chunk_size", int)

    use_amp, amp_dtype, scaler = setup_runtime(device)

    constrained_classes = inputs.constrained_classes
    num_classes = inputs.num_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids

    # Read the SAME key fioretto_alm reads, in BOTH SCOPES. ALM's own docstring
    # says it is "identical to fioretto_ldf EXCEPT the dual update", so the
    # initial multipliers have to match or that sentence is false. The fix was
    # half-applied: the global scope took lam0 while the local scope stayed
    # hardcoded at 0.0, so sweeping the key would have initialised ALM's locals
    # and not LDF's -- a scope asymmetry that survives the unit-norm clip
    # (the clip rescales the step, it cannot restore a term that is absent).
    # Dormant at 0.0 today; live the moment anyone sweeps it.
    lam0 = float(hp.get("fioretto_lambda_init", 0.0))
    lambda_g = {c: lam0 for c in constrained_classes if global_con[c] < UNLIMITED}
    lambda_l = {}
    for group_id, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                lambda_l[(group_id, c)] = lam0

    log.info("Fioretto LDF: %d epochs, lr=%.2e, step_size=%.4f, "
             "%d global + %d local multipliers",
             constraint_epochs, lr_c, step_size, len(lambda_g), len(lambda_l))

    optimizer = make_optimizer(model.parameters(), lr_c, device)
    # Built from the config, exactly as tralo does. Constructing
    # nn.CrossEntropyLoss() directly here silently ignored
    # class_weighted_ce, so three of the four trained arms would have
    # run a different CE from the fourth the moment that key was
    # turned on -- and no gate could see it, because the key IS read
    # (by tralo) and IS emitted.
    criterion_ce = make_ce_criterion(inputs.config, inputs.y_train,
                                     num_classes, device)
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, batch_size)

    X_test_dev = inputs.X_test.to(device)
    unique_groups = np.unique(groups_np)

    satisfaction_epoch = None
    # Best-checkpoint restore (apples-to-apples with TraLO): snapshot model
    # state BEFORE the constraint step at every epoch that satisfies or that
    # improves on the lowest total excess seen so far. After the loop, if the
    # final epoch violates we restore best_sat; else if final excess exceeds
    # the lowest seen, we restore min_excess. Prior implementation tracked
    # only best_excess + F1-of-final-vs-best (double-dipped on the eval
    # metric); this matches TraLO and removes that bias.
    best_sat_state = None
    best_sat_epoch = None
    min_excess_state = None
    min_excess_epoch = None
    min_total_excess = float("inf")

    log_path = inputs.experiment_path / "training_log.csv"
    last_grad_norm = 0.0
    log_fields = ["epoch", "ce_loss", "constraint_loss", "total_excess",
                  "all_satisfied", "max_lambda_g",
                  # The raw norm BEFORE the unit clip. It is the whole dose
                  # question: FRAMEWORK measures the clip delivering exactly
                  # 1.000 against a raw norm of thousands, which makes the
                  # lambda ratchet a no-op. tralo logged it and these three
                  # discarded it, so the comparison was one arm wide.
                  "grad_norm"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, log_fields).writeheader()

    stable_count = 0  # consecutive epochs with all_satisfied for early-stop parity with TraLO

    for epoch in range(constraint_epochs):
        epoch_start = time.time()

        # ---- Step 1: CE on TRAIN data (batched) ----
        ce_losses = ce_epoch(model, train_loader, optimizer, criterion_ce,
                             device, amp_dtype, use_amp, scaler)


        # ---- Step 2: constraint gradient on TEST data (transductive) ----
        total_soft, group_soft, hard_preds = transductive_counts(
            model, X_test_dev, groups_np, unique_groups, num_classes,
            chunk_size, device)

        violations_g = {}
        violated_global = set()
        for c in constrained_classes:
            K = global_con[c]
            if K >= UNLIMITED:
                continue
            excess = total_soft[c].item() - K
            violations_g[c] = max(0.0, excess)
            if excess > 0:
                violated_global.add(c)

        violations_l = {}
        violated_local = set()
        for g in unique_groups:
            bounds = local_con.get(g, [UNLIMITED] * num_classes)
            for c in constrained_classes:
                key = (g, c)
                if key not in lambda_l:
                    continue
                K_local = bounds[c]
                if K_local >= UNLIMITED:
                    continue
                excess = group_soft[g][c].item() - K_local
                violations_l[key] = max(0.0, excess)
                if excess > 0:
                    violated_local.add(key)

        # Compute hard-count satisfaction from pass-1 predictions BEFORE the
        # constraint step. Required so the snapshot state below reflects the
        # exact model that produced the satisfaction status (saving post-step
        # state would mismatch the next-pass counts).
        total_excess = count_excess(hard_preds, groups_np, constrained_classes,
                                    global_con, local_con)
        all_satisfied = (total_excess == 0)

        # Snapshot model state BEFORE constraint step (mirrors TraLO). Saved
        # iff this epoch is satisfied OR improves on the lowest total excess
        # seen so far. Used to restore best checkpoint after training.
        snapshot_state = None
        # Only clone when a restore could use it: every generated config
        # sets enable_checkpoint_restore=false, and a full state_dict()
        # copied to CPU each epoch is ~344 MB on ViTB16 for a checkpoint
        # nothing ever reads.
        if allow_restore and (all_satisfied or total_excess < min_total_excess):
            snapshot_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

        has_work = (
            any(lambda_g.get(c, 0) > 0 for c in violated_global) or
            any(lambda_l.get(k, 0) > 0 for k in violated_local)
        )
        constraint_loss_val = 0.0
        did_backward = False
        if has_work:
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(X_test_dev), chunk_size):
                with constraint_autocast(amp_dtype, use_amp, CONSTRAINT_FP32):
                    chunk_logits = model(X_test_dev[i:i + chunk_size])
                    chunk_proba = F.softmax(chunk_logits, dim=1)
                    chunk_loss = torch.zeros(1, device=device)
                    for c in violated_global:
                        if lambda_g[c] > 0:
                            chunk_loss = chunk_loss + lambda_g[c] * chunk_proba[:, c].sum()
                    chunk_groups = groups_np[i:i + chunk_size]
                    for key in violated_local:
                        g, c = key
                        if lambda_l[key] > 0:
                            mask = (chunk_groups == g)
                            if mask.any():
                                chunk_loss = chunk_loss + lambda_l[key] * chunk_proba[mask, c].sum()
                if chunk_loss.item() > 0:
                    constraint_backward(chunk_loss, scaler, CONSTRAINT_FP32)
                    constraint_loss_val += chunk_loss.item()
                    did_backward = True
            if did_backward:
                # Grad clip + grad_norm>0 gate + scaler.update() always called
                # (mirrors TraLO's recovery pattern). Prevents bad-step state
                # leakage and unbounded step magnitudes.
                last_grad_norm, _applied = finish_constraint_step(
                    model, optimizer, scaler, CLIP,
                    mode=CONSTRAINT_GRAD_MODE, fp32=CONSTRAINT_FP32,
                    step_rule=CONSTRAINT_STEP_RULE, lr=LR_CONSTRAINT,
                random_direction=CONSTRAINT_RANDOM_DIR)

        # ---- Step 3: subgradient dual update (Fioretto Eq. 5) ----
        for c, viol in violations_g.items():
            lambda_g[c] += step_size * viol
        for key, viol in violations_l.items():
            lambda_l[key] += step_size * viol

        if all_satisfied and satisfaction_epoch is None:
            # +1: align with TraLO's convention so cross-method tables
            # report the SAME epoch number for the same training step.
            satisfaction_epoch = epoch + 1
            log.info("Fioretto: first satisfaction at epoch %d", epoch + 1)
        # Apples-to-apples early stop: 5 consecutive satisfied epochs.
        if all_satisfied:
            stable_count += 1
            if snapshot_state is not None:
                best_sat_state = snapshot_state
                best_sat_epoch = epoch + 1
        else:
            stable_count = 0
        if total_excess < min_total_excess and snapshot_state is not None:
            min_total_excess = total_excess
            min_excess_state = snapshot_state
            min_excess_epoch = epoch + 1

        row = {
            "epoch": epoch,
            "ce_loss": round(np.mean(ce_losses), 6),
            "constraint_loss": round(constraint_loss_val, 6),
            "total_excess": total_excess,
            "all_satisfied": int(all_satisfied),
            "max_lambda_g": round(max(lambda_g.values()) if lambda_g else 0, 6),
            "grad_norm": round(float(last_grad_norm), 6),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_fields).writerow(row)

        if epoch < 5 or (epoch + 1) % 25 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join(f"c{c}={lambda_g[c]:.3f}" for c in sorted(lambda_g))
            log.info("Fioretto %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s stable=%d lam=[%s] [%.1fs]",
                     epoch + 1, constraint_epochs, np.mean(ce_losses),
                     constraint_loss_val, total_excess, all_satisfied,
                     stable_count, lam_str, time.time() - epoch_start)

        if stable_count >= stable_count_threshold:
            log.info("Fioretto: converged (constraints stable for %d epochs at ep %d)",
                     stable_count, epoch + 1)
            break

    return (satisfaction_epoch, best_sat_state, best_sat_epoch,
            min_excess_state, min_excess_epoch, min_total_excess)


def train(inputs: TrainInputs) -> TrainOutputs:
    return run_dual_arm(inputs, _train_constraints, "Fioretto")
