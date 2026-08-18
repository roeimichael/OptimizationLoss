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
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.pipeline.setup import setup_runtime
from src.pipeline.warmup import make_dataloader, make_optimizer
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _required(hp, key, cast=float):
    """Read a protocol value that must never fall back to an inline default.

    The inline defaults here were the retracted ones -- lr_constraint 1e-5
    against the protocol's 1e-4, constraint_epochs 150 against 29,
    stable_count_threshold 5 against 31 (low enough that the early stop would
    actually fire). A missing key is a generator bug; failing loudly is the
    only safe behaviour.
    """
    if key not in hp:
        raise KeyError(
            "%s is required and has no safe default. configs/protocol.yml is "
            "the source of truth; generate the campaign with "
            "configs.gen_campaign rather than hand-writing a config." % key)
    return cast(hp[key])


def _train_constraints(model, config, inputs, device):
    """Fioretto Algorithm 1/2: linear penalty + per-constraint subgradient dual ascent."""
    hp = inputs.hyperparams
    # Hoisted: the per-epoch snapshot clone is gated on this, and a
    # state_dict() copied to CPU each epoch for a checkpoint nothing
    # reads is ~344 MB per epoch on ViTB16.
    allow_restore = bool(hp.get("enable_checkpoint_restore", True))
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
    chunk_size = hp.get("constraint_chunk_size", 256)

    use_amp, amp_dtype, scaler = setup_runtime(device)

    constrained_classes = inputs.constrained_classes
    num_classes = inputs.num_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids

    lambda_g = {c: 0.0 for c in constrained_classes if global_con[c] < UNLIMITED}
    lambda_l = {}
    for group_id, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                lambda_l[(group_id, c)] = 0.0

    log.info("Fioretto LDF: %d epochs, lr=%.2e, step_size=%.4f, "
             "%d global + %d local multipliers",
             constraint_epochs, lr_c, step_size, len(lambda_g), len(lambda_l))

    optimizer = make_optimizer(model.parameters(), lr_c, device)
    criterion_ce = nn.CrossEntropyLoss()
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
    log_fields = ["epoch", "ce_loss", "constraint_loss", "total_excess",
                  "all_satisfied", "max_lambda_g"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, log_fields).writeheader()

    stable_count = 0  # consecutive epochs with all_satisfied for early-stop parity with TraLO
    for epoch in range(constraint_epochs):
        epoch_start = time.time()

        # ---- Step 1: CE on TRAIN data (batched) ----
        # CE saturation skip (mirrors TraLO): once train_acc >= 0.995 for 2
        # consecutive epochs, disable CE so only constraint pressure remains.
        model.train()
        ce_losses = []
        train_correct, train_total = 0, 0
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
            with torch.no_grad():
                train_correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
                train_total += batch_y.size(0)
        cached_train_acc = train_correct / train_total if train_total > 0 else 1.0

        # ---- Step 2: constraint gradient on TEST data (transductive) ----
        # Apples-to-apples with TraLO: use model.eval() during the transductive
        # pass so dropout is off and BN running stats are NOT updated from test
        # data (would be a data-leakage source that flips a few borderline
        # samples and corrupts the lambda update). TraLO does this for the same
        # reason (AUDIT C1).
        model.eval()
        total_soft = torch.zeros(num_classes, device=device)
        group_soft = {g: torch.zeros(num_classes, device=device) for g in unique_groups}
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
                        group_soft[g] += chunk_proba[mask].sum(dim=0)
            hard_preds = torch.cat(all_hard).cpu().numpy()

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
        hard_counts = {c: int((hard_preds == c).sum()) for c in constrained_classes}
        total_excess = sum(
            max(0, hard_counts[c] - int(global_con[c]))
            for c in constrained_classes if global_con[c] < UNLIMITED
        )
        if local_con:
            for g_id, bounds in local_con.items():
                for c in constrained_classes:
                    if bounds[c] < UNLIMITED:
                        gc = int(((hard_preds == c) & (groups_np == g_id)).sum())
                        total_excess += max(0, gc - int(bounds[c]))
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
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
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
                    if scaler:
                        scaler.scale(chunk_loss).backward()
                    else:
                        chunk_loss.backward()
                    constraint_loss_val += chunk_loss.item()
                    did_backward = True
            if did_backward:
                # Grad clip + grad_norm>0 gate + scaler.update() always called
                # (mirrors TraLO's recovery pattern). Prevents bad-step state
                # leakage and unbounded step magnitudes.
                if scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    if grad_norm > 0:
                        scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    if grad_norm > 0:
                        optimizer.step()

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
    hp = inputs.hyperparams
    allow_restore = bool(hp.get("enable_checkpoint_restore", True))
    model = inputs.model
    device = inputs.device

    (satisfaction_epoch, best_sat_state, best_sat_epoch,
     min_excess_state, min_excess_epoch, min_total_excess
     ) = _train_constraints(model, inputs.config, inputs, device)

    # Apples-to-apples checkpoint restore (mirrors TraLO). Selection rule is
    # deterministic on the constraint excess axis -- NOT on F1. Picking the
    # checkpoint by F1 over multiple candidates is double-dipping on the
    # evaluation metric.
    constrained_classes = inputs.constrained_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids
    X_test_dev = inputs.X_test.to(device)
    num_classes = inputs.num_classes

    model.eval()
    with torch.no_grad():
        chunk_size = inputs.hyperparams.get("constraint_chunk_size", 256)
        all_hard = []
        for i in range(0, len(X_test_dev), chunk_size):
            chunk_logits = model(X_test_dev[i:i + chunk_size])
            all_hard.append(chunk_logits.argmax(dim=1))
        hard_preds = torch.cat(all_hard).cpu().numpy()
    final_total_excess = sum(
        max(0, int((hard_preds == c).sum()) - int(global_con[c]))
        for c in constrained_classes if global_con[c] < UNLIMITED
    )
    if local_con:
        for g_id, bounds in local_con.items():
            for c in constrained_classes:
                if bounds[c] < UNLIMITED:
                    gc = int(((hard_preds == c) & (groups_np == g_id)).sum())
                    final_total_excess += max(0, gc - int(bounds[c]))
    final_violates = final_total_excess > 0

    restored_from_epoch = None
    restore_kind = None
    # PARITY with tralo, which gates this and whose campaigns set it False.
    # Unconditional restore here meant that in a head-to-head only tralo kept
    # its trained model, while these two were swapped for a checkpoint chosen
    # on constraint satisfaction -- measured at -0.0351 AP within-run. Any
    # tralo win over the duals would have carried that advantage for free.
    # Default True: runs predating the flag keep their behaviour bit for bit.
    if not allow_restore:
        log.info("Fioretto: enable_checkpoint_restore=False, keeping the trained model")
    if allow_restore and best_sat_state is not None and final_violates:
        log.info("Fioretto: final violates; restoring best-satisfied checkpoint from epoch %d",
                 best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
    elif (allow_restore and min_excess_state is not None
          and final_total_excess > min_total_excess):
        log.info("Fioretto: final excess=%d > min seen excess=%d (epoch %d); "
                 "restoring lowest-excess checkpoint",
                 int(final_total_excess), int(min_total_excess), min_excess_epoch)
        model.load_state_dict({k: v.to(device) for k, v in min_excess_state.items()})
        restored_from_epoch = min_excess_epoch
        restore_kind = "min_excess"

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
        },
    )
