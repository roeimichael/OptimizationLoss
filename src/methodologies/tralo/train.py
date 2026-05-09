"""TraLO (Transductive Lagrangian Optimization).

Bounded saturated penalty + per-class lambda ratchet + optional KL anchor to
warmup distribution. Constraint phase only — warmup is shared with the
baselines via pipeline.warmup.run_warmup. State flows through TrainInputs.

Loss per constrained class c (and per (group, c) for local):
    L_c = lambda_c * [ E/(E+K) + rho * (E/K)^2 / (1 + (E/K)^2) ]
  where E = ReLU(soft_count_c - K_c).
Both terms bounded in [0, 1) so a single very-violated class cannot hijack
the gradient. Per-class lambdas (not a single scalar) prevent one violator
from dominating the multi-class case.
"""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import MulticlassTransductiveLoss
from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.pipeline.setup import setup_runtime
from src.pipeline.warmup import make_ce_criterion, make_dataloader, make_optimizer
from src.training.logging import log_progress_to_csv, write_csv_header
from src.training.metrics import compute_prediction_statistics
from src.utils.constants import UNLIMITED
from src.utils.error_handler import logger
from src.utils.inference import chunked_forward

log = logging.getLogger(__name__)

CONSTRAINT_CHUNK_SIZE = 256


def _cache_warmup_logits(model, X_test, amp_dtype, use_amp):
    """Cache RAW warmup logits so KL anchor can be applied symmetrically
    to BOTH current and warmup distributions inside the KL term. AUDIT C2.
    """
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        warmup_logits = chunked_forward(model, X_test).float()
    log.info("Cached warmup logits: shape=%s", warmup_logits.shape)
    return warmup_logits.detach()


@logger()
def train(inputs: TrainInputs) -> TrainOutputs:
    config = inputs.config
    hp = inputs.hyperparams
    device = inputs.device
    num_classes = inputs.num_classes
    model = inputs.model
    csv_log_path = str(inputs.csv_log_path)

    use_amp, amp_dtype, scaler = setup_runtime(device)

    warmup_epochs = hp["warmup_epochs"]
    total_epochs = warmup_epochs + hp.get("constraint_epochs", 300)
    lambda_step = hp["lambda_step"]
    criterion_ce = make_ce_criterion(config, inputs.y_train, num_classes, device)
    optimizer = make_optimizer(model.parameters(), hp.get("lr_constraint", 1e-5), device)
    log.info("Reset optimizer for constraint phase (lr=%.2e)", hp.get("lr_constraint", 1e-5))
    lr_constraint = hp.get("lr_constraint", 1e-5)
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, hp["batch_size"])
    X_test = inputs.X_test.to(device)
    group_ids = torch.LongTensor(inputs.group_ids).to(device)
    global_con = inputs.global_con
    local_con = inputs.local_con
    criterion_constraint = MulticlassTransductiveLoss(
        global_constraints=global_con, local_constraints=local_con,
        num_classes=num_classes,
        initial_rho=hp.get("initial_rho", 0.5), alpha_kl=hp.get("alpha_kl", 0.0),
        penalty_mode=hp.get("penalty_mode", "both"),
    ).to(device)
    log.info("Using FULL test set (%d samples) for constraint gradient", len(X_test))
    alpha_kl = hp.get("alpha_kl", 0.0)
    warmup_logits_cache = None
    if alpha_kl > 0:
        warmup_logits_cache = _cache_warmup_logits(model, X_test, amp_dtype, use_amp)
    satisfaction_epoch = None
    stable_count = 0
    best_sat_state = None
    best_sat_epoch = None
    min_excess_state = None
    min_excess_epoch = None
    min_total_excess = float("inf")
    training_start = time.time()
    constrained_classes = [c for c in range(num_classes) if global_con[c] < UNLIMITED]
    rho_frozen = False
    init_g = hp.get("lambda_global", 0.01)
    init_l = hp.get("lambda_local", 0.01)
    for c in constrained_classes:
        criterion_constraint.set_lambda_per_class(c, init_g, scope="global")
    for gid, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                criterion_constraint.set_lambda_per_class(c, init_l, scope="local", group_id=gid)
    log.info("Per-class lambdas: %d global + %d local",
             len(criterion_constraint.lambda_global_per_class),
             len(criterion_constraint.lambda_local_per_key))
    constraint_epochs = hp.get("constraint_epochs", 300)
    rho_target = hp.get("rho_target", 100.0)
    initial_rho = hp.get("initial_rho", 0.5)
    rho_step = (rho_target - initial_rho) / max(constraint_epochs, 1)
    # Ablation flags (default off = full TraLO behavior)
    disable_freeze_on_satisfy = hp.get("disable_freeze_on_satisfy", False)  # A3
    disable_min_excess_restore = hp.get("disable_min_excess_restore", False)  # A5
    shared_lambda = hp.get("shared_lambda", False)  # A6
    log.info("Constraint training: epochs %d to %d (ablations: freeze=%s restore=%s shared_lam=%s)",
             warmup_epochs + 1, total_epochs,
             not disable_freeze_on_satisfy, not disable_min_excess_restore, shared_lambda)
    write_csv_header(csv_log_path, num_classes, local_con)

    for epoch in range(warmup_epochs, total_epochs):
        model.train()
        current_lr = lr_constraint
        for pg in optimizer.param_groups:
            pg["lr"] = lr_constraint
        epoch_ce = 0.0
        num_batches = len(train_loader)
        train_correct, train_total = 0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits_ce = model(batch_X)
                loss_ce = criterion_ce(logits_ce, batch_y)
            if scaler:
                scaler.scale(loss_ce).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_ce.backward()
                optimizer.step()
            epoch_ce += loss_ce.item()
            with torch.no_grad():
                train_correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
                train_total += batch_y.size(0)
        cached_train_acc = train_correct / train_total if train_total > 0 else 1.0

        # Pass A + Pass B in eval() so dropout off and BN stats untouched (AUDIT C1).
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        chunk_size = hp.get("constraint_chunk_size", CONSTRAINT_CHUNK_SIZE)
        n_test = len(X_test)
        n_chunks = (n_test + chunk_size - 1) // chunk_size

        with torch.no_grad():
            total_global_soft = torch.zeros(num_classes, device=device)
            total_global_hard = torch.zeros(num_classes, device=device)
            total_local_soft = {gid: torch.zeros(num_classes, device=device)
                                for gid in criterion_constraint.local_groups}
            total_local_hard = {gid: torch.zeros(num_classes, device=device)
                                for gid in criterion_constraint.local_groups}
            for ci in range(n_chunks):
                start = ci * chunk_size
                end = min(start + chunk_size, n_test)
                # FP32 forward for count consistency with eval (no autocast).
                # AUDIT C7: BF16 argmax flips a few borderline samples vs FP32,
                # so satisfied-during-training did not survive FP32 evaluation.
                chunk_logits = model(X_test[start:end])
                chunk_proba = F.softmax(chunk_logits, dim=1)
                chunk_preds = chunk_logits.argmax(dim=1)
                total_global_soft += chunk_proba.sum(dim=0)
                total_global_hard += torch.bincount(
                    chunk_preds, minlength=num_classes).float()
                chunk_gids = group_ids[start:end]
                for gid in total_local_soft:
                    mask = (chunk_gids == gid)
                    if mask.any():
                        total_local_soft[gid] += chunk_proba[mask].sum(dim=0)
                        total_local_hard[gid] += torch.bincount(
                            chunk_preds[mask], minlength=num_classes).float()

        # Snapshot model state BEFORE constraint backward+step. The hard counts
        # in total_global_hard / total_local_hard reflect THIS state; saving
        # post-step state would mismatch (the state-dict no longer produces
        # those counts on a forward pass).
        snapshot_global_satisfied = True
        for c in range(num_classes):
            if c < len(criterion_constraint.global_constraints) and \
                    criterion_constraint.global_constraints[c] < UNLIMITED:
                if total_global_hard[c].item() > criterion_constraint.global_constraints[c].item():
                    snapshot_global_satisfied = False
                    break
        snapshot_local_satisfied = True
        for gid_s, buffer_name_s in criterion_constraint.local_groups.items():
            lc_s = getattr(criterion_constraint, buffer_name_s)
            for c in range(num_classes):
                if c < len(lc_s) and lc_s[c] < UNLIMITED:
                    if total_local_hard[gid_s][c].item() > lc_s[c].item():
                        snapshot_local_satisfied = False
                        break
            if not snapshot_local_satisfied:
                break
        snapshot_total_excess = 0.0
        for c in range(num_classes):
            if c < len(criterion_constraint.global_constraints) and \
                    criterion_constraint.global_constraints[c] < UNLIMITED:
                snapshot_total_excess += max(
                    0.0, total_global_hard[c].item()
                    - criterion_constraint.global_constraints[c].item())
        for gid_s, buffer_name_s in criterion_constraint.local_groups.items():
            lc_s = getattr(criterion_constraint, buffer_name_s)
            for c in range(num_classes):
                if c < len(lc_s) and lc_s[c] < UNLIMITED:
                    snapshot_total_excess += max(
                        0.0, total_local_hard[gid_s][c].item() - lc_s[c].item())
        snapshot_state = None
        snapshot_is_sat = snapshot_global_satisfied and snapshot_local_satisfied
        if snapshot_is_sat or snapshot_total_excess < min_total_excess:
            snapshot_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        loss_global_val = criterion_constraint.compute_global_from_counts(total_global_soft).item()
        loss_local_val = criterion_constraint.compute_local_from_counts(total_local_soft).item()
        total_constraint = loss_global_val + loss_local_val

        loss_kl_val = 0.0
        has_constraint = total_constraint > 0
        has_kl = alpha_kl > 0 and warmup_logits_cache is not None
        if has_constraint or has_kl:
            for ci in range(n_chunks):
                start = ci * chunk_size
                end = min(start + chunk_size, n_test)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    chunk_logits = model(X_test[start:end])
                chunk_logits_f = chunk_logits.float()
                chunk_proba = F.softmax(chunk_logits_f, dim=1)
                chunk_loss = torch.tensor(0.0, device=device)
                if has_constraint:
                    chunk_global = chunk_proba.sum(dim=0)
                    chunk_local = {}
                    chunk_gids = group_ids[start:end]
                    for gid in criterion_constraint.local_groups:
                        mask = (chunk_gids == gid)
                        if mask.any():
                            chunk_local[gid] = chunk_proba[mask].sum(dim=0)
                        else:
                            chunk_local[gid] = torch.zeros(num_classes, device=device)
                    g_soft = total_global_soft.detach() - chunk_proba.sum(dim=0).detach() + chunk_global
                    l_soft = {}
                    for gid in total_local_soft:
                        l_soft[gid] = total_local_soft[gid].detach() - chunk_local[gid].detach() + chunk_local[gid]
                    lg = criterion_constraint.compute_global_from_counts(g_soft)
                    ll = criterion_constraint.compute_local_from_counts(l_soft)
                    chunk_loss = chunk_loss + (lg + ll) / n_chunks
                if has_kl:
                    log_p_cur = F.log_softmax(chunk_logits_f, dim=1)
                    p_cur = F.softmax(chunk_logits_f, dim=1)
                    log_p_warm = F.log_softmax(warmup_logits_cache[start:end], dim=1)
                    kl_chunk = (p_cur * (log_p_cur - log_p_warm)).sum(dim=1).mean()
                    chunk_loss = chunk_loss + alpha_kl * kl_chunk / n_chunks
                    loss_kl_val += kl_chunk.item() / n_chunks
                if scaler:
                    scaler.scale(chunk_loss).backward()
                else:
                    chunk_loss.backward()

        did_backward = has_constraint or has_kl
        if scaler and did_backward:
            try:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if grad_norm > 0:
                    scaler.step(optimizer)
                scaler.update()
            except (AssertionError, RuntimeError):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if grad_norm > 0:
                    optimizer.step()
        elif not scaler and did_backward:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if grad_norm > 0:
                optimizer.step()

        avg_ce = epoch_ce / num_batches
        avg_global = loss_global_val
        avg_local = loss_local_val
        avg_kl = loss_kl_val

        global_satisfied = True
        for c in range(num_classes):
            if c < len(criterion_constraint.global_constraints) and \
                    criterion_constraint.global_constraints[c] < UNLIMITED:
                if total_global_hard[c].item() > criterion_constraint.global_constraints[c].item():
                    global_satisfied = False
                    break
        local_satisfied = True
        for gid, buffer_name in criterion_constraint.local_groups.items():
            lc = getattr(criterion_constraint, buffer_name)
            for c in range(num_classes):
                if c < len(lc) and lc[c] < UNLIMITED:
                    if total_local_hard[gid][c].item() > lc[c].item():
                        local_satisfied = False
                        break
            if not local_satisfied:
                break

        is_satisfied = global_satisfied and local_satisfied
        if is_satisfied:
            stable_count += 1
            if snapshot_state is not None:
                best_sat_state = snapshot_state
                best_sat_epoch = epoch + 1
        else:
            stable_count = 0
        if snapshot_total_excess < min_total_excess and snapshot_state is not None:
            min_total_excess = snapshot_total_excess
            min_excess_state = snapshot_state
            min_excess_epoch = epoch + 1

        # Per-class lambda ratchet: increment only on violation, freeze on first satisfaction.
        # Ablation A3 disable_freeze_on_satisfy: keep ratcheting after satisfaction.
        # Ablation A6 shared_lambda: increment all classes by the same step on ANY violation.
        ratchet_gate = (satisfaction_epoch is None) or disable_freeze_on_satisfy
        any_global_viol = any(
            (total_global_hard[c].item() > criterion_constraint.global_constraints[c].item())
            for c in constrained_classes
            if c < len(criterion_constraint.global_constraints)
            and criterion_constraint.global_constraints[c] < UNLIMITED
        )
        for c in constrained_classes:
            if c < len(criterion_constraint.global_constraints) and \
                    criterion_constraint.global_constraints[c] < UNLIMITED:
                hard_c = total_global_hard[c].item()
                limit_c = criterion_constraint.global_constraints[c].item()
                trigger = (any_global_viol if shared_lambda else hard_c > limit_c)
                if trigger and ratchet_gate:
                    old = criterion_constraint.get_lambda_per_class(c, scope="global")
                    criterion_constraint.set_lambda_per_class(c, old + lambda_step, scope="global")
        for gid, buffer_name in criterion_constraint.local_groups.items():
            lc = getattr(criterion_constraint, buffer_name)
            any_local_viol = any(
                (total_local_hard[gid][c].item() > lc[c].item())
                for c in constrained_classes
                if c < len(lc) and lc[c] < UNLIMITED
            )
            for c in constrained_classes:
                if c < len(lc) and lc[c] < UNLIMITED:
                    hard_c = total_local_hard[gid][c].item()
                    limit_c = lc[c].item()
                    trigger = (any_local_viol if shared_lambda else hard_c > limit_c)
                    if trigger and ratchet_gate:
                        old = criterion_constraint.get_lambda_per_class(c, scope="local", group_id=gid)
                        criterion_constraint.set_lambda_per_class(c, old + lambda_step, scope="local", group_id=gid)
        if is_satisfied and satisfaction_epoch is None:
            satisfaction_epoch = epoch + 1
            if not rho_frozen and not disable_freeze_on_satisfy:
                rho_frozen = True
                log.info("First satisfied at epoch %d, freezing rho=%.3f and per-class lambdas",
                         epoch + 1, criterion_constraint.get_rho())
            elif disable_freeze_on_satisfy:
                log.info("First satisfied at epoch %d, NOT freezing (ablation A3)", epoch + 1)

        if not rho_frozen:
            criterion_constraint.increment_rho(rho_step)

        if stable_count >= 5:
            log.info("Converged: constraints stable for %d epochs (lambdas frozen)", stable_count)
            break

        if (epoch + 1) % 5 == 0 or is_satisfied or epoch == warmup_epochs:
            train_acc = cached_train_acc
            g_counts = {c: int(total_global_hard[c].item()) for c in range(num_classes)}
            l_counts = {}
            for gid in total_local_hard:
                l_counts[gid] = {c: int(total_local_hard[gid][c].item()) for c in range(num_classes)}
            g_soft = {c: total_global_soft[c].item() for c in range(num_classes)}
            l_soft = {}
            for gid in total_local_soft:
                l_soft[gid] = {c: total_local_soft[gid][c].item() for c in range(num_classes)}
            mode = "Satisfied" if is_satisfied else "Constraint"
            kl_str = f" kl={avg_kl:.4f}" if alpha_kl > 0 else ""
            lam_g_mean = (sum(criterion_constraint.lambda_global_per_class.values())
                          / max(1, len(criterion_constraint.lambda_global_per_class)))
            lam_l_mean = (sum(criterion_constraint.lambda_local_per_key.values())
                          / max(1, len(criterion_constraint.lambda_local_per_key)))
            log.info("Epoch %d [%s] lr=%.2e ce=%.4f g=%.4f l=%.4f%s "
                     "lambda_mean(g=%.3f l=%.3f rho=%.3f) acc=%.4f g_%s l_%s",
                     epoch + 1, mode, current_lr, avg_ce, avg_global, avg_local, kl_str,
                     lam_g_mean, lam_l_mean,
                     criterion_constraint.get_rho(), train_acc,
                     "OK" if global_satisfied else "VIOL",
                     "OK" if local_satisfied else "VIOL")
            for c in range(num_classes):
                if global_con[c] < UNLIMITED:
                    log.info("  Global class %d: pred=%d limit=%d", c, g_counts.get(c, 0), int(global_con[c]))
            for gid in sorted(l_counts.keys()):
                group_name = f"group_{gid}"
                for c in range(num_classes):
                    if local_con and gid in local_con and local_con[gid][c] < UNLIMITED:
                        log.info("  Local %s class %d: pred=%d limit=%d",
                                 group_name, c,
                                 l_counts.get(gid, {}).get(c, 0),
                                 int(local_con[gid][c]))
            log_progress_to_csv(
                csv_log_path, epoch, avg_ce, train_acc, avg_global, avg_local,
                g_counts, l_counts, g_soft, l_soft,
                lam_g_mean, lam_l_mean,
                global_con, global_satisfied, local_satisfied,
                kl_loss=avg_kl, local_constraints=local_con)
        model.train()

    elapsed = time.time() - training_start
    log.info("Training complete: %.1fs, satisfaction epoch: %s",
             elapsed, satisfaction_epoch or "N/A")
    model.eval()
    g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
        model, X_test, group_ids, num_classes=num_classes)
    final_violates = False
    for c in range(num_classes):
        if global_con[c] < UNLIMITED and g_counts.get(c, 0) > int(global_con[c]):
            final_violates = True
            break
    if not final_violates and local_con:
        for gid, bounds in local_con.items():
            for c in range(num_classes):
                if bounds[c] < UNLIMITED and l_counts.get(gid, {}).get(c, 0) > int(bounds[c]):
                    final_violates = True
                    break
            if final_violates:
                break
    final_total_excess = 0.0
    for c in range(num_classes):
        if global_con[c] < UNLIMITED:
            final_total_excess += max(0, g_counts.get(c, 0) - int(global_con[c]))
    if local_con:
        for gid, bounds in local_con.items():
            for c in range(num_classes):
                if bounds[c] < UNLIMITED:
                    final_total_excess += max(0, l_counts.get(gid, {}).get(c, 0) - int(bounds[c]))
    restored_from_epoch = None
    restore_kind = None
    if disable_min_excess_restore:
        log.info("Ablation A5: skipping checkpoint restore, using final epoch state")
    elif best_sat_state is not None and final_violates:
        log.info("Final epoch violates; restoring best-satisfied checkpoint from epoch %d",
                 best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            model, X_test, group_ids, num_classes=num_classes)
    elif min_excess_state is not None and final_total_excess > min_total_excess:
        log.info("Final epoch excess=%d > min seen excess=%d (epoch %d); "
                 "restoring lowest-excess checkpoint",
                 int(final_total_excess), int(min_total_excess), min_excess_epoch)
        model.load_state_dict({k: v.to(device) for k, v in min_excess_state.items()})
        restored_from_epoch = min_excess_epoch
        restore_kind = "min_excess"
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            model, X_test, group_ids, num_classes=num_classes)
    log.info("=== Final prediction summary ===")
    for c in range(num_classes):
        limit = int(global_con[c]) if global_con[c] < UNLIMITED else "INF"
        log.info("  Global class %d: hard=%d soft=%.2f limit=%s",
                 c, g_counts.get(c, 0), g_soft.get(c, 0), limit)
    for gid in sorted(l_counts.keys()):
        group_name = f"group_{gid}"
        for c in range(num_classes):
            if local_con and gid in local_con and local_con[gid][c] < UNLIMITED:
                log.info("  Local %s class %d: hard=%d soft=%.2f limit=%d",
                         group_name, c,
                         l_counts.get(gid, {}).get(c, 0),
                         l_soft.get(gid, {}).get(c, 0.0),
                         int(local_con[gid][c]))

    final_soft_hard_gap = {}
    for c in [c for c in range(num_classes) if global_con[c] < UNLIMITED]:
        final_soft_hard_gap[c] = abs(g_soft.get(c, 0) - g_counts.get(c, 0))

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
            "soft_hard_gap": final_soft_hard_gap,
        },
    )
