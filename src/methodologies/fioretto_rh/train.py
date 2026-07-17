"""fioretto_rh methodology: fioretto_ldf + TraLO's two load-bearing components.

REVIEW-RESPONSE GRAFT EXPERIMENT (P0-2 / DA-CRITICAL-2 / R1-W1 / R2-Q2).
Identical to fioretto_ldf (linear penalty + per-constraint subgradient ascent,
same harness, same checkpoint policy) plus exactly two additions, both lifted
verbatim in functional form from src/methodologies/tralo/train.py:

  1. optimizer reset at first satisfaction (reset_optimizer_at_sat):
     rebuild Adam with fresh m/v buffers the first epoch the hard counts
     satisfy. TraLO ablation credits this +0.079 cc-F1 (Table S1).

  2. undershoot hinge:  + lambda_c * beta * relu(K_c - soft_count_c) / K_c
     for every active global and local constraint, using the HOST method's
     own multipliers lambda_c (Fioretto's duals, which are monotone
     nondecreasing here, so post-satisfaction the hinge has a working
     multiplier exactly as TraLO's frozen lambda does). TraLO ablation
     credits the hinge +0.036 cc-F1.

Everything else is byte-identical in behavior to fioretto_ldf. The question
this arm answers: does fioretto_ldf + reset + hinge recover TraLO's
OctMNIST tight-cap edge, i.e. is the bounded penalty needed for the quality
gain or only for stability?
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


def _train_constraints(model, config, inputs, device):
    """Fioretto Algorithm 1/2 + grafted TraLO reset & undershoot hinge."""
    hp = inputs.hyperparams
    constraint_epochs = hp.get("constraint_epochs", 150)
    stable_count_threshold = int(hp.get("stable_count_threshold", 5))
    lr_c = hp.get("lr_constraint", 1e-5)
    if "fioretto_step_size" not in hp:
        raise ValueError(
            "fioretto_step_size is required in hyperparams (inherit it verbatim "
            "from the frozen paper_final config of the same cell).")
    step_size = float(hp["fioretto_step_size"])
    batch_size = hp.get("batch_size", 64)
    chunk_size = hp.get("constraint_chunk_size", 256)
    enable_ce_skip = bool(hp.get("enable_ce_skip", True))
    # --- graft knobs (TraLO values: fior_beta=0.5, reset on) ---
    fior_beta = float(hp.get("fior_beta", 0.5))
    reset_optimizer_at_sat = bool(hp.get("reset_optimizer_at_sat", True))

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

    log.info("Fioretto-RH (graft): %d epochs, lr=%.2e, step_size=%.4f, beta=%.2f, "
             "reset_at_sat=%s, %d global + %d local multipliers",
             constraint_epochs, lr_c, step_size, fior_beta, reset_optimizer_at_sat,
             len(lambda_g), len(lambda_l))

    optimizer = make_optimizer(model.parameters(), lr_c, device)
    criterion_ce = nn.CrossEntropyLoss()
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, batch_size)

    X_test_dev = inputs.X_test.to(device)
    unique_groups = np.unique(groups_np)
    n_chunks = (len(X_test_dev) + chunk_size - 1) // chunk_size

    satisfaction_epoch = None
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

    ce_skip_counter = 0
    skip_ce = False
    stable_count = 0
    for epoch in range(constraint_epochs):
        epoch_start = time.time()

        # ---- Step 1: CE on TRAIN data (batched; CE saturation skip) ----
        model.train()
        ce_losses = []
        train_correct, train_total = 0, 0
        for batch_X, batch_y in (train_loader if not skip_ce else []):
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
        if enable_ce_skip and not skip_ce:
            if cached_train_acc >= 0.995:
                ce_skip_counter += 1
                if ce_skip_counter >= 2:
                    skip_ce = True
                    log.info("Fioretto-RH epoch %d: CE saturated (acc=%.4f), "
                             "disabling CE batch loop", epoch + 1, cached_train_acc)
            else:
                ce_skip_counter = 0

        # ---- Step 2: constraint gradient on TEST data (transductive, eval mode) ----
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

        snapshot_state = None
        if all_satisfied or total_excess < min_total_excess:
            snapshot_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

        # --- graft: the hinge also creates work when soft < K and lambda > 0 ---
        hinge_active = False
        if fior_beta > 0:
            for c, lam in lambda_g.items():
                K = global_con[c]
                if lam > 0 and K > 0 and total_soft[c].item() < K:
                    hinge_active = True
                    break
            if not hinge_active:
                for g in unique_groups:
                    bounds = local_con.get(g, [UNLIMITED] * num_classes)
                    for c in constrained_classes:
                        key = (g, c)
                        if key not in lambda_l:
                            continue
                        K_local = bounds[c]
                        if K_local >= UNLIMITED or K_local <= 0:
                            continue
                        if lambda_l[key] > 0 and group_soft[g][c].item() < K_local:
                            hinge_active = True
                            break
                    if hinge_active:
                        break

        has_work = (
            any(lambda_g.get(c, 0) > 0 for c in violated_global) or
            any(lambda_l.get(k, 0) > 0 for k in violated_local) or
            hinge_active
        )
        constraint_loss_val = 0.0
        did_backward = False
        if has_work:
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(X_test_dev), chunk_size):
                chunk_groups = groups_np[i:i + chunk_size]
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    chunk_logits = model(X_test_dev[i:i + chunk_size])
                    chunk_proba = F.softmax(chunk_logits, dim=1)
                    chunk_loss = torch.zeros(1, device=device)
                    for c in violated_global:
                        if lambda_g[c] > 0:
                            chunk_loss = chunk_loss + lambda_g[c] * chunk_proba[:, c].sum()
                    for key in violated_local:
                        g, c = key
                        if lambda_l[key] > 0:
                            mask = (chunk_groups == g)
                            if mask.any():
                                chunk_loss = chunk_loss + lambda_l[key] * chunk_proba[mask, c].sum()
                    # --- graft: TraLO undershoot hinge, lambda*beta*relu(K-soft)/K.
                    # relu(total) is nonlinear, so gradient flows through this
                    # chunk's partial while the total is the detached aggregate
                    # (same g_soft trick as tralo/train.py). /n_chunks keeps the
                    # per-epoch hinge magnitude identical to TraLO's.
                    if fior_beta > 0:
                        for c, lam in lambda_g.items():
                            K = global_con[c]
                            if lam <= 0 or K <= 0:
                                continue
                            chunk_part = chunk_proba[:, c].sum()
                            g_soft_c = (total_soft[c].detach()
                                        - chunk_part.detach() + chunk_part)
                            chunk_loss = chunk_loss + (
                                lam * fior_beta * F.relu(K - g_soft_c) / K / n_chunks)
                        for g in unique_groups:
                            bounds = local_con.get(g, [UNLIMITED] * num_classes)
                            for c in constrained_classes:
                                key = (g, c)
                                if key not in lambda_l:
                                    continue
                                K_local = bounds[c]
                                lam = lambda_l[key]
                                if lam <= 0 or K_local >= UNLIMITED or K_local <= 0:
                                    continue
                                mask = (chunk_groups == g)
                                if mask.any():
                                    chunk_part = chunk_proba[mask, c].sum()
                                else:
                                    chunk_part = torch.zeros((), device=device)
                                l_soft_c = (group_soft[g][c].detach()
                                            - chunk_part.detach() + chunk_part)
                                chunk_loss = chunk_loss + (
                                    lam * fior_beta * F.relu(K_local - l_soft_c)
                                    / K_local / n_chunks)
                if chunk_loss.item() > 0:
                    if scaler:
                        scaler.scale(chunk_loss).backward()
                    else:
                        chunk_loss.backward()
                    constraint_loss_val += chunk_loss.item()
                    did_backward = True
            if did_backward:
                if scaler:
                    try:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0)
                        if grad_norm > 0:
                            scaler.step(optimizer)
                        scaler.update()
                    except (AssertionError, RuntimeError):
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1.0)
                        if grad_norm > 0:
                            optimizer.step()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0)
                    if grad_norm > 0:
                        optimizer.step()

        # ---- Step 3: subgradient dual update (Fioretto Eq. 5, unchanged) ----
        for c, viol in violations_g.items():
            lambda_g[c] += step_size * viol
        for key, viol in violations_l.items():
            lambda_l[key] += step_size * viol

        if all_satisfied and satisfaction_epoch is None:
            satisfaction_epoch = epoch + 1
            log.info("Fioretto-RH: first satisfaction at epoch %d", epoch + 1)
            # --- graft: TraLO optimizer reset. Clears Adam m/v buffers so the
            # post-satisfaction hinge gradient is not fighting stale descent
            # momentum (tralo/train.py, hybrid_v2 diagnosis).
            if reset_optimizer_at_sat:
                optimizer = make_optimizer(model.parameters(), lr_c, device)
                log.info("Fioretto-RH: reset Adam state at sat E%d", epoch + 1)
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
            "ce_loss": round(np.mean(ce_losses), 6) if ce_losses else 0.0,
            "constraint_loss": round(constraint_loss_val, 6),
            "total_excess": total_excess,
            "all_satisfied": int(all_satisfied),
            "max_lambda_g": round(max(lambda_g.values()) if lambda_g else 0, 6),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_fields).writerow(row)

        if epoch < 5 or (epoch + 1) % 25 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join(f"c{c}={lambda_g[c]:.3f}" for c in sorted(lambda_g))
            log.info("Fioretto-RH %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s stable=%d lam=[%s] [%.1fs]",
                     epoch + 1, constraint_epochs,
                     np.mean(ce_losses) if ce_losses else 0.0,
                     constraint_loss_val, total_excess, all_satisfied,
                     stable_count, lam_str, time.time() - epoch_start)

        if stable_count >= stable_count_threshold:
            log.info("Fioretto-RH: converged (constraints stable for %d epochs at ep %d)",
                     stable_count, epoch + 1)
            break

    return (satisfaction_epoch, best_sat_state, best_sat_epoch,
            min_excess_state, min_excess_epoch, min_total_excess)


def train(inputs: TrainInputs) -> TrainOutputs:
    model = inputs.model
    device = inputs.device

    (satisfaction_epoch, best_sat_state, best_sat_epoch,
     min_excess_state, min_excess_epoch, min_total_excess
     ) = _train_constraints(model, inputs.config, inputs, device)

    # Checkpoint restore on the constraint-excess axis (identical to
    # fioretto_ldf / TraLO -- never on F1).
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
    if best_sat_state is not None and final_violates:
        log.info("Fioretto-RH: final violates; restoring best-satisfied checkpoint from epoch %d",
                 best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
    elif min_excess_state is not None and final_total_excess > min_total_excess:
        log.info("Fioretto-RH: final excess=%d > min seen excess=%d (epoch %d); "
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
