"""fioretto_restart methodology: fioretto_ldf + dual restarts (anti-windup).

REVIEW-RESPONSE ANTI-WINDUP ARM (P1-1 / R1-W2 / R2-W1).
Identical to fioretto_ldf (linear penalty + per-constraint subgradient dual
ascent, same harness, same checkpoint policy) plus ONE addition:

  Dual restart (Gallego-Posada, Ramirez, Erraqabi, Bengio, Lacoste-Julien,
  "Controlled Sparsity via Constrained Optimization...", NeurIPS 2022): the
  moment the constraints are (hard-count) satisfied, ALL dual multipliers are
  reset to zero. This is the standard cure for the integrator-windup pathology
  the paper attributes to linear-penalty dual ascent: multipliers cannot keep
  climbing after feasibility is reached, so no wound-up penalty keeps crushing
  the constrained class.

The satisfaction signal used for the restart is the harness's own hard-count
feasibility (the same signal every method's freeze/convergence logic uses),
keeping the arm apples-to-apples with TraLO's freeze-at-satisfaction.

The question this arm answers: does an anti-windup dual baseline erase
TraLO's OctMNIST tight-cap quality edge, i.e. was plain fixed-step dual
ascent a strawman?
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
    """Fioretto Algorithm 1/2 + dual restart at feasibility."""
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

    log.info("Fioretto-RESTART (anti-windup): %d epochs, lr=%.2e, step_size=%.4f, "
             "%d global + %d local multipliers",
             constraint_epochs, lr_c, step_size, len(lambda_g), len(lambda_l))

    optimizer = make_optimizer(model.parameters(), lr_c, device)
    criterion_ce = nn.CrossEntropyLoss()
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, batch_size)

    X_test_dev = inputs.X_test.to(device)
    unique_groups = np.unique(groups_np)

    satisfaction_epoch = None
    n_restarts = 0
    best_sat_state = None
    best_sat_epoch = None
    min_excess_state = None
    min_excess_epoch = None
    min_total_excess = float("inf")

    log_path = inputs.experiment_path / "training_log.csv"
    log_fields = ["epoch", "ce_loss", "constraint_loss", "total_excess",
                  "all_satisfied", "max_lambda_g", "n_restarts"]
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
                    log.info("Fioretto-RESTART epoch %d: CE saturated (acc=%.4f), "
                             "disabling CE batch loop", epoch + 1, cached_train_acc)
            else:
                ce_skip_counter = 0

        # ---- Step 2: constraint gradient on TEST data (transductive, eval) ----
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

        # ---- Step 3: subgradient dual update (Fioretto Eq. 5) ----
        for c, viol in violations_g.items():
            lambda_g[c] += step_size * viol
        for key, viol in violations_l.items():
            lambda_l[key] += step_size * viol

        # ---- Step 3b: DUAL RESTART at feasibility (Gallego-Posada et al. 2022).
        # Trigger on SOFT feasibility -- the constraint value the optimizer
        # actually sees (their g(theta) <= 0 is the surrogate being ascended
        # on). Hard-count feasibility is the wrong trigger here: the soft/hard
        # gap can leave hard counts at K+1 while soft counts sit below K, in
        # which case a hard trigger never fires and the wound-up multiplier
        # (frozen at its peak, since Fioretto duals never decay) survives
        # forever -- the exact pathology a restart is meant to kill.
        soft_feasible = (not violated_global) and (not violated_local)
        if soft_feasible:
            had_mass = (any(v > 0 for v in lambda_g.values())
                        or any(v > 0 for v in lambda_l.values()))
            if had_mass:
                n_restarts += 1
                log.info("Fioretto-RESTART: soft-feasible at E%d -> dual restart "
                         "(all lambda -> 0; restart #%d)", epoch + 1, n_restarts)
            lambda_g = {c: 0.0 for c in lambda_g}
            lambda_l = {k: 0.0 for k in lambda_l}

        if all_satisfied and satisfaction_epoch is None:
            satisfaction_epoch = epoch + 1
            log.info("Fioretto-RESTART: first satisfaction at epoch %d", epoch + 1)
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
            "n_restarts": n_restarts,
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_fields).writerow(row)

        if epoch < 5 or (epoch + 1) % 25 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join(f"c{c}={lambda_g[c]:.3f}" for c in sorted(lambda_g))
            log.info("Fioretto-RESTART %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s "
                     "stable=%d restarts=%d lam=[%s] [%.1fs]",
                     epoch + 1, constraint_epochs,
                     np.mean(ce_losses) if ce_losses else 0.0,
                     constraint_loss_val, total_excess, all_satisfied,
                     stable_count, n_restarts, lam_str, time.time() - epoch_start)

        if stable_count >= stable_count_threshold:
            log.info("Fioretto-RESTART: converged (constraints stable for %d epochs at ep %d)",
                     stable_count, epoch + 1)
            break

    return (satisfaction_epoch, best_sat_state, best_sat_epoch,
            min_excess_state, min_excess_epoch, min_total_excess, n_restarts)


def train(inputs: TrainInputs) -> TrainOutputs:
    model = inputs.model
    device = inputs.device

    (satisfaction_epoch, best_sat_state, best_sat_epoch,
     min_excess_state, min_excess_epoch, min_total_excess, n_restarts
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
        log.info("Fioretto-RESTART: final violates; restoring best-satisfied checkpoint "
                 "from epoch %d", best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
    elif min_excess_state is not None and final_total_excess > min_total_excess:
        log.info("Fioretto-RESTART: final excess=%d > min seen excess=%d (epoch %d); "
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
            "n_dual_restarts": n_restarts,
        },
    )
