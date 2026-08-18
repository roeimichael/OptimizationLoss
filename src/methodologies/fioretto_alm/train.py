"""fioretto_alm methodology: augmented-Lagrangian dual update (Track B / B3).

Identical to fioretto_ldf EXCEPT the dual (multiplier) update. Fioretto-LDF
accumulates the positive-part subgradient (lambda += step * excess^+), which
can only grow -- the "linear-penalty windup" the paper discusses. The
augmented-Lagrangian method (Hestenes 1969 / Powell 1969 / Rockafellar 1974)
is the standard literature fix, and R2 asked for it as a baseline. It

  (a) updates the multiplier on the RAW residual with a nonnegativity
      projection, so the multiplier can SHRINK when the constraint goes slack
      (dual descent), and
  (b) adds an augmentation penalty whose coefficient mu grows linearly, giving
      feasibility pressure without requiring the multiplier itself to wind up.

Update rule (advisor handoff B3):

    lambda_c <- max(0, lambda_c + eta (S_c - K_c)) + mu_t (S_c - K_c)^+
    mu_t = alm_mu0 + alm_mu_step * epoch     (linear growth)

where S_c is the soft (probability-sum) count and K_c the cap. The rule is
self-limiting: as the model reaches feasibility the residual (S_c - K_c)^+ -> 0,
the augmentation term vanishes, and the projected ascent term bleeds the
multiplier back down. Applied to both global and per-group (local) caps.

Everything else -- the two-pass transductive constraint gradient, CE-saturation
skip, grad-clip recovery, best-checkpoint restore on the excess axis, and the
5-consecutive-satisfied early stop -- is copied verbatim from fioretto_ldf so
the ALM/Fioretto comparison isolates ONLY the dual rule.
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
    """Augmented-Lagrangian dual optimization (ALM variant of Fioretto Alg. 1/2)."""
    hp = inputs.hyperparams
    constraint_epochs = hp.get("constraint_epochs", 150)
    # Apples-to-apples: same early-stop policy as TraLO/Fioretto (5 consecutive
    # satisfied epochs). Default matches TraLO.
    stable_count_threshold = int(hp.get("stable_count_threshold", 5))
    lr_c = hp.get("lr_constraint", 1e-5)
    # ALM update hyperparameters. eta falls back to the Fioretto step size so a
    # config cloned from a Fioretto/TraLO cell runs without extra keys.
    eta = float(hp.get("alm_eta", hp.get("fioretto_step_size", 0.005)))
    mu0 = float(hp.get("alm_mu0", 0.01))
    mu_step = float(hp.get("alm_mu_step", 0.01))
    batch_size = hp.get("batch_size", 64)
    chunk_size = hp.get("constraint_chunk_size", 256)
    # Apples-to-apples with TraLO/Fioretto: CE saturation skip.

    use_amp, amp_dtype, scaler = setup_runtime(device)

    constrained_classes = inputs.constrained_classes
    num_classes = inputs.num_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids

    # ALM starts the multipliers at 0 (zero-start dual ascent); the augmentation
    # term supplies the initial feasibility pressure once a violation appears.
    lam0 = float(hp.get("fioretto_lambda_init", 0.0))
    lambda_g = {c: lam0 for c in constrained_classes if global_con[c] < UNLIMITED}
    lambda_l = {}
    for group_id, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                lambda_l[(group_id, c)] = lam0

    log.info("Fioretto ALM: %d epochs, lr=%.2e, eta=%.4f, mu0=%.4f, mu_step=%.4f, "
             "%d global + %d local multipliers",
             constraint_epochs, lr_c, eta, mu0, mu_step, len(lambda_g), len(lambda_l))

    optimizer = make_optimizer(model.parameters(), lr_c, device)
    criterion_ce = nn.CrossEntropyLoss()
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, batch_size)

    X_test_dev = inputs.X_test.to(device)
    unique_groups = np.unique(groups_np)

    satisfaction_epoch = None
    # Best-checkpoint restore (apples-to-apples with TraLO/Fioretto): snapshot
    # model state BEFORE the constraint step at every epoch that satisfies or
    # improves on the lowest total excess seen so far.
    best_sat_state = None
    best_sat_epoch = None
    min_excess_state = None
    min_excess_epoch = None
    min_total_excess = float("inf")

    log_path = inputs.experiment_path / "training_log.csv"
    log_fields = ["epoch", "ce_loss", "constraint_loss", "total_excess",
                  "all_satisfied", "max_lambda_g", "mu_t"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, log_fields).writeheader()

    stable_count = 0  # consecutive satisfied epochs for early-stop parity
    for epoch in range(constraint_epochs):
        epoch_start = time.time()
        mu_t = mu0 + mu_step * epoch  # linearly growing augmentation coefficient

        # ---- Step 1: CE on TRAIN data (batched) ----
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

        # Raw residuals r_c = S_c - K_c (kept signed for the ALM ascent term);
        # violations_* hold the positive part for the loss gate (parity with
        # Fioretto: the constraint LOSS pushes only classes above the cap).
        residual_g = {}
        violations_g = {}
        violated_global = set()
        for c in constrained_classes:
            K = global_con[c]
            if K >= UNLIMITED:
                continue
            excess = total_soft[c].item() - K
            residual_g[c] = excess
            violations_g[c] = max(0.0, excess)
            if excess > 0:
                violated_global.add(c)

        residual_l = {}
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
                residual_l[key] = excess
                violations_l[key] = max(0.0, excess)
                if excess > 0:
                    violated_local.add(key)

        # Hard-count satisfaction from pass-1 predictions BEFORE the step.
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

        # ---- Step 3: augmented-Lagrangian dual update ----
        # lambda_c <- max(0, lambda_c + eta * r_c) + mu_t * (r_c)^+   (r_c signed)
        for c, r in residual_g.items():
            lambda_g[c] = max(0.0, lambda_g[c] + eta * r) + mu_t * max(0.0, r)
        for key, r in residual_l.items():
            lambda_l[key] = max(0.0, lambda_l[key] + eta * r) + mu_t * max(0.0, r)

        if all_satisfied and satisfaction_epoch is None:
            satisfaction_epoch = epoch + 1
            log.info("Fioretto ALM: first satisfaction at epoch %d", epoch + 1)
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
            "mu_t": round(mu_t, 6),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_fields).writerow(row)

        if epoch < 5 or (epoch + 1) % 25 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join(f"c{c}={lambda_g[c]:.3f}" for c in sorted(lambda_g))
            log.info("Fioretto ALM %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s stable=%d "
                     "mu=%.3f lam=[%s] [%.1fs]",
                     epoch + 1, constraint_epochs, np.mean(ce_losses),
                     constraint_loss_val, total_excess, all_satisfied,
                     stable_count, mu_t, lam_str, time.time() - epoch_start)

        if stable_count >= stable_count_threshold:
            log.info("Fioretto ALM: converged (constraints stable for %d epochs at ep %d)",
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

    # Apples-to-apples checkpoint restore (mirrors TraLO/Fioretto): selection on
    # the constraint-excess axis, NOT on F1.
    constrained_classes = inputs.constrained_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids
    X_test_dev = inputs.X_test.to(device)

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
        log.info("Fioretto ALM: final violates; restoring best-satisfied checkpoint from epoch %d",
                 best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
    elif min_excess_state is not None and final_total_excess > min_total_excess:
        log.info("Fioretto ALM: final excess=%d > min seen excess=%d (epoch %d); "
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
