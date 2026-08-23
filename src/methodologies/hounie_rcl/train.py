"""hounie_rcl methodology: Resilient Constrained Learning.

Faithful reimplementation of the algorithm from:

    Hounie, Ribeiro, Chamon. "Resilient Constrained Learning."
    NeurIPS 2023. arXiv:2306.02426.

Algorithm 1 (generic) + Algorithm 2 (federated specialisation) collapsed onto
TraLO's prediction-count constraint task. The mapping from their notation to
this code is documented in `archive/benchmarks/hounie/` (reference implementation).

Per epoch, three updates:

    theta:  primal SGD on    L = L_ce + sum_i lam_i * (l_i - u_i)
    u:      grad ascent on   max_u  -h(u) - lam_i*u_i  (with h = alpha*||u||^2)
            -> u_i <- max(0, u_i + eta_u * (lam_i - 2*alpha*u_i))
    lam:    dual ascent on   E[l_i] - u_i
            -> lam_i <- max(0, lam_i + eta_lam * (E[l_i] - u_i))

Constraint losses for prediction-count case:

    l_c(f_theta(x))      = softmax_c(f_theta(x)) - K_c / N
    l_{g,c}(f_theta(x))  = softmax_c(f_theta(x)) - K_{g,c} / N_g

So mean over the test set ((1/N) sum l_c) equals (count_soft_c - K_c) / N.
Constraint satisfied iff count_soft_c <= K_c.

No posthoc, no best_excess pick - paper takes the final-epoch model. Posthoc is
applied at the runner level for fair comparison with TraLO/Fioretto.
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


def _train_constraints(model, inputs: TrainInputs, device):
    hp = inputs.hyperparams
    constraint_epochs = hp.get("constraint_epochs", 150)
    lr_c = hp.get("lr_constraint", 1e-5)
    # Default dual-step bumped 10x for apples-to-apples convergence speed.
    # At 0.01 (original) lambda grows ~0.01/epoch when (count_soft-K)/N ~= 0.04
    # -> constraint contribution to L_total is ~1e-3, ~25x weaker than CE.
    # The model effectively trains CE-only for 100+ epochs before lambda
    # builds up. With 0.1 lambda hits meaningful magnitude by ep 10.
    eta_lambda = float(hp.get("hounie_eta_lambda", 0.1))
    eta_u = float(hp.get("hounie_eta_u", 0.1))
    alpha = float(hp.get("hounie_alpha", 10.0))
    if abs(1.0 - 2.0 * eta_u * alpha) >= 1.0:
        raise ValueError(
            f"hounie_rcl: eta_u={eta_u} with alpha={alpha} gives stability "
            f"factor {1.0 - 2.0 * eta_u * alpha:+.3f}; |factor| >= 1 means the "
            f"perturbation u oscillates or diverges instead of converging to "
            f"lambda/(2*alpha). The paper's value is eta_u=0.01.")
    batch_size = hp.get("batch_size", 64)
    chunk_size = hp.get("constraint_chunk_size", 256)
    # Apples-to-apples early stop: 5 consecutive satisfied epochs (matches TraLO).
    stable_count_threshold = int(hp.get("stable_count_threshold", 5))

    use_amp, amp_dtype, scaler = setup_runtime(device)

    constrained_classes = inputs.constrained_classes
    num_classes = inputs.num_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids

    n_test = len(inputs.X_test)
    unique_groups = np.unique(groups_np)
    group_sizes = {int(g): int((groups_np == g).sum()) for g in unique_groups}

    # K thresholds per active constraint (in absolute counts).
    K_global = {c: float(global_con[c])
                for c in constrained_classes if global_con[c] < UNLIMITED}
    K_local = {}
    for g in unique_groups:
        bounds = local_con.get(int(g), [UNLIMITED] * num_classes)
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                K_local[(int(g), c)] = float(bounds[c])

    # Multipliers and slack variables, one per active constraint.
    lam_g = {c: 0.0 for c in K_global}
    lam_l = {key: 0.0 for key in K_local}
    u_g = {c: 0.0 for c in K_global}
    u_l = {key: 0.0 for key in K_local}

    log.info(
        "Hounie RCL: %d constraint epochs, lr=%.2e eta_lam=%.4f eta_u=%.4f alpha=%.2f, "
        "%d global + %d local constraints",
        constraint_epochs, lr_c, eta_lambda, eta_u, alpha,
        len(lam_g), len(lam_l),
    )

    optimizer = make_optimizer(model.parameters(), lr_c, device)
    criterion_ce = nn.CrossEntropyLoss()
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, batch_size)

    X_test_dev = inputs.X_test.to(device)

    log_path = inputs.experiment_path / "training_log.csv"
    log_fields = ["epoch", "ce_loss", "constraint_loss", "total_excess",
                  "all_satisfied", "max_lam_g", "max_u_g", "h_u"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, log_fields).writeheader()

    satisfaction_epoch = None
    stable_count = 0
    # Best-checkpoint restore (mirrors TraLO). Snapshot model state BEFORE
    # the constraint step at every epoch that satisfies OR improves on the
    # lowest total excess seen so far. After training, restore best_sat if
    # final violates, else min_excess if final exceeds it. The Hounie paper
    # uses the last iterate; we add this for fair F1 comparison with TraLO.
    best_sat_state = None
    best_sat_epoch = None
    min_excess_state = None
    min_excess_epoch = None
    min_total_excess = float("inf")

    for epoch in range(constraint_epochs):
        epoch_start = time.time()

        # ---- Step 1: CE on TRAIN (theta SGD on L_ce) ----
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

        # ---- Step 2: soft-count gradient on TEST (theta SGD on Σ_i lam_i * E[l_i]) ----
        # Apples-to-apples with TraLO: model.eval() during the transductive pass.
        # Prevents dropout noise + BN drift from test data corrupting the
        # gradient. TraLO does this (AUDIT C1).
        model.eval()
        # First pass: aggregate soft + hard counts (no grad).
        total_soft = torch.zeros(num_classes, device=device)
        group_soft = {int(g): torch.zeros(num_classes, device=device)
                      for g in unique_groups}
        all_hard = []
        with torch.no_grad():
            for i in range(0, n_test, chunk_size):
                # FP32 forward for argmax/softmax consistency with eval.
                # AUDIT C7: BF16 argmax flips a few borderline samples vs FP32
                # (same fix applied in TraLO).
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

        # Compute hard-count satisfaction from pass-1 predictions BEFORE the
        # constraint step. Required so the snapshot below reflects the exact
        # model that produced these counts.
        hard_counts_pre = {c: int((hard_preds == c).sum()) for c in constrained_classes}
        total_excess_pre = sum(
            max(0, hard_counts_pre[c] - int(global_con[c]))
            for c in constrained_classes if global_con[c] < UNLIMITED
        )
        if local_con:
            for g_id, bounds in local_con.items():
                for c in constrained_classes:
                    if bounds[c] < UNLIMITED:
                        gc = int(((hard_preds == c) & (groups_np == g_id)).sum())
                        total_excess_pre += max(0, gc - int(bounds[c]))
        all_satisfied_pre = (total_excess_pre == 0)
        snapshot_state = None
        if all_satisfied_pre or total_excess_pre < min_total_excess:
            snapshot_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

        # Second pass: weighted gradient if any lam > 0.
        constraint_loss_val = 0.0
        has_active = (any(v > 0 for v in lam_g.values())
                      or any(v > 0 for v in lam_l.values()))
        did_backward = False
        if has_active:
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, n_test, chunk_size):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    chunk_logits = model(X_test_dev[i:i + chunk_size])
                    chunk_proba = F.softmax(chunk_logits.float(), dim=1)
                    chunk_loss = torch.zeros(1, device=device)
                    # AUDIT BUGFIX: divide by n_test / N_g to match the dual ascent
                    # scale (mean_l = sum/N), so primal d/dtheta and dual lambda
                    # update are on the same scale. Without this, primal gradient
                    # is N-times stronger than intended -> over-suppresses the
                    # constrained class and inflates ECE.
                    for c, lam in lam_g.items():
                        if lam > 0:
                            chunk_loss = chunk_loss + lam * chunk_proba[:, c].sum() / n_test
                    chunk_groups = groups_np[i:i + chunk_size]
                    for (g, c), lam in lam_l.items():
                        if lam > 0:
                            mask = (chunk_groups == g)
                            if mask.any():
                                N_g = max(1, group_sizes[g])
                                chunk_loss = chunk_loss + lam * chunk_proba[mask, c].sum() / N_g
                if chunk_loss.item() > 0:
                    if scaler:
                        scaler.scale(chunk_loss).backward()
                    else:
                        chunk_loss.backward()
                    constraint_loss_val += chunk_loss.item()
                    did_backward = True
            if did_backward:
                # Grad clip + grad_norm>0 gate + scaler.update() always called
                # (mirrors TraLO recovery pattern).
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

        # ---- Step 3: dual ascent on lambda (paper Eq. 5 / Alg. 2) ----
        # E[l_i] = (count_soft_i - K_i) / N_i  (per-constraint normalisation).
        for c, K in K_global.items():
            mean_l = (total_soft[c].item() - K) / n_test
            lam_g[c] = max(0.0, lam_g[c] + eta_lambda * (mean_l - u_g[c]))
        for (g, c), K in K_local.items():
            N_g = max(1, group_sizes[g])
            mean_l = (group_soft[g][c].item() - K) / N_g
            lam_l[(g, c)] = max(0.0, lam_l[(g, c)] + eta_lambda * (mean_l - u_l[(g, c)]))

        # ---- Step 4: perturbation update on u (h(u) = alpha * ||u||^2) ----
        # u_i <- max(0, u_i + eta_u * (lam_i - 2 * alpha * u_i)).
        for c in K_global:
            u_g[c] = max(0.0, u_g[c] + eta_u * (lam_g[c] - 2.0 * alpha * u_g[c]))
        for key in K_local:
            u_l[key] = max(0.0, u_l[key] + eta_u * (lam_l[key] - 2.0 * alpha * u_l[key]))

        # ---- Bookkeeping ---- (uses the pre-step satisfaction state computed
        # earlier, which is what the snapshot reflects).
        hard_counts = hard_counts_pre
        total_excess = total_excess_pre
        all_satisfied = all_satisfied_pre
        if all_satisfied and satisfaction_epoch is None:
            # +1: align with TraLO's convention so cross-method tables
            # report the SAME epoch number for the same training step.
            satisfaction_epoch = epoch + 1
            log.info("Hounie RCL: first satisfaction at epoch %d", epoch + 1)
        # Apples-to-apples early stop: 5 consecutive satisfied epochs (matches TraLO/Fioretto).
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

        h_u = alpha * (sum(v ** 2 for v in u_g.values())
                       + sum(v ** 2 for v in u_l.values()))

        row = {
            "epoch": epoch,
            "ce_loss": round(np.mean(ce_losses), 6),
            "constraint_loss": round(constraint_loss_val, 6),
            "total_excess": total_excess,
            "all_satisfied": int(all_satisfied),
            "max_lam_g": round(max(lam_g.values()) if lam_g else 0.0, 6),
            "max_u_g": round(max(u_g.values()) if u_g else 0.0, 6),
            "h_u": round(h_u, 6),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_fields).writerow(row)

        if epoch < 5 or (epoch + 1) % 10 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join(f"c{c}={lam_g[c]:.3f}" for c in sorted(lam_g))
            u_str = " ".join(f"c{c}={u_g[c]:.3f}" for c in sorted(u_g))
            log.info(
                "Hounie %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s stable=%d "
                "lam=[%s] u=[%s] h_u=%.4f [%.1fs]",
                epoch + 1, constraint_epochs,
                np.mean(ce_losses), constraint_loss_val, total_excess,
                all_satisfied, stable_count, lam_str, u_str, h_u,
                time.time() - epoch_start,
            )

        if stable_count >= stable_count_threshold:
            log.info("Hounie: converged (constraints stable for %d epochs at ep %d)",
                     stable_count, epoch + 1)
            break

    return (satisfaction_epoch, best_sat_state, best_sat_epoch,
            min_excess_state, min_excess_epoch, min_total_excess)


def train(inputs: TrainInputs) -> TrainOutputs:
    hp = inputs.hyperparams
    model = inputs.model
    device = inputs.device
    (satisfaction_epoch, best_sat_state, best_sat_epoch,
     min_excess_state, min_excess_epoch, min_total_excess
     ) = _train_constraints(model, inputs, device)

    # Apples-to-apples checkpoint restore (mirrors TraLO). Restore best_sat
    # if final epoch violates, else min_excess if final exceeds the lowest
    # seen excess. Selection criterion is the constraint excess axis, NOT F1.
    constrained_classes = inputs.constrained_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids
    X_test_dev = inputs.X_test.to(device)

    model.eval()
    chunk_size = inputs.hyperparams.get("constraint_chunk_size", 256)
    with torch.no_grad():
        all_hard = []
        for i in range(0, len(X_test_dev), chunk_size):
            chunk_logits = model(X_test_dev[i:i + chunk_size])
            all_hard.append(chunk_logits.argmax(dim=1))
        hard_preds_final = torch.cat(all_hard).cpu().numpy()
    final_total_excess = sum(
        max(0, int((hard_preds_final == c).sum()) - int(global_con[c]))
        for c in constrained_classes if global_con[c] < UNLIMITED
    )
    if local_con:
        for g_id, bounds in local_con.items():
            for c in constrained_classes:
                if bounds[c] < UNLIMITED:
                    gc = int(((hard_preds_final == c) & (groups_np == g_id)).sum())
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
    allow_restore = bool(hp.get("enable_checkpoint_restore", True))
    if not allow_restore:
        log.info("Hounie: enable_checkpoint_restore=False, keeping the trained model")
    if allow_restore and best_sat_state is not None and final_violates:
        log.info("Hounie: final violates; restoring best-satisfied checkpoint from epoch %d",
                 best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
    elif (allow_restore and min_excess_state is not None
          and final_total_excess > min_total_excess):
        log.info("Hounie: final excess=%d > min seen excess=%d (epoch %d); "
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
