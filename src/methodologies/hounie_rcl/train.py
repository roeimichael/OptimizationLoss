"""hounie_rcl methodology: Resilient Constrained Learning.

Faithful reimplementation of the algorithm from:

    Hounie, Ribeiro, Chamon. "Resilient Constrained Learning."
    NeurIPS 2023. arXiv:2306.02426.

Algorithm 1 (generic) + Algorithm 2 (federated specialisation) collapsed onto
TraLO's prediction-count constraint task. The mapping from their notation to
this code is documented in `benchmarks/hounie/benchmark_fix/algorithm_derivation.md`.

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
    constraint_epochs = hp.get("constraint_epochs", 100)
    lr_c = hp.get("lr_constraint", 1e-5)
    eta_lambda = float(hp.get("hounie_eta_lambda", 0.01))
    eta_u = float(hp.get("hounie_eta_u", 0.01))
    alpha = float(hp.get("hounie_alpha", 10.0))
    batch_size = hp.get("batch_size", 64)
    chunk_size = hp.get("constraint_chunk_size", 256)

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

    for epoch in range(constraint_epochs):
        epoch_start = time.time()

        # ---- Step 1: CE on TRAIN (theta SGD on L_ce) ----
        model.train()
        ce_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                ce_loss = criterion_ce(model(batch_X), batch_y)
            ce_losses.append(ce_loss.item())
            if scaler:
                scaler.scale(ce_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                ce_loss.backward()
                optimizer.step()

        # ---- Step 2: soft-count gradient on TEST (theta SGD on Σ_i lam_i * E[l_i]) ----
        model.train()
        # First pass: aggregate soft + hard counts (no grad).
        total_soft = torch.zeros(num_classes, device=device)
        group_soft = {int(g): torch.zeros(num_classes, device=device)
                      for g in unique_groups}
        all_hard = []
        with torch.no_grad():
            for i in range(0, n_test, chunk_size):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    chunk_logits = model(X_test_dev[i:i + chunk_size])
                chunk_proba = F.softmax(chunk_logits.float(), dim=1)
                total_soft += chunk_proba.sum(dim=0)
                all_hard.append(chunk_logits.argmax(dim=1))
                chunk_groups = groups_np[i:i + chunk_size]
                for g in unique_groups:
                    mask = (chunk_groups == g)
                    if mask.any():
                        group_soft[int(g)] += chunk_proba[mask].sum(dim=0)
            hard_preds = torch.cat(all_hard).cpu().numpy()

        # Second pass: weighted gradient if any lam > 0.
        constraint_loss_val = 0.0
        has_active = (any(v > 0 for v in lam_g.values())
                      or any(v > 0 for v in lam_l.values()))
        if has_active:
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, n_test, chunk_size):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    chunk_logits = model(X_test_dev[i:i + chunk_size])
                    chunk_proba = F.softmax(chunk_logits.float(), dim=1)
                    chunk_loss = torch.zeros(1, device=device)
                    for c, lam in lam_g.items():
                        if lam > 0:
                            chunk_loss = chunk_loss + lam * chunk_proba[:, c].sum()
                    chunk_groups = groups_np[i:i + chunk_size]
                    for (g, c), lam in lam_l.items():
                        if lam > 0:
                            mask = (chunk_groups == g)
                            if mask.any():
                                chunk_loss = chunk_loss + lam * chunk_proba[mask, c].sum()
                if chunk_loss.item() > 0:
                    if scaler:
                        scaler.scale(chunk_loss).backward()
                    else:
                        chunk_loss.backward()
                    constraint_loss_val += chunk_loss.item()
            if scaler:
                try:
                    scaler.step(optimizer)
                    scaler.update()
                except (AssertionError, RuntimeError):
                    optimizer.step()
            else:
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

        # ---- Bookkeeping ----
        hard_counts = {c: int((hard_preds == c).sum()) for c in constrained_classes}
        total_excess = sum(
            max(0, hard_counts[c] - int(global_con[c]))
            for c in constrained_classes if global_con[c] < UNLIMITED
        )
        all_satisfied = all(
            hard_counts[c] <= int(global_con[c])
            for c in constrained_classes if global_con[c] < UNLIMITED
        )
        if all_satisfied and satisfaction_epoch is None:
            satisfaction_epoch = epoch
            log.info("Hounie RCL: first satisfaction at epoch %d", epoch)

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
                "Hounie %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s "
                "lam=[%s] u=[%s] h_u=%.4f [%.1fs]",
                epoch + 1, constraint_epochs,
                np.mean(ce_losses), constraint_loss_val, total_excess,
                all_satisfied, lam_str, u_str, h_u,
                time.time() - epoch_start,
            )

    return satisfaction_epoch


def train(inputs: TrainInputs) -> TrainOutputs:
    satisfaction_epoch = _train_constraints(inputs.model, inputs, inputs.device)
    return TrainOutputs(
        model=inputs.model,
        summary={
            "satisfaction_epoch": satisfaction_epoch,
        },
    )
