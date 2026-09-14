"""Fioretto-LDF: linear penalty with per-constraint subgradient ascent."""

import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.pipeline.setup import setup_runtime
from src.training.constraint_step import (
    constraint_autocast,
    constraint_backward,
    finish_constraint_step,
)
from src.methodologies.dual_common import (
    TrainingState,
    ce_epoch,
    count_excess,
    count_fields,
    count_row,
    dual_setup,
    open_epoch_log,
    read_step_config,
    run_dual_arm,
    transductive_counts,
)
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _train_constraints(model, inputs, device):
    """Fioretto Algorithm 1/2: linear penalty + per-constraint subgradient dual ascent."""
    hp = inputs.hyperparams
    step_cfg = read_step_config(hp)
    constraint_epochs = _required(hp, "constraint_epochs", int)
    lr_c = _required(hp, "lr_constraint", float)
    if "fioretto_step_size" not in hp:
        raise ValueError(
            "fioretto_step_size is required in hyperparams. The runner used to default to 0.01 while the multi-methodology generator defaulted to 0.005, producing inconsistent baselines silently. Set it explicitly in your config (typical sweep: 0.001/0.005/0.01)."
        )
    step_size = float(hp["fioretto_step_size"])
    batch_size = hp.get("batch_size", 64)
    chunk_size = _required(hp, "constraint_chunk_size", int)
    (use_amp, amp_dtype, scaler) = setup_runtime(device)
    constrained_classes = inputs.constrained_classes
    num_classes = inputs.num_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids
    lam0 = float(hp.get("fioretto_lambda_init", 0.0))
    lambda_g = {c: lam0 for c in constrained_classes if global_con[c] < UNLIMITED}
    lambda_l = {}
    for group_id, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                lambda_l[group_id, c] = lam0
    log.info(
        "Fioretto LDF: %d epochs, lr=%.2e, step_size=%.4f, %d global + %d local multipliers",
        constraint_epochs,
        lr_c,
        step_size,
        len(lambda_g),
        len(lambda_l),
    )
    (optimizer, criterion_ce, train_loader) = dual_setup(
        model, inputs, device, lr_c, batch_size
    )
    X_test_dev = inputs.X_test.to(device)
    unique_groups = np.unique(groups_np)
    ck = TrainingState("Fioretto")
    log_fields = [
        "epoch",
        "train_acc",
        "ce_loss",
        "constraint_loss",
        "total_excess",
        "all_satisfied",
        "max_lambda_g",
        "grad_norm",
    ]
    log_fields = log_fields + count_fields(constrained_classes)
    write_row = open_epoch_log(inputs.experiment_path, log_fields)
    stable_count = 0
    for epoch in range(constraint_epochs):
        last_grad_norm = 0.0
        epoch_start = time.time()
        (ce_losses, train_acc) = ce_epoch(
            model,
            train_loader,
            optimizer,
            criterion_ce,
            device,
            amp_dtype,
            use_amp,
            scaler,
        )
        (total_soft, group_soft, hard_preds) = transductive_counts(
            model, X_test_dev, groups_np, unique_groups, num_classes, chunk_size, device
        )
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
        total_excess = count_excess(
            hard_preds, groups_np, constrained_classes, global_con, local_con
        )
        all_satisfied = total_excess == 0
        for c, viol in violations_g.items():
            lambda_g[c] += step_size * viol
        for key, viol in violations_l.items():
            lambda_l[key] += step_size * viol
        has_work = any((lambda_g.get(c, 0) > 0 for c in violated_global)) or any(
            (lambda_l.get(k, 0) > 0 for k in violated_local)
        )
        constraint_loss_val = 0.0
        did_backward = False
        if has_work:
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(X_test_dev), chunk_size):
                with constraint_autocast(amp_dtype, use_amp, step_cfg["fp32"]):
                    chunk_logits = model(X_test_dev[i : i + chunk_size])
                    chunk_proba = F.softmax(chunk_logits, dim=1)
                    chunk_loss = torch.zeros(1, device=device)
                    for c in violated_global:
                        if lambda_g[c] > 0:
                            chunk_loss = (
                                chunk_loss + lambda_g[c] * chunk_proba[:, c].sum()
                            )
                    chunk_groups = groups_np[i : i + chunk_size]
                    for key in violated_local:
                        (g, c) = key
                        if lambda_l[key] > 0:
                            mask = chunk_groups == g
                            if mask.any():
                                chunk_loss = (
                                    chunk_loss
                                    + lambda_l[key] * chunk_proba[mask, c].sum()
                                )
                if chunk_loss.item() > 0:
                    constraint_backward(chunk_loss, scaler, step_cfg["fp32"])
                    constraint_loss_val += chunk_loss.item()
                    did_backward = True
            if did_backward:
                (last_grad_norm, applied) = finish_constraint_step(
                    model, optimizer, scaler, **step_cfg
                )
                ck.record_step(applied)
        ck.record(all_satisfied, epoch)
        stable_count = stable_count + 1 if all_satisfied else 0
        row = {
            "epoch": epoch,
            "train_acc": round(train_acc, 4),
            "ce_loss": round(np.mean(ce_losses), 6),
            "constraint_loss": round(constraint_loss_val, 6),
            "total_excess": total_excess,
            "all_satisfied": int(all_satisfied),
            "max_lambda_g": round(max(lambda_g.values()) if lambda_g else 0, 6),
            "grad_norm": round(float(last_grad_norm), 6),
        }
        row.update(count_row(hard_preds, total_soft, constrained_classes, global_con))
        write_row(row)
        if epoch < 5 or (epoch + 1) % 25 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join((f"c{c}={lambda_g[c]:.3f}" for c in sorted(lambda_g)))
            log.info(
                "Fioretto %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s stable=%d lam=[%s] [%.1fs]",
                epoch + 1,
                constraint_epochs,
                np.mean(ce_losses),
                constraint_loss_val,
                total_excess,
                all_satisfied,
                stable_count,
                lam_str,
                time.time() - epoch_start,
            )
    return ck


def train(inputs: TrainInputs) -> TrainOutputs:
    return run_dual_arm(inputs, _train_constraints, "Fioretto")
