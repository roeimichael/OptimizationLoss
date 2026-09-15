"""TraLO reference: sum of probabilities, bounded penalty, constant hard-count ratchet.
Two-pass chunking preserves the full-population gradient. JSONL events record actual optimizer displacement."""

import logging
import time
import hashlib
import json
import uuid
import torch
import torch.nn.functional as F
from src.losses import MulticlassTransductiveLoss
from src.methodologies.dual_common import read_step_config
from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.pipeline.setup import setup_runtime
from src.pipeline.warmup import make_ce_criterion, make_dataloader, make_optimizer
from src.training.item_weights import (
    apply as apply_weights,
    features_and_preds,
    item_weights,
    read_weight_config,
    weight_spread,
)
from src.training.constraint_step import (
    constraint_autocast,
    constraint_backward,
    finish_constraint_step,
)
from src.training.logging import (
    append_constraint_event,
    log_progress_to_csv,
    write_csv_header,
)
from src.training.metrics import compute_prediction_statistics
from src.training.reordering import capped_scores, reordering_report
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _scope_events(criterion, global_soft, global_hard, local_soft, local_hard):
    scopes = [("global", None, criterion.global_constraints, global_soft, global_hard)]
    scopes += [
        ("local", gid, getattr(criterion, name), local_soft[gid], local_hard[gid])
        for (gid, name) in criterion.local_groups.items()
    ]
    rows = []
    for scope, gid, bounds, soft, hard in scopes:
        for c, bound in enumerate(bounds):
            if bound >= UNLIMITED:
                continue
            (budget, soft_count, hard_count) = (
                int(bound),
                float(soft[c]),
                int(hard[c]),
            )
            rows.append(
                {
                    "scope": scope,
                    "group": str(gid) if gid is not None else None,
                    "class": c,
                    "budget": budget,
                    "soft_count": soft_count,
                    "hard_count": hard_count,
                    "soft_residual": soft_count - budget,
                    "hard_residual": hard_count - budget,
                    "multiplier_before": criterion.get_lambda_per_class(
                        c, scope=scope, group_id=gid
                    ),
                }
            )
    return rows


def train(inputs: TrainInputs) -> TrainOutputs:
    config = inputs.config
    hp = inputs.hyperparams
    step_cfg = read_step_config(hp)
    weight_cfg = read_weight_config(hp)
    device = inputs.device
    num_classes = inputs.num_classes
    model = inputs.model
    csv_log_path = str(inputs.csv_log_path)
    (use_amp, amp_dtype, scaler) = setup_runtime(device)
    warmup_epochs = hp["warmup_epochs"]
    constraint_epochs = _required(hp, "constraint_epochs", int)
    total_epochs = warmup_epochs + constraint_epochs
    lambda_step = hp["lambda_step"]
    criterion_ce = make_ce_criterion(config, inputs.y_train, num_classes, device)
    lr_constraint = _required(hp, "lr_constraint", float)
    optimizer = make_optimizer(model.parameters(), lr_constraint, device)
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, hp["batch_size"],
                                   augment=hp.get("augment", False))
    X_test = inputs.X_test.to(device)
    group_ids = torch.LongTensor(inputs.group_ids).to(device)
    global_con = inputs.global_con
    local_con = inputs.local_con
    criterion_constraint = MulticlassTransductiveLoss(
        global_constraints=global_con,
        local_constraints=local_con,
        num_classes=num_classes,
        initial_rho=hp.get("initial_rho", 0.5),
    ).to(device)
    constrained_classes = sorted(
        {c for c in range(num_classes) if global_con[c] < UNLIMITED}
        | {
            c
            for bounds in local_con.values()
            for c in range(num_classes)
            if bounds[c] < UNLIMITED
        }
    )
    init_g = _required(hp, "lambda_global")
    init_l = _required(hp, "lambda_local")
    for c in constrained_classes:
        criterion_constraint.set_lambda_per_class(c, init_g, scope="global")
    for gid, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                criterion_constraint.set_lambda_per_class(
                    c, init_l, scope="local", group_id=gid
                )
    rho_target = hp.get("rho_target", 100.0)
    initial_rho = hp.get("initial_rho", 0.5)
    rho_step = (rho_target - initial_rho) / max(constraint_epochs, 1)
    rho_frozen = False
    satisfaction_epoch = None
    stable_count = 0
    constraint_steps_applied = constraint_steps_attempted = 0
    training_start = time.time()
    attempt_id = uuid.uuid4().hex
    config_sha256 = hashlib.sha256(
        json.dumps(inputs.config, sort_keys=True).encode()
    ).hexdigest()
    write_csv_header(csv_log_path, num_classes, local_con)
    chunk_size = _required(hp, "constraint_chunk_size", int)
    warmup_scores = capped_scores(model, X_test, constrained_classes, chunk_size)
    cached_train_acc = 0.0
    for epoch in range(warmup_epochs, total_epochs):
        epoch_start = time.time()
        task_attempted = task_applied = 0
        model.train()
        for pg in optimizer.param_groups:
            pg["lr"] = lr_constraint
        epoch_ce = 0.0
        num_batches = max(len(train_loader), 1)
        (train_correct, train_total) = (0, 0)
        for batch_X, batch_y in train_loader:
            (batch_X, batch_y) = (batch_X.to(device), batch_y.to(device))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits_ce = model(batch_X)
                loss_ce = criterion_ce(logits_ce, batch_y)
            if scaler:
                scaler.scale(loss_ce).backward()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                task_applied += int(scaler.get_scale() >= scale_before)
            else:
                loss_ce.backward()
                optimizer.step()
                task_applied += 1
            task_attempted += 1
            epoch_ce += loss_ce.item()
            with torch.no_grad():
                train_correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
                train_total += batch_y.size(0)
        cached_train_acc = (
            train_correct / train_total if train_total > 0 else cached_train_acc
        )
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        n_test = len(X_test)
        n_chunks = (n_test + chunk_size - 1) // chunk_size
        # The soft count is sum_i w_i p_i(c). In the reference arm w is exactly
        # ones and this costs nothing; see src/training/item_weights.py for why
        # a non-uniform w is the only escape from the harm lemma.
        if weight_cfg["mode"] == "uniform":
            item_w = torch.ones(n_test, device=device, dtype=torch.float32)
        else:
            w_feats, w_preds = features_and_preds(model, X_test, chunk_size, device)
            item_w = item_weights(weight_cfg, w_feats, w_preds, group_ids)
            del w_feats, w_preds
        weight_cv = weight_spread(item_w)
        with torch.no_grad():
            total_global_soft = torch.zeros(num_classes, device=device)
            total_global_hard = torch.zeros(num_classes, device=device)
            total_local_soft = {
                gid: torch.zeros(num_classes, device=device)
                for gid in criterion_constraint.local_groups
            }
            total_local_hard = {
                gid: torch.zeros(num_classes, device=device)
                for gid in criterion_constraint.local_groups
            }
            for ci in range(n_chunks):
                start = ci * chunk_size
                end = min(start + chunk_size, n_test)
                chunk_logits = model(X_test[start:end])
                chunk_proba = F.softmax(chunk_logits, dim=1)
                chunk_preds = chunk_logits.argmax(dim=1)
                chunk_proba = apply_weights(item_w, chunk_proba, start, end)
                total_global_soft += chunk_proba.sum(dim=0)
                total_global_hard += torch.bincount(
                    chunk_preds, minlength=num_classes
                ).float()
                chunk_gids = group_ids[start:end]
                for gid in total_local_soft:
                    mask = chunk_gids == gid
                    if mask.any():
                        total_local_soft[gid] += chunk_proba[mask].sum(dim=0)
                        total_local_hard[gid] += torch.bincount(
                            chunk_preds[mask], minlength=num_classes
                        ).float()
        snapshot_global_satisfied = True
        for c in constrained_classes:
            if (
                total_global_hard[c].item()
                > criterion_constraint.global_constraints[c].item()
            ):
                snapshot_global_satisfied = False
                break
        snapshot_local_satisfied = True
        for gid_s, buffer_name_s in criterion_constraint.local_groups.items():
            lc_s = getattr(criterion_constraint, buffer_name_s)
            for c in constrained_classes:
                if c < len(lc_s) and lc_s[c] < UNLIMITED:
                    if total_local_hard[gid_s][c].item() > lc_s[c].item():
                        snapshot_local_satisfied = False
                        break
            if not snapshot_local_satisfied:
                break
        loss_global_val = criterion_constraint.compute_global_from_counts(
            total_global_soft
        ).item()
        loss_local_val = criterion_constraint.compute_local_from_counts(
            total_local_soft
        ).item()
        bounded_total = loss_global_val + loss_local_val
        total_constraint = bounded_total
        scope_events = _scope_events(
            criterion_constraint,
            total_global_soft,
            total_global_hard,
            total_local_soft,
            total_local_hard,
        )
        rho_before = float(criterion_constraint.get_rho())
        has_constraint = total_constraint > 0
        if has_constraint:
            for ci in range(n_chunks):
                start = ci * chunk_size
                end = min(start + chunk_size, n_test)
                with constraint_autocast(amp_dtype, use_amp, step_cfg["fp32"]):
                    chunk_logits = model(X_test[start:end])
                chunk_logits_f = chunk_logits.float()
                chunk_proba = F.softmax(chunk_logits_f, dim=1)
                chunk_loss = torch.tensor(0.0, device=device)
                chunk_eff = apply_weights(item_w, chunk_proba, start, end)
                chunk_global = chunk_eff.sum(dim=0)
                chunk_gids = group_ids[start:end]
                chunk_local_soft = {}
                for gid in criterion_constraint.local_groups:
                    mask = chunk_gids == gid
                    if mask.any():
                        chunk_local_soft[gid] = chunk_eff[mask].sum(dim=0)
                    else:
                        chunk_local_soft[gid] = torch.zeros(num_classes, device=device)
                g_soft = (
                    total_global_soft.detach()
                    - chunk_eff.sum(dim=0).detach()
                    + chunk_global
                )
                l_soft = {}
                for gid in total_local_soft:
                    l_soft[gid] = (
                        total_local_soft[gid].detach()
                        - chunk_local_soft[gid].detach()
                        + chunk_local_soft[gid]
                    )
                lg = criterion_constraint.compute_global_from_counts(g_soft)
                ll = criterion_constraint.compute_local_from_counts(l_soft)
                chunk_loss = chunk_loss + lg + ll
                constraint_backward(chunk_loss, scaler, step_cfg["fp32"])
        last_grad_norm = 0.0
        step_diagnostics = {
            "optimizer_step_applied": False,
            "parameter_delta_norm": 0.0,
            "pre_clip_grad_norm": None,
            "transformed_grad_norm": None,
            "descent_alignment": None,
            "nonfinite_gradient": False,
            "amp_overflow_detected": False,
        }
        did_backward = has_constraint
        if did_backward:
            (last_grad_norm, applied) = finish_constraint_step(
                model, optimizer, scaler, diagnostics=step_diagnostics, **step_cfg
            )
            constraint_steps_attempted += 1
            constraint_steps_applied += 1 if applied else 0
        avg_ce = epoch_ce / num_batches
        global_satisfied = snapshot_global_satisfied
        local_satisfied = snapshot_local_satisfied
        is_satisfied = global_satisfied and local_satisfied
        if is_satisfied:
            stable_count += 1
        else:
            stable_count = 0
        ratchet_gate = satisfaction_epoch is None
        for c in constrained_classes:
            hard_c = total_global_hard[c].item()
            limit_c = criterion_constraint.global_constraints[c].item()
            if hard_c > limit_c and ratchet_gate:
                old = criterion_constraint.get_lambda_per_class(c, scope="global")
                criterion_constraint.set_lambda_per_class(
                    c, old + lambda_step, scope="global"
                )
        for gid, buffer_name in criterion_constraint.local_groups.items():
            lc = getattr(criterion_constraint, buffer_name)
            for c in constrained_classes:
                if c < len(lc) and lc[c] < UNLIMITED:
                    hard_c = total_local_hard[gid][c].item()
                    if hard_c > lc[c].item() and ratchet_gate:
                        old = criterion_constraint.get_lambda_per_class(
                            c, scope="local", group_id=gid
                        )
                        criterion_constraint.set_lambda_per_class(
                            c, old + lambda_step, scope="local", group_id=gid
                        )
        if is_satisfied and satisfaction_epoch is None:
            satisfaction_epoch = epoch + 1
            if not rho_frozen:
                rho_frozen = True
                log.info(
                    "First satisfied at epoch %d, freezing rho=%.3f",
                    epoch + 1,
                    criterion_constraint.get_rho(),
                )
        if not rho_frozen:
            criterion_constraint.increment_rho(rho_step)
        after_scopes = _scope_events(
            criterion_constraint,
            total_global_soft,
            total_global_hard,
            total_local_soft,
            total_local_hard,
        )
        for row, after in zip(scope_events, after_scopes):
            row["multiplier_after"] = after["multiplier_before"]
        append_constraint_event(
            inputs.experiment_path,
            {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "phase": "constraint",
                "method": "tralo",
                "seed": hp["seed"],
                "config_sha256": config_sha256,
                "code_version": inputs.config.get("code_version"),
                "device": str(device),
                "constraint_fp32": step_cfg["fp32"],
                "constraint_grad_mode": step_cfg["mode"],
                "constraint_weight": weight_cfg["mode"],
                # 0.0 exactly when the weights are inert. gate:weight_bites
                # reads this, because five earlier flags were inert unnoticed.
                "constraint_weight_cv": weight_cv,
                "epoch_absolute_1based": epoch + 1,
                "constraint_epoch_1based": epoch - warmup_epochs + 1,
                "counts_state": "post_task_pre_constraint",
                "count_values": "penalty_input",
                "task_updates_planned": len(train_loader),
                "task_updates_attempted": task_attempted,
                "task_updates_applied": task_applied,
                "task_updates_skipped": task_attempted - task_applied,
                "constraint_updates_planned": 1,
                "constraint_updates_attempted": int(did_backward),
                "step": step_diagnostics,
                "rho_before": rho_before,
                "rho_after": float(criterion_constraint.get_rho()),
                "scopes": scope_events,
                "constraint_objective_pre_step": bounded_total,
                "task_loss_online_mean": avg_ce,
                "train_accuracy_online": cached_train_acc,
                "elapsed_seconds": time.time() - epoch_start,
            },
        )
        train_acc = cached_train_acc
        g_counts = {c: int(total_global_hard[c].item()) for c in range(num_classes)}
        l_counts = {
            gid: {c: int(total_local_hard[gid][c].item()) for c in range(num_classes)}
            for gid in total_local_hard
        }
        g_soft_d = {c: total_global_soft[c].item() for c in range(num_classes)}
        l_soft_d = {
            gid: {c: total_local_soft[gid][c].item() for c in range(num_classes)}
            for gid in total_local_soft
        }
        mode_tag = "Satisfied" if is_satisfied else "Constraint"
        lam_local = criterion_constraint.lambda_local_per_key
        lam_L_mean = sum(lam_local.values()) / len(lam_local) if lam_local else 0.0
        lam_T_mean = sum(criterion_constraint.lambda_global_per_class.values()) / max(
            1, len(criterion_constraint.lambda_global_per_class)
        )
        log.info(
            "Epoch %d [%s] ce=%.4f bounded=%.4f lam_T=%.3f rho=%.3f acc=%.4f stable=%d g_%s l_%s",
            epoch + 1,
            mode_tag,
            avg_ce,
            bounded_total,
            lam_T_mean,
            criterion_constraint.get_rho(),
            train_acc,
            stable_count,
            "OK" if global_satisfied else "VIOL",
            "OK" if local_satisfied else "VIOL",
        )
        log_progress_to_csv(
            csv_log_path,
            epoch,
            avg_ce,
            train_acc,
            loss_global_val,
            loss_local_val,
            g_counts,
            l_counts,
            g_soft_d,
            l_soft_d,
            lam_T_mean,
            lam_L_mean,
            global_con,
            global_satisfied,
            local_satisfied,
            grad_norm=last_grad_norm,
            local_constraints=local_con,
        )
        model.train()
    elapsed = time.time() - training_start
    log.info(
        "Training complete: %.1fs, satisfaction epoch: %s",
        elapsed,
        satisfaction_epoch or "N/A",
    )
    model.eval()
    (g_counts, l_counts, g_soft, l_soft) = compute_prediction_statistics(
        model, X_test, group_ids, num_classes=num_classes
    )
    reorder = reordering_report(
        model, X_test, warmup_scores, constrained_classes, chunk_size
    )
    final_soft_hard_gap = {
        c: abs(g_soft.get(c, 0) - g_counts.get(c, 0)) for c in constrained_classes
    }
    return TrainOutputs(
        model=model,
        summary={
            "satisfaction_epoch": satisfaction_epoch,
            "soft_hard_gap": final_soft_hard_gap,
            "constraint_steps_applied": int(constraint_steps_applied),
            "constraint_steps_attempted": int(constraint_steps_attempted),
            "reordering": reorder,
        },
    )
