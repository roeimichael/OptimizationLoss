"""fioretto_ldf methodology: linear penalty + per-constraint subgradient ascent.

Lifted from the prior fioretto_research/run_fioretto.py module. The dual-checkpoint pick
(final vs best_excess by F1 after post-hoc) lives here because it is
methodology-specific.
"""

import csv
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.pipeline.eval import evaluate_with_posthoc
from src.pipeline.setup import setup_runtime
from src.pipeline.warmup import make_dataloader, make_optimizer
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _train_constraints(model, config, inputs, device):
    """Fioretto Algorithm 1/2: linear penalty + per-constraint subgradient dual ascent."""
    hp = inputs.hyperparams
    constraint_epochs = hp.get("constraint_epochs", 300)
    lr_c = hp.get("lr_constraint", 1e-5)
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
    best_model_state = None
    best_excess = float("inf")

    log_path = inputs.experiment_path / "training_log.csv"
    log_fields = ["epoch", "ce_loss", "constraint_loss", "total_excess",
                  "all_satisfied", "max_lambda_g"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, log_fields).writeheader()

    for epoch in range(constraint_epochs):
        epoch_start = time.time()

        # ---- Step 1: CE on TRAIN data (batched) ----
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

        # ---- Step 2: constraint gradient on TEST data (transductive) ----
        model.train()
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
                        scaler.step(optimizer)
                        scaler.update()
                    except (AssertionError, RuntimeError):
                        optimizer.step()
                else:
                    optimizer.step()

        # ---- Step 3: subgradient dual update (Fioretto Eq. 5) ----
        for c, viol in violations_g.items():
            lambda_g[c] += step_size * viol
        for key, viol in violations_l.items():
            lambda_l[key] += step_size * viol

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
            log.info("Fioretto: first satisfaction at epoch %d", epoch)
        if total_excess < best_excess:
            best_excess = total_excess
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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
            log.info("Fioretto %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s lam=[%s] [%.1fs]",
                     epoch + 1, constraint_epochs, np.mean(ce_losses),
                     constraint_loss_val, total_excess, all_satisfied,
                     lam_str, time.time() - epoch_start)

    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return satisfaction_epoch, final_state, best_model_state, best_excess


def train(inputs: TrainInputs) -> TrainOutputs:
    model = inputs.model
    device = inputs.device

    satisfaction_epoch, final_state, best_excess_state, best_excess = _train_constraints(
        model, inputs.config, inputs, device,
    )

    # Dual-checkpoint pick: F1 after post-hoc decides between final and
    # best_excess. Lives here because it is methodology-specific (our_approach
    # uses the final epoch only, by design).
    X_test_dev = inputs.X_test.to(device)

    def _eval_candidate(state, label):
        model.load_state_dict(state)
        model.to(device)
        return label, evaluate_with_posthoc(
            model, X_test_dev, inputs.y_test, inputs.group_ids,
            inputs.global_con, inputs.local_con,
            inputs.constrained_classes, inputs.num_classes,
            label=label,
        )

    candidates = [_eval_candidate(final_state, "final")]
    if best_excess_state is not None:
        candidates.append(_eval_candidate(best_excess_state, "best_excess"))

    best_source, best_result = max(candidates, key=lambda kv: kv[1]["metrics"]["f1_macro"])
    log.info("Selected checkpoint: %s (f1_macro=%.4f from %d candidates)",
             best_source, best_result["metrics"]["f1_macro"], len(candidates))

    # Restore winner state on the model so the runner's downstream eval matches.
    winner_state = final_state if best_source == "final" else best_excess_state
    model.load_state_dict(winner_state)
    model.to(device)

    results_comparison = {
        label: {
            "f1_macro": float(res["metrics"]["f1_macro"]),
            "accuracy": float(res["metrics"]["accuracy"]),
            "adjusted": int(res["adj"]),
            "lp_fallback_used": res["posthoc_meta"].get("lp_fallback_used", False),
        }
        for label, res in candidates
    }

    return TrainOutputs(
        model=model,
        summary={
            "satisfaction_epoch": satisfaction_epoch,
            "checkpoint_source": best_source,
            "results_comparison": results_comparison,
        },
    )
