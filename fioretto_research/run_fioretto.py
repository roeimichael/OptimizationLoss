# Fioretto LDF benchmark: Lagrangian dual with linear penalty + subgradient ascent.
# Trains CE-only warmup (shared cache), then constraint phase with Fioretto's algorithm.
# Post-hoc adjustment + evaluation identical to our_approach for fair comparison.
#
# Key differences vs our_approach:
#   Penalty:   lambda * max(0, soft_count - K)           (linear, unbounded)
#   Update:    lambda += step_size * violation             (subgradient ascent)
#   Lambdas:   per-constraint (one per constrained class + one per group×class)
#   No:        rho schedule, rational saturation, lambda toggle, KL regularization

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data_loader import load_experiment_data
from src.utils.filesystem_manager import load_config_from_path, save_config_to_path
from src.models import get_model
from src.training.metrics import (
    compute_metrics, compute_train_accuracy,
    get_predictions_with_probabilities,
    compute_flips, compute_raw_constraint_satisfaction,
)
from src.training.logging import save_final_predictions, save_evaluation_metrics
from src.training.model_cache import load_from_cache, save_to_cache
from src.utils.posthoc_adjustment import targeted_correction
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: warmup (identical to run_heuristic.py — shares base_model_id cache)
# ---------------------------------------------------------------------------

def _train_warmup(config, input_dim, num_classes, X_train, y_train, device):
    warmup_epochs = config['hyperparams'].get('warmup_epochs', 50)
    cache_id = config['base_model_id']
    model = load_from_cache(cache_id, config, input_dim, num_classes, device)
    if model is not None:
        log.info("Loaded cached warmup model: %s", cache_id)
        return model

    hp = config['hyperparams']
    lr = hp.get('lr', 0.0001)
    use_amp = device.type == 'cuda'
    gpu_arch = torch.cuda.get_device_capability(0)[0] if use_amp else 0
    use_bf16 = gpu_arch >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda') if (use_amp and not use_bf16) else None

    model = get_model(
        config['model_name'], input_dim=input_dim, n_classes=num_classes,
        hidden_dims=hp.get('hidden_dims'), dropout=hp['dropout'],
        pretrained=hp.get('pretrained', False),
    ).to(device)

    use_fused = device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=use_fused)
    except Exception:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    num_workers = 2 if os.name != 'nt' else 0
    loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=hp['batch_size'],
        shuffle=True, pin_memory=True, num_workers=num_workers,
    )

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False

    for epoch in range(warmup_epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = criterion(model(batch_X), batch_y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        if epoch < 3 or (epoch + 1) % 10 == 0 or epoch == warmup_epochs - 1:
            log.info("Fioretto warmup %d/%d", epoch + 1, warmup_epochs)

    save_to_cache(model, cache_id, config)
    return model


# ---------------------------------------------------------------------------
# Phase 2: Fioretto LDF constraint training
# ---------------------------------------------------------------------------

def _train_fioretto_constraints(
    model: nn.Module,
    config: dict,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    groups_test: np.ndarray,
    global_con: list,
    local_con: Dict[int, list],
    constrained_classes: List[int],
    num_classes: int,
    device: torch.device,
    experiment_path: Path,
) -> Tuple[nn.Module, Optional[int]]:
    """Fioretto Algorithm 1/2: linear penalty + per-constraint subgradient dual ascent."""

    hp = config['hyperparams']
    constraint_epochs = hp.get('constraint_epochs', 300)
    lr_c = hp.get('lr_constraint', 1e-5)
    step_size = hp.get('fioretto_step_size', 0.01)
    batch_size = hp.get('batch_size', 64)
    chunk_size = hp.get('constraint_chunk_size', 256)

    # AMP setup
    use_amp = device.type == 'cuda'
    gpu_arch = torch.cuda.get_device_capability(0)[0] if use_amp else 0
    use_bf16 = gpu_arch >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler('cuda') if (use_amp and not use_bf16) else None

    # --- Per-constraint Lagrange multipliers (Fioretto's design) ---
    lambda_g = {c: 0.0 for c in constrained_classes if global_con[c] < UNLIMITED}
    lambda_l = {}
    for group_id, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                lambda_l[(group_id, c)] = 0.0

    log.info(
        "Fioretto LDF: %d epochs, lr=%.2e, step_size=%.4f, "
        "%d global + %d local multipliers",
        constraint_epochs, lr_c, step_size, len(lambda_g), len(lambda_l),
    )

    # Optimizer — fresh, same as our_approach constraint phase
    use_fused = device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_c, fused=use_fused)
    except Exception:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_c)

    criterion_ce = nn.CrossEntropyLoss()
    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=num_workers,
    )

    X_test_dev = X_test.to(device)
    groups_np = groups_test
    unique_groups = np.unique(groups_np)
    # Precompute group masks (on CPU, index into later)
    group_indices = {
        g: np.where(groups_np == g)[0] for g in unique_groups
    }

    satisfaction_epoch = None
    best_model_state = None
    best_excess = float('inf')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False

    # Training log CSV
    log_path = experiment_path / 'training_log.csv'
    log_fields = ['epoch', 'ce_loss', 'constraint_loss', 'total_excess',
                  'all_satisfied', 'max_lambda_g']
    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, log_fields).writeheader()

    for epoch in range(constraint_epochs):
        epoch_start = time.time()

        # ---- Step 1: CE training on TRAIN data (batched) ----
        model.train()
        ce_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                ce_loss = criterion_ce(model(batch_X), batch_y)
            ce_losses.append(ce_loss.item())
            if scaler:
                scaler.scale(ce_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                ce_loss.backward()
                optimizer.step()

        # ---- Step 2: Constraint gradient on TEST data (transductive) ----
        # Two-pass approach for memory efficiency:
        #   Pass A (no_grad): compute total soft counts + hard counts
        #   Pass B (grad):    accumulate constraint gradients per chunk
        model.train()

        # Pass A: no-grad counts
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

        # Determine violations from Pass A (for dual update + gradient gating)
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

        # Pass B: gradient accumulation per chunk
        # d/d(theta) [lambda * ReLU(total_soft - K)] = lambda * I[violated] * d(total_soft)/d(theta)
        # Since total_soft = sum_chunks chunk_proba.sum(), gradients decompose per chunk.
        # Gate on any positive lambda for violated constraints (lambda=0 at epoch 0 → skip)
        has_work = (
            any(lambda_g.get(c, 0) > 0 for c in violated_global) or
            any(lambda_l.get(k, 0) > 0 for k in violated_local)
        )
        constraint_loss_val = 0.0
        did_backward = False
        if has_work:
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(X_test_dev), chunk_size):
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    chunk_logits = model(X_test_dev[i:i + chunk_size])
                    chunk_proba = F.softmax(chunk_logits, dim=1)

                    chunk_loss = torch.zeros(1, device=device)
                    # Global: accumulate gradient for violated classes
                    for c in violated_global:
                        if lambda_g[c] > 0:
                            chunk_loss = chunk_loss + lambda_g[c] * chunk_proba[:, c].sum()
                    # Local: accumulate gradient for violated group×class
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

            # Single optimizer step after all chunks (only if backward ran)
            if did_backward:
                if scaler:
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except (AssertionError, RuntimeError):
                        optimizer.step()
                else:
                    optimizer.step()

        # ---- Step 3: Subgradient dual update (Fioretto Eq. 5) ----
        # lambda_c += step_size * violation_c  (monotonic ascent)
        for c, viol in violations_g.items():
            lambda_g[c] += step_size * viol

        for key, viol in violations_l.items():
            lambda_l[key] += step_size * viol

        # ---- Tracking ----
        hard_counts = {
            c: int((hard_preds == c).sum())
            for c in constrained_classes
        }
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

        # Log to CSV
        row = {
            'epoch': epoch,
            'ce_loss': round(np.mean(ce_losses), 6),
            'constraint_loss': round(constraint_loss_val, 6),
            'total_excess': total_excess,
            'all_satisfied': int(all_satisfied),
            'max_lambda_g': round(max(lambda_g.values()) if lambda_g else 0, 6),
        }
        with open(log_path, 'a', newline='') as f:
            csv.DictWriter(f, log_fields).writerow(row)

        if epoch < 5 or (epoch + 1) % 25 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join(f"c{c}={lambda_g[c]:.3f}" for c in sorted(lambda_g))
            log.info(
                "Fioretto %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s lam=[%s] [%.1fs]",
                epoch + 1, constraint_epochs, np.mean(ce_losses),
                constraint_loss_val, total_excess, all_satisfied,
                lam_str, time.time() - epoch_start,
            )

    # Restore best-excess model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        log.info("Restored best model (excess=%d)", best_excess)

    return model, satisfaction_epoch


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _to_numpy(arr):
    return arr.values if hasattr(arr, 'values') else arr


def run_fioretto(config_path: str) -> None:
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)

    seed = config.get('hyperparams', {}).get('seed', None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(
        "Fioretto LDF: %s on %s (model=%s, seed=%s)",
        config_path, device, config['model_name'], seed,
    )
    if torch.cuda.is_available():
        log.info(
            "GPU: %s | CUDA: %s | BF16: %s",
            torch.cuda.get_device_name(0), torch.version.cuda,
            torch.cuda.is_bf16_supported(),
        )

    # ---- Load data ----
    t0 = time.time()
    data = load_experiment_data(config)
    X_train, X_test, y_train, y_test, groups_test, global_con, local_con, num_classes = data
    log.info("TIMING data_load=%.2fs train=%s test=%s",
             time.time() - t0, X_train.shape, X_test.shape)

    ds = config.get('dataset_config', {})
    constrained_class = ds.get('constrained_class', 4)
    if isinstance(constrained_class, int):
        constrained_classes = [constrained_class]
    elif isinstance(constrained_class, list):
        constrained_classes = constrained_class
    else:
        constrained_classes = []

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(_to_numpy(y_train))
    X_test_tensor = torch.FloatTensor(X_test)

    input_dim = None

    # ---- Phase 1: warmup (shared cache) ----
    start_time = time.time()
    model = _train_warmup(config, input_dim, num_classes,
                          X_train_tensor, y_train_tensor, device)

    # ---- Phase 2: Fioretto constraint training ----
    model, satisfaction_epoch = _train_fioretto_constraints(
        model, config, X_train_tensor, y_train_tensor,
        X_test_tensor, groups_test, global_con, local_con,
        constrained_classes, num_classes, device, experiment_path,
    )
    training_time = time.time() - start_time

    # ---- Post-hoc adjustment + evaluation (identical to our_approach) ----
    y_true = _to_numpy(y_test)
    group_ids = _to_numpy(groups_test)
    X_test_dev = X_test_tensor.to(device)

    model.eval()
    y_pred, y_proba = get_predictions_with_probabilities(model, X_test_dev)

    needs_adjustment = any(global_con[c] < UNLIMITED for c in constrained_classes)
    adj = 0
    posthoc_meta = {}
    if needs_adjustment:
        y_pred, adj, posthoc_meta = targeted_correction(
            y_proba, group_ids, global_con, local_con, constrained_classes)

    # Save raw (pre-post-hoc) predictions
    raw_pred = y_proba.argmax(axis=1)
    save_final_predictions(
        experiment_path / 'final_predictions_raw.csv',
        y_true, raw_pred, y_proba, group_ids,
    )
    # Save final predictions
    save_final_predictions(
        experiment_path / 'final_predictions.csv',
        y_true, y_pred, y_proba, group_ids,
    )

    # Constraint verification
    for c in range(num_classes):
        pred_count = (y_pred == c).sum()
        limit = int(global_con[c]) if global_con[c] < UNLIMITED else 'INF'
        status = ('OK' if (isinstance(limit, str) or pred_count <= limit)
                  else f'VIOLATED by {pred_count - limit}')
        log.info("Class %d: pred=%d limit=%s %s", c, pred_count, limit, status)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    flips = compute_flips(raw_pred, y_pred)
    raw_sat = compute_raw_constraint_satisfaction(
        raw_pred, global_con, local_con, group_ids, constrained_classes)
    metrics['flips_required'] = flips
    metrics.update(raw_sat)
    metrics['satisfaction_epoch'] = satisfaction_epoch
    log.info(
        "[Track1] flips=%d raw_satisfied=%s excess=%d sat_epoch=%s",
        flips, raw_sat['raw_all_satisfied'], raw_sat['raw_total_excess'],
        satisfaction_epoch or 'N/A',
    )
    save_evaluation_metrics(experiment_path / 'evaluation_metrics.csv', metrics)

    # Save results to config
    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(training_time),
        'samples_adjusted': int(flips),
        'lp_fallback_used': posthoc_meta.get('lp_fallback_used', False),
    }
    config['status'] = 'completed'
    save_config_to_path(config, experiment_path)
    log.info("Fioretto LDF done: acc=%.4f flips=%d time=%.2fs",
             metrics['accuracy'], flips, training_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    try:
        run_fioretto(args.config_path)
    except Exception as e:
        log.error("Fioretto LDF failed: %s", e, exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
