# Heuristic baseline: greedy allocation on a fixed warmup model.
# Trains CE-only model, then assigns predictions via top-K constrained allocation.
# Processes constrained classes first to ensure optimal budget utilization.

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data_loader import load_experiment_data
from src.utils.filesystem_manager import load_config_from_path, save_config_to_path
from src.models import get_model
from src.training.metrics import compute_metrics, compute_train_accuracy
from src.training.logging import save_final_predictions, save_evaluation_metrics
from src.training.model_cache import load_from_cache, save_to_cache
from src.utils.posthoc_adjustment import lp_constrained_assignment

log = logging.getLogger(__name__)

UNLIMITED = 1e10


def train_fixed_warmup(config, input_dim, num_classes, X_train, y_train, device):
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
    log.info("Training heuristic warmup: %d epochs lr=%.2e (AMP=%s dtype=%s)",
             warmup_epochs, lr, use_amp, amp_dtype)
    model = get_model(
        config['model_name'], input_dim=input_dim, n_classes=num_classes,
        hidden_dims=hp.get('hidden_dims'), dropout=hp['dropout'],
        pretrained=hp.get('pretrained', False)
    ).to(device)
    use_fused = device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=use_fused)
    except Exception:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=hp['batch_size'],
                        shuffle=True, pin_memory=True, num_workers=2 if os.name != 'nt' else 0)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False  # Disabled: Blackwell sm_120 VBIOS temp bug
    for epoch in range(warmup_epochs):
        epoch_start = time.time()
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
            log.info("Heuristic warmup %d/%d [%.2fs/epoch]",
                     epoch + 1, warmup_epochs, time.time() - epoch_start)
    train_acc = compute_train_accuracy(model, loader, device)
    log.info("Heuristic warmup done: %d epochs, train_acc=%.4f", warmup_epochs, train_acc)
    save_to_cache(model, cache_id, config)
    return model


def _build_hierarchy(num_classes, global_constraints, constrained_classes):
    constrained_sorted = sorted(constrained_classes,
                                key=lambda c: global_constraints[c])
    unconstrained = [c for c in range(num_classes) if c not in constrained_classes]
    return constrained_sorted + unconstrained


def apply_allocation_heuristic(probs: np.ndarray, groups: np.ndarray, hierarchy: List[int],
                               global_constraints: List[float], local_constraints: Dict[int, List[float]],
                               num_classes: int = 7) -> Tuple[np.ndarray, float]:
    start_time = time.time()
    n_samples, n_classes = probs.shape
    y_pred = np.full(n_samples, -1, dtype=int)
    assigned_mask = np.zeros(n_samples, dtype=bool)
    current_global = {c: 0 for c in range(n_classes)}
    current_local = {}
    argmax_preds = np.argmax(probs, axis=1)
    for class_idx in hierarchy:
        g_limit = global_constraints[class_idx]
        is_constrained = g_limit < UNLIMITED
        unassigned = np.where(~assigned_mask)[0]
        if len(unassigned) == 0:
            break
        if is_constrained:
            class_probs = probs[unassigned, class_idx]
            sorted_indices = unassigned[np.argsort(class_probs)[::-1]]
        else:
            prefer = argmax_preds[unassigned] == class_idx
            candidates = unassigned[prefer]
            if len(candidates) == 0:
                continue
            class_probs = probs[candidates, class_idx]
            sorted_indices = candidates[np.argsort(class_probs)[::-1]]
        for idx in sorted_indices:
            group_id = groups[idx]
            if group_id not in current_local:
                current_local[group_id] = {c: 0 for c in range(n_classes)}
            if is_constrained and current_global[class_idx] >= g_limit:
                break
            l_limit = local_constraints.get(group_id, [UNLIMITED] * num_classes)[class_idx]
            if l_limit is None or np.isnan(l_limit):
                l_limit = UNLIMITED
            if l_limit < UNLIMITED and current_local[group_id][class_idx] >= l_limit:
                continue
            y_pred[idx] = class_idx
            assigned_mask[idx] = True
            current_global[class_idx] += 1
            current_local[group_id][class_idx] += 1
    remaining = np.where(~assigned_mask)[0]
    for idx in remaining:
        sample_probs = probs[idx].copy()
        group_id = groups[idx]
        # Ensure group exists in current_local before checking — otherwise
        # first-time groups skipped local limit check and could overflow.
        if group_id not in current_local:
            current_local[group_id] = {c: 0 for c in range(n_classes)}
        for c in range(n_classes):
            if global_constraints[c] < UNLIMITED and current_global[c] >= global_constraints[c]:
                sample_probs[c] = -1
            if global_constraints[c] < UNLIMITED:
                l_limit = local_constraints.get(group_id, [UNLIMITED] * n_classes)[c]
                if l_limit < UNLIMITED and current_local[group_id].get(c, 0) >= l_limit:
                    sample_probs[c] = -1
        best = np.argmax(sample_probs)
        y_pred[idx] = best
        current_global[best] = current_global.get(best, 0) + 1
        current_local[group_id][best] = current_local[group_id].get(best, 0) + 1
    return y_pred, time.time() - start_time


def _to_numpy(arr):
    return arr.values if hasattr(arr, 'values') else arr


def run_heuristic(config_path: str) -> None:
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    # Seed control — matches run_experiment.py so both paths produce
    # deterministic warmup models when a seed is specified.
    seed = config.get('hyperparams', {}).get('seed', None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Running heuristic %s on %s (model=%s, seed=%s)", config_path, device, config['model_name'], seed)
    if torch.cuda.is_available():
        log.info("GPU: %s | CUDA: %s | BF16: %s",
                 torch.cuda.get_device_name(0), torch.version.cuda,
                 torch.cuda.is_bf16_supported())
    t0 = time.time()
    data = load_experiment_data(config)
    X_train, X_test, y_train, y_test, groups_test, global_con, local_con, num_classes = data
    log.info("TIMING data_load=%.2fs train=%s test=%s", time.time() - t0,
             X_train.shape, X_test.shape)
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
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    input_dim = None
    model = train_fixed_warmup(config, input_dim, num_classes,
                               X_train_tensor, y_train_tensor, device)
    model.eval()
    with torch.no_grad():
        chunk_size = 256
        logit_chunks = [model(X_test_tensor[i:i + chunk_size])
                        for i in range(0, len(X_test_tensor), chunk_size)]
        probs = torch.softmax(torch.cat(logit_chunks, dim=0), dim=1).cpu().numpy()
    groups_np = _to_numpy(groups_test)
    y_true = _to_numpy(y_test)
    methodology = config.get('methodology', 'heuristic')
    if methodology == 'po_lp':
        t_alloc = time.time()
        y_pred, adj_count = lp_constrained_assignment(
            probs, groups_np, global_con, local_con, constrained_classes)
        exec_time = time.time() - t_alloc
        log.info("PO-LP: %d predictions changed in %.3fs", adj_count, exec_time)
    elif methodology == 'danits_lp':
        # Paper [5] (Shifman et al. 2025) LP post-hoc with arbitrary cost matrix.
        # This branches from the SAME cached warmup as 'heuristic' and 'po_lp'
        # (via base_model_id), so all four methodologies can be compared
        # head-to-head on identical Phase-1 weights.
        from danits_research import (
            DERMMNIST_PRESETS,
            build_priority_cost_matrix,
            build_psi_phi_from_percentages,
            solve_lp_assignment,
        )
        # Cost matrix preset name (see danits_research.cost_matrices).
        # Default 'identity' == minimise expected error rate subject to Psi/Phi.
        # The identity matrix is built dynamically from num_classes so it works
        # for ANY dataset (DermMNIST=7, TissueMNIST=8, etc.) without hardcoding.
        cost_preset = config.get('hyperparams', {}).get('danits_cost_preset', 'identity')
        if cost_preset == 'identity':
            omega = np.ones((num_classes, num_classes), dtype=np.float64) - np.eye(num_classes, dtype=np.float64)
        elif cost_preset in DERMMNIST_PRESETS:
            omega = DERMMNIST_PRESETS[cost_preset]
        else:
            raise ValueError(
                f"Unknown danits_cost_preset {cost_preset!r}. "
                f"Available: ['identity'] + {list(DERMMNIST_PRESETS.keys())}"
            )
        # Convert the project's (global_con, local_con) format — which uses
        # UNLIMITED=1e10 sentinel — into paper-[5] Psi/Phi (None = unbounded).
        psi_list = [int(v) if v < UNLIMITED else None for v in global_con]
        phi_dict: dict = {}
        if local_con:
            for g, bounds in local_con.items():
                phi_dict[g] = [int(v) if v < UNLIMITED else None for v in bounds]
        t_alloc = time.time()
        lp_res = solve_lp_assignment(
            y_proba=probs, groups=groups_np, cost_matrix=omega,
            psi=psi_list, phi=phi_dict, verbose=False,
        )
        exec_time = time.time() - t_alloc
        if lp_res.status != "OPTIMAL":
            raise RuntimeError(
                f"danits_lp: LP solver returned status={lp_res.status}"
            )
        y_pred = lp_res.y_pred
        log.info(
            "DANITS-LP [%s]: obj=%.4f status=%s runtime=%.3fs vars=%d constraints=%d",
            cost_preset, lp_res.objective_value, lp_res.status,
            exec_time, lp_res.num_variables, lp_res.num_constraints,
        )
    else:
        hierarchy = _build_hierarchy(num_classes, global_con, constrained_classes)
        y_pred, exec_time = apply_allocation_heuristic(
            probs, groups_np, hierarchy, global_con, local_con, num_classes)
    # Save raw argmax predictions (before heuristic/LP reallocation)
    argmax_preds = np.argmax(probs, axis=1)
    save_final_predictions(Path(experiment_path) / 'final_predictions_raw.csv',
                           y_true, argmax_preds, probs, groups_np)
    for c in range(num_classes):
        pred_count = (y_pred == c).sum()
        limit = int(global_con[c]) if global_con[c] < 1e9 else 'INF'
        status = 'OK' if (isinstance(limit, str) or pred_count <= limit) else f'VIOLATED by {pred_count - limit}'
        log.info("Heuristic class %d: pred=%d limit=%s %s", c, pred_count, limit, status)
    metrics = compute_metrics(y_true, y_pred, probs)
    save_final_predictions(Path(experiment_path) / 'final_predictions.csv',
                           y_true, y_pred, probs, groups_np)
    # Track 1: constraint-specific metrics
    from src.training.metrics import compute_flips, compute_raw_constraint_satisfaction
    flips = compute_flips(argmax_preds, y_pred)
    raw_sat = compute_raw_constraint_satisfaction(
        argmax_preds, global_con, local_con, groups_np, constrained_classes)
    metrics['flips_required'] = flips
    metrics.update(raw_sat)
    log.info("[Track1] flips=%d raw_satisfied=%s excess=%d",
             flips, raw_sat['raw_all_satisfied'], raw_sat['raw_total_excess'])
    save_evaluation_metrics(Path(experiment_path) / 'evaluation_metrics.csv', metrics)
    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(exec_time),
        'samples_adjusted': int(flips),
    }
    config['status'] = 'completed'
    save_config_to_path(config, experiment_path)
    log.info("Heuristic: acc=%.4f time=%.2fs", metrics['accuracy'], exec_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    try:
        run_heuristic(args.config_path)
    except Exception as e:
        log.error("Heuristic failed: %s", e, exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
