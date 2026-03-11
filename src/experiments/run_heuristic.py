# Heuristic baseline: greedy allocation on a fixed warmup model.
# Trains CE-only model, then assigns predictions via top-K constrained allocation.

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

log = logging.getLogger(__name__)

HEURISTIC_WARMUP_EPOCHS = 50
HEURISTIC_LR = 0.0001


def train_fixed_warmup(config, input_dim, num_classes, X_train, y_train, device):
    warmup_epochs = config['hyperparams'].get('warmup_epochs', HEURISTIC_WARMUP_EPOCHS)
    cache_id = config['base_model_id']
    model = load_from_cache(cache_id, config, input_dim, num_classes, device)
    if model is not None:
        log.info("Loaded cached warmup model: %s", cache_id)
        return model
    log.info("Training heuristic warmup: %d epochs", warmup_epochs)
    hp = config['hyperparams']
    model = get_model(
        config['model_name'], input_dim=input_dim, n_classes=num_classes,
        hidden_dims=hp.get('hidden_dims'), dropout=hp['dropout'],
        pretrained=hp.get('pretrained', False)
    ).to(device)
    use_fused = device.type == 'cuda' and hasattr(torch.optim.Adam, 'fused')
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=HEURISTIC_LR, fused=use_fused)
    except Exception:
        optimizer = torch.optim.Adam(model.parameters(), lr=HEURISTIC_LR)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=hp['batch_size'],
                        shuffle=True, pin_memory=True, num_workers=2 if os.name != 'nt' else 0)
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda') if (use_amp and amp_dtype == torch.float16) else None
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
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
    train_acc = compute_train_accuracy(model, loader, device)
    log.info("Heuristic warmup done: %d epochs, train_acc=%.4f", warmup_epochs, train_acc)
    save_to_cache(model, cache_id, config)
    return model


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
        is_constrained = g_limit < 1e8
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
            l_limit = local_constraints.get(group_id, [1e9] * num_classes)[class_idx]
            if l_limit is None or np.isnan(l_limit):
                l_limit = 1e9
            if l_limit < 1e8 and current_local[group_id][class_idx] >= l_limit:
                continue
            y_pred[idx] = class_idx
            assigned_mask[idx] = True
            current_global[class_idx] += 1
            current_local[group_id][class_idx] += 1
    remaining = np.where(~assigned_mask)[0]
    for idx in remaining:
        sample_probs = probs[idx].copy()
        for c in range(n_classes):
            if global_constraints[c] < 1e8:
                sample_probs[c] = -1
        y_pred[idx] = np.argmax(sample_probs)
    return y_pred, time.time() - start_time


def _to_numpy(arr):
    return arr.values if hasattr(arr, 'values') else arr


def run_heuristic(config_path: str) -> None:
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_experiment_data(config)
    X_train, X_test, y_train, y_test, groups_test, global_con, local_con, num_classes = data
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
    hierarchy = list(range(num_classes - 1, -1, -1))
    groups_np = _to_numpy(groups_test)
    y_pred, exec_time = apply_allocation_heuristic(
        probs, groups_np, hierarchy, global_con, local_con, num_classes)
    y_true = _to_numpy(y_test)
    for c in range(num_classes):
        pred_count = (y_pred == c).sum()
        limit = int(global_con[c]) if global_con[c] < 1e9 else 'INF'
        status = 'OK' if (isinstance(limit, str) or pred_count <= limit) else f'VIOLATED by {pred_count - limit}'
        log.info("Heuristic class %d: pred=%d limit=%s %s", c, pred_count, limit, status)
    metrics = compute_metrics(y_true, y_pred, probs)
    save_final_predictions(Path(experiment_path) / 'final_predictions.csv',
                           y_true, y_pred, probs, groups_np)
    save_evaluation_metrics(Path(experiment_path) / 'evaluation_metrics.csv', metrics)
    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(exec_time),
        'methodology': 'heuristic'
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
