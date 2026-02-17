"""Heuristic baseline: Apply allocation heuristic on top of warmup model predictions.

This script uses the cached warmup model (trained with CE loss only) and applies
a post-hoc allocation heuristic to satisfy constraints. This serves as a baseline
comparison against the end-to-end constraint optimization approach.

Labels are 0-indexed (0-4) throughout.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import load_experiment_data
from src.utils.filesystem_manager import load_config_from_path, save_config_to_path, update_experiment_status
from src.training.trainer import ConstraintTrainer
from src.training.metrics import compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics

NUM_CLASSES = 2  # Binary: 0 = no churn, 1 = churn


def apply_allocation_heuristic(probs: np.ndarray, groups: np.ndarray, hierarchy: List[int],
                               global_constraints: List[float], local_constraints: Dict[int, List[float]]) -> Tuple[
    np.ndarray, float]:
    """Apply greedy allocation heuristic to satisfy constraints.

    Strategy:
    1. First, identify which class each sample "prefers" (argmax)
    2. For constrained classes: assign top-K samples (by probability) that prefer that class
    3. For unconstrained classes: assign all samples that prefer that class
    4. This respects both preferences and constraints

    Args:
        probs: Model output probabilities (n_samples, n_classes)
        groups: Group IDs for each sample
        hierarchy: Order to process classes (constrained class should be first)
        global_constraints: Per-class global limits
        local_constraints: Per-group per-class limits

    Returns:
        Tuple of (predictions, execution_time)
    """
    start_time = time.time()
    n_samples, n_classes = probs.shape
    y_pred = np.full(n_samples, -1, dtype=int)
    assigned_mask = np.zeros(n_samples, dtype=bool)
    current_global = {c: 0 for c in range(n_classes)}
    current_local = {}

    # Get argmax prediction for each sample
    argmax_preds = np.argmax(probs, axis=1)

    for class_idx in hierarchy:
        g_limit = global_constraints[class_idx]
        is_constrained = g_limit < 1e8

        # Find unassigned samples that prefer this class (or all if constrained)
        unassigned_indices = np.where(~assigned_mask)[0]
        if len(unassigned_indices) == 0:
            break

        if is_constrained:
            # For constrained class: consider all unassigned, sorted by prob for this class
            class_probs = probs[unassigned_indices, class_idx]
            sorted_absolute_indices = unassigned_indices[np.argsort(class_probs)[::-1]]
        else:
            # For unconstrained class: only consider samples that prefer this class
            prefer_this_class = argmax_preds[unassigned_indices] == class_idx
            candidate_indices = unassigned_indices[prefer_this_class]
            if len(candidate_indices) == 0:
                continue
            # Sort by probability (highest first)
            class_probs = probs[candidate_indices, class_idx]
            sorted_absolute_indices = candidate_indices[np.argsort(class_probs)[::-1]]

        for idx in sorted_absolute_indices:
            group_id = groups[idx]
            if group_id not in current_local:
                current_local[group_id] = {c: 0 for c in range(n_classes)}

            # Check global constraint
            if is_constrained and current_global[class_idx] >= g_limit:
                break  # Stop processing this class

            # Check local constraint
            l_limit = local_constraints.get(group_id, [1e9] * NUM_CLASSES)[class_idx]
            if l_limit is None or np.isnan(l_limit):
                l_limit = 1e9

            if l_limit < 1e8 and current_local[group_id][class_idx] >= l_limit:
                continue

            y_pred[idx] = class_idx
            assigned_mask[idx] = True
            current_global[class_idx] += 1
            current_local[group_id][class_idx] += 1

    # Assign remaining samples by argmax (these are samples whose preferred class was full)
    remaining_indices = np.where(~assigned_mask)[0]
    if len(remaining_indices) > 0:
        for idx in remaining_indices:
            # Find best unconstrained class
            sample_probs = probs[idx].copy()
            for c in range(n_classes):
                if global_constraints[c] < 1e8:  # Skip constrained classes
                    sample_probs[c] = -1
            y_pred[idx] = np.argmax(sample_probs)

    return y_pred, time.time() - start_time


def run_heuristic(config_path: str) -> None:
    print(f"Heuristic Config: {config_path}")
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_clean, X_test_clean, _, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(config)

    scaler = StandardScaler()
    scaler.fit(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    trainer = ConstraintTrainer(config, str(experiment_path), device)
    # This loads the EXISTING warmup model from cache
    trainer.setup_model(X_train_clean.shape[1], config['base_model_id'])

    if trainer.model is None:
        print("[ERROR] Cached model not found. Run optimization experiment first.")
        return

    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(X_test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Define Hierarchy: Process constrained class (1 = churn) FIRST to reserve slots
    # Then fill remaining samples with unconstrained class (0 = no churn)
    # Binary: constrained class first, then unconstrained
    class_hierarchy = [1, 0]

    y_pred, exec_time = apply_allocation_heuristic(
        probs, groups_test.values, class_hierarchy, global_constraint, local_constraint
    )

    y_true_np = y_test.values if hasattr(y_test, 'values') else y_test

    # Constraint verification
    print("\n[CONSTRAINT VERIFICATION - HEURISTIC]")
    for c in range(NUM_CLASSES):
        pred_count = (y_pred == c).sum()
        true_count = (y_true_np == c).sum()
        limit = int(global_constraint[c]) if global_constraint[c] < 1e9 else 'INF'
        if isinstance(limit, int):
            status = 'OK' if pred_count <= limit else f'VIOLATED by {pred_count - limit}'
        else:
            status = 'OK (unlimited)'
        print(f"  Class {c}: predicted={pred_count:>5}, actual={true_count:>5}, limit={str(limit):>5}, {status}")

    metrics = compute_metrics(y_true_np, y_pred)

    save_final_predictions(
        Path(experiment_path) / 'final_predictions.csv',
        y_true_np, y_pred, probs, groups_test.values
    )

    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(exec_time),
        'methodology': 'heuristic'
    }

    # Save full metrics to CSV (needs all per-class and weighted metrics)
    save_evaluation_metrics(Path(experiment_path) / 'evaluation_metrics.csv', metrics)
    config['status'] = 'completed'
    save_config_to_path(config, experiment_path)

    print(f"Heuristic | Acc: {metrics['accuracy']:.4f} | Time: {exec_time:.2f}s | Saved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    try:
        run_heuristic(args.config_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()