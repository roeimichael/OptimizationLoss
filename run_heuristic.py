import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN
from src.training.constraints import compute_global_constraints, compute_local_constraints
from src.utils.data_loader import load_presplit_data
from src.utils.filesystem_manager import load_config_from_path, save_config_to_path, mark_experiment_complete
from src.training.trainer import ConstraintTrainer
from src.training.metrics import compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics


def apply_allocation_heuristic(probs: np.ndarray, groups: np.ndarray, hierarchy: List[int],
                               global_constraints: List[float], local_constraints: Dict[int, List[float]]) -> Tuple[
    np.ndarray, float]:
    start_time = time.time()
    n_samples, n_classes = probs.shape
    y_pred = np.full(n_samples, -1, dtype=int)
    assigned_mask = np.zeros(n_samples, dtype=bool)
    current_global = {c: 0 for c in range(n_classes)}
    current_local = {}

    for class_idx in hierarchy:
        g_limit = global_constraints[class_idx]
        unassigned_indices = np.where(~assigned_mask)[0]
        if len(unassigned_indices) == 0:
            break

        class_probs = probs[unassigned_indices, class_idx]
        sorted_absolute_indices = unassigned_indices[np.argsort(class_probs)[::-1]]

        for idx in sorted_absolute_indices:
            group_id = groups[idx]
            if group_id not in current_local:
                current_local[group_id] = {c: 0 for c in range(n_classes)}

            if g_limit < 1e8 and current_global[class_idx] >= g_limit:
                continue

            l_limit = local_constraints.get(group_id, [1e9] * 3)[class_idx]
            if l_limit is None or np.isnan(l_limit):
                l_limit = 1e9

            if l_limit < 1e8 and current_local[group_id][class_idx] >= l_limit:
                continue

            y_pred[idx] = class_idx
            assigned_mask[idx] = True
            current_global[class_idx] += 1
            current_local[group_id][class_idx] += 1

    remaining_indices = np.where(~assigned_mask)[0]
    if len(remaining_indices) > 0:
        for idx in remaining_indices:
            y_pred[idx] = np.argmax(probs[idx])

    return y_pred, time.time() - start_time


def run_heuristic(config_path: str) -> None:
    print(f"Heuristic Config: {config_path}")
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df, test_df = load_presplit_data(TRAIN_PATH, TEST_PATH)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    local_percent, global_percent = config['constraint']
    groups = full_df['Course'].unique()
    global_constraint = compute_global_constraints(full_df, TARGET_COLUMN, global_percent)
    local_constraint = compute_local_constraints(full_df, TARGET_COLUMN, local_percent, groups)

    drop_cols = [TARGET_COLUMN, 'Course']
    X_train_clean = train_df.drop(columns=drop_cols)
    X_test_clean = test_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COLUMN]
    groups_test = test_df['Course']

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

    # Define Hierarchy: Graduate(2) -> Dropout(0) -> Enrolled(1)
    class_hierarchy = [2, 0, 1]

    y_pred, exec_time = apply_allocation_heuristic(
        probs, groups_test.values, class_hierarchy, global_constraint, local_constraint
    )

    y_true_np = y_test.values if hasattr(y_test, 'values') else y_test
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

    save_evaluation_metrics(Path(experiment_path) / 'evaluation_metrics.csv', config['results'])
    save_config_to_path(config, experiment_path)
    mark_experiment_complete(experiment_path)

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