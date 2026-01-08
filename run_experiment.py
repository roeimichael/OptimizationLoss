import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN
from src.training.constraints import compute_global_constraints, compute_local_constraints
from src.utils.data_loader import load_presplit_data

from src.training.trainer import ConstraintTrainer

from src.training.metrics import get_predictions_with_probabilities, compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics
from src.utils.filesystem_manager import load_config_from_path, save_config_to_path, mark_experiment_complete, \
    is_experiment_complete, update_experiment_status


def load_experiment_data(config: Dict[str, Any]):
    print("\nLoading dataset...")
    train_df, test_df = load_presplit_data(TRAIN_PATH, TEST_PATH)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    local_percent, global_percent = config['constraint']
    groups = full_df['Course'].unique()
    global_constraint = compute_global_constraints(full_df, TARGET_COLUMN, global_percent)
    local_constraint = compute_local_constraints(full_df, TARGET_COLUMN, local_percent, groups)
    print(f"Global constraint: {global_constraint}")
    print(f"Local constraints: {len(local_constraint)} courses")
    drop_cols = [TARGET_COLUMN, 'Course']
    y_train = train_df[TARGET_COLUMN]
    X_train_clean = train_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COLUMN]
    groups_test = test_df['Course']
    X_test_clean = test_df.drop(columns=drop_cols)
    return X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint


def run_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    print(f"Config: {config_path}")
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    if is_experiment_complete(experiment_path):
        print(f"\n[SKIP] Already completed")
        return None
    update_experiment_status(experiment_path, 'running')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")
    X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(
        config)
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    if y_train.dtype == 'O' or isinstance(y_train.iloc[0], str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
    else:
        y_train_encoded = y_train.values
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    input_dim = X_train_clean.shape[1]
    trainer = ConstraintTrainer(config, str(experiment_path), device)
    trainer.setup_model(input_dim=input_dim, base_model_id=config['base_model_id'])
    start_time = time.time()
    trainer.train_warmup(X_train_tensor, y_train_tensor, config['base_model_id'])
    model = trainer.train_constraints(
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        X_test=X_test_tensor,
        groups_test=groups_test,
        global_con=global_constraint,
        local_con=local_constraint
    )

    training_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    model.eval()

    y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
    y_true_np = y_test.values if hasattr(y_test, 'values') else y_test
    course_ids_np = groups_test.values if hasattr(groups_test, 'values') else groups_test

    save_final_predictions(Path(experiment_path) / 'final_predictions.csv', y_true_np, y_pred, y_proba, course_ids_np)
    metrics = compute_metrics(y_true_np, y_pred)
    save_evaluation_metrics(Path(experiment_path) / 'evaluation_metrics.csv', metrics)

    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(training_time),
        'used_cached_model': trainer.from_cache  # Access property from trainer instance
    }

    save_config_to_path(config, experiment_path)
    mark_experiment_complete(experiment_path)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Results saved: {experiment_path}")

    return config['results']


def main() -> None:
    parser = argparse.ArgumentParser(description='Run single experiment')
    parser.add_argument('config_path', type=str, help='Path to config.json')
    args = parser.parse_args()
    experiment_path = Path(args.config_path).parent
    try:
        results = run_experiment(args.config_path)
        if results:
            print("\n" + "=" * 80)
            print("COMPLETED")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("SKIPPED")
            print("=" * 80)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        update_experiment_status(str(experiment_path), 'pending')
        print("\n[STATUS] Reset to 'pending' for retry")
        exit(1)


if __name__ == "__main__":
    main()