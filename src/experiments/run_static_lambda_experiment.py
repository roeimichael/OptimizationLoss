#!/usr/bin/env python3

import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.data_loader import load_experiment_data
from src.training.static_lambda_trainer import StaticLambdaTrainer, ConstraintsNotMetError
from src.training.metrics import get_predictions_with_probabilities, compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics
from src.utils.filesystem_manager import (
    load_config_from_path,
    save_config_to_path,
    mark_experiment_complete,
    update_experiment_status
)


def run_static_lambda_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)

    if config.get('status', 'pending') == 'completed':
        return None

    if config.get('status') == 'constraints_not_met':
        return None

    update_experiment_status(experiment_path, 'running')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running: {config['model_name']} | lambda_g={config['hyperparams']['lambda_global']} lambda_l={config['hyperparams']['lambda_local']}")

    X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(config)

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

    trainer = StaticLambdaTrainer(config, str(experiment_path), device)
    trainer.setup_model(input_dim=input_dim, base_model_id=config['base_model_id'])

    try:
        start_time = time.time()
        model = trainer.train_static_lambda(
            X_train=X_train_tensor,
            y_train=y_train_tensor,
            X_test=X_test_tensor,
            groups_test=groups_test.values if hasattr(groups_test, 'values') else groups_test,
            global_con=global_constraint,
            local_con=local_constraint
        )
        training_time = time.time() - start_time

        model.eval()
        y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
        y_true_np = y_test.values if hasattr(y_test, 'values') else y_test
        course_ids_np = groups_test.values if hasattr(groups_test, 'values') else groups_test

        save_final_predictions(
            Path(experiment_path) / 'final_predictions.csv',
            y_true_np,
            y_pred,
            y_proba,
            course_ids_np
        )

        metrics = compute_metrics(y_true_np, y_pred)
        save_evaluation_metrics(Path(experiment_path) / 'evaluation_metrics.csv', metrics)

        config['results'] = {
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'training_time': float(training_time)
        }

        save_config_to_path(config, experiment_path)
        mark_experiment_complete(experiment_path)

        print(f"Completed: Acc={metrics['accuracy']:.4f} F1={metrics['f1_macro']:.4f} Time={training_time:.1f}s")
        return config['results']

    except ConstraintsNotMetError as e:
        config['results'] = {
            'constraints_satisfied': False,
            'global_satisfied': e.global_satisfied,
            'local_satisfied': e.local_satisfied,
            'final_global_loss': e.final_global_loss,
            'final_local_loss': e.final_local_loss
        }
        save_config_to_path(config, experiment_path)
        update_experiment_status(experiment_path, 'constraints_not_met')
        print(f"Failed: Constraints not met (global={e.global_satisfied} local={e.local_satisfied})")
        return None


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_static_lambda_experiment.py <config_path>")
        sys.exit(1)

    try:
        run_static_lambda_experiment(sys.argv[1])
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
