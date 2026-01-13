"""Run script for static lambda experiments.

This script runs experiments using the static lambda methodology where:
- Lambda values are constant throughout training
- No warmup phase (constraints applied from epoch 0)
- Fixed training duration (300 epochs)
- Raises error if constraints not satisfied after training

Usage:
    python run_static_lambda_experiment.py <path_to_config.json>

Example:
    python run_static_lambda_experiment.py results/static_lambda/BasicNN/constraint_0.9_0.8/lambda_search/medium/config.json
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import torch
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
    """Run a single static lambda experiment.

    Args:
        config_path: Path to experiment config.json file

    Returns:
        Dictionary with experiment results, or None if failed/skipped

    Raises:
        Exception: For unexpected errors (not constraint failures)
    """
    print(f"Config: {config_path}")
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)

    # Skip if already completed
    if config.get('status', 'pending') == 'completed':
        print(f"\n[SKIP] Already completed")
        return None

    # Skip if constraints were previously not met
    if config.get('status') == 'constraints_not_met':
        print(f"\n[SKIP] Constraints not met in previous run")
        return None

    update_experiment_status(experiment_path, 'running')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")
    print(f"Methodology: static_lambda")
    print(f"Lambda Global: {config['hyperparams']['lambda_global']}")
    print(f"Lambda Local: {config['hyperparams']['lambda_local']}")

    # Load and preprocess data
    X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(
        config)

    print("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    # Encode labels if necessary
    if y_train.dtype == 'O' or isinstance(y_train.iloc[0], str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
    else:
        y_train_encoded = y_train.values

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    input_dim = X_train_clean.shape[1]

    # Initialize trainer
    trainer = StaticLambdaTrainer(config, str(experiment_path), device)
    trainer.setup_model(input_dim=input_dim, base_model_id=config['base_model_id'])

    try:
        # Train with static lambda (no warmup phase)
        start_time = time.time()
        model = trainer.train_static_lambda(
            X_train=X_train_tensor,
            y_train=y_train_tensor,
            X_test=X_test_tensor,
            groups_test=groups_test,
            global_con=global_constraint,
            local_con=local_constraint
        )
        training_time = time.time() - start_time

        # If we get here, constraints were satisfied
        print("\n" + "=" * 80)
        print("EVALUATION")
        print("=" * 80)

        model.eval()
        y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
        y_true_np = y_test.values if hasattr(y_test, 'values') else y_test
        course_ids_np = groups_test.values if hasattr(groups_test, 'values') else groups_test

        # Save predictions and metrics
        save_final_predictions(
            Path(experiment_path) / 'final_predictions.csv',
            y_true_np, y_pred, y_proba, course_ids_np
        )
        metrics = compute_metrics(y_true_np, y_pred)
        save_evaluation_metrics(
            Path(experiment_path) / 'evaluation_metrics.csv',
            metrics
        )

        # Update config with results
        config['results'] = {
            'accuracy': float(metrics['accuracy']),
            'precision_macro': float(metrics['precision_macro']),
            'recall_macro': float(metrics['recall_macro']),
            'f1_macro': float(metrics['f1_macro']),
            'training_time': float(training_time),
            'constraints_satisfied': True,
            'methodology': 'static_lambda'
        }

        save_config_to_path(config, experiment_path)
        mark_experiment_complete(experiment_path)

        print(f"\n✓ Accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ Training Time: {training_time:.2f}s")
        print(f"✓ Constraints Satisfied: YES")
        print(f"✓ Results saved: {experiment_path}")

        return config['results']

    except ConstraintsNotMetError as e:
        # Constraints not satisfied - this is an expected outcome
        training_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("CONSTRAINTS NOT SATISFIED")
        print("=" * 80)
        print(f"✗ Global Satisfied: {e.global_satisfied}")
        print(f"✗ Local Satisfied: {e.local_satisfied}")
        print(f"✗ Final Global Loss: {e.final_global_loss:.6f}")
        print(f"✗ Final Local Loss: {e.final_local_loss:.6f}")
        print(f"Training Time: {training_time:.2f}s")
        print("=" * 80)

        # Save failure information
        config['results'] = {
            'constraints_satisfied': False,
            'global_satisfied': e.global_satisfied,
            'local_satisfied': e.local_satisfied,
            'final_global_loss': e.final_global_loss,
            'final_local_loss': e.final_local_loss,
            'training_time': float(training_time),
            'methodology': 'static_lambda',
            'error_message': str(e)
        }

        save_config_to_path(config, experiment_path)
        update_experiment_status(experiment_path, 'constraints_not_met')

        print(f"✗ Status saved as 'constraints_not_met': {experiment_path}")
        return None


def main() -> None:
    """Main entry point for running static lambda experiments."""
    parser = argparse.ArgumentParser(
        description='Run single static lambda experiment'
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to config.json file'
    )
    args = parser.parse_args()

    experiment_path = Path(args.config_path).parent

    try:
        results = run_static_lambda_experiment(args.config_path)

        if results:
            print("\n" + "=" * 80)
            print("✓ EXPERIMENT COMPLETED SUCCESSFULLY")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("✗ EXPERIMENT FAILED (Constraints Not Met)")
            print("=" * 80)

    except Exception as e:
        # Unexpected error (not constraint failure)
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        update_experiment_status(str(experiment_path), 'pending')
        print("\n[STATUS] Reset to 'pending' for retry")
        exit(1)


if __name__ == "__main__":
    main()
