#!/usr/bin/env python3
"""
Runner script for loss-proportional adaptive lambda experiments.

This script runs a single experiment using the loss-proportional adaptive lambda
methodology where lambda increases proportionally to constraint loss magnitude.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.data_loader import load_experiment_data
from src.models import get_model
from src.training.loss_proportional_trainer import LossProportionalTrainer
from src.training.metrics import get_predictions_with_probabilities, compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics
from src.utils.filesystem_manager import (
    load_config_from_path,
    save_config_to_path,
    mark_experiment_complete,
    update_experiment_status
)


def run_loss_proportional_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Run a single loss-proportional adaptive lambda experiment.

    Args:
        config_path: Path to experiment configuration JSON file

    Returns:
        Dictionary with experiment results, or None if experiment failed/skipped
    """
    print(f"Config: {config_path}")
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)

    # Skip if already completed
    if config.get('status', 'pending') == 'completed':
        print(f"\n[SKIP] Already completed")
        return None

    update_experiment_status(experiment_path, 'running')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")
    print(f"Methodology: loss_proportional")
    print(f"Lambda learning rate: {config.get('lambda_learning_rate', 0.01)}")

    # Load and preprocess data
    X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(config)

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

    input_dim = X_train_clean.shape[1]

    # Create model
    print(f"Creating model: {config['model_name']}...")
    model = get_model(
        model_name=config['model_name'],
        input_dim=input_dim,
        num_classes=config['num_classes'],
        **config.get('model_kwargs', {})
    ).to(device)

    # Create trainer
    trainer = LossProportionalTrainer(
        config=config,
        experiment_path=str(experiment_path),
        device=device
    )

    # Train model
    print("Starting training...")
    start_time = time.time()

    model = trainer.train(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train_encoded,
        X_test=X_test_scaled,
        groups_test=groups_test.values if hasattr(groups_test, 'values') else groups_test,
        global_con=global_constraint,
        local_con=local_constraint
    )

    training_time = time.time() - start_time

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    model.eval()

    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
    y_true_np = y_test.values if hasattr(y_test, 'values') else y_test
    course_ids_np = groups_test.values if hasattr(groups_test, 'values') else groups_test

    # Save predictions and metrics
    save_final_predictions(
        Path(experiment_path) / 'final_predictions.csv',
        y_true_np,
        y_pred,
        y_proba,
        course_ids_np
    )

    metrics = compute_metrics(y_true_np, y_pred)
    save_evaluation_metrics(Path(experiment_path) / 'evaluation_metrics.csv', metrics)

    # Save results to config
    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(training_time)
    }

    save_config_to_path(config, experiment_path)
    mark_experiment_complete(experiment_path)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (Macro): {metrics['f1_macro']:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Results saved: {experiment_path}")

    return config['results']


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_loss_proportional_experiment.py <config_path>")
        sys.exit(1)

    try:
        run_loss_proportional_experiment(sys.argv[1])
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
