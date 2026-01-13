#!/usr/bin/env python3
"""
Runner script for loss-proportional adaptive lambda experiments.

This script runs a single experiment using the loss-proportional adaptive lambda
methodology where lambda increases proportionally to constraint loss magnitude.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data import load_and_preprocess_data, stratified_split_data
from src.models import get_model
from src.training.loss_proportional_trainer import LossProportionalTrainer
from src.evaluation import evaluate_model, generate_confusion_matrix
from src.utils.experiment_utils import (
    mark_experiment_complete,
    update_experiment_status,
    is_experiment_complete
)


def run_loss_proportional_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Run a single loss-proportional adaptive lambda experiment.

    Args:
        config_path: Path to experiment configuration JSON file

    Returns:
        Dictionary with experiment results, or None if experiment failed/skipped
    """
    config_path = Path(config_path)

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    experiment_path = config_path.parent

    # Check if experiment already completed
    if is_experiment_complete(experiment_path):
        print(f"[SKIP] Experiment already completed: {experiment_path}")
        return None

    print(f"\n{'='*80}")
    print(f"RUNNING LOSS-PROPORTIONAL EXPERIMENT")
    print(f"{'='*80}")
    print(f"Experiment path: {experiment_path}")
    print(f"Model: {config['model_name']}")
    print(f"Constraints: Global {config['global_constraints']}, Local {config['local_constraints']}")
    print(f"Lambda learning rate: {config.get('lambda_learning_rate', 0.01)}")
    print(f"{'='*80}\n")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    try:
        # Load and preprocess data
        print("Loading data...")
        df = load_and_preprocess_data(config['data_path'])

        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test, groups_train, groups_test = stratified_split_data(
            df,
            test_size=config['test_size'],
            random_state=config['seed']
        )

        # Get input dimension
        input_dim = X_train.shape[1]

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
        model = trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            groups_test=groups_test,
            global_con=tuple(config['global_constraints']),
            local_con=config['local_constraints']
        )

        # Evaluate on test set
        print("\nEvaluating model on test set...")
        test_probs = model(torch.FloatTensor(X_test).to(device))
        test_probs = torch.softmax(test_probs, dim=1).detach().cpu().numpy()
        test_pred = np.argmax(test_probs, axis=1)

        # Compute metrics
        metrics = evaluate_model(y_test, test_pred, test_probs)

        # Save predictions
        predictions_file = experiment_path / 'final_predictions.csv'
        np.savetxt(
            predictions_file,
            np.column_stack([y_test, test_pred]),
            delimiter=',',
            header='True_Label,Predicted_Label',
            comments='',
            fmt='%d'
        )

        # Save evaluation metrics
        metrics_file = experiment_path / 'evaluation_metrics.csv'
        with open(metrics_file, 'w') as f:
            f.write("Metric,Value\n")
            f.write(f"Overall Accuracy,{metrics['accuracy']:.4f}\n")
            f.write("\n")
            f.write("Macro Averaged Metrics,\n")
            f.write(f"Precision (Macro),{metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro),{metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro),{metrics['f1_macro']:.4f}\n")
            f.write("\n")
            f.write("Weighted Averaged Metrics,\n")
            f.write(f"Precision (Weighted),{metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (Weighted),{metrics['recall_weighted']:.4f}\n")
            f.write(f"F1-Score (Weighted),{metrics['f1_weighted']:.4f}\n")
            f.write("\n")
            f.write("Per-Class Metrics,\n")
            f.write("Class,Precision,Recall,F1-Score,Support\n")
            for cls_name in ['Dropout', 'Enrolled', 'Graduate']:
                cls_idx = ['Dropout', 'Enrolled', 'Graduate'].index(cls_name)
                f.write(f"{cls_name},"
                       f"{metrics['per_class_precision'][cls_idx]:.4f},"
                       f"{metrics['per_class_recall'][cls_idx]:.4f},"
                       f"{metrics['per_class_f1'][cls_idx]:.4f},"
                       f"{metrics['per_class_support'][cls_idx]}\n")
            f.write("\n")
            f.write("Confusion Matrix,\n")
            cm = metrics['confusion_matrix']
            f.write(",Dropout,Enrolled,Graduate\n")
            f.write(f"Dropout,{cm[0,0]},{cm[0,1]},{cm[0,2]}\n")
            f.write(f"Enrolled,{cm[1,0]},{cm[1,1]},{cm[1,2]}\n")
            f.write(f"Graduate,{cm[2,0]},{cm[2,1]},{cm[2,2]}\n")

        # Mark experiment as complete
        mark_experiment_complete(experiment_path)

        print(f"\n{'='*80}")
        print(f"✓ EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Results saved to: {experiment_path}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"{'='*80}\n")

        return metrics

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ EXPERIMENT FAILED")
        print(f"{'='*80}")
        print(f"Error: {str(e)}")
        print(f"{'='*80}\n")

        # Mark as failed
        update_experiment_status(experiment_path, 'failed')

        # Re-raise exception for debugging
        raise


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_loss_proportional_experiment.py <config_path>")
        sys.exit(1)

    run_loss_proportional_experiment(sys.argv[1])
