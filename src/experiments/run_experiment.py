"""
Single experiment runner with centralized error handling.

This module executes individual experiments based on configuration files.
Error handling is managed through the @logger decorator and helper functions,
keeping the core logic clean and focused on the experiment workflow.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.data_loader import load_experiment_data
from src.utils.error_handler import logger, log_exception, safe_execute
from src.training.trainer import ConstraintTrainer
from src.training.metrics import get_predictions_with_probabilities, compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics, save_run_status
from src.utils.filesystem_manager import (
    load_config_from_path,
    save_config_to_path,
    mark_experiment_complete,
    update_experiment_status,
    save_stop_reason
)


def _get_last_training_state(experiment_path: Path) -> Dict[str, Any]:
    """Extract last known training state from log file."""
    log_path = experiment_path / 'training_log.csv'
    state = {'epoch': 0, 'global_satisfied': False, 'local_satisfied': False}

    if not log_path.exists():
        return state

    df = safe_execute(pd.read_csv, log_path, default=None, context="Reading training log")
    if df is not None and len(df) > 0:
        last_row = df.iloc[-1]
        state['epoch'] = int(last_row.get('Epoch', 0))
        state['global_satisfied'] = int(last_row.get('Global_Satisfied', 0)) == 1
        state['local_satisfied'] = int(last_row.get('Local_Satisfied', 0)) == 1

    return state


def _save_interruption(experiment_path: Path, exception: Exception, reason: str) -> None:
    """Save interruption status to experiment files."""
    state = _get_last_training_state(experiment_path)
    exception_type = type(exception).__name__

    save_run_status(
        str(experiment_path),
        status='interrupted',
        epoch=state['epoch'],
        global_satisfied=state['global_satisfied'],
        local_satisfied=state['local_satisfied'],
        details=reason
    )

    save_stop_reason(
        str(experiment_path),
        status='interrupted',
        reason=reason,
        exception_type=exception_type,
        final_epoch=state['epoch'],
        global_satisfied=state['global_satisfied'],
        local_satisfied=state['local_satisfied']
    )


@logger()
def run_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Run a single experiment based on configuration.

    Args:
        config_path: Path to the experiment's config.json file

    Returns:
        Dictionary of results if successful, None if skipped
    """
    print(f"Config: {config_path}")
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)

    if config.get('status', 'pending') == 'completed':
        print("[SKIP] Already completed")
        return None

    update_experiment_status(experiment_path, 'running')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")

    X_train, X_test, y_train, y_test, groups_test, global_con, local_con = load_experiment_data(config)

    print("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    input_dim = X_train.shape[1]

    trainer = ConstraintTrainer(config, str(experiment_path), device)
    trainer.setup_model(input_dim=input_dim, base_model_id=config['base_model_id'])

    start_time = time.time()

    try:
        trainer.train_warmup(X_train_tensor, y_train_tensor, config['base_model_id'])
        model = trainer.train_constraints(
            X_train=X_train_tensor,
            y_train=y_train_tensor,
            X_test=X_test_tensor,
            groups_test=groups_test,
            global_con=global_con,
            local_con=local_con
        )

    except KeyboardInterrupt as e:
        print("\n[INTERRUPTED] Training interrupted by user (Ctrl+C)")
        _save_interruption(
            experiment_path, e,
            "User manually interrupted training with Ctrl+C (KeyboardInterrupt)"
        )
        raise

    except MemoryError as e:
        print("\n[INTERRUPTED] Out of Memory Error")
        _save_interruption(
            experiment_path, e,
            f"Out of Memory (OOM) error - system ran out of RAM. Error: {e}"
        )
        raise

    except RuntimeError as e:
        error_str = str(e).lower()
        if 'out of memory' in error_str or 'cuda' in error_str:
            reason = f"CUDA Out of Memory - GPU ran out of memory. Error: {e}"
        else:
            reason = f"Runtime error during training. Error: {e}"
        print(f"\n[INTERRUPTED] {reason}")
        _save_interruption(experiment_path, e, reason)
        raise

    except SystemExit as e:
        print(f"\n[INTERRUPTED] System Exit (code: {e.code})")
        _save_interruption(
            experiment_path, e,
            f"Process exited with code {e.code} - possibly killed by system or timeout"
        )
        raise

    training_time = time.time() - start_time

    print("\nEvaluation...")
    model.eval()

    y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
    y_true_np = y_test.values if hasattr(y_test, 'values') else y_test
    group_ids_np = groups_test.values if hasattr(groups_test, 'values') else groups_test

    save_final_predictions(
        experiment_path / 'final_predictions.csv',
        y_true_np, y_pred, y_proba, group_ids_np
    )

    metrics = compute_metrics(y_true_np, y_pred)
    save_evaluation_metrics(experiment_path / 'evaluation_metrics.csv', metrics)

    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(training_time),
        'used_cached_model': trainer.from_cache
    }

    save_config_to_path(config, experiment_path)
    mark_experiment_complete(experiment_path)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Results saved: {experiment_path}")

    return config['results']


def main() -> None:
    """Main entry point for single experiment execution."""
    parser = argparse.ArgumentParser(description='Run single experiment')
    parser.add_argument('config_path', type=str, help='Path to config.json')
    args = parser.parse_args()
    experiment_path = Path(args.config_path).parent

    try:
        results = run_experiment(args.config_path)
        if results:
            print("\nCompleted")
        else:
            print("\nSkipped")

    except Exception as e:
        log_exception(e, context=f"Experiment: {experiment_path}")
        update_experiment_status(str(experiment_path), 'pending')
        print("[STATUS] Reset to 'pending' for retry")
        exit(1)


if __name__ == "__main__":
    main()
