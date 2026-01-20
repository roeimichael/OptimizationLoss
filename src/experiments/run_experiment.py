import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.data_loader import load_experiment_data
from src.training.trainer import ConstraintTrainer
from src.training.metrics import get_predictions_with_probabilities, compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics, save_run_status
from src.utils.filesystem_manager import load_config_from_path, save_config_to_path, mark_experiment_complete, \
    update_experiment_status, save_stop_reason


def run_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    print(f"Config: {config_path}")
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)

    if config.get('status', 'pending') == 'completed':
        print(f"[SKIP] Already completed")
        return None

    if 'base_model_id' not in config:
        raise ValueError(f"Config missing 'base_model_id': {config_path}")

    update_experiment_status(experiment_path, 'running')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")

    X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(config)

    print("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    try:
        if len(y_train) == 0:
            raise ValueError("Training data is empty")
        if y_train.dtype == 'O' or (hasattr(y_train, 'iloc') and isinstance(y_train.iloc[0], str)):
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
        else:
            y_train_encoded = y_train.values
    except Exception as e:
        raise ValueError(f"Failed to encode labels: {e}")

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train_encoded)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    input_dim = X_train_clean.shape[1]

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
            global_con=global_constraint,
            local_con=local_constraint
        )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user (Ctrl+C)")
        # Check if we have a training log to get the last epoch and constraint status
        log_path = Path(experiment_path) / 'training_log.csv'
        last_epoch = 0
        global_sat = False
        local_sat = False
        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    last_epoch = int(last_row['Epoch'])
                    global_sat = int(last_row['Global_Satisfied']) == 1
                    local_sat = int(last_row['Local_Satisfied']) == 1
            except:
                pass

        # Save to run_status.json
        save_run_status(
            str(experiment_path),
            status='interrupted',
            epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat,
            details="Training interrupted by user (KeyboardInterrupt / Ctrl+C)"
        )

        # Save to config.json
        save_stop_reason(
            str(experiment_path),
            status='interrupted',
            reason="User manually interrupted training with Ctrl+C (KeyboardInterrupt)",
            exception_type='KeyboardInterrupt',
            final_epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat
        )
        raise

    except MemoryError as e:
        print(f"\n[INTERRUPTED] Out of Memory Error")
        log_path = Path(experiment_path) / 'training_log.csv'
        last_epoch = 0
        global_sat = False
        local_sat = False
        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    last_epoch = int(last_row['Epoch'])
                    global_sat = int(last_row['Global_Satisfied']) == 1
                    local_sat = int(last_row['Local_Satisfied']) == 1
            except:
                pass

        save_run_status(
            str(experiment_path),
            status='interrupted',
            epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat,
            details=f"Out of Memory (OOM): {str(e)}"
        )

        save_stop_reason(
            str(experiment_path),
            status='interrupted',
            reason=f"Out of Memory (OOM) error - system ran out of RAM during training. Error: {str(e)}",
            exception_type='MemoryError',
            final_epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat
        )
        raise

    except RuntimeError as e:
        print(f"\n[INTERRUPTED] Runtime Error: {e}")
        log_path = Path(experiment_path) / 'training_log.csv'
        last_epoch = 0
        global_sat = False
        local_sat = False
        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    last_epoch = int(last_row['Epoch'])
                    global_sat = int(last_row['Global_Satisfied']) == 1
                    local_sat = int(last_row['Local_Satisfied']) == 1
            except:
                pass

        # Check if it's a CUDA OOM error
        error_str = str(e).lower()
        if 'out of memory' in error_str or 'cuda' in error_str:
            reason = f"CUDA Out of Memory error - GPU ran out of memory during training. Error: {str(e)}"
            exception_type = 'RuntimeError (CUDA OOM)'
        else:
            reason = f"Runtime error during training. Error: {str(e)}"
            exception_type = 'RuntimeError'

        save_run_status(
            str(experiment_path),
            status='interrupted',
            epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat,
            details=reason
        )

        save_stop_reason(
            str(experiment_path),
            status='interrupted',
            reason=reason,
            exception_type=exception_type,
            final_epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat
        )
        raise

    except SystemExit as e:
        print(f"\n[INTERRUPTED] System Exit (code: {e.code})")
        log_path = Path(experiment_path) / 'training_log.csv'
        last_epoch = 0
        global_sat = False
        local_sat = False
        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    last_epoch = int(last_row['Epoch'])
                    global_sat = int(last_row['Global_Satisfied']) == 1
                    local_sat = int(last_row['Local_Satisfied']) == 1
            except:
                pass

        save_run_status(
            str(experiment_path),
            status='interrupted',
            epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat,
            details=f"System exit (code: {e.code})"
        )

        save_stop_reason(
            str(experiment_path),
            status='interrupted',
            reason=f"Process exited with code {e.code} - possibly killed by system, timeout, or external signal",
            exception_type='SystemExit',
            final_epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat
        )
        raise

    except Exception as e:
        exception_type_name = type(e).__name__
        print(f"\n[INTERRUPTED] Unexpected exception ({exception_type_name}): {e}")
        traceback.print_exc()

        log_path = Path(experiment_path) / 'training_log.csv'
        last_epoch = 0
        global_sat = False
        local_sat = False
        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    last_epoch = int(last_row['Epoch'])
                    global_sat = int(last_row['Global_Satisfied']) == 1
                    local_sat = int(last_row['Local_Satisfied']) == 1
            except:
                pass

        save_run_status(
            str(experiment_path),
            status='interrupted',
            epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat,
            details=f"Unexpected exception ({exception_type_name}): {str(e)}"
        )

        save_stop_reason(
            str(experiment_path),
            status='interrupted',
            reason=f"Unexpected exception during training: {exception_type_name}: {str(e)}",
            exception_type=exception_type_name,
            final_epoch=last_epoch,
            global_satisfied=global_sat,
            local_satisfied=local_sat
        )
        raise

    training_time = time.time() - start_time

    print("\nEvaluation...")
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
        'used_cached_model': trainer.from_cache
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
            print("\nCompleted")
        else:
            print("\nSkipped")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        update_experiment_status(str(experiment_path), 'pending')
        print("[STATUS] Reset to 'pending' for retry")
        exit(1)


if __name__ == "__main__":
    main()