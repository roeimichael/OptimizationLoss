import os
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path

from config.experiment_config import *
from src.data import load_presplit_data
from src.training import (
    compute_global_constraints,
    compute_local_constraints,
    train_model_transductive,
    predict,
    evaluate_accuracy
)


def save_results(results_file, model_name, method_name, accuracy, local_percent, global_percent, training_time):
    output = f"{model_name},{method_name},{accuracy},{local_percent},{global_percent},{training_time}"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    if not os.path.exists(results_file):
        with open(results_file, 'a') as f:
            headlines = "model_name,method_name,accuracy,local_percent,global_percent,training_time"
            f.write(headlines + "\n" + output + "\n")
    else:
        with open(results_file, 'a') as f:
            f.write(output + "\n")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading datasets...")
    print(f"  Train: {TRAIN_PATH}")
    print(f"  Test: {TEST_PATH}")
    X_train, X_test, y_train, y_test, train_df, test_df = load_presplit_data(
        TRAIN_PATH, TEST_PATH, TARGET_COLUMN
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    groups = full_df['Course'].unique()
    print(f"Combined dataset: {len(full_df)} samples")
    print(f"Number of courses: {len(groups)}")

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    all_results = {}

    for config_idx, config in enumerate(NN_CONFIGS):
        config_name = config.get('name', f"nn_config{config_idx + 1}")

        lambda_global = config['lambda_global']
        lambda_local = config['lambda_local']
        hidden_dims = config['hidden_dims']
        lr = config.get('lr', TRAINING_PARAMS['lr'])
        dropout = config.get('dropout', TRAINING_PARAMS['dropout'])
        batch_size = config.get('batch_size', TRAINING_PARAMS['batch_size'])

        print(f"\n{'='*80}")
        print(f"Config: {config_name}")
        print(f"  Lambda: global={lambda_global}, local={lambda_local}")
        print(f"  Architecture: {hidden_dims}")
        print(f"  Learning rate: {lr}, Dropout: {dropout}, Batch size: {batch_size}")
        print(f"{'='*80}")

        config_results = {}

        for constraint_pair in CONSTRAINTS:
            local_percent, global_percent = constraint_pair
            print(f"\nConstraint: local={local_percent}, global={global_percent}")

            constraint_suffix = f"_c{local_percent}_{global_percent}" if len(CONSTRAINTS) > 1 else ""
            exp_name = f"{config_name}{constraint_suffix}"

            global_constraint = compute_global_constraints(full_df, TARGET_COLUMN, global_percent)
            local_constraint = compute_local_constraints(full_df, TARGET_COLUMN, local_percent, groups)

            print(f"  Global constraint: {global_constraint}")
            print(f"  Local constraints: {len(local_constraint)} courses")

            X_train_clean = X_train.drop("Course", axis=1)
            X_test_clean = X_test.drop("Course", axis=1)
            groups_test = X_test["Course"]

            print(f"  Training with transductive approach...")
            model, scaler, training_time, history, metrics = train_model_transductive(
                X_train_clean, y_train,
                X_test_clean, groups_test, y_test,
                global_constraint, local_constraint,
                lambda_global=lambda_global,
                lambda_local=lambda_local,
                hidden_dims=hidden_dims,
                epochs=TRAINING_PARAMS['epochs'],
                batch_size=batch_size,
                lr=lr,
                dropout=dropout,
                device=device,
                constraint_dropout_pct=local_percent,
                constraint_enrolled_pct=global_percent,
                hyperparam_name=exp_name
            )

            y_test_pred = predict(model, scaler, X_test_clean, device)
            accuracy = evaluate_accuracy(y_test.values, y_test_pred)

            results_file = f"{RESULTS_DIR}/students__train__{exp_name}__transductive.csv"
            save_results(results_file, exp_name, "transductive", accuracy,
                       local_percent, global_percent, training_time)

            print(f"  Test Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")

            benchmark_metrics_path = f"./results/hyperparam_{exp_name}/benchmark_metrics.csv"
            benchmark_metrics = {}
            if os.path.exists(benchmark_metrics_path):
                with open(benchmark_metrics_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) == 2:
                            metric, value = parts
                            if metric == 'Overall Accuracy':
                                benchmark_metrics['benchmark_accuracy'] = float(value)
                            elif metric == 'Precision (Macro)':
                                benchmark_metrics['benchmark_precision_macro'] = float(value)
                            elif metric == 'Recall (Macro)':
                                benchmark_metrics['benchmark_recall_macro'] = float(value)
                            elif metric == 'F1-Score (Macro)':
                                benchmark_metrics['benchmark_f1_macro'] = float(value)

            config_results[str(constraint_pair)] = {
                'hyperparameters': {
                    'lambda_global': lambda_global,
                    'lambda_local': lambda_local,
                    'hidden_dims': hidden_dims,
                    'learning_rate': lr,
                    'dropout': dropout,
                    'batch_size': batch_size
                },
                'accuracy': float(metrics['accuracy']),
                'precision_macro': float(metrics['precision_macro']),
                'recall_macro': float(metrics['recall_macro']),
                'f1_macro': float(metrics['f1_macro']),
                'precision_weighted': float(metrics['precision_weighted']),
                'recall_weighted': float(metrics['recall_weighted']),
                'f1_weighted': float(metrics['f1_weighted']),
                'training_time': float(training_time),
                **benchmark_metrics
            }

        all_results[config_name] = config_results

    results_json_path = f"{RESULTS_DIR}/nn_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n{'='*80}")
    print(f"Results saved to {results_json_path}")
    print(f"{'='*80}")

    print("\nSummary:")
    for config_name, results in all_results.items():
        print(f"\n{config_name}:")
        for constraint, metrics in results.items():
            summary = f"  {constraint}:\n"
            summary += f"    Optimized: Acc={metrics['accuracy']:.4f}, P={metrics['precision_macro']:.4f}, "
            summary += f"R={metrics['recall_macro']:.4f}, F1={metrics['f1_macro']:.4f}, Time={metrics['training_time']:.2f}s"
            if 'benchmark_accuracy' in metrics:
                summary += f"\n    Benchmark: Acc={metrics['benchmark_accuracy']:.4f}, P={metrics['benchmark_precision_macro']:.4f}, "
                summary += f"R={metrics['benchmark_recall_macro']:.4f}, F1={metrics['benchmark_f1_macro']:.4f}"
            print(summary)

    print(f"\n{'='*80}")
    print("Running comprehensive analysis and comparison...")
    print(f"{'='*80}\n")

    try:
        import subprocess
        if len(CONSTRAINTS) > 1:
            print("Multiple constraints detected - running constraint-grouped analysis...\n")
            result = subprocess.run(['python', 'experiments/analyze_by_constraints.py'],
                                  capture_output=False, text=True)
            if result.returncode == 0:
                print("\nConstraint-grouped analysis completed successfully")
            else:
                print("\nConstraint-grouped analysis completed with warnings")
        else:
            print("Single constraint detected - running standard top 5 analysis...\n")
            result = subprocess.run(['python', 'experiments/analyze_top5.py'],
                                  capture_output=False, text=True)
            if result.returncode == 0:
                print("\nTop 5 analysis completed successfully")
            else:
                print("\nTop 5 analysis completed with warnings")
    except Exception as e:
        print(f"\nCould not run analysis: {e}")
        if len(CONSTRAINTS) > 1:
            print("You can run it manually with: python experiments/analyze_by_constraints.py")
        else:
            print("You can run it manually with: python experiments/analyze_top5.py")


if __name__ == "__main__":
    main()
