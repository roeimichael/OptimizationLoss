import os
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path

from config.experiment_config import *
from src.data import load_and_preprocess_data, split_data, load_presplit_data
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

    # Load data - use pre-split or split dynamically
    if USE_PRESPLIT_DATA:
        print("Loading pre-split datasets...")
        print(f"  Train: {TRAIN_PATH}")
        print(f"  Test: {TEST_PATH}")
        X_train, X_test, y_train, y_test = load_presplit_data(
            TRAIN_PATH, TEST_PATH, TARGET_COLUMN
        )
        print(f"Train: {len(y_train)}, Test: {len(y_test)} (Test used for constraints only)")

        # Load full dataset to get groups
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        df = pd.concat([train_df, test_df], ignore_index=True)
        groups = df['Course'].unique()

    else:
        print("Loading and splitting data dynamically...")
        df = load_and_preprocess_data(DATA_PATH, TARGET_COLUMN)
        print(f"Data loaded: {df.shape}")

        X_train, X_test, y_train, y_test = split_data(
            df, TARGET_COLUMN,
            test_size=TRAINING_PARAMS['test_size'],
            random_state=42
        )
        print(f"Train: {len(y_train)}, Test: {len(y_test)} (Test used for constraints only)")
        groups = df['Course'].unique()

    print(f"Number of courses: {len(groups)}")

    Path(RESULTS_DIR).mkdir(exist_ok=True)

    all_results = {}

    for config_idx, config in enumerate(NN_CONFIGS):
        config_name = f"nn_config{config_idx + 1}"
        print(f"\n{'='*80}")
        print(f"Config {config_idx + 1}: lambda_g={config['lambda_global']}, lambda_l={config['lambda_local']}")
        print(f"Hidden dims: {config['hidden_dims']}")
        print(f"{'='*80}")

        config_results = {}

        for constraint_pair in CONSTRAINTS:
            local_percent, global_percent = constraint_pair
            print(f"\nConstraint: local={local_percent}, global={global_percent}")

            df_test = df.loc[X_test.index]
            global_constraint = compute_global_constraints(df_test, TARGET_COLUMN, global_percent)
            local_constraint = compute_local_constraints(df_test, TARGET_COLUMN, local_percent, groups)

            print(f"  Global constraint: {global_constraint}")
            print(f"  Local constraints: {len(local_constraint)} courses")

            X_train_clean = X_train.drop("Course", axis=1)
            X_test_clean = X_test.drop("Course", axis=1)
            groups_test = X_test["Course"]

            print(f"  Training with transductive approach...")
            model, scaler, training_time, history = train_model_transductive(
                X_train_clean, y_train,
                X_test_clean, groups_test,
                global_constraint, local_constraint,
                lambda_global=config['lambda_global'],
                lambda_local=config['lambda_local'],
                hidden_dims=config['hidden_dims'],
                epochs=TRAINING_PARAMS['epochs'],
                batch_size=TRAINING_PARAMS['batch_size'],
                lr=TRAINING_PARAMS['lr'],
                dropout=TRAINING_PARAMS['dropout'],
                patience=TRAINING_PARAMS['patience'],
                device=device
            )

            y_test_pred = predict(model, scaler, X_test_clean, device)
            accuracy = evaluate_accuracy(y_test.values, y_test_pred)

            results_file = f"{RESULTS_DIR}/students__train__{config_name}__transductive.csv"
            save_results(results_file, config_name, "transductive", accuracy,
                       local_percent, global_percent, training_time)

            print(f"  Test Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")

            config_results[str(constraint_pair)] = {
                'accuracy': float(accuracy),
                'training_time': float(training_time)
            }

        all_results[f"{config_name}_transductive"] = config_results

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
            print(f"  {constraint}: Acc={metrics['accuracy']:.4f}, Time={metrics['training_time']:.2f}s")


if __name__ == "__main__":
    main()
