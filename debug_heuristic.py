#!/usr/bin/env python3
"""
Debug version of run_heuristic.py to investigate model prediction issues.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import load_experiment_data
from src.utils.filesystem_manager import load_config_from_path
from src.training.trainer import ConstraintTrainer


def debug_model_predictions(config_path: str) -> None:
    print("=" * 80)
    print("HEURISTIC DEBUG MODE")
    print("=" * 80)
    print(f"\nConfig: {config_path}")

    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")
    print(f"Base Model ID: {config['base_model_id']}")
    print(f"Constraint: {config['constraint']}")
    print(f"LR: {config['hyperparams']['lr']}")
    print(f"Lambda Strategy: {config['hyperparams']['lambda_strategy']}")

    # Load data
    print("\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)
    X_train_clean, X_test_clean, _, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(config)

    print(f"Train shape: {X_train_clean.shape}")
    print(f"Test shape: {X_test_clean.shape}")
    print(f"Test labels distribution:")
    print(pd.Series(y_test).value_counts().sort_index())

    # Scale data
    print("\n" + "-" * 80)
    print("SCALING DATA")
    print("-" * 80)
    scaler = StandardScaler()
    scaler.fit(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)

    print(f"Original test data - mean: {X_test_clean.values.mean():.4f}, std: {X_test_clean.values.std():.4f}")
    print(f"Scaled test data - mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")
    print(f"Min: {X_test_scaled.min():.4f}, Max: {X_test_scaled.max():.4f}")

    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    print(f"Tensor shape: {X_test_tensor.shape}")

    # Load model
    print("\n" + "-" * 80)
    print("LOADING MODEL")
    print("-" * 80)
    trainer = ConstraintTrainer(config, str(experiment_path), device)
    trainer.setup_model(X_train_clean.shape[1], config['base_model_id'])

    if trainer.model is None:
        print("[ERROR] Model not loaded from cache!")
        return

    print(f"Model loaded successfully")
    print(f"From cache: {trainer.from_cache}")
    print(f"Model type: {type(trainer.model).__name__}")

    # Check model weights
    print("\nModel parameter check:")
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Sample some weights to verify they're not zero/uniform
    first_layer = None
    for name, param in trainer.model.named_parameters():
        if 'weight' in name:
            first_layer = param
            print(f"\n{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Mean: {param.mean().item():.6f}")
            print(f"  Std: {param.std().item():.6f}")
            print(f"  Min: {param.min().item():.6f}")
            print(f"  Max: {param.max().item():.6f}")
            break

    # Get predictions
    print("\n" + "-" * 80)
    print("MODEL PREDICTIONS")
    print("-" * 80)
    trainer.model.eval()

    with torch.no_grad():
        logits = trainer.model(X_test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    print(f"Logits shape: {logits.shape}")
    print(f"Probs shape: {probs.shape}")

    # Analyze logits
    print("\nLogits statistics (raw model output):")
    logits_np = logits.cpu().numpy()
    for class_idx in range(3):
        print(f"  Class {class_idx}: mean={logits_np[:, class_idx].mean():.4f}, "
              f"std={logits_np[:, class_idx].std():.4f}, "
              f"min={logits_np[:, class_idx].min():.4f}, "
              f"max={logits_np[:, class_idx].max():.4f}")

    # Analyze probabilities
    print("\nProbability statistics (after softmax):")
    for class_idx in range(3):
        print(f"  Class {class_idx}: mean={probs[:, class_idx].mean():.4f}, "
              f"std={probs[:, class_idx].std():.4f}, "
              f"min={probs[:, class_idx].min():.4f}, "
              f"max={probs[:, class_idx].max():.4f}")

    # Predicted classes
    predicted_classes = probs.argmax(axis=1)
    print("\nPredicted class distribution:")
    unique, counts = np.unique(predicted_classes, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(predicted_classes)*100:.1f}%)")

    # Sample predictions
    print("\nFirst 20 predictions:")
    print("Index | True | Pred | Prob[0]  | Prob[1]  | Prob[2]")
    print("-" * 60)
    for i in range(min(20, len(y_test))):
        y_true = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
        y_pred = predicted_classes[i]
        print(f"{i:5d} | {y_true:4d} | {y_pred:4d} | {probs[i,0]:.6f} | {probs[i,1]:.6f} | {probs[i,2]:.6f}")

    # Compare with training accuracy from optimization results
    print("\n" + "-" * 80)
    print("COMPARISON WITH OPTIMIZATION RESULTS")
    print("-" * 80)

    if 'optimization_results' in config:
        opt_results = config['optimization_results']
        print(f"Optimization training accuracy: {opt_results.get('accuracy', 'N/A')}")
    elif 'results' in config:
        print(f"Results accuracy: {config['results'].get('accuracy', 'N/A')}")

    # Check if there's a training log
    training_log = experiment_path / 'training_log.csv'
    if training_log.exists():
        print(f"\nFound training log: {training_log}")
        log_df = pd.read_csv(training_log)
        if 'Epoch' in log_df.columns:
            print(f"Training epochs logged: {len(log_df)}")
            if 'Train_Acc' in log_df.columns:
                print(f"Final warmup train accuracy: {log_df['Train_Acc'].iloc[-1]:.4f}")

    # Test on training data to see if model works there
    print("\n" + "-" * 80)
    print("TESTING ON TRAINING DATA (SANITY CHECK)")
    print("-" * 80)
    X_train_scaled = scaler.transform(X_train_clean)
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)

    with torch.no_grad():
        train_logits = trainer.model(X_train_tensor)
        train_probs = torch.softmax(train_logits, dim=1).cpu().numpy()

    train_predicted = train_probs.argmax(axis=1)
    print("Predicted class distribution on TRAIN data:")
    unique, counts = np.unique(train_predicted, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(train_predicted)*100:.1f}%)")

    # Check if model is frozen or in eval mode incorrectly
    print("\n" + "-" * 80)
    print("MODEL STATE CHECK")
    print("-" * 80)
    print(f"Model training mode: {trainer.model.training}")
    print(f"Requires grad:")
    for name, param in trainer.model.named_parameters():
        if not param.requires_grad:
            print(f"  ⚠️  {name}: requires_grad = False")
            break
    else:
        print("  ✓ All parameters require grad")

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    try:
        debug_model_predictions(args.config_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
