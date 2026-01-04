import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from models import get_model
from src.losses import MulticlassTransductiveLoss
from src.training.metrics import compute_prediction_statistics, compute_train_accuracy, get_predictions_with_probabilities, compute_metrics
from src.training.logging import save_final_predictions, save_evaluation_metrics
from utils.filesystem_manager import load_config_from_path, save_config_to_path, mark_experiment_complete, is_experiment_complete

def get_model_cache_path(base_model_id: str) -> Path:
    cache_dir = Path('model_cache')
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{base_model_id}.pt"

def save_model_to_cache(model: nn.Module, scaler: StandardScaler, base_model_id: str, config: Dict[str, Any]) -> None:
    cache_path = get_model_cache_path(base_model_id)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'base_model_id': base_model_id,
        'model_name': config['model_name'],
        'hyperparams': {
            'hidden_dims': config['hyperparams']['hidden_dims'],
            'dropout': config['hyperparams']['dropout'],
            'lr': config['hyperparams']['lr'],
            'batch_size': config['hyperparams']['batch_size']
        },
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    torch.save(checkpoint, cache_path)
    print(f"\n[CACHE] Model saved to: {cache_path}")
    print(f"[CACHE] Base Model ID: {base_model_id}")

def load_model_from_cache(base_model_id: str, input_dim: int, config: Dict[str, Any], device: torch.device) -> Tuple[Optional[nn.Module], Optional[StandardScaler]]:
    cache_path = get_model_cache_path(base_model_id)
    if not cache_path.exists():
        print(f"\n[CACHE] No cached model found for {base_model_id}")
        return None, None
    try:
        print(f"\n[CACHE] Loading cached model from: {cache_path}")
        checkpoint = torch.load(cache_path, map_location=device)
        if checkpoint['base_model_id'] != base_model_id:
            print(f"[CACHE] Warning: Model ID mismatch. Training from scratch.")
            return None, None
        hyperparams = config['hyperparams']
        model = get_model(
            config['model_name'],
            input_dim=input_dim,
            hidden_dims=hyperparams['hidden_dims'],
            n_classes=3,
            dropout=hyperparams['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        scaler = StandardScaler()
        scaler.mean_ = checkpoint['scaler_mean']
        scaler.scale_ = checkpoint['scaler_scale']
        print(f"[CACHE] Successfully loaded model: {config['model_name']}")
        print(f"[CACHE] Saved at: {checkpoint['saved_at']}")
        print(f"[CACHE] Skipping warmup training - loading pre-trained weights")
        return model, scaler
    except Exception as e:
        print(f"[CACHE] Error loading cached model: {e}")
        print(f"[CACHE] Training from scratch...")
        return None, None

def train_warmup_phase(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion_ce: nn.Module, device: torch.device, warmup_epochs: int) -> nn.Module:
    print("\n" + "="*80)
    print("WARMUP PHASE: Training with pure cross-entropy loss")
    print(f"Epochs: {warmup_epochs}")
    print("="*80)
    for epoch in range(warmup_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion_ce(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            avg_loss = epoch_loss / len(train_loader)
            train_acc = compute_train_accuracy(model, train_loader, device)
            print(f"Warmup Epoch {epoch + 1}/{warmup_epochs}: Loss={avg_loss:.4f}, Acc={train_acc:.4f}")
    print(f"\n[WARMUP] Completed {warmup_epochs} warmup epochs")
    return model

def train_constraint_phase(model: nn.Module, train_loader: DataLoader, X_test_tensor: torch.Tensor, group_ids_test: torch.Tensor, optimizer: torch.optim.Optimizer, criterion_ce: nn.Module, criterion_constraint: MulticlassTransductiveLoss, device: torch.device, epochs: int, warmup_epochs: int, constraint_threshold: float, lambda_step: float) -> nn.Module:
    print("\n" + "="*80)
    print("CONSTRAINT OPTIMIZATION PHASE")
    print(f"Continuing training from epoch {warmup_epochs} to {epochs}")
    print(f"Lambda will adapt based on constraint violations")
    print("="*80)
    for epoch in range(warmup_epochs, epochs):
        model.train()
        epoch_loss_ce = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            train_logits = model(batch_features)
            loss_ce = criterion_ce(train_logits, batch_labels)
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test_tensor)
            model.train()
            _, _, loss_global, loss_local = criterion_constraint(
                test_logits, y_true=None, group_ids=group_ids_test
            )
            loss = loss_ce + criterion_constraint.lambda_global * loss_global + criterion_constraint.lambda_local * loss_local
            loss.backward()
            optimizer.step()
            epoch_loss_ce += loss_ce.item()
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor)
            _, _, loss_global, loss_local = criterion_constraint(
                test_logits, y_true=None, group_ids=group_ids_test
            )
        avg_global = loss_global.item()
        avg_local = loss_local.item()
        if avg_global > constraint_threshold:
            new_lambda_global = criterion_constraint.lambda_global + lambda_step
            criterion_constraint.set_lambda(lambda_global=new_lambda_global)
        if avg_local > constraint_threshold:
            new_lambda_local = criterion_constraint.lambda_local + lambda_step
            criterion_constraint.set_lambda(lambda_local=new_lambda_local)
        if (epoch + 1) % 10 == 0:
            train_acc = compute_train_accuracy(model, train_loader, device)
            print(f"Epoch {epoch + 1}: CE={epoch_loss_ce/len(train_loader):.4f}, "
                  f"Train Acc={train_acc:.4f}, "
                  f"Lambda=({criterion_constraint.lambda_global:.3f}, {criterion_constraint.lambda_local:.3f})")
        if criterion_constraint.global_constraints_satisfied and criterion_constraint.local_constraints_satisfied:
            print(f"\n[SUCCESS] All constraints satisfied at epoch {epoch + 1}!")
            break
    return model

def run_single_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    print("\n" + "="*80)
    print("EXPERIMENT RUNNER")
    print("="*80)
    print(f"Loading config: {config_path}")
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    if is_experiment_complete(experiment_path):
        print(f"\n[SKIP] Experiment already completed: {experiment_path}")
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")
    print(f"Constraint: {config['constraint']}")
    print(f"Regime: {config['hyperparam_regime']} / {config['variation_name']}")
    print(f"Base Model ID: {config['base_model_id']}")
    from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN
    from src.data import load_presplit_data
    from src.training.constraints import compute_global_constraints, compute_local_constraints
    print("\nLoading dataset...")
    X_train, X_test, y_train, y_test, train_df, test_df = load_presplit_data(
        TRAIN_PATH, TEST_PATH, TARGET_COLUMN
    )
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    constraint = config['constraint']
    local_percent, global_percent = constraint
    groups = full_df['Course'].unique()
    global_constraint = compute_global_constraints(full_df, TARGET_COLUMN, global_percent)
    local_constraint = compute_local_constraints(full_df, TARGET_COLUMN, local_percent, groups)
    print(f"Global constraint: {global_constraint}")
    print(f"Local constraints: {len(local_constraint)} courses")
    X_train_clean = X_train.drop("Course", axis=1)
    X_test_clean = X_test.drop("Course", axis=1)
    groups_test = X_test["Course"]
    hyperparams = config['hyperparams']
    lr = hyperparams['lr']
    dropout = hyperparams['dropout']
    batch_size = hyperparams['batch_size']
    hidden_dims = hyperparams['hidden_dims']
    epochs = hyperparams['epochs']
    lambda_global = hyperparams['lambda_global']
    lambda_local = hyperparams['lambda_local']
    warmup_epochs = hyperparams['warmup_epochs']
    constraint_threshold = hyperparams['constraint_threshold']
    lambda_step = hyperparams['lambda_step']
    input_dim = X_train_clean.shape[1]
    base_model_id = config['base_model_id']
    start_time = time.time()
    model, scaler = load_model_from_cache(base_model_id, input_dim, config, device)
    if model is None:
        print("\n[TRAINING] No cached model found - training from scratch")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)
        if y_train.dtype == 'O' or isinstance(y_train.iloc[0], str):
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
        else:
            y_train_encoded = y_train.values
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train_encoded)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = get_model(
            config['model_name'],
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            n_classes=3,
            dropout=dropout
        ).to(device)
        criterion_ce = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model = train_warmup_phase(model, train_loader, optimizer, criterion_ce, device, warmup_epochs)
        save_model_to_cache(model, scaler, base_model_id, config)
    else:
        print("\n[CACHE] Using cached model weights")
        X_train_scaled = scaler.transform(X_train_clean)
        X_test_scaled = scaler.transform(X_test_clean)
        if y_train.dtype == 'O' or isinstance(y_train.iloc[0], str):
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
        else:
            y_train_encoded = y_train.values
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train_encoded)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    X_test_tensor = torch.FloatTensor(scaler.transform(X_test_clean)).to(device)
    group_ids_test = torch.LongTensor(groups_test.values).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_constraint = MulticlassTransductiveLoss(
        global_constraints=global_constraint,
        local_constraints=local_constraint,
        lambda_global=lambda_global,
        lambda_local=lambda_local,
        use_ce=False
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = train_constraint_phase(
        model, train_loader, X_test_tensor, group_ids_test,
        optimizer, criterion_ce, criterion_constraint, device,
        epochs, warmup_epochs, constraint_threshold, lambda_step
    )
    training_time = time.time() - start_time
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    model.eval()
    y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
    y_true_np = y_test.values if hasattr(y_test, 'values') else y_test
    course_ids_np = groups_test.values if hasattr(groups_test, 'values') else groups_test
    save_final_predictions(
        Path(experiment_path) / 'final_predictions.csv',
        y_true_np, y_pred, y_proba, course_ids_np
    )
    metrics = compute_metrics(y_true_np, y_pred)
    save_evaluation_metrics(
        Path(experiment_path) / 'evaluation_metrics.csv',
        metrics
    )
    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(training_time),
        'final_lambda_global': float(criterion_constraint.lambda_global),
        'final_lambda_local': float(criterion_constraint.lambda_local),
        'used_cached_model': model is not None and scaler is not None
    }
    save_config_to_path(config, experiment_path)
    mark_experiment_complete(experiment_path)
    print(f"\nFinal Accuracy: {metrics['accuracy']:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Results saved to: {experiment_path}")
    return config['results']

def main() -> None:
    parser = argparse.ArgumentParser(description='Run single experiment from config file')
    parser.add_argument('config_path', type=str, help='Path to config.json file')
    args = parser.parse_args()
    try:
        results = run_single_experiment(args.config_path)
        if results:
            print("\n" + "="*80)
            print("EXPERIMENT COMPLETED SUCCESSFULLY")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("EXPERIMENT SKIPPED (already completed)")
            print("="*80)
    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()
