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
from utils.filesystem_manager import load_config_from_path, save_config_to_path, mark_experiment_complete, is_experiment_complete, update_experiment_status

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
    update_experiment_status(experiment_path, 'running')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {config['model_name']}")
    print(f"Constraint: {config['constraint']}")
    print(f"Regime: {config['hyperparam_regime']} / {config['variation_name']}")
    print(f"Base Model ID: {config['base_model_id']}")
