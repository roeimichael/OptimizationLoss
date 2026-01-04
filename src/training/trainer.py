import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.models import get_model
from src.losses import MulticlassTransductiveLoss
from .metrics import compute_train_accuracy

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
    print(f"[CACHE] Model saved: {base_model_id}")

def load_model_from_cache(base_model_id: str, input_dim: int, config: Dict[str, Any], device: torch.device) -> Tuple[Optional[nn.Module], Optional[StandardScaler]]:
    cache_path = get_model_cache_path(base_model_id)
    if not cache_path.exists():
        return None, None
    try:
        checkpoint = torch.load(cache_path, map_location=device)
        if checkpoint['base_model_id'] != base_model_id:
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
        return model, scaler
    except Exception as e:
        print(f"[CACHE] Error loading: {e}")
        return None, None

def train_warmup(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, input_dim: int, hidden_dims: list, dropout: float, lr: float, batch_size: int, warmup_epochs: int, device: torch.device) -> Tuple[nn.Module, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
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
        model_name,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        n_classes=3,
        dropout=dropout
    ).to(device)
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
    print(f"Warmup complete: {warmup_epochs} epochs")
    return model, scaler

def train_with_constraints(model: nn.Module, scaler: StandardScaler, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, groups_test: pd.Series, global_constraint: Dict[str, float], local_constraint: Dict[str, Dict[str, float]], lr: float, batch_size: int, epochs: int, warmup_epochs: int, lambda_global: float, lambda_local: float, constraint_threshold: float, lambda_step: float, device: torch.device) -> nn.Module:
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
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
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
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
            print(f"Epoch {epoch + 1}: CE={epoch_loss_ce/len(train_loader):.4f}, Acc={train_acc:.4f}, Lambda=({criterion_constraint.lambda_global:.3f}, {criterion_constraint.lambda_local:.3f})")
        if criterion_constraint.global_constraints_satisfied and criterion_constraint.local_constraints_satisfied:
            print(f"All constraints satisfied at epoch {epoch + 1}")
            break
    return model
