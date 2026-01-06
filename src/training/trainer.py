import os
import csv
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.models import get_model
from src.losses import MulticlassTransductiveLoss
from .metrics import compute_train_accuracy, compute_prediction_statistics

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

    print("\n" + "="*80)
    print("WARMUP TRAINING")
    print("="*80)
    print(f"Epochs: {warmup_epochs}")
    print(f"Pure cross-entropy (no constraint pressure)")
    print("="*80 + "\n")

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
    print(f"\nWarmup complete: {warmup_epochs} epochs")
    return model, scaler

def log_progress_to_csv(csv_path: str, epoch: int, avg_global: float, avg_local: float, avg_ce: float,
                        global_counts: Dict[int, int], local_counts: Dict[int, Dict[int, int]],
                        global_soft_counts: Dict[int, float], local_soft_counts: Dict[int, Dict[int, float]],
                        lambda_global: float, lambda_local: float, global_constraints: list,
                        global_satisfied: bool, local_satisfied: bool, train_acc: float,
                        tracked_course_id: int = 1) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = [
                'Epoch', 'Train_Acc', 'L_pred_CE', 'L_target_Global', 'L_feat_Local',
                'Lambda_Global', 'Lambda_Local', 'Global_Satisfied', 'Local_Satisfied',
                'Limit_Dropout', 'Limit_Enrolled', 'Limit_Graduate',
                'Hard_Dropout', 'Hard_Enrolled', 'Hard_Graduate',
                'Soft_Dropout', 'Soft_Enrolled', 'Soft_Graduate',
                'Excess_Dropout', 'Excess_Enrolled', 'Excess_Graduate',
                'Course_ID', 'Course_Hard_Dropout', 'Course_Hard_Enrolled', 'Course_Hard_Graduate',
                'Course_Soft_Dropout', 'Course_Soft_Enrolled', 'Course_Soft_Graduate'
            ]
            writer.writerow(header)

        excess_dropout = max(0, global_soft_counts[0] - global_constraints[0])
        excess_enrolled = max(0, global_soft_counts[1] - global_constraints[1])
        excess_graduate = max(0, global_soft_counts[2] - global_constraints[2]) if global_constraints[2] < 1e9 else 0

        course_hard = [0, 0, 0]
        course_soft = [0.0, 0.0, 0.0]
        if tracked_course_id in local_counts:
            course_hard = [local_counts[tracked_course_id][i] for i in range(3)]
            course_soft = [local_soft_counts[tracked_course_id][i] for i in range(3)]

        row = [
            epoch + 1,
            f"{train_acc:.4f}",
            f"{avg_ce:.6f}", f"{avg_global:.6f}", f"{avg_local:.6f}",
            f"{lambda_global:.2f}", f"{lambda_local:.2f}",
            1 if global_satisfied else 0, 1 if local_satisfied else 0,
            int(global_constraints[0]) if global_constraints[0] < 1e9 else 'inf',
            int(global_constraints[1]) if global_constraints[1] < 1e9 else 'inf',
            int(global_constraints[2]) if global_constraints[2] < 1e9 else 'inf',
            global_counts[0], global_counts[1], global_counts[2],
            f"{global_soft_counts[0]:.2f}", f"{global_soft_counts[1]:.2f}", f"{global_soft_counts[2]:.2f}",
            f"{excess_dropout:.2f}", f"{excess_enrolled:.2f}", f"{excess_graduate:.2f}",
            tracked_course_id,
            course_hard[0], course_hard[1], course_hard[2],
            f"{course_soft[0]:.2f}", f"{course_soft[1]:.2f}", f"{course_soft[2]:.2f}"
        ]
        writer.writerow(row)

def print_progress(epoch: int, avg_ce: float, avg_global: float, avg_local: float,
                   lambda_global: float, lambda_local: float, train_acc: float,
                   global_counts: Dict[int, int], global_soft_counts: Dict[int, float],
                   global_constraints: list, global_satisfied: bool, local_satisfied: bool) -> None:
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}")
    print(f"{'='*80}")
    print(f"Train Accuracy:     {train_acc:.4f}")
    print(f"L_target (Global):  {avg_global:.6f}")
    print(f"L_feat (Local):     {avg_local:.6f}")
    print(f"L_pred (CE):        {avg_ce:.6f}")
    print(f"\n{'-'*80}")
    print("GLOBAL CONSTRAINTS")
    print(f"{'-'*80}")
    print(f"{'Class':<12} {'Limit':<8} {'Hard':<8} {'Soft':<10} {'Excess':<10} {'Status':<15}")
    print(f"{'-'*80}")

    class_names = ['Dropout', 'Enrolled', 'Graduate']
    for idx, class_name in enumerate(class_names):
        limit = int(global_constraints[idx]) if global_constraints[idx] < 1e9 else 'inf'
        hard = global_counts[idx]
        soft = global_soft_counts[idx]
        excess = max(0, soft - global_constraints[idx]) if global_constraints[idx] < 1e9 else 0

        if limit == 'inf':
            status = "N/A"
        elif excess == 0:
            status = "OK"
        else:
            status = f"Over by {excess:.1f}"

        print(f"{class_name:<12} {str(limit):<8} {hard:<8} {soft:<10.2f} {excess:<10.2f} {status:<15}")

    print(f"{'-'*80}")
    total_hard = sum(global_counts.values())
    total_soft = sum(global_soft_counts.values())
    print(f"{'Total':<12} {'':<8} {total_hard:<8} {total_soft:<10.2f}")
    print(f"\nLambda Weights: lambda_global={lambda_global:.2f}, lambda_local={lambda_local:.2f}")
    constraint_global = "Satisfied" if global_satisfied else "Violated"
    constraint_local = "Satisfied" if local_satisfied else "Violated"
    print(f"Constraint Status: Global={constraint_global}, Local={constraint_local}")
    print(f"{'='*80}\n")

def train_with_constraints(model: nn.Module, scaler: StandardScaler, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, groups_test: pd.Series, global_constraint: list,
                           local_constraint: Dict[int, list], lr: float, batch_size: int, epochs: int,
                           warmup_epochs: int, lambda_global: float, lambda_local: float,
                           constraint_threshold: float, lambda_step: float, device: torch.device,
                           experiment_path: str) -> nn.Module:
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

    csv_log_path = Path(experiment_path) / 'training_log.csv'

    print("\n" + "="*80)
    print("CONSTRAINT OPTIMIZATION TRAINING")
    print("="*80)
    print(f"Epochs: {warmup_epochs} -> {epochs}")
    print(f"Initial Lambda: global={lambda_global:.2f}, local={lambda_local:.2f}")
    print(f"Lambda Step: {lambda_step}")
    print(f"Constraint Threshold: {constraint_threshold}")
    print(f"Log: {csv_log_path}")
    print("="*80 + "\n")

    for epoch in range(warmup_epochs, epochs):
        model.train()
        epoch_loss_ce = 0.0
        num_batches = len(train_loader)

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            train_logits = model(batch_features)
            loss_ce = criterion_ce(train_logits, batch_labels)

            model.eval()
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
        model.train()

        avg_ce = epoch_loss_ce / num_batches
        avg_global = loss_global.item()
        avg_local = loss_local.item()

        if avg_global > constraint_threshold:
            new_lambda_global = criterion_constraint.lambda_global + lambda_step
            criterion_constraint.set_lambda(lambda_global=new_lambda_global)

        if avg_local > constraint_threshold:
            new_lambda_local = criterion_constraint.lambda_local + lambda_step
            criterion_constraint.set_lambda(lambda_local=new_lambda_local)

        if (epoch + 1) % 3 == 0 or (epoch + 1) == warmup_epochs + 1:
            train_acc = compute_train_accuracy(model, train_loader, device)
            global_counts, local_counts, global_soft_counts, local_soft_counts = compute_prediction_statistics(
                model, X_test_tensor, group_ids_test
            )

            log_progress_to_csv(
                str(csv_log_path), epoch, avg_global, avg_local, avg_ce,
                global_counts, local_counts, global_soft_counts, local_soft_counts,
                criterion_constraint.lambda_global, criterion_constraint.lambda_local,
                global_constraint,
                criterion_constraint.global_constraints_satisfied,
                criterion_constraint.local_constraints_satisfied,
                train_acc
            )

            print_progress(
                epoch, avg_ce, avg_global, avg_local,
                criterion_constraint.lambda_global, criterion_constraint.lambda_local, train_acc,
                global_counts, global_soft_counts, global_constraint,
                criterion_constraint.global_constraints_satisfied,
                criterion_constraint.local_constraints_satisfied
            )

        if criterion_constraint.global_constraints_satisfied and criterion_constraint.local_constraints_satisfied:
            print(f"\n{'='*80}")
            print(f"ALL CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
            print(f"{'='*80}\n")
            break

    return model
