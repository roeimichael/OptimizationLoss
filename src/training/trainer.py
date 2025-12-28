"""
Core training logic for transductive learning with constraints.

This module contains the main training loop and model initialization
for neural network training with global and local constraints.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import numpy as np
import os

from src.models import NeuralNetClassifier
from src.losses import MulticlassTransductiveLoss
from src.utils import create_all_visualizations
from config.experiment_config import CONSTRAINT_THRESHOLD, LAMBDA_STEP, WARMUP_EPOCHS

# Import from new modules
from .metrics import compute_prediction_statistics, compute_train_accuracy, evaluate_accuracy
from .logging import (log_progress_to_csv, print_progress_from_csv, load_history_from_csv)


def prepare_training_data(X_train, y_train, X_test, groups_test, batch_size, device):
    """
    Prepare and scale training/test data for PyTorch.

    Args:
        X_train: Training features (pandas DataFrame)
        y_train: Training labels (pandas Series)
        X_test: Test features (pandas DataFrame)
        groups_test: Test group IDs (pandas Series)
        batch_size: Batch size for training
        device: torch device (cpu/cuda)

    Returns:
        tuple: (train_loader, X_test_tensor, group_ids_test, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
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

    return train_loader, X_test_tensor, group_ids_test, scaler


def initialize_model_and_optimizer(input_dim, hidden_dims, dropout, lr, device,
                                    global_constraint, local_constraint,
                                    lambda_global, lambda_local):
    """
    Initialize model, loss criteria, and optimizer.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
        lr: Learning rate
        device: torch device (cpu/cuda)
        global_constraint: Global constraint values
        local_constraint: Local constraint dict
        lambda_global: Initial global lambda weight
        lambda_local: Initial local lambda weight

    Returns:
        tuple: (model, criterion_ce, criterion_constraint, optimizer)
    """
    model = NeuralNetClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        n_classes=3,
        dropout=dropout
    ).to(device)

    criterion_ce = nn.CrossEntropyLoss()

    criterion_constraint = MulticlassTransductiveLoss(
        global_constraints=global_constraint,
        local_constraints=local_constraint,
        lambda_global=lambda_global,
        lambda_local=lambda_local,
        use_ce=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion_ce, criterion_constraint, optimizer


def update_lambda_weights(avg_global, avg_local, criterion_constraint, epoch):
    """
    Update lambda weights based on constraint violation.

    During warmup period, lambdas are forced to 0. After warmup, lambdas
    increase when constraints are violated.

    Args:
        avg_global: Average global constraint loss
        avg_local: Average local constraint loss
        criterion_constraint: Constraint loss criterion
        epoch: Current epoch number

    Returns:
        bool: True if lambda was updated, False otherwise
    """
    # During warmup, force lambdas to 0
    if epoch < WARMUP_EPOCHS:
        criterion_constraint.set_lambda(lambda_global=0.0, lambda_local=0.0)
        return False

    lambda_updated = False
    if avg_global > CONSTRAINT_THRESHOLD:
        new_lambda_global = criterion_constraint.lambda_global + LAMBDA_STEP
        criterion_constraint.set_lambda(lambda_global=new_lambda_global)
        lambda_updated = True

    if avg_local > CONSTRAINT_THRESHOLD:
        new_lambda_local = criterion_constraint.lambda_local + LAMBDA_STEP
        criterion_constraint.set_lambda(lambda_local=new_lambda_local)
        lambda_updated = True

    return lambda_updated


def train_single_epoch(model, train_loader, criterion_ce, criterion_constraint,
                       optimizer, X_test_tensor, group_ids_test, device):
    """
    Train for a single epoch with transductive constraints.

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion_ce: Cross-entropy loss criterion
        criterion_constraint: Constraint loss criterion
        optimizer: Model optimizer
        X_test_tensor: Test features tensor
        group_ids_test: Test group IDs tensor
        device: torch device (cpu/cuda)

    Returns:
        tuple: (avg_ce, avg_global, avg_local) - average losses for the epoch
    """
    model.train()
    epoch_loss_ce = 0.0
    num_batches = len(train_loader)

    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        # Supervised loss on labeled training data
        train_logits = model(batch_features)
        loss_ce = criterion_ce(train_logits, batch_labels)

        # Constraint loss on unlabeled test data (in eval mode for stable predictions)
        model.eval()
        test_logits = model(X_test_tensor)  # Gradients flow through to model weights
        model.train()

        _, _, loss_global, loss_local = criterion_constraint(
            test_logits, y_true=None, group_ids=group_ids_test
        )

        # Combined loss
        loss = loss_ce + criterion_constraint.lambda_global * loss_global + criterion_constraint.lambda_local * loss_local
        loss.backward()
        optimizer.step()

        epoch_loss_ce += loss_ce.item()

    # Compute constraint losses once at end of epoch for reporting
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

    return avg_ce, avg_global, avg_local


def train_model_transductive(X_train, y_train, X_test, groups_test,
                             global_constraint, local_constraint,
                             lambda_global, lambda_local, hidden_dims, epochs,
                             batch_size, lr, dropout, device,
                             constraint_dropout_pct, constraint_enrolled_pct):
    """
    Train a neural network with transductive learning constraints.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (unlabeled)
        groups_test: Test group IDs for local constraints
        global_constraint: Global constraint values
        local_constraint: Local constraint dict
        lambda_global: Initial global lambda weight
        lambda_local: Initial local lambda weight
        hidden_dims: List of hidden layer dimensions
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        dropout: Dropout probability
        device: torch device (cpu/cuda)
        constraint_dropout_pct: Dropout constraint percentage (for folder naming)
        constraint_enrolled_pct: Enrolled constraint percentage (for folder naming)

    Returns:
        tuple: (model, scaler, training_time, history)
    """
    start_time = time.time()

    # Prepare data
    train_loader, X_test_tensor, group_ids_test, scaler = prepare_training_data(
        X_train, y_train, X_test, groups_test, batch_size, device
    )

    # Initialize model and optimizer
    model, criterion_ce, criterion_constraint, optimizer = \
        initialize_model_and_optimizer(
            input_dim=X_train.shape[1],
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            device=device,
            global_constraint=global_constraint,
            local_constraint=local_constraint,
            lambda_global=lambda_global,
            lambda_local=lambda_local
        )

    # Setup experiment folder and logging
    num_local_courses = len(local_constraint) if local_constraint else 0
    experiment_folder = f'./results/constraints_{constraint_dropout_pct}_{constraint_enrolled_pct}'
    os.makedirs(experiment_folder, exist_ok=True)
    csv_log_path = f'{experiment_folder}/training_log.csv'

    # Print training configuration
    print("\n" + "="*80)
    print("Starting Training with Adaptive Lambda Weights")
    print(f"Constraint Configuration: Dropout≤{int(global_constraint[0])}, Enrolled≤{int(global_constraint[1])}, {num_local_courses} local courses")
    print(f"Initial: λ_global={criterion_constraint.lambda_global:.2f}, "
          f"λ_local={criterion_constraint.lambda_local:.2f}")
    print(f"Lambda adjustment: +{LAMBDA_STEP} per epoch when constraints violated")
    print(f"Results folder: {experiment_folder}")
    print(f"Progress logged to: {csv_log_path}")

    # Print training data distribution
    import pandas as pd
    train_labels_list = []
    for _, batch_labels in train_loader:
        train_labels_list.extend(batch_labels.numpy())
    train_labels_array = np.array(train_labels_list)
    unique, counts = np.unique(train_labels_array, return_counts=True)
    print(f"\nTraining Data Distribution:")
    for label, count in zip(unique, counts):
        class_name = ['Dropout', 'Enrolled', 'Graduate'][int(label)]
        print(f"  {class_name}: {count} ({count/len(train_labels_array)*100:.1f}%)")

    print(f"\nWarmup Period: First {WARMUP_EPOCHS} epochs will train with λ=0 (no constraint pressure)")
    print("="*80 + "\n")

    # Training loop
    for epoch in range(epochs):
        avg_ce, avg_global, avg_local = train_single_epoch(
            model, train_loader, criterion_ce, criterion_constraint,
            optimizer, X_test_tensor, group_ids_test, device
        )

        update_lambda_weights(avg_global, avg_local, criterion_constraint, epoch)

        # Log and print progress every 3 epochs
        if (epoch + 1) % 3 == 0:
            global_counts, local_counts, global_soft_counts, local_soft_counts = compute_prediction_statistics(
                model, X_test_tensor, group_ids_test
            )

            log_progress_to_csv(csv_log_path, epoch, avg_global, avg_local, avg_ce,
                               global_counts, local_counts, global_soft_counts, local_soft_counts,
                               criterion_constraint.lambda_global, criterion_constraint.lambda_local,
                               global_constraint,
                               criterion_constraint.global_constraints_satisfied,
                               criterion_constraint.local_constraints_satisfied)

            # Only print after warmup completes
            if epoch >= WARMUP_EPOCHS:
                print(f"\nDEBUG - Epoch {epoch + 1}:")
                print(f"  avg_ce={avg_ce:.6f}, avg_global={avg_global:.6f}, avg_local={avg_local:.6f}")
                print(f"  λ_global={criterion_constraint.lambda_global:.2f}, λ_local={criterion_constraint.lambda_local:.2f}")
                print(f"  Weighted: CE={avg_ce:.6f}, Global={criterion_constraint.lambda_global * avg_global:.6f}, Local={criterion_constraint.lambda_local * avg_local:.6f}")
                print(f"  Total weighted loss: {avg_ce + criterion_constraint.lambda_global * avg_global + criterion_constraint.lambda_local * avg_local:.6f}")

                # Check train accuracy
                train_acc = compute_train_accuracy(model, train_loader, device)
                print(f"  Train Accuracy: {train_acc:.4f}")

                print_progress_from_csv(csv_log_path, criterion_constraint)

        # Check for early stopping
        if criterion_constraint.global_constraints_satisfied and criterion_constraint.local_constraints_satisfied:
            print(f"\n{'='*80}")
            print(f"✓ ALL CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
            print(f"{'='*80}")
            print(f"Global constraints: ✓ Satisfied")
            print(f"Local constraints: ✓ Satisfied")
            print(f"{'='*80}\n")
            break

    training_time = time.time() - start_time

    # Load history from CSV and create visualizations
    history = load_history_from_csv(csv_log_path)
    if history is not None:
        g_cons_np = criterion_constraint.global_constraints.cpu().numpy()
        local_cons_dict = {}
        if criterion_constraint.local_constraint_dict is not None:
            for group_id, buffer_name in criterion_constraint.local_constraint_dict.items():
                l_cons = getattr(criterion_constraint, buffer_name).cpu().numpy()
                local_cons_dict[group_id] = l_cons
        create_all_visualizations(history, g_cons_np, local_cons_dict, output_dir=experiment_folder)

    return model, scaler, training_time, history


def predict(model, scaler, X_test, device):
    """
    Make predictions on test data.

    Args:
        model: Trained neural network model
        scaler: Fitted StandardScaler
        X_test: Test features
        device: torch device (cpu/cuda)

    Returns:
        np.ndarray: Predicted class labels
    """
    model.eval()
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions
