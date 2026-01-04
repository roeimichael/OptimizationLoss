"""
Our Approach: Transductive Learning with Constraint-Based Optimization
Main implementation for the thesis methodology
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import numpy as np
import json
from pathlib import Path

from models import get_model
from src.losses import MulticlassTransductiveLoss
from src.training.metrics import compute_prediction_statistics, compute_train_accuracy, get_predictions_with_probabilities, compute_metrics
from src.training.logging import log_progress_to_csv, save_final_predictions, save_constraint_comparison, save_evaluation_metrics
from src.benchmark import greedy_constraint_selection
from utils.filesystem_manager import save_config_to_path, mark_experiment_complete


def train_with_our_approach(config, X_train, y_train, X_test, groups_test, y_test,
                            global_constraint, local_constraint):
    """
    Train model using our transductive constraint-based approach

    Args:
        config: Experiment configuration dictionary
        X_train, y_train: Training data
        X_test, groups_test, y_test: Test data with group assignments
        global_constraint: Global constraint targets
        local_constraint: Local (per-course) constraint targets

    Returns:
        dict: Results including metrics, training time, etc.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Extract hyperparameters from config
    hyperparams = config['hyperparams']
    lr = hyperparams.get('lr', 0.001)
    dropout = hyperparams.get('dropout', 0.3)
    batch_size = hyperparams.get('batch_size', 64)
    hidden_dims = hyperparams.get('hidden_dims', [128, 64])
    epochs = hyperparams.get('epochs', 10000)
    lambda_global = hyperparams.get('lambda_global', 0.01)
    lambda_local = hyperparams.get('lambda_local', 0.01)

    # Training stability parameters
    max_lambda_global = hyperparams.get('max_lambda_global', 0.5)
    max_lambda_local = hyperparams.get('max_lambda_local', 0.5)
    gradient_clip = hyperparams.get('gradient_clip', 1.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 250)
    constraint_threshold = hyperparams.get('constraint_threshold', 1e-6)
    lambda_step = hyperparams.get('lambda_step', 0.01)

    # Get experiment path
    experiment_path = config['experiment_path']

    start_time = time.time()

    # Prepare data
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

    # Initialize model
    input_dim = X_train.shape[1]
    model = get_model(
        config['model_name'],
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        n_classes=3,
        dropout=dropout
    ).to(device)

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_constraint = MulticlassTransductiveLoss(
        global_constraints=global_constraint,
        local_constraints=local_constraint,
        lambda_global=lambda_global,
        lambda_local=lambda_local,
        use_ce=False
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    # Setup logging
    csv_log_path = Path(experiment_path) / 'training_log.csv'

    print("\n" + "="*80)
    print("Starting Training with Our Approach (Transductive + Constraints)")
    print(f"Model: {config['model_name']}")
    print(f"Constraint: {config['constraint']}")
    print(f"Hyperparameters: LR={lr}, Dropout={dropout}, Batch={batch_size}")
    print(f"Lambda caps: {max_lambda_global}/{max_lambda_local}")
    print("="*80 + "\n")

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss_ce = 0.0

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

            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            epoch_loss_ce += loss_ce.item()

        # Update lambda weights
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor)
            _, _, loss_global, loss_local = criterion_constraint(
                test_logits, y_true=None, group_ids=group_ids_test
            )

        avg_global = loss_global.item()
        avg_local = loss_local.item()

        if epoch >= warmup_epochs:
            if avg_global > constraint_threshold:
                new_lambda_global = min(
                    criterion_constraint.lambda_global + lambda_step,
                    max_lambda_global
                )
                criterion_constraint.set_lambda(lambda_global=new_lambda_global)

            if avg_local > constraint_threshold:
                new_lambda_local = min(
                    criterion_constraint.lambda_local + lambda_step,
                    max_lambda_local
                )
                criterion_constraint.set_lambda(lambda_local=new_lambda_local)

        # Scheduler step
        if epoch >= warmup_epochs:
            total_loss = epoch_loss_ce / len(train_loader) + criterion_constraint.lambda_global * avg_global + criterion_constraint.lambda_local * avg_local
            scheduler.step(total_loss)

        # Logging every 10 epochs
        if (epoch + 1) % 10 == 0:
            train_acc = compute_train_accuracy(model, train_loader, device)
            print(f"Epoch {epoch + 1}: CE={epoch_loss_ce/len(train_loader):.4f}, "
                  f"Train Acc={train_acc:.4f}, "
                  f"Lambda=({criterion_constraint.lambda_global:.3f}, {criterion_constraint.lambda_local:.3f})")

        # Check if constraints satisfied
        if criterion_constraint.global_constraints_satisfied and criterion_constraint.local_constraints_satisfied:
            print(f"\nAll constraints satisfied at epoch {epoch + 1}!")
            break

    training_time = time.time() - start_time

    # Final evaluation
    model.eval()
    y_pred, y_proba = get_predictions_with_probabilities(model, X_test_tensor)
    y_true_np = y_test.values if hasattr(y_test, 'values') else y_test
    course_ids_np = groups_test.values if hasattr(groups_test, 'values') else groups_test

    # Save results
    save_final_predictions(
        Path(experiment_path) / 'final_predictions.csv',
        y_true_np, y_pred, y_proba, course_ids_np
    )

    metrics = compute_metrics(y_true_np, y_pred)
    save_evaluation_metrics(
        Path(experiment_path) / 'evaluation_metrics.csv',
        metrics
    )

    # Save config with results
    config['results'] = {
        'accuracy': float(metrics['accuracy']),
        'training_time': float(training_time),
        'final_lambda_global': float(criterion_constraint.lambda_global),
        'final_lambda_local': float(criterion_constraint.lambda_local)
    }
    save_config_to_path(config, experiment_path)

    # Mark experiment complete
    mark_experiment_complete(experiment_path)

    print(f"\nExperiment complete! Results saved to {experiment_path}")
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"Training Time: {training_time:.2f}s")

    return config['results']
