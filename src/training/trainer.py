import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import numpy as np
import csv
import os

from src.models import NeuralNetClassifier
from src.losses import MulticlassTransductiveLoss
from src.utils import create_all_visualizations


# =============================================================================
# Helper Functions for Statistics and Logging
# =============================================================================

def compute_prediction_statistics(model, X_test_tensor, group_ids_test):
    """
    Compute prediction statistics for tracking.

    Args:
        model: Neural network model
        X_test_tensor: Test features tensor
        group_ids_test: Course/group IDs for test samples

    Returns:
        global_counts: Dict mapping class_id -> hard count
        local_counts: Dict mapping course_id -> {class_id: hard count}
        global_soft_counts: Dict mapping class_id -> soft count (sum of probabilities)
        local_soft_counts: Dict mapping course_id -> {class_id: soft count}
    """
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)
        test_proba = torch.nn.functional.softmax(test_logits, dim=1)

        # Global hard counts
        global_counts = {}
        for class_id in range(3):
            count = (test_preds == class_id).sum().item()
            global_counts[class_id] = count

        # Global soft counts (sum of probabilities)
        global_soft_counts = {}
        for class_id in range(3):
            soft_count = test_proba[:, class_id].sum().item()
            global_soft_counts[class_id] = soft_count

        # Local hard counts per course
        unique_groups = torch.unique(group_ids_test)
        local_counts = {}
        local_soft_counts = {}

        for group_id in unique_groups:
            group_id_val = group_id.item()
            group_mask = (group_ids_test == group_id_val)
            group_preds = test_preds[group_mask]
            group_proba = test_proba[group_mask]

            course_counts = {}
            course_soft_counts = {}
            for class_id in range(3):
                # Hard count
                count = (group_preds == class_id).sum().item()
                course_counts[class_id] = count

                # Soft count
                soft_count = group_proba[:, class_id].sum().item()
                course_soft_counts[class_id] = soft_count

            local_counts[group_id_val] = course_counts
            local_soft_counts[group_id_val] = course_soft_counts

    model.train()
    return global_counts, local_counts, global_soft_counts, local_soft_counts


def print_progress(epoch, avg_global, avg_local, avg_ce, avg_loss,
                   global_counts, local_counts, global_soft_counts, local_soft_counts,
                   criterion_constraint):
    """
    Print detailed progress every N epochs.
    """
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}")
    print(f"{'='*80}")
    print(f"L_target (Global):  {avg_global:.6f}")
    print(f"L_feat (Local):     {avg_local:.6f}")
    print(f"L_pred (CE):        {avg_ce:.6f}")
    print(f"L_total:            {avg_loss:.6f}")

    # Global constraints vs predictions
    print(f"\n{'─'*80}")
    print("GLOBAL CONSTRAINTS vs PREDICTIONS (Hard vs Soft)")
    print(f"{'─'*80}")
    print(f"{'Class':<12} {'Limit':<8} {'Hard':<8} {'Soft':<10} {'Diff':<8} {'Status':<15}")
    print(f"{'─'*80}")

    g_cons = criterion_constraint.global_constraints.cpu().numpy()
    for class_id in range(3):
        class_name = ['Dropout', 'Enrolled', 'Graduate'][class_id]
        constraint_val = g_cons[class_id]
        hard_pred = global_counts[class_id]
        soft_pred = global_soft_counts[class_id]
        diff = soft_pred - hard_pred

        if constraint_val > 1e9:
            limit_str = "∞"
            status = "N/A"
        else:
            limit_str = f"{int(constraint_val)}"
            # Check soft predictions (what loss function sees)
            if soft_pred <= constraint_val:
                status = "✓ OK"
            else:
                excess = soft_pred - constraint_val
                status = f"✗ Over by {excess:.1f}"

        print(f"{class_name:<12} {limit_str:<8} {hard_pred:<8} {soft_pred:<10.2f} {diff:<8.2f} {status:<15}")

    print(f"{'─'*80}")
    print(f"{'Total':<12} {'':<8} {sum(global_counts.values()):<8} {sum(global_soft_counts.values()):<10.2f}")

    # Local constraints vs predictions (TABLE FORMAT)
    print(f"\n{'─'*80}")
    print("LOCAL CONSTRAINTS vs PREDICTIONS (Per Course)")
    print(f"{'─'*80}")
    print(f"{'Course':<8} {'Class':<10} {'Hard':<6} {'Soft':<8} {'Limit':<6} {'Excess':<8} {'Status':<10}")
    print(f"{'─'*80}")

    violations_count = 0
    satisfactions_count = 0

    for group_id in sorted(local_counts.keys()):
        buffer_name = f'local_constraint_{group_id}'
        if hasattr(criterion_constraint, buffer_name):
            l_cons = getattr(criterion_constraint, buffer_name).cpu().numpy()
            hard_preds = local_counts[group_id]
            soft_preds = local_soft_counts[group_id]

            # Track if this course has any violations
            course_has_violation = False
            course_rows = []

            for class_id in range(3):
                class_name = ['Dropout', 'Enrolled', 'Graduate'][class_id]
                constraint_val = l_cons[class_id]
                hard_count = hard_preds[class_id]
                soft_count = soft_preds[class_id]

                if constraint_val > 1e9:
                    continue

                # Check soft predictions (what loss sees)
                if soft_count > constraint_val:
                    excess = soft_count - constraint_val
                    status = f"✗ +{excess:.1f}"
                    course_has_violation = True
                else:
                    excess = 0.0
                    status = "✓ OK"

                course_rows.append((class_name, hard_count, soft_count, int(constraint_val), excess, status))

            # Print rows for this course
            if course_rows:
                for idx, (class_name, hard, soft, limit, excess, status) in enumerate(course_rows):
                    if idx == 0:
                        print(f"{group_id:<8} {class_name:<10} {hard:<6} {soft:<8.2f} {limit:<6} {excess:<8.2f} {status:<10}")
                    else:
                        print(f"{'':<8} {class_name:<10} {hard:<6} {soft:<8.2f} {limit:<6} {excess:<8.2f} {status:<10}")

                # Add separator between courses for readability
                print(f"{'─'*80}")

                if course_has_violation:
                    violations_count += 1
                else:
                    satisfactions_count += 1

    print(f"\nSummary: {violations_count} courses with violations, {satisfactions_count} courses satisfied")
    print(f"{'='*80}\n")


def log_progress_to_csv(csv_path, epoch, avg_global, avg_local, avg_ce, avg_loss,
                        global_counts, local_counts, global_soft_counts, local_soft_counts,
                        lambda_global, lambda_local):
    """
    Log training progress to CSV file every 50 epochs.

    Logs: epoch, losses, lambda values, and predictions (hard and soft) for each class.
    """
    # Check if file exists to determine if we need to write header
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header if file doesn't exist
        if not file_exists:
            header = [
                'Epoch',
                'L_total', 'L_pred_CE', 'L_target_Global', 'L_feat_Local',
                'Lambda_Global', 'Lambda_Local',
                'Hard_Dropout', 'Hard_Enrolled', 'Hard_Graduate',
                'Soft_Dropout', 'Soft_Enrolled', 'Soft_Graduate'
            ]

            # Add per-course columns (optional - could be large)
            # For now, just include global predictions

            writer.writerow(header)

        # Write data row
        row = [
            epoch + 1,
            f"{avg_loss:.6f}",
            f"{avg_ce:.6f}",
            f"{avg_global:.6f}",
            f"{avg_local:.6f}",
            f"{lambda_global:.2f}",
            f"{lambda_local:.2f}",
            global_counts[0],  # Hard Dropout
            global_counts[1],  # Hard Enrolled
            global_counts[2],  # Hard Graduate
            f"{global_soft_counts[0]:.2f}",  # Soft Dropout
            f"{global_soft_counts[1]:.2f}",  # Soft Enrolled
            f"{global_soft_counts[2]:.2f}"   # Soft Graduate
        ]

        writer.writerow(row)


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_training_data(X_train, y_train, X_test, groups_test, batch_size, device):
    """
    Prepare and preprocess data for training.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (for transductive learning)
        groups_test: Course IDs for test samples
        batch_size: Batch size for training
        device: PyTorch device

    Returns:
        train_loader: DataLoader for training
        X_test_tensor: Test features tensor on device
        group_ids_test: Test group IDs tensor on device
        scaler: Fitted StandardScaler
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels if needed
    if y_train.dtype == 'O' or isinstance(y_train.iloc[0], str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
    else:
        y_train_encoded = y_train.values

    # Create training dataset and loader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train_encoded)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare test tensors (for transductive learning)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    group_ids_test = torch.LongTensor(groups_test.values).to(device)

    return train_loader, X_test_tensor, group_ids_test, scaler


# =============================================================================
# Model and Optimizer Initialization
# =============================================================================

def initialize_model_and_optimizer(input_dim, hidden_dims, dropout, lr, device,
                                    global_constraint, local_constraint,
                                    lambda_global, lambda_local):
    """
    Initialize model, loss functions, and optimizer.

    Returns:
        model: Neural network model
        criterion_ce: Cross-entropy loss
        criterion_constraint: Transductive constraint loss
        optimizer: Adam optimizer
    """
    # Initialize model
    model = NeuralNetClassifier(
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

    return model, criterion_ce, criterion_constraint, optimizer


# =============================================================================
# Training State Management
# =============================================================================

def initialize_training_state(lambda_global, lambda_local):
    """
    Initialize training state variables and history tracking.

    Returns:
        state: Dictionary containing all training state variables
        history: Dictionary for tracking training metrics
    """
    state = {
        'best_loss': float('inf'),
        'best_model_state': None,
        'patience_counter': 0,
        'constraint_threshold': 1e-6,
        'current_lambda_global': lambda_global,
        'current_lambda_local': lambda_local,
        'lambda_step': 0.1
    }

    history = {
        'epochs': [],
        'loss_total': [],
        'loss_ce': [],
        'loss_global': [],
        'loss_local': [],
        'lambda_global': [],
        'lambda_local': [],
        'global_predictions': [],
        'local_predictions': []
    }

    return state, history


def update_lambda_weights(state, avg_global, avg_local, criterion_constraint):
    """
    Adaptively adjust lambda weights based on constraint violations.

    Lambdas increase unbounded until constraints are satisfied.

    Args:
        state: Training state dictionary
        avg_global: Average global constraint loss
        avg_local: Average local constraint loss
        criterion_constraint: Constraint loss function

    Returns:
        lambda_updated: Boolean indicating if lambdas were updated
    """
    lambda_updated = False
    threshold = state['constraint_threshold']
    lambda_step = state['lambda_step']

    # Increase global lambda if constraint violated (no cap)
    if avg_global > threshold:
        state['current_lambda_global'] += lambda_step
        criterion_constraint.set_lambda(lambda_global=state['current_lambda_global'])
        lambda_updated = True

    # Increase local lambda if constraint violated (no cap)
    if avg_local > threshold:
        state['current_lambda_local'] += lambda_step
        criterion_constraint.set_lambda(lambda_local=state['current_lambda_local'])
        lambda_updated = True

    return lambda_updated


# =============================================================================
# Training Loop Components
# =============================================================================

def train_single_epoch(model, train_loader, criterion_ce, criterion_constraint,
                       optimizer, X_test_tensor, group_ids_test, device):
    """
    Execute one epoch of training.

    Training pattern:
    1. Loop through training batches, optimize CE loss
    2. Compute constraint loss ONCE on test set per epoch
    3. Backprop constraint loss and update weights
    4. Compute constraint loss again for accurate reporting

    Returns:
        avg_ce: Average cross-entropy loss
        loss_global: Global constraint loss (AFTER weight update)
        loss_local: Local constraint loss (AFTER weight update)
    """
    model.train()
    epoch_loss_ce = 0

    # Step 1: Train on supervised batches
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        # Supervised loss on training data
        train_logits = model(batch_features)
        loss_ce = criterion_ce(train_logits, batch_labels)

        # Backprop and update
        loss_ce.backward()
        optimizer.step()

        # Accumulate CE loss
        epoch_loss_ce += loss_ce.item()

    # Average CE loss
    num_batches = len(train_loader)
    avg_ce = epoch_loss_ce / num_batches

    # Step 2: Compute constraint loss ONCE on test set
    optimizer.zero_grad()

    test_logits = model(X_test_tensor)
    loss_constraint, _, loss_global, loss_local = criterion_constraint(
        test_logits, y_true=None, group_ids=group_ids_test
    )

    # Backprop constraint loss and update
    loss_constraint.backward()
    optimizer.step()

    # Step 3: Recompute constraint loss for accurate reporting (after weight update)
    # IMPORTANT: Use eval mode to match prediction statistics computation
    model.eval()
    with torch.no_grad():
        test_logits_final = model(X_test_tensor)
        _, _, loss_global_final, loss_local_final = criterion_constraint(
            test_logits_final, y_true=None, group_ids=group_ids_test
        )
    model.train()

    return avg_ce, loss_global_final.item(), loss_local_final.item()


def update_training_history(history, epoch, avg_loss, avg_ce, avg_global, avg_local,
                            current_lambda_global, current_lambda_local,
                            global_counts, local_counts):
    """
    Update training history with current epoch metrics.
    """
    history['epochs'].append(epoch + 1)
    history['loss_total'].append(avg_loss)
    history['loss_ce'].append(avg_ce)
    history['loss_global'].append(avg_global)
    history['loss_local'].append(avg_local)
    history['lambda_global'].append(current_lambda_global)
    history['lambda_local'].append(current_lambda_local)
    history['global_predictions'].append(global_counts)
    history['local_predictions'].append(local_counts)


def check_constraint_satisfaction(avg_global, avg_local, constraint_threshold,
                                   epoch, model, X_test_tensor, criterion_constraint):
    """
    Check if constraints are satisfied and print final results.

    Returns:
        satisfied: Boolean indicating if constraints are satisfied
    """
    if avg_global >= constraint_threshold or avg_local >= constraint_threshold:
        return False

    # Constraints satisfied!
    print(f"\n{'='*80}")
    print(f"✓ CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
    print(f"{'='*80}")
    print(f"L_target (Global): {avg_global:.8f} < {constraint_threshold}")
    print(f"L_feat (Local):    {avg_local:.8f} < {constraint_threshold}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)

        # Global counts
        global_class_counts = {}
        for class_id in range(3):
            count = (test_preds == class_id).sum().item()
            global_class_counts[class_id] = count

    print(f"\nFinal Global Predictions vs Constraints:")
    g_cons = criterion_constraint.global_constraints.cpu().numpy()
    all_satisfied = True
    for class_id in range(3):
        class_name = ['Dropout', 'Enrolled', 'Graduate'][class_id]
        constraint_val = g_cons[class_id]
        predicted = global_class_counts[class_id]

        if constraint_val > 1e9:
            print(f"  {class_name}: {predicted} (unconstrained)")
        else:
            # Actually check if constraint is satisfied
            if predicted <= constraint_val:
                print(f"  {class_name}: {predicted} ≤ {int(constraint_val)} ✓")
            else:
                print(f"  {class_name}: {predicted} > {int(constraint_val)} ✗ VIOLATED!")
                all_satisfied = False

    print(f"{'='*80}\n")

    if not all_satisfied:
        print("WARNING: Constraints appear satisfied by loss but violated by hard predictions!")
        print("This indicates the loss computation or stopping criteria has a bug.\n")

    return True


# =============================================================================
# Main Training Function
# =============================================================================

def train_model_transductive(X_train, y_train, X_test, groups_test,
                             global_constraint, local_constraint,
                             lambda_global, lambda_local, hidden_dims, epochs,
                             batch_size, lr, dropout, patience, device):
    """
    Train model with transductive learning.

    Stops when BOTH L_target (global) and L_feat (local) constraints reach 0.
    Prints detailed progress every 50 epochs including predicted class counts.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (for transductive constraints)
        groups_test: Course IDs for test samples
        global_constraint: Global constraint limits
        local_constraint: Local (per-course) constraint limits
        lambda_global: Weight for global constraints
        lambda_local: Weight for local constraints
        hidden_dims: Hidden layer dimensions
        epochs: Maximum number of epochs
        batch_size: Batch size
        lr: Learning rate
        dropout: Dropout rate
        patience: Early stopping patience
        device: PyTorch device

    Returns:
        model: Trained model
        scaler: Fitted scaler
        training_time: Training duration in seconds
        history: Training metrics history
    """
    start_time = time.time()

    # Step 1: Prepare data
    train_loader, X_test_tensor, group_ids_test, scaler = prepare_training_data(
        X_train, y_train, X_test, groups_test, batch_size, device
    )

    # Step 2: Initialize model and optimizer
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

    # Step 3: Initialize training state
    state, history = initialize_training_state(lambda_global, lambda_local)

    # Create CSV log file path
    os.makedirs('./results', exist_ok=True)
    csv_log_path = f'./results/training_log_lambda{lambda_global}_{lambda_local}.csv'

    # Print training start
    print("\n" + "="*80)
    print("Starting Training with Adaptive Lambda Weights")
    print(f"Initial: λ_global={state['current_lambda_global']:.2f}, "
          f"λ_local={state['current_lambda_local']:.2f}")
    print(f"Lambda adjustment: +{state['lambda_step']} per epoch when constraints violated")
    print(f"Progress logged to: {csv_log_path}")
    print("="*80 + "\n")

    # Step 4: Main training loop
    for epoch in range(epochs):
        # Train one epoch
        avg_ce, avg_global, avg_local = train_single_epoch(
            model, train_loader, criterion_ce, criterion_constraint,
            optimizer, X_test_tensor, group_ids_test, device
        )

        # Compute total loss for tracking
        avg_loss = avg_ce + state['current_lambda_global'] * avg_global + state['current_lambda_local'] * avg_local

        # Adaptive lambda adjustment
        update_lambda_weights(state, avg_global, avg_local, criterion_constraint)

        # Track and log progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            # Compute prediction statistics (both hard and soft)
            global_counts, local_counts, global_soft_counts, local_soft_counts = compute_prediction_statistics(
                model, X_test_tensor, group_ids_test
            )

            # Update history
            update_training_history(
                history, epoch, avg_loss, avg_ce, avg_global, avg_local,
                state['current_lambda_global'], state['current_lambda_local'],
                global_counts, local_counts
            )

            # Print progress (show both hard and soft predictions)
            print_progress(epoch, avg_global, avg_local, avg_ce, avg_loss,
                          global_counts, local_counts, global_soft_counts, local_soft_counts,
                          criterion_constraint)

            # Log progress to CSV
            log_progress_to_csv(csv_log_path, epoch, avg_global, avg_local, avg_ce, avg_loss,
                               global_counts, local_counts, global_soft_counts, local_soft_counts,
                               state['current_lambda_global'], state['current_lambda_local'])

            print(f"Current Lambda Weights: λ_global={state['current_lambda_global']:.2f}, "
                  f"λ_local={state['current_lambda_local']:.2f}\n")

        # Track best model
        if avg_loss < state['best_loss']:
            state['best_loss'] = avg_loss
            state['best_model_state'] = model.state_dict().copy()
            state['patience_counter'] = 0
        else:
            state['patience_counter'] += 1

        # Check for early stopping (constraint satisfaction)
        if check_constraint_satisfaction(
            avg_global, avg_local, state['constraint_threshold'],
            epoch, model, X_test_tensor, criterion_constraint
        ):
            break

    # Step 5: Restore best model
    if state['best_model_state'] is not None:
        model.load_state_dict(state['best_model_state'])

    training_time = time.time() - start_time

    # Step 6: Create visualizations
    if len(history['epochs']) > 0:
        g_cons_np = criterion_constraint.global_constraints.cpu().numpy()

        local_cons_dict = {}
        if criterion_constraint.local_constraint_dict is not None:
            for group_id, buffer_name in criterion_constraint.local_constraint_dict.items():
                l_cons = getattr(criterion_constraint, buffer_name).cpu().numpy()
                local_cons_dict[group_id] = l_cons

        create_all_visualizations(history, g_cons_np, local_cons_dict)

    return model, scaler, training_time, history


# =============================================================================
# Inference Functions
# =============================================================================

def predict(model, scaler, X_test, device):
    """Make predictions on test data."""
    model.eval()
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions


def evaluate_accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)
