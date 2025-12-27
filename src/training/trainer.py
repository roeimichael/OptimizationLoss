import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import numpy as np

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
        global_counts: Dict mapping class_id -> count
        local_counts: Dict mapping course_id -> {class_id: count}
    """
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)

        # Global counts
        global_counts = {}
        for class_id in range(3):
            count = (test_preds == class_id).sum().item()
            global_counts[class_id] = count

        # Local counts per course
        unique_groups = torch.unique(group_ids_test)
        local_counts = {}
        for group_id in unique_groups:
            group_id_val = group_id.item()
            group_mask = (group_ids_test == group_id_val)
            group_preds = test_preds[group_mask]

            course_counts = {}
            for class_id in range(3):
                count = (group_preds == class_id).sum().item()
                course_counts[class_id] = count
            local_counts[group_id_val] = course_counts

    model.train()
    return global_counts, local_counts


def print_progress(epoch, avg_global, avg_local, avg_ce, avg_loss,
                   global_counts, local_counts, criterion_constraint):
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
    print("GLOBAL CONSTRAINTS vs PREDICTIONS")
    print(f"{'─'*80}")
    print(f"{'Class':<15} {'Constraint':<15} {'Predicted':<15} {'Status':<15}")
    print(f"{'─'*80}")

    g_cons = criterion_constraint.global_constraints.cpu().numpy()
    for class_id in range(3):
        class_name = ['Dropout', 'Enrolled', 'Graduate'][class_id]
        constraint_val = g_cons[class_id]
        predicted = global_counts[class_id]

        if constraint_val > 1e9:
            constraint_str = "None (unconstrained)"
            status = "N/A"
        else:
            constraint_str = f"{int(constraint_val)}"
            if predicted <= constraint_val:
                status = "✓ OK"
            else:
                excess = predicted - constraint_val
                status = f"✗ Over by {int(excess)}"

        print(f"{class_name:<15} {constraint_str:<15} {predicted:<15} {status:<15}")

    print(f"{'─'*80}")
    print(f"{'Total':<15} {'':<15} {sum(global_counts.values()):<15}")

    # Local constraints vs predictions
    print(f"\n{'─'*80}")
    print("LOCAL CONSTRAINTS vs PREDICTIONS (Per Course)")
    print(f"{'─'*80}")

    violations = []
    satisfactions = []

    for group_id in sorted(local_counts.keys()):
        buffer_name = f'local_constraint_{group_id}'
        if hasattr(criterion_constraint, buffer_name):
            l_cons = getattr(criterion_constraint, buffer_name).cpu().numpy()
            preds = local_counts[group_id]

            has_violation = False
            course_info = f"Course {group_id}: "
            details = []

            for class_id in range(3):
                class_name = ['Drop', 'Enrl', 'Grad'][class_id]
                constraint_val = l_cons[class_id]
                predicted = preds[class_id]

                if constraint_val > 1e9:
                    continue

                if predicted > constraint_val:
                    has_violation = True
                    excess = predicted - constraint_val
                    details.append(f"{class_name}:{int(predicted)}/{int(constraint_val)} (✗+{int(excess)})")
                else:
                    details.append(f"{class_name}:{int(predicted)}/{int(constraint_val)} (✓)")

            if details:
                course_info += ", ".join(details)
                if has_violation:
                    violations.append(course_info)
                else:
                    satisfactions.append(course_info)

    if violations:
        print("Courses with VIOLATIONS:")
        for v in violations:
            print(f"  {v}")

    if satisfactions and len(satisfactions) <= 10:
        print("\nCourses SATISFYING constraints:")
        for s in satisfactions:
            print(f"  {s}")
    elif satisfactions:
        print(f"\nCourses SATISFYING constraints: {len(satisfactions)} courses (all OK)")

    print(f"{'='*80}\n")


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
        'lambda_step': 0.1,
        'lambda_max': 100.0
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
    lambda_max = state['lambda_max']

    # Increase global lambda if constraint violated
    if avg_global > threshold and state['current_lambda_global'] < lambda_max:
        state['current_lambda_global'] = min(
            state['current_lambda_global'] + lambda_step, lambda_max
        )
        criterion_constraint.set_lambda(lambda_global=state['current_lambda_global'])
        lambda_updated = True

    # Increase local lambda if constraint violated
    if avg_local > threshold and state['current_lambda_local'] < lambda_max:
        state['current_lambda_local'] = min(
            state['current_lambda_local'] + lambda_step, lambda_max
        )
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

    Returns:
        avg_loss: Average total loss for the epoch
        avg_ce: Average cross-entropy loss
        avg_global: Average global constraint loss
        avg_local: Average local constraint loss
    """
    model.train()
    epoch_loss_total = 0
    epoch_loss_ce = 0
    epoch_loss_global = 0
    epoch_loss_local = 0

    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        # Supervised loss on training data
        train_logits = model(batch_features)
        loss_ce = criterion_ce(train_logits, batch_labels)

        # Constraint loss on test data (transductive)
        test_logits = model(X_test_tensor)
        loss_constraint, _, loss_global, loss_local = criterion_constraint(
            test_logits, y_true=None, group_ids=group_ids_test
        )

        # Total loss
        loss_total = loss_ce + loss_constraint

        # Optimization step
        loss_total.backward()
        optimizer.step()

        # Accumulate losses
        epoch_loss_total += loss_total.item()
        epoch_loss_ce += loss_ce.item()
        epoch_loss_global += loss_global.item()
        epoch_loss_local += loss_local.item()

    # Compute averages
    num_batches = len(train_loader)
    avg_loss = epoch_loss_total / num_batches
    avg_ce = epoch_loss_ce / num_batches
    avg_global = epoch_loss_global / num_batches
    avg_local = epoch_loss_local / num_batches

    return avg_loss, avg_ce, avg_global, avg_local


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
    for class_id in range(3):
        class_name = ['Dropout', 'Enrolled', 'Graduate'][class_id]
        constraint_val = g_cons[class_id]
        predicted = global_class_counts[class_id]

        if constraint_val > 1e9:
            print(f"  {class_name}: {predicted} (unconstrained)")
        else:
            print(f"  {class_name}: {predicted} ≤ {int(constraint_val)} ✓")

    print(f"{'='*80}\n")
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

    # Print training start
    print("\n" + "="*80)
    print("Starting Training with Adaptive Lambda Weights")
    print(f"Initial: λ_global={state['current_lambda_global']:.2f}, "
          f"λ_local={state['current_lambda_local']:.2f}")
    print(f"Lambda adjustment: +{state['lambda_step']} per epoch when constraints violated")
    print("="*80 + "\n")

    # Step 4: Main training loop
    for epoch in range(epochs):
        # Train one epoch
        avg_loss, avg_ce, avg_global, avg_local = train_single_epoch(
            model, train_loader, criterion_ce, criterion_constraint,
            optimizer, X_test_tensor, group_ids_test, device
        )

        # Adaptive lambda adjustment
        update_lambda_weights(state, avg_global, avg_local, criterion_constraint)

        # Track and log progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            # Compute prediction statistics
            global_counts, local_counts = compute_prediction_statistics(
                model, X_test_tensor, group_ids_test
            )

            # Update history
            update_training_history(
                history, epoch, avg_loss, avg_ce, avg_global, avg_local,
                state['current_lambda_global'], state['current_lambda_local'],
                global_counts, local_counts
            )

            # Print progress
            print_progress(epoch, avg_global, avg_local, avg_ce, avg_loss,
                          global_counts, local_counts, criterion_constraint)

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
