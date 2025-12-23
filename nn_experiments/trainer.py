import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import numpy as np

from model import NeuralNetClassifier
from transductive_loss import MulticlassTransductiveLoss


def train_model_transductive(X_train, y_train, X_test, groups_test,
                             global_constraint, local_constraint,
                             lambda_global, lambda_local, hidden_dims, epochs,
                             batch_size, lr, dropout, patience, device):
    """
    Train model with transductive learning.

    Stops when BOTH L_target (global) and L_feat (local) constraints reach 0.
    Prints detailed progress every 50 epochs including predicted class counts.
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

    model = NeuralNetClassifier(
        input_dim=X_train_scaled.shape[1],
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    start_time = time.time()
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Constraint satisfaction threshold
    constraint_threshold = 1e-6

    print("\n" + "="*80)
    print("Starting Training - Will stop when constraints are satisfied")
    print("="*80 + "\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss_total = 0
        epoch_loss_ce = 0
        epoch_loss_global = 0
        epoch_loss_local = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            train_logits = model(batch_features)
            loss_ce = criterion_ce(train_logits, batch_labels)

            test_logits = model(X_test_tensor)
            loss_constraint, _, loss_global, loss_local = criterion_constraint(
                test_logits, y_true=None, group_ids=group_ids_test
            )

            loss_total = loss_ce + loss_constraint

            loss_total.backward()
            optimizer.step()

            epoch_loss_total += loss_total.item()
            epoch_loss_ce += loss_ce.item()
            epoch_loss_global += loss_global.item()
            epoch_loss_local += loss_local.item()

        avg_loss = epoch_loss_total / len(train_loader)
        avg_ce = epoch_loss_ce / len(train_loader)
        avg_global = epoch_loss_global / len(train_loader)
        avg_local = epoch_loss_local / len(train_loader)

        scheduler.step(avg_loss)

        # Print detailed progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            # Get predicted class counts on test set
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test_tensor)
                test_preds = torch.argmax(test_logits, dim=1)

                # Count predictions per class (global)
                global_class_counts = {}
                for class_id in range(3):
                    count = (test_preds == class_id).sum().item()
                    global_class_counts[class_id] = count

                # Count predictions per course per class (local)
                unique_groups = torch.unique(group_ids_test)
                local_predictions = {}
                for group_id in unique_groups:
                    group_id_val = group_id.item()
                    group_mask = (group_ids_test == group_id_val)
                    group_preds = test_preds[group_mask]

                    course_counts = {}
                    for class_id in range(3):
                        count = (group_preds == class_id).sum().item()
                        course_counts[class_id] = count
                    local_predictions[group_id_val] = course_counts

            model.train()

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

            # Get global constraints from the loss function
            g_cons = criterion_constraint.global_constraints.cpu().numpy()
            for class_id in range(3):
                class_name = ['Dropout', 'Enrolled', 'Graduate'][class_id]
                constraint_val = g_cons[class_id]
                predicted = global_class_counts[class_id]

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
            print(f"{'Total':<15} {'':<15} {sum(global_class_counts.values()):<15}")

            # Local constraints vs predictions (per course)
            print(f"\n{'─'*80}")
            print("LOCAL CONSTRAINTS vs PREDICTIONS (Per Course)")
            print(f"{'─'*80}")

            # Get local constraints
            violations = []
            satisfactions = []

            for group_id in sorted(local_predictions.keys()):
                buffer_name = f'local_constraint_{group_id}'
                if hasattr(criterion_constraint, buffer_name):
                    l_cons = getattr(criterion_constraint, buffer_name).cpu().numpy()
                    preds = local_predictions[group_id]

                    # Check if any constraint is violated
                    has_violation = False
                    course_info = f"Course {group_id}: "
                    details = []

                    for class_id in range(3):
                        class_name = ['Drop', 'Enrl', 'Grad'][class_id]
                        constraint_val = l_cons[class_id]
                        predicted = preds[class_id]

                        if constraint_val > 1e9:
                            continue  # Skip unconstrained

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

            # Print violations first (these are the problem areas)
            if violations:
                print("Courses with VIOLATIONS:")
                for v in violations:
                    print(f"  {v}")

            # Print satisfactions (optional, can be hidden if too many)
            if satisfactions and len(satisfactions) <= 10:
                print("\nCourses SATISFYING constraints:")
                for s in satisfactions:
                    print(f"  {s}")
            elif satisfactions:
                print(f"\nCourses SATISFYING constraints: {len(satisfactions)} courses (all OK)")

            print(f"{'='*80}\n")

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping when BOTH constraints are satisfied
        if avg_global < constraint_threshold and avg_local < constraint_threshold:
            print(f"\n{'='*80}")
            print(f"✓ CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
            print(f"{'='*80}")
            print(f"L_target (Global): {avg_global:.8f} < {constraint_threshold}")
            print(f"L_feat (Local):    {avg_local:.8f} < {constraint_threshold}")

            # Final evaluation with full details
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
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_time = time.time() - start_time

    return model, scaler, training_time


def predict(model, scaler, X_test, device):
    model.eval()
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions


def evaluate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
