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
from config.experiment_config import CONSTRAINT_THRESHOLD, LAMBDA_STEP, WARMUP_EPOCHS

def compute_prediction_statistics(model, X_test_tensor, group_ids_test):
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)
        test_proba = torch.nn.functional.softmax(test_logits, dim=1)
        global_counts = {}
        for class_id in range(3):
            count = (test_preds == class_id).sum().item()
            global_counts[class_id] = count
        global_soft_counts = {}
        for class_id in range(3):
            soft_count = test_proba[:, class_id].sum().item()
            global_soft_counts[class_id] = soft_count
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
                count = (group_preds == class_id).sum().item()
                course_counts[class_id] = count
                soft_count = group_proba[:, class_id].sum().item()
                course_soft_counts[class_id] = soft_count
            local_counts[group_id_val] = course_counts
            local_soft_counts[group_id_val] = course_soft_counts
    model.train()
    return global_counts, local_counts, global_soft_counts, local_soft_counts

def print_progress(epoch, avg_global, avg_local, avg_ce,
                   global_counts, local_counts, global_soft_counts, local_soft_counts,
                   criterion_constraint):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}")
    print(f"{'='*80}")
    print(f"L_target (Global):  {avg_global:.6f}")
    print(f"L_feat (Local):     {avg_local:.6f}")
    print(f"L_pred (CE):        {avg_ce:.6f}")
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
            if soft_pred <= constraint_val:
                status = "✓ OK"
            else:
                excess = soft_pred - constraint_val
                status = f"✗ Over by {excess:.1f}"
        print(f"{class_name:<12} {limit_str:<8} {hard_pred:<8} {soft_pred:<10.2f} {diff:<8.2f} {status:<15}")
    print(f"{'─'*80}")
    print(f"{'Total':<12} {'':<8} {sum(global_counts.values()):<8} {sum(global_soft_counts.values()):<10.2f}")
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
            course_has_violation = False
            course_rows = []
            for class_id in range(3):
                class_name = ['Dropout', 'Enrolled', 'Graduate'][class_id]
                constraint_val = l_cons[class_id]
                hard_count = hard_preds[class_id]
                soft_count = soft_preds[class_id]
                if constraint_val > 1e9:
                    continue
                if soft_count > constraint_val:
                    excess = soft_count - constraint_val
                    status = f"✗ +{excess:.1f}"
                    course_has_violation = True
                else:
                    excess = 0.0
                    status = "✓ OK"
                course_rows.append((class_name, hard_count, soft_count, int(constraint_val), excess, status))
            if course_rows:
                for idx, (class_name, hard, soft, limit, excess, status) in enumerate(course_rows):
                    if idx == 0:
                        print(f"{group_id:<8} {class_name:<10} {hard:<6} {soft:<8.2f} {limit:<6} {excess:<8.2f} {status:<10}")
                    else:
                        print(f"{'':<8} {class_name:<10} {hard:<6} {soft:<8.2f} {limit:<6} {excess:<8.2f} {status:<10}")
                print(f"{'─'*80}")
                if course_has_violation:
                    violations_count += 1
                else:
                    satisfactions_count += 1
    print(f"\nSummary: {violations_count} courses with violations, {satisfactions_count} courses satisfied")
    print(f"{'='*80}\n")

def log_progress_to_csv(csv_path, epoch, avg_global, avg_local, avg_ce,
                        global_counts, local_counts, global_soft_counts, local_soft_counts,
                        lambda_global, lambda_local, global_constraints,
                        global_satisfied, local_satisfied):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = [
                'Epoch',
                'L_pred_CE', 'L_target_Global', 'L_feat_Local',
                'Lambda_Global', 'Lambda_Local',
                'Global_Satisfied', 'Local_Satisfied',
                'Limit_Dropout', 'Limit_Enrolled', 'Limit_Graduate',
                'Hard_Dropout', 'Hard_Enrolled', 'Hard_Graduate',
                'Soft_Dropout', 'Soft_Enrolled', 'Soft_Graduate',
                'Excess_Dropout', 'Excess_Enrolled', 'Excess_Graduate'
            ]
            writer.writerow(header)
        excess_dropout = max(0, global_soft_counts[0] - global_constraints[0])
        excess_enrolled = max(0, global_soft_counts[1] - global_constraints[1])
        excess_graduate = max(0, global_soft_counts[2] - global_constraints[2]) if global_constraints[2] < 1e9 else 0
        row = [
            epoch + 1,
            f"{avg_ce:.6f}",
            f"{avg_global:.6f}",
            f"{avg_local:.6f}",
            f"{lambda_global:.2f}",
            f"{lambda_local:.2f}",
            1 if global_satisfied else 0,
            1 if local_satisfied else 0,
            int(global_constraints[0]) if global_constraints[0] < 1e9 else 'inf',
            int(global_constraints[1]) if global_constraints[1] < 1e9 else 'inf',
            int(global_constraints[2]) if global_constraints[2] < 1e9 else 'inf',
            global_counts[0],
            global_counts[1],
            global_counts[2],
            f"{global_soft_counts[0]:.2f}",
            f"{global_soft_counts[1]:.2f}",
            f"{global_soft_counts[2]:.2f}",
            f"{excess_dropout:.2f}",
            f"{excess_enrolled:.2f}",
            f"{excess_graduate:.2f}"
        ]
        writer.writerow(row)

def prepare_training_data(X_train, y_train, X_test, groups_test, batch_size, device):
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

def read_last_csv_row(csv_path):
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        if len(rows) == 0:
            return None
        return rows[-1]

def print_progress_from_csv(csv_path, criterion_constraint):
    last_row = read_last_csv_row(csv_path)
    if last_row is None:
        return
    epoch = int(last_row['Epoch'])
    avg_ce = float(last_row['L_pred_CE'])
    avg_global = float(last_row['L_target_Global'])
    avg_local = float(last_row['L_feat_Local'])
    lambda_global = float(last_row['Lambda_Global'])
    lambda_local = float(last_row['Lambda_Local'])
    global_satisfied = int(last_row['Global_Satisfied']) == 1
    local_satisfied = int(last_row['Local_Satisfied']) == 1
    print(f"\n{'='*80}")
    print(f"Epoch {epoch}")
    print(f"{'='*80}")
    print(f"L_target (Global):  {avg_global:.6f}")
    print(f"L_feat (Local):     {avg_local:.6f}")
    print(f"L_pred (CE):        {avg_ce:.6f}")
    print(f"\n{'─'*80}")
    print("GLOBAL CONSTRAINTS")
    print(f"{'─'*80}")
    print(f"{'Class':<12} {'Limit':<8} {'Hard':<8} {'Soft':<10} {'Excess':<10} {'Status':<15}")
    print(f"{'─'*80}")
    g_cons = criterion_constraint.global_constraints.cpu().numpy()
    for idx, class_name in enumerate(['Dropout', 'Enrolled', 'Graduate']):
        limit = last_row[f'Limit_{class_name}']
        hard = int(last_row[f'Hard_{class_name}'])
        soft = float(last_row[f'Soft_{class_name}'])
        excess = float(last_row[f'Excess_{class_name}'])
        if limit == 'inf':
            status = "N/A"
        elif excess == 0:
            status = "✓ OK"
        else:
            status = f"✗ Over by {excess:.1f}"
        print(f"{class_name:<12} {limit:<8} {hard:<8} {soft:<10.2f} {excess:<10.2f} {status:<15}")
    print(f"{'─'*80}")
    print(f"{'Total':<12} {'':<8} {int(last_row['Hard_Dropout']) + int(last_row['Hard_Enrolled']) + int(last_row['Hard_Graduate']):<8} "
          f"{float(last_row['Soft_Dropout']) + float(last_row['Soft_Enrolled']) + float(last_row['Soft_Graduate']):<10.2f}")
    print(f"\nCurrent Lambda Weights: λ_global={lambda_global:.2f}, λ_local={lambda_local:.2f}")
    print(f"Constraint Status: Global={'✓' if global_satisfied else '✗'}, Local={'✓' if local_satisfied else '✗'}")
    print(f"{'='*80}\n")

def load_history_from_csv(csv_path):
    if not os.path.isfile(csv_path):
        return None
    import pandas as pd
    df = pd.read_csv(csv_path)
    history = {
        'epochs': df['Epoch'].tolist(),
        'loss_ce': df['L_pred_CE'].tolist(),
        'loss_global': df['L_target_Global'].tolist(),
        'loss_local': df['L_feat_Local'].tolist(),
        'lambda_global': df['Lambda_Global'].tolist(),
        'lambda_local': df['Lambda_Local'].tolist(),
        'global_predictions': [],
        'local_predictions': []
    }
    for _, row in df.iterrows():
        global_counts = {
            0: int(row['Hard_Dropout']),
            1: int(row['Hard_Enrolled']),
            2: int(row['Hard_Graduate'])
        }
        history['global_predictions'].append(global_counts)
        history['local_predictions'].append({})
    return history

def update_lambda_weights(avg_global, avg_local, criterion_constraint, epoch):
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
    model.train()
    epoch_loss_ce = 0.0
    num_batches = len(train_loader)
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        train_logits = model(batch_features)
        loss_ce = criterion_ce(train_logits, batch_labels)

        # Compute constraint loss in eval mode on full test set
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor)
        # Back to train mode for gradient computation
        test_logits.requires_grad_(True)
        model.train()

        _, _, loss_global, loss_local = criterion_constraint(
            test_logits, y_true=None, group_ids=group_ids_test
        )
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
    start_time = time.time()
    train_loader, X_test_tensor, group_ids_test, scaler = prepare_training_data(
        X_train, y_train, X_test, groups_test, batch_size, device
    )
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
    num_local_courses = len(local_constraint) if local_constraint else 0
    experiment_folder = f'./results/constraints_{constraint_dropout_pct}_{constraint_enrolled_pct}'
    os.makedirs(experiment_folder, exist_ok=True)
    csv_log_path = f'{experiment_folder}/training_log.csv'
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
    for epoch in range(epochs):
        avg_ce, avg_global, avg_local = train_single_epoch(
            model, train_loader, criterion_ce, criterion_constraint,
            optimizer, X_test_tensor, group_ids_test, device
        )
        update_lambda_weights(avg_global, avg_local, criterion_constraint, epoch)
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
            print(f"\nDEBUG - Epoch {epoch + 1}:" + (" [WARMUP]" if epoch < WARMUP_EPOCHS else ""))
            print(f"  avg_ce={avg_ce:.6f}, avg_global={avg_global:.6f}, avg_local={avg_local:.6f}")
            print(f"  λ_global={criterion_constraint.lambda_global:.2f}, λ_local={criterion_constraint.lambda_local:.2f}")
            print(f"  Weighted: CE={avg_ce:.6f}, Global={criterion_constraint.lambda_global * avg_global:.6f}, Local={criterion_constraint.lambda_local * avg_local:.6f}")
            print(f"  Total weighted loss: {avg_ce + criterion_constraint.lambda_global * avg_global + criterion_constraint.lambda_local * avg_local:.6f}")

            # Check train accuracy
            model.eval()
            with torch.no_grad():
                train_correct = 0
                train_total = 0
                for batch_features, batch_labels in train_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs, 1)
                    train_total += batch_labels.size(0)
                    train_correct += (predicted == batch_labels).sum().item()
                train_acc = train_correct / train_total
            model.train()
            print(f"  Train Accuracy: {train_acc:.4f}")

            print_progress_from_csv(csv_log_path, criterion_constraint)
        if criterion_constraint.global_constraints_satisfied and criterion_constraint.local_constraints_satisfied:
            print(f"\n{'='*80}")
            print(f"✓ ALL CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
            print(f"{'='*80}")
            print(f"Global constraints: ✓ Satisfied")
            print(f"Local constraints: ✓ Satisfied")
            print(f"{'='*80}\n")
            break
    training_time = time.time() - start_time
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
    model.eval()
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return predictions

def evaluate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
