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

def print_progress(epoch, avg_global, avg_local, avg_ce, avg_loss,
                   global_counts, local_counts, global_soft_counts, local_soft_counts,
                   criterion_constraint):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch + 1}")
    print(f"{'='*80}")
    print(f"L_target (Global):  {avg_global:.6f}")
    print(f"L_feat (Local):     {avg_local:.6f}")
    print(f"L_pred (CE):        {avg_ce:.6f}")
    print(f"L_total:            {avg_loss:.6f}")
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

def log_progress_to_csv(csv_path, epoch, avg_global, avg_local, avg_ce, avg_loss,
                        global_counts, local_counts, global_soft_counts, local_soft_counts,
                        lambda_global, lambda_local, global_constraints):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = [
                'Epoch',
                'L_total', 'L_pred_CE', 'L_target_Global', 'L_feat_Local',
                'Lambda_Global', 'Lambda_Local',
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
            f"{avg_loss:.6f}",
            f"{avg_ce:.6f}",
            f"{avg_global:.6f}",
            f"{avg_local:.6f}",
            f"{lambda_global:.2f}",
            f"{lambda_local:.2f}",
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

def initialize_training_state(lambda_global, lambda_local):
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
    lambda_updated = False
    threshold = state['constraint_threshold']
    lambda_step = state['lambda_step']
    if avg_global > threshold:
        state['current_lambda_global'] += lambda_step
        criterion_constraint.set_lambda(lambda_global=state['current_lambda_global'])
        lambda_updated = True
    if avg_local > threshold:
        state['current_lambda_local'] += lambda_step
        criterion_constraint.set_lambda(lambda_local=state['current_lambda_local'])
        lambda_updated = True
    return lambda_updated

def train_single_epoch(model, train_loader, criterion_ce, criterion_constraint,
                       optimizer, X_test_tensor, group_ids_test, device):
    model.train()
    epoch_loss_ce = 0
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        train_logits = model(batch_features)
        loss_ce = criterion_ce(train_logits, batch_labels)
        loss_ce.backward()
        optimizer.step()
        epoch_loss_ce += loss_ce.item()
    num_batches = len(train_loader)
    avg_ce = epoch_loss_ce / num_batches
    optimizer.zero_grad()
    test_logits = model(X_test_tensor)
    loss_constraint, _, loss_global, loss_local = criterion_constraint(
        test_logits, y_true=None, group_ids=group_ids_test
    )
    loss_constraint.backward()
    optimizer.step()
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
    if avg_global >= constraint_threshold or avg_local >= constraint_threshold:
        return False
    print(f"\n{'='*80}")
    print(f"✓ CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
    print(f"{'='*80}")
    print(f"L_target (Global): {avg_global:.8f} < {constraint_threshold}")
    print(f"L_feat (Local):    {avg_local:.8f} < {constraint_threshold}")
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)
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

def train_model_transductive(X_train, y_train, X_test, groups_test,
                             global_constraint, local_constraint,
                             lambda_global, lambda_local, hidden_dims, epochs,
                             batch_size, lr, dropout, patience, device):
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
    state, history = initialize_training_state(lambda_global, lambda_local)
    constraint_dropout = int(global_constraint[0]) if global_constraint[0] < 1e9 else 'inf'
    constraint_enrolled = int(global_constraint[1]) if global_constraint[1] < 1e9 else 'inf'
    num_local_courses = len(local_constraint) if local_constraint else 0
    experiment_folder = f'./results/constraints_dropout{constraint_dropout}_enrolled{constraint_enrolled}_local{num_local_courses}courses'
    os.makedirs(experiment_folder, exist_ok=True)
    csv_log_path = f'{experiment_folder}/training_log.csv'
    print("\n" + "="*80)
    print("Starting Training with Adaptive Lambda Weights")
    print(f"Constraint Configuration: Dropout≤{constraint_dropout}, Enrolled≤{constraint_enrolled}, {num_local_courses} local courses")
    print(f"Initial: λ_global={state['current_lambda_global']:.2f}, "
          f"λ_local={state['current_lambda_local']:.2f}")
    print(f"Lambda adjustment: +{state['lambda_step']} per epoch when constraints violated")
    print(f"Results folder: {experiment_folder}")
    print(f"Progress logged to: {csv_log_path}")
    print("="*80 + "\n")
    for epoch in range(epochs):
        avg_ce, avg_global, avg_local = train_single_epoch(
            model, train_loader, criterion_ce, criterion_constraint,
            optimizer, X_test_tensor, group_ids_test, device
        )
        avg_loss = avg_ce + state['current_lambda_global'] * avg_global + state['current_lambda_local'] * avg_local
        update_lambda_weights(state, avg_global, avg_local, criterion_constraint)
        if (epoch + 1) % 10 == 0:
            global_counts, local_counts, global_soft_counts, local_soft_counts = compute_prediction_statistics(
                model, X_test_tensor, group_ids_test
            )
            update_training_history(
                history, epoch, avg_loss, avg_ce, avg_global, avg_local,
                state['current_lambda_global'], state['current_lambda_local'],
                global_counts, local_counts
            )
            log_progress_to_csv(csv_log_path, epoch, avg_global, avg_local, avg_ce, avg_loss,
                               global_counts, local_counts, global_soft_counts, local_soft_counts,
                               state['current_lambda_global'], state['current_lambda_local'],
                               global_constraint)
        if (epoch + 1) % 50 == 0:
            if (epoch + 1) % 10 != 0:
                global_counts, local_counts, global_soft_counts, local_soft_counts = compute_prediction_statistics(
                    model, X_test_tensor, group_ids_test
                )
            print_progress(epoch, avg_global, avg_local, avg_ce, avg_loss,
                          global_counts, local_counts, global_soft_counts, local_soft_counts,
                          criterion_constraint)
            print(f"Current Lambda Weights: λ_global={state['current_lambda_global']:.2f}, "
                  f"λ_local={state['current_lambda_local']:.2f}\n")
        if avg_loss < state['best_loss']:
            state['best_loss'] = avg_loss
            state['best_model_state'] = model.state_dict().copy()
            state['patience_counter'] = 0
        else:
            state['patience_counter'] += 1
        if check_constraint_satisfaction(
            avg_global, avg_local, state['constraint_threshold'],
            epoch, model, X_test_tensor, criterion_constraint
        ):
            break
    if state['best_model_state'] is not None:
        model.load_state_dict(state['best_model_state'])
    training_time = time.time() - start_time
    if len(history['epochs']) > 0:
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
