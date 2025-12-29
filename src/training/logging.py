import csv
import os
import torch
from config.experiment_config import TRACKED_COURSE_ID

def log_progress_to_csv(csv_path, epoch, avg_global, avg_local, avg_ce,
                        global_counts, local_counts, global_soft_counts, local_soft_counts,
                        lambda_global, lambda_local, global_constraints,
                        global_satisfied, local_satisfied):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = [
                'Epoch', 'L_pred_CE', 'L_target_Global', 'L_feat_Local',
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
        if TRACKED_COURSE_ID in local_counts:
            course_hard = [local_counts[TRACKED_COURSE_ID][i] for i in range(3)]
            course_soft = [local_soft_counts[TRACKED_COURSE_ID][i] for i in range(3)]

        row = [
            epoch + 1,
            f"{avg_ce:.6f}", f"{avg_global:.6f}", f"{avg_local:.6f}",
            f"{lambda_global:.2f}", f"{lambda_local:.2f}",
            1 if global_satisfied else 0, 1 if local_satisfied else 0,
            int(global_constraints[0]) if global_constraints[0] < 1e9 else 'inf',
            int(global_constraints[1]) if global_constraints[1] < 1e9 else 'inf',
            int(global_constraints[2]) if global_constraints[2] < 1e9 else 'inf',
            global_counts[0], global_counts[1], global_counts[2],
            f"{global_soft_counts[0]:.2f}", f"{global_soft_counts[1]:.2f}", f"{global_soft_counts[2]:.2f}",
            f"{excess_dropout:.2f}", f"{excess_enrolled:.2f}", f"{excess_graduate:.2f}",
            TRACKED_COURSE_ID,
            course_hard[0], course_hard[1], course_hard[2],
            f"{course_soft[0]:.2f}", f"{course_soft[1]:.2f}", f"{course_soft[2]:.2f}"
        ]
        writer.writerow(row)

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
    print(f"\n{'-'*80}")
    print("GLOBAL CONSTRAINTS")
    print(f"{'-'*80}")
    print(f"{'Class':<12} {'Limit':<8} {'Hard':<8} {'Soft':<10} {'Excess':<10} {'Status':<15}")
    print(f"{'-'*80}")

    for idx, class_name in enumerate(['Dropout', 'Enrolled', 'Graduate']):
        limit = last_row[f'Limit_{class_name}']
        hard = int(last_row[f'Hard_{class_name}'])
        soft = float(last_row[f'Soft_{class_name}'])
        excess = float(last_row[f'Excess_{class_name}'])

        if limit == 'inf':
            status = "N/A"
        elif excess == 0:
            status = "OK"
        else:
            status = f"Over by {excess:.1f}"

        print(f"{class_name:<12} {limit:<8} {hard:<8} {soft:<10.2f} {excess:<10.2f} {status:<15}")

    print(f"{'-'*80}")
    total_hard = int(last_row['Hard_Dropout']) + int(last_row['Hard_Enrolled']) + int(last_row['Hard_Graduate'])
    total_soft = float(last_row['Soft_Dropout']) + float(last_row['Soft_Enrolled']) + float(last_row['Soft_Graduate'])
    print(f"{'Total':<12} {'':<8} {total_hard:<8} {total_soft:<10.2f}")
    print(f"\nCurrent Lambda Weights: lambda_global={lambda_global:.2f}, lambda_local={lambda_local:.2f}")
    constraint_global = "Satisfied" if global_satisfied else "Violated"
    constraint_local = "Satisfied" if local_satisfied else "Violated"
    print(f"Constraint Status: Global={constraint_global}, Local={constraint_local}")
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
        local_data = {}
        if 'Course_ID' in row and 'Course_Hard_Dropout' in row:
            course_id = int(row['Course_ID'])
            local_data[course_id] = [
                int(row['Course_Hard_Dropout']),
                int(row['Course_Hard_Enrolled']),
                int(row['Course_Hard_Graduate'])
            ]
        history['local_predictions'].append(local_data)
    return history

def save_final_predictions(save_path, y_true, y_pred, y_proba, course_ids=None):
    import pandas as pd
    df_data = {
        'Sample_Index': list(range(len(y_true))),
        'True_Label': y_true,
        'Predicted_Label': y_pred,
        'Prob_Dropout': y_proba[:, 0],
        'Prob_Enrolled': y_proba[:, 1],
        'Prob_Graduate': y_proba[:, 2],
        'Correct': (y_true == y_pred).astype(int)
    }
    if course_ids is not None:
        df_data['Course_ID'] = course_ids
    df = pd.DataFrame(df_data)
    df.to_csv(save_path, index=False)
    print(f"Final predictions saved to: {save_path}")

def save_constraint_comparison(save_path, model, X_test_tensor, group_ids_test,
                                global_constraints, local_constraints_dict):
    import pandas as pd
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        test_preds = torch.argmax(test_logits, dim=1)

        unique_groups = torch.unique(group_ids_test)
        rows = []

        for group_id in unique_groups:
            group_id_val = group_id.item()
            group_mask = (group_ids_test == group_id_val)
            group_preds = test_preds[group_mask]

            course_counts = {}
            for class_id in range(3):
                count = (group_preds == class_id).sum().item()
                course_counts[class_id] = count

            buffer_name = f'local_constraint_{group_id_val}'
            local_cons = local_constraints_dict.get(group_id_val, [float('inf')] * 3)

            class_names = ['Dropout', 'Enrolled', 'Graduate']
            for class_id in range(3):
                predicted = course_counts[class_id]
                constraint = local_cons[class_id]

                if constraint < 1e9:
                    overprediction = max(0, predicted - constraint)
                    status = "OK" if predicted <= constraint else "OVER"
                else:
                    overprediction = 0
                    constraint = 'Unlimited'
                    status = "N/A"

                rows.append({
                    'Course_ID': group_id_val,
                    'Class': class_names[class_id],
                    'Constraint': constraint,
                    'Predicted': predicted,
                    'Overprediction': overprediction,
                    'Status': status
                })

    model.train()
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Constraint comparison saved to: {save_path}")

def save_evaluation_metrics(save_path, metrics):
    class_names = ['Dropout', 'Enrolled', 'Graduate']

    rows = []
    rows.append(['Metric', 'Value'])
    rows.append(['Overall Accuracy', f"{metrics['accuracy']:.4f}"])
    rows.append([''])
    rows.append(['Macro Averaged Metrics', ''])
    rows.append(['Precision (Macro)', f"{metrics['precision_macro']:.4f}"])
    rows.append(['Recall (Macro)', f"{metrics['recall_macro']:.4f}"])
    rows.append(['F1-Score (Macro)', f"{metrics['f1_macro']:.4f}"])
    rows.append([''])
    rows.append(['Weighted Averaged Metrics', ''])
    rows.append(['Precision (Weighted)', f"{metrics['precision_weighted']:.4f}"])
    rows.append(['Recall (Weighted)', f"{metrics['recall_weighted']:.4f}"])
    rows.append(['F1-Score (Weighted)', f"{metrics['f1_weighted']:.4f}"])
    rows.append([''])
    rows.append(['Per-Class Metrics', ''])
    rows.append(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    for i, class_name in enumerate(class_names):
        rows.append([
            class_name,
            f"{metrics['precision_per_class'][i]:.4f}",
            f"{metrics['recall_per_class'][i]:.4f}",
            f"{metrics['f1_per_class'][i]:.4f}",
            int(metrics['support_per_class'][i])
        ])
    rows.append([''])
    rows.append(['Confusion Matrix', ''])
    rows.append([''] + class_names)
    cm = metrics['confusion_matrix']
    for i, class_name in enumerate(class_names):
        rows.append([class_name] + [int(cm[i][j]) for j in range(3)])

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Evaluation metrics saved to: {save_path}")
