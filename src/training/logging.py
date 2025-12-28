import csv
import os
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
