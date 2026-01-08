import torch
import os
import csv
from typing import Dict, Optional


def log_progress_to_csv(csv_path: str, epoch: int, avg_ce: float, train_acc: float,
                        avg_global: float = 0.0, avg_local: float = 0.0,
                        global_counts: Optional[Dict[int, int]] = None,
                        local_counts: Optional[Dict[int, Dict[int, int]]] = None,
                        global_soft_counts: Optional[Dict[int, float]] = None,
                        local_soft_counts: Optional[Dict[int, Dict[int, float]]] = None,
                        lambda_global: float = 0.0, lambda_local: float = 0.0,
                        global_constraints: Optional[list] = None,
                        global_satisfied: bool = True, local_satisfied: bool = True,
                        tracked_course_id: int = 1) -> None:
    file_exists = os.path.isfile(csv_path)

    if global_counts is None:
        global_counts = {0: 0, 1: 0, 2: 0}
    if global_soft_counts is None:
        global_soft_counts = {0: 0.0, 1: 0.0, 2: 0.0}
    if global_constraints is None:
        global_constraints = [1e9, 1e9, 1e9]

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

        excess_dropout = max(0, global_soft_counts[0] - global_constraints[0]) if global_constraints[0] < 1e9 else 0
        excess_enrolled = max(0, global_soft_counts[1] - global_constraints[1]) if global_constraints[1] < 1e9 else 0
        excess_graduate = max(0, global_soft_counts[2] - global_constraints[2]) if global_constraints[2] < 1e9 else 0

        course_hard = [0, 0, 0]
        course_soft = [0.0, 0.0, 0.0]
        if local_counts and tracked_course_id in local_counts:
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
    print(f"\n{'=' * 80}")
    print(f"Epoch {epoch + 1}")
    print(f"{'=' * 80}")
    print(f"Train Accuracy:     {train_acc:.4f}")
    print(f"L_target (Global):  {avg_global:.6f}")
    print(f"L_feat (Local):     {avg_local:.6f}")
    print(f"L_pred (CE):        {avg_ce:.6f}")
    print(f"\n{'-' * 80}")
    print("GLOBAL CONSTRAINTS")
    print(f"{'-' * 80}")
    print(f"{'Class':<12} {'Limit':<8} {'Hard':<8} {'Soft':<10} {'Excess':<10} {'Status':<15}")
    print(f"{'-' * 80}")

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

    print(f"{'-' * 80}")
    total_hard = sum(global_counts.values())
    total_soft = sum(global_soft_counts.values())
    print(f"{'Total':<12} {'':<8} {total_hard:<8} {total_soft:<10.2f}")
    print(f"\nLambda Weights: lambda_global={lambda_global:.2f}, lambda_local={lambda_local:.2f}")
    constraint_global = "Satisfied" if global_satisfied else "Violated"
    constraint_local = "Satisfied" if local_satisfied else "Violated"
    print(f"Constraint Status: Global={constraint_global}, Local={constraint_local}")
    print(f"{'=' * 80}\n")


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
