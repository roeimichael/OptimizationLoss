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
                        tracked_group_id: int = 1) -> None:
    file_exists = os.path.isfile(csv_path)

    num_classes = len(global_constraints) if global_constraints else 5
    if global_counts is None:
        global_counts = {i: 0 for i in range(num_classes)}
    if global_soft_counts is None:
        global_soft_counts = {i: 0.0 for i in range(num_classes)}
    if global_constraints is None:
        global_constraints = [1e9] * num_classes

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = [
                'Epoch', 'Train_Acc', 'L_pred_CE', 'L_target_Global', 'L_feat_Local',
                'Lambda_Global', 'Lambda_Local', 'Global_Satisfied', 'Local_Satisfied'
            ]
            for i in range(num_classes):
                header.extend([f'Limit_Class{i+1}', f'Hard_Class{i+1}', f'Soft_Class{i+1}'])
            writer.writerow(header)

        row = [
            epoch + 1,
            f"{train_acc:.4f}",
            f"{avg_ce:.6f}", f"{avg_global:.6f}", f"{avg_local:.6f}",
            f"{lambda_global:.2f}", f"{lambda_local:.2f}",
            1 if global_satisfied else 0, 1 if local_satisfied else 0
        ]
        for i in range(num_classes):
            row.extend([
                int(global_constraints[i]) if global_constraints[i] < 1e9 else 'inf',
                global_counts.get(i, 0),
                f"{global_soft_counts.get(i, 0.0):.2f}"
            ])
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

    num_classes = len(global_constraints)
    for idx in range(num_classes):
        class_name = f"Class_{idx+1}"
        limit = int(global_constraints[idx]) if global_constraints[idx] < 1e9 else 'inf'
        hard = global_counts.get(idx, 0)
        soft = global_soft_counts.get(idx, 0.0)
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


def save_final_predictions(save_path, y_true, y_pred, y_proba, group_ids=None):
    import pandas as pd
    df_data = {
        'Sample_Index': list(range(len(y_true))),
        'True_Label': y_true,
        'Predicted_Label': y_pred
    }
    num_classes = y_proba.shape[1]
    for i in range(num_classes):
        df_data[f'Prob_Class_{i+1}'] = y_proba[:, i]
    df_data['Correct'] = (y_true == y_pred).astype(int)
    if group_ids is not None:
        df_data['Group_ID'] = group_ids
    df = pd.DataFrame(df_data)
    df.to_csv(save_path, index=False)
    print(f"Final predictions saved to: {save_path}")


def save_evaluation_metrics(save_path, metrics):
    num_classes = len(metrics['precision_per_class'])
    class_names = [f'Class_{i+1}' for i in range(num_classes)]
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
        rows.append([class_name] + [int(cm[i][j]) for j in range(num_classes)])

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Evaluation metrics saved to: {save_path}")


def save_run_status(experiment_path, status: str, epoch: int, global_satisfied: bool, local_satisfied: bool, details: str = ""):
    """
    Save the final status of a training run.

    Args:
        experiment_path: Path to the experiment directory
        status: One of 'converged', 'failed', 'interrupted'
        epoch: Final epoch number
        global_satisfied: Whether global constraint was satisfied
        local_satisfied: Whether local constraint was satisfied
        details: Additional details about the run status
    """
    from pathlib import Path
    import json
    from datetime import datetime

    status_path = Path(experiment_path) / 'run_status.json'

    status_data = {
        'status': status,
        'final_epoch': epoch,
        'global_constraint_satisfied': global_satisfied,
        'local_constraint_satisfied': local_satisfied,
        'details': details,
        'timestamp': datetime.now().isoformat()
    }

    with open(status_path, 'w') as f:
        json.dump(status_data, f, indent=2)

    print(f"[STATUS] Run status saved: {status}")
