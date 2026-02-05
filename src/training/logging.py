"""Training progress logging utilities."""

import csv
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def log_progress_to_csv(
    csv_path: str,
    epoch: int,
    ce_loss: float,
    train_acc: float,
    global_loss: float = 0.0,
    local_loss: float = 0.0,
    global_counts: Optional[Dict] = None,
    local_counts: Optional[Dict] = None,
    global_soft: Optional[Dict] = None,
    local_soft: Optional[Dict] = None,
    lambda_global: float = 0.0,
    lambda_local: float = 0.0,
    constraints: Optional[list] = None,
    global_satisfied: bool = True,
    local_satisfied: bool = True
):
    """Log training progress to CSV file."""
    num_classes = len(constraints) if constraints else 5
    global_counts = global_counts or {i: 0 for i in range(num_classes)}
    global_soft = global_soft or {i: 0.0 for i in range(num_classes)}
    constraints = constraints or [1e9] * num_classes

    file_exists = Path(csv_path).exists()

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ['Epoch', 'Train_Acc', 'L_pred_CE', 'L_target_Global', 'L_feat_Local',
                      'Lambda_Global', 'Lambda_Local', 'Global_Satisfied', 'Local_Satisfied']
            for i in range(num_classes):
                header += [f'Limit_Class{i+1}', f'Hard_Class{i+1}', f'Soft_Class{i+1}']
            writer.writerow(header)

        row = [epoch + 1, f"{train_acc:.4f}", f"{ce_loss:.6f}", f"{global_loss:.6f}",
               f"{local_loss:.6f}", f"{lambda_global:.2f}", f"{lambda_local:.2f}",
               1 if global_satisfied else 0, 1 if local_satisfied else 0]

        for i in range(num_classes):
            limit = int(constraints[i]) if constraints[i] < 1e9 else 'inf'
            row += [limit, global_counts.get(i, 0), f"{global_soft.get(i, 0.0):.2f}"]

        writer.writerow(row)


def print_progress(
    epoch: int,
    ce_loss: float,
    global_loss: float,
    local_loss: float,
    lambda_global: float,
    lambda_local: float,
    train_acc: float,
    global_counts: Dict,
    global_soft: Dict,
    constraints: list,
    global_satisfied: bool,
    local_satisfied: bool
):
    """Print formatted training progress."""
    print(f"\n{'='*60}")
    print(f"Epoch {epoch + 1} | Acc: {train_acc:.4f} | CE: {ce_loss:.4f}")
    print(f"Global Loss: {global_loss:.6f} | Local Loss: {local_loss:.6f}")
    print(f"Lambda: global={lambda_global:.2f}, local={lambda_local:.2f}")
    print(f"Constraints: Global={'OK' if global_satisfied else 'VIOLATED'}, "
          f"Local={'OK' if local_satisfied else 'VIOLATED'}")
    print(f"{'='*60}\n")


def save_final_predictions(save_path, y_true, y_pred, y_proba, group_ids=None):
    """Save predictions to CSV."""
    data = {
        'True_Label': y_true,
        'Predicted_Label': y_pred,
        'Correct': (y_true == y_pred).astype(int)
    }
    for i in range(y_proba.shape[1]):
        data[f'Prob_Class_{i+1}'] = y_proba[:, i]
    if group_ids is not None:
        data['Group_ID'] = group_ids

    pd.DataFrame(data).to_csv(save_path, index=False)
    print(f"Predictions saved: {save_path}")


def save_evaluation_metrics(save_path, metrics):
    """Save evaluation metrics to CSV."""
    rows = [
        ['Metric', 'Value'],
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Precision (Macro)', f"{metrics['precision_macro']:.4f}"],
        ['Recall (Macro)', f"{metrics['recall_macro']:.4f}"],
        ['F1 (Macro)', f"{metrics['f1_macro']:.4f}"],
    ]

    with open(save_path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"Metrics saved: {save_path}")
