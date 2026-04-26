# CSV logging for training progress and evaluation metric export.
# Handles warmup and constraint phase logging with per-group columns.

import csv
import logging
from pathlib import Path

import pandas as pd

from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def build_csv_header(num_classes, local_constraints=None):
    header = ['Epoch', 'Train_Acc', 'L_CE', 'L_Global', 'L_Local',
              'L_KL', 'Lambda_Global', 'Lambda_Local', 'Global_Satisfied', 'Local_Satisfied']
    for i in range(num_classes):
        header += [f'Limit_Class{i}', f'Hard_Class{i}', f'Soft_Class{i}']
    if local_constraints:
        for gid in sorted(local_constraints.keys()):
            for c in range(num_classes):
                l_limit = local_constraints[gid][c] if c < len(local_constraints[gid]) else UNLIMITED
                if l_limit < UNLIMITED:
                    header += [f'Group{gid}_Hard_Class{c}', f'Group{gid}_Soft_Class{c}',
                               f'Group{gid}_Limit_Class{c}']
    return header


def write_csv_header(csv_path, num_classes, local_constraints=None):
    header = build_csv_header(num_classes, local_constraints)
    csv_file = Path(csv_path)
    if csv_file.exists():
        with open(csv_file, 'r') as f:
            existing_lines = f.readlines()
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for line in existing_lines:
                fields = line.strip().split(',')
                while len(fields) < len(header):
                    fields.append('')
                writer.writerow(fields[:len(header)])
    else:
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(header)


def log_progress_to_csv(csv_path, epoch, ce_loss, train_acc,
                        global_loss=0.0, local_loss=0.0,
                        global_counts=None, local_counts=None,
                        global_soft=None, local_soft=None,
                        lambda_global=0.0, lambda_local=0.0,
                        constraints=None, global_satisfied=True, local_satisfied=True,
                        num_classes=7, kl_loss=0.0, local_constraints=None):
    num_classes = len(constraints) if constraints else num_classes
    global_counts = global_counts or {i: 0 for i in range(num_classes)}
    global_soft = global_soft or {i: 0.0 for i in range(num_classes)}
    local_counts = local_counts or {}
    local_soft = local_soft or {}
    constraints = constraints or [UNLIMITED] * num_classes
    group_ids_sorted = sorted(local_constraints.keys()) if local_constraints else []
    row = [epoch + 1, f"{train_acc:.4f}", f"{ce_loss:.6f}", f"{global_loss:.6f}",
           f"{local_loss:.6f}", f"{kl_loss:.6f}",
           f"{lambda_global:.2f}", f"{lambda_local:.2f}",
           1 if global_satisfied else 0, 1 if local_satisfied else 0]
    for i in range(num_classes):
        limit = int(constraints[i]) if constraints[i] < UNLIMITED else 'inf'
        row += [limit, global_counts.get(i, 0), f"{global_soft.get(i, 0.0):.2f}"]
    for gid in group_ids_sorted:
        group_hard = local_counts.get(gid, {})
        group_soft_data = local_soft.get(gid, {})
        for c in range(num_classes):
            l_limit = local_constraints[gid][c] if c < len(local_constraints[gid]) else UNLIMITED
            if l_limit < UNLIMITED:
                row += [group_hard.get(c, 0),
                        f"{group_soft_data.get(c, 0.0):.2f}",
                        int(l_limit)]
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def save_final_predictions(save_path, y_true, y_pred, y_proba, group_ids=None):
    data = {
        'True_Label': y_true,
        'Predicted_Label': y_pred,
        'Correct': (y_true == y_pred).astype(int)
    }
    for i in range(y_proba.shape[1]):
        data[f'Prob_Class_{i}'] = y_proba[:, i]
    if group_ids is not None:
        data['Group_ID'] = group_ids
    pd.DataFrame(data).to_csv(save_path, index=False)


def save_evaluation_metrics(save_path, metrics):
    rows = [
        ['Metric', 'Value'],
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Precision (Macro)', f"{metrics['precision_macro']:.4f}"],
        ['Recall (Macro)', f"{metrics['recall_macro']:.4f}"],
        ['F1 (Macro)', f"{metrics['f1_macro']:.4f}"],
        ['Precision (Weighted)', f"{metrics.get('precision_weighted', 0):.4f}"],
        ['Recall (Weighted)', f"{metrics.get('recall_weighted', 0):.4f}"],
        ['F1 (Weighted)', f"{metrics.get('f1_weighted', 0):.4f}"],
    ]
    if 'precision_per_class' in metrics:
        prec = metrics['precision_per_class']
        rec = metrics['recall_per_class']
        f1 = metrics['f1_per_class']
        sup = metrics['support_per_class']
        for c in range(len(prec)):
            rows.append([f'Precision_Class{c}', f"{prec[c]:.4f}"])
            rows.append([f'Recall_Class{c}', f"{rec[c]:.4f}"])
            rows.append([f'F1_Class{c}', f"{f1[c]:.4f}"])
            rows.append([f'Support_Class{c}', int(sup[c])])
    if 'ece' in metrics:
        rows.append(['ECE', f"{metrics['ece']:.4f}"])
        rows.append(['Brier Score', f"{metrics['brier_score']:.4f}"])
        rows.append(['Mean Entropy', f"{metrics['mean_entropy']:.4f}"])
        rows.append(['Mean Confidence', f"{metrics['mean_confidence']:.4f}"])
        rows.append(['Confidence (Correct)', f"{metrics['confidence_correct']:.4f}"])
        rows.append(['Confidence (Incorrect)', f"{metrics['confidence_incorrect']:.4f}"])
        rows.append(['Confidence Gap', f"{metrics['confidence_gap']:.4f}"])
        rows.append(['Pct High Confidence', f"{metrics['pct_high_confidence']:.4f}"])
        rows.append(['Pct Low Confidence', f"{metrics['pct_low_confidence']:.4f}"])
    # Constraint-specific metrics (Track 1)
    if 'flips_required' in metrics:
        rows.append(['Flips Required', metrics['flips_required']])
    if 'raw_global_satisfied_pct' in metrics:
        rows.append(['Raw Global Satisfied %', f"{metrics['raw_global_satisfied_pct']:.4f}"])
        rows.append(['Raw Local Satisfied %', f"{metrics['raw_local_satisfied_pct']:.4f}"])
        rows.append(['Raw All Satisfied', int(metrics['raw_all_satisfied'])])
        rows.append(['Raw Total Excess', metrics['raw_total_excess']])
    if 'satisfaction_epoch' in metrics:
        rows.append(['Satisfaction Epoch', metrics.get('satisfaction_epoch', 'N/A')])
    if 'soft_hard_gap' in metrics:
        for c, gap in metrics['soft_hard_gap'].items():
            rows.append([f'Soft-Hard Gap Class{c}', f"{gap:.2f}"])
    with open(save_path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
