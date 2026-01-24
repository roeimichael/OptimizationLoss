#!/usr/bin/env python3
"""
Hybrid approach: Optimization predictions + Heuristic saturation of unused constraint budget.

This script:
1. Loads predictions from optimization approach
2. Checks how much constraint budget is unused
3. Applies heuristic saturation to fill the remaining budget
4. Saves saturated predictions
"""

import argparse
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.filesystem_manager import load_config_from_path
from src.training.metrics import compute_metrics


def load_optimization_predictions(experiment_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions and probabilities from optimization results."""

    predictions_file = experiment_path / 'final_predictions.csv'

    if not predictions_file.exists():
        raise FileNotFoundError(f"No predictions found at {predictions_file}")

    # Read predictions CSV
    y_true = []
    y_pred = []
    probs = []
    groups = []

    with open(predictions_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row['True_Label']))
            y_pred.append(int(row['Predicted_Label']))
            probs.append([
                float(row['Prob_Dropout']),
                float(row['Prob_Enrolled']),
                float(row['Prob_Graduate'])
            ])
            groups.append(int(row['Course_ID']))

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(probs),
        np.array(groups)
    )


def compute_current_allocations(y_pred: np.ndarray, groups: np.ndarray) -> Tuple[Dict[int, int], Dict[int, Dict[int, int]]]:
    """Compute current global and local allocations from predictions."""

    # Global counts
    global_counts = {c: int(np.sum(y_pred == c)) for c in range(3)}

    # Local counts per group
    local_counts = {}
    for group_id in np.unique(groups):
        group_mask = groups == group_id
        group_preds = y_pred[group_mask]
        local_counts[group_id] = {c: int(np.sum(group_preds == c)) for c in range(3)}

    return global_counts, local_counts


def apply_heuristic_saturation(
    y_pred: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    global_constraint: List[float],
    local_constraint: Dict[int, List[float]]
) -> np.ndarray:
    """
    Apply heuristic saturation to fill unused constraint budget.

    Strategy:
    1. Keep all current predictions
    2. For each constrained class (in order 0, 1), check remaining budget
    3. Find samples predicted as class 2 (unlimited) with high probability for constrained class
    4. Reassign them to constrained class up to budget limit
    """

    print("\n" + "=" * 80)
    print("HEURISTIC SATURATION")
    print("=" * 80)

    # Make a copy to modify
    y_saturated = y_pred.copy()

    # Get current allocations
    current_global, current_local = compute_current_allocations(y_pred, groups)

    print("\nCurrent allocations (from optimization):")
    for c in range(3):
        budget = global_constraint[c]
        current = current_global[c]
        if budget < 1e8:
            remaining = int(budget - current)
            print(f"  Class {c}: {current}/{int(budget)} (unused: {remaining})")
        else:
            print(f"  Class {c}: {current}/unlimited")

    # Process constrained classes in order [0, 1]
    saturation_hierarchy = [0, 1]

    for class_idx in saturation_hierarchy:
        g_limit = global_constraint[class_idx]

        if g_limit >= 1e8:
            continue  # Skip unlimited

        current_count = current_global[class_idx]
        remaining_budget = int(g_limit - current_count)

        if remaining_budget <= 0:
            print(f"\nClass {class_idx}: No remaining budget")
            continue

        print(f"\n--- Saturating Class {class_idx} ---")
        print(f"Remaining budget: {remaining_budget}")

        # Find candidates: currently predicted as class 2, sorted by probability for class_idx
        candidates = []
        for idx in range(len(y_saturated)):
            if y_saturated[idx] == 2:  # Currently assigned to Graduate (unlimited)
                prob = probs[idx, class_idx]
                group_id = groups[idx]

                # Check local constraint if applicable
                if group_id in local_constraint:
                    local_limit = local_constraint[group_id][class_idx]
                    if local_limit < 1e8:
                        local_current = current_local[group_id][class_idx]
                        if local_current >= local_limit:
                            continue  # Local budget exhausted

                candidates.append((idx, prob, group_id))

        # Sort by probability (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Reassign top candidates
        reassigned = 0
        for idx, prob, group_id in candidates:
            if reassigned >= remaining_budget:
                break

            # Double-check local constraint
            if group_id in local_constraint:
                local_limit = local_constraint[group_id][class_idx]
                if local_limit < 1e8:
                    if current_local[group_id][class_idx] >= local_limit:
                        continue
                    current_local[group_id][class_idx] += 1

            # Reassign
            y_saturated[idx] = class_idx
            current_global[class_idx] += 1
            reassigned += 1

            if reassigned <= 5:
                print(f"  Reassign idx={idx}: class 2→{class_idx} (prob={prob:.4f})")

        print(f"Reassigned {reassigned} samples to class {class_idx}")

    print("\nFinal allocations (after saturation):")
    for c in range(3):
        budget = global_constraint[c]
        final = current_global[c]
        if budget < 1e8:
            print(f"  Class {c}: {final}/{int(budget)} (utilization: {final/budget*100:.1f}%)")
        else:
            print(f"  Class {c}: {final}/unlimited")

    return y_saturated


def main():
    parser = argparse.ArgumentParser(description="Apply heuristic saturation to optimization predictions")
    parser.add_argument('experiment_path', type=str, help='Path to experiment directory with optimization results')
    args = parser.parse_args()

    experiment_path = Path(args.experiment_path)

    print("=" * 80)
    print("OPTIMIZATION + HEURISTIC SATURATION")
    print("=" * 80)
    print(f"\nExperiment: {experiment_path}")

    # Load config
    config = load_config_from_path(experiment_path)

    # Load optimization predictions
    print("\nLoading optimization predictions...")
    y_true, y_pred_opt, probs, groups = load_optimization_predictions(experiment_path)

    print(f"Samples: {len(y_true)}")
    print("\nOptimization predictions:")
    unique, counts = np.unique(y_pred_opt, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y_pred_opt)*100:.1f}%)")

    opt_metrics = compute_metrics(y_true, y_pred_opt)
    print(f"Optimization accuracy: {opt_metrics['accuracy']:.4f}")

    # Load constraints
    from src.utils.data_loader import load_experiment_data
    _, _, _, _, _, global_constraint, local_constraint = load_experiment_data(config)

    # Apply saturation
    y_pred_saturated = apply_heuristic_saturation(
        y_pred_opt, probs, groups, global_constraint, local_constraint
    )

    # Compute saturated metrics
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    sat_metrics = compute_metrics(y_true, y_pred_saturated)

    print("\nOptimization (before saturation):")
    print(f"  Accuracy: {opt_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {opt_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {opt_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro): {opt_metrics['f1_macro']:.4f}")

    print("\nOptimization + Saturation (after):")
    print(f"  Accuracy: {sat_metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {sat_metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {sat_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro): {sat_metrics['f1_macro']:.4f}")

    print("\nImprovement:")
    print(f"  Accuracy: {(sat_metrics['accuracy'] - opt_metrics['accuracy']):.4f} "
          f"({(sat_metrics['accuracy'] - opt_metrics['accuracy'])/opt_metrics['accuracy']*100:+.1f}%)")

    # Save saturated predictions
    output_file = experiment_path / 'final_predictions_saturated.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'True', 'Predicted_Original', 'Predicted_Saturated',
                        'Prob_0', 'Prob_1', 'Prob_2', 'Group'])
        for i in range(len(y_true)):
            writer.writerow([
                i, y_true[i], y_pred_opt[i], y_pred_saturated[i],
                f"{probs[i, 0]:.6f}", f"{probs[i, 1]:.6f}", f"{probs[i, 2]:.6f}",
                groups[i]
            ])

    print(f"\n✓ Saturated predictions saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
