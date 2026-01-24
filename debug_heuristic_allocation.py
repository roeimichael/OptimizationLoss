#!/usr/bin/env python3
"""
Debug the heuristic allocation to see why everything is assigned to class 2.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.utils.data_loader import load_experiment_data
from src.utils.filesystem_manager import load_config_from_path
from src.training.trainer import ConstraintTrainer


def debug_allocation(config_path: str):
    print("=" * 80)
    print("HEURISTIC ALLOCATION DEBUG")
    print("=" * 80)

    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    device = torch.device('cpu')

    print(f"\nConfig: {config_path}")
    print(f"Constraint: {config['constraint']}")

    # Load data
    X_train_clean, X_test_clean, _, y_test, groups_test, global_constraint, local_constraint = load_experiment_data(config)

    print(f"\nTest set size: {len(y_test)}")
    print(f"True label distribution:")
    print(pd.Series(y_test).value_counts().sort_index())

    # Get constraints
    print(f"\n" + "-" * 80)
    print("CONSTRAINTS:")
    print("-" * 80)
    print(f"Global constraint: {global_constraint}")
    print(f"  Class 0 (Dropout): {global_constraint[0]}")
    print(f"  Class 1 (Enrolled): {global_constraint[1]}")
    print(f"  Class 2 (Graduate): {global_constraint[2]}")

    print(f"\nLocal constraints: {len(local_constraint)} groups")
    sample_group = list(local_constraint.keys())[0]
    print(f"Sample group {sample_group}: {local_constraint[sample_group]}")

    # Load model and get predictions
    scaler = StandardScaler()
    scaler.fit(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    trainer = ConstraintTrainer(config, str(experiment_path), device)
    trainer.setup_model(X_train_clean.shape[1], config['base_model_id'])

    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(X_test_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    print(f"\n" + "-" * 80)
    print("MODEL PREDICTIONS (before heuristic):")
    print("-" * 80)
    model_preds = probs.argmax(axis=1)
    print("Predicted class distribution (from model directly):")
    unique, counts = np.unique(model_preds, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(model_preds)*100:.1f}%)")

    print("\nProbability statistics:")
    for class_idx in range(3):
        print(f"  Class {class_idx}: mean={probs[:, class_idx].mean():.4f}, "
              f"std={probs[:, class_idx].std():.4f}")

    # Now apply heuristic allocation with detailed tracking
    print(f"\n" + "-" * 80)
    print("HEURISTIC ALLOCATION PROCESS:")
    print("-" * 80)

    class_hierarchy = [2, 0, 1]
    print(f"Class hierarchy: {class_hierarchy} (Graduate -> Dropout -> Enrolled)")

    n_samples = len(probs)
    y_pred = np.full(n_samples, -1, dtype=int)
    assigned_mask = np.zeros(n_samples, dtype=bool)
    current_global = {c: 0 for c in range(3)}

    for class_idx in class_hierarchy:
        print(f"\n--- Processing Class {class_idx} ---")
        g_limit = global_constraint[class_idx]
        print(f"Global limit: {g_limit}")

        unassigned_indices = np.where(~assigned_mask)[0]
        print(f"Unassigned samples: {len(unassigned_indices)}")

        if len(unassigned_indices) == 0:
            print("No more unassigned samples")
            break

        # Get probabilities for this class
        class_probs = probs[unassigned_indices, class_idx]
        print(f"Probability range for class {class_idx}: [{class_probs.min():.4f}, {class_probs.max():.4f}]")

        # Sort by probability (descending)
        sorted_indices = unassigned_indices[np.argsort(class_probs)[::-1]]

        # Try to assign
        assigned_to_this_class = 0
        rejected_global = 0
        rejected_local = 0

        for idx in sorted_indices[:min(20, len(sorted_indices))]:  # Show first 20 attempts
            group_id = groups_test.iloc[idx]
            prob = probs[idx, class_idx]

            # Check global constraint
            if g_limit < 1e8 and current_global[class_idx] >= g_limit:
                rejected_global += 1
                continue

            # Check local constraint
            # (simplified - not tracking local counts in this debug)

            # Would assign here
            assigned_to_this_class += 1
            if assigned_to_this_class <= 5:
                print(f"  Assign idx={idx}, prob={prob:.4f}, group={group_id}")

        # Now actually count how many get assigned
        actual_assigned = 0
        for idx in sorted_indices:
            if g_limit < 1e8 and current_global[class_idx] >= g_limit:
                break
            y_pred[idx] = class_idx
            assigned_mask[idx] = True
            current_global[class_idx] += 1
            actual_assigned += 1

        print(f"Assigned {actual_assigned} samples to class {class_idx}")
        print(f"Current global counts: {current_global}")

    # Assign remaining to argmax
    remaining_indices = np.where(~assigned_mask)[0]
    print(f"\nRemaining unassigned: {len(remaining_indices)}")
    if len(remaining_indices) > 0:
        for idx in remaining_indices:
            y_pred[idx] = np.argmax(probs[idx])
        print("Assigned remaining to argmax of probabilities")

    print(f"\n" + "-" * 80)
    print("FINAL ALLOCATION:")
    print("-" * 80)
    unique, counts = np.unique(y_pred, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} ({cnt/len(y_pred)*100:.1f}%)")

    # Compare with true labels
    accuracy = np.mean(y_pred == y_test.values)
    print(f"\nAccuracy: {accuracy:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python debug_heuristic_allocation.py <config_path>")
        sys.exit(1)

    debug_allocation(sys.argv[1])
