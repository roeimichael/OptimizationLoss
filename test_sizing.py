"""Constraint Sizing Validation Script

This script validates that constraint values are correctly scaled for the test dataset.
Since constraints are computed on the full dataset (train+test) but applied only to
the test dataset (which is 1/10th the size), the CONSTRAINT_SCALE_FACTOR=10 accounts
for this size difference.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path to import constraint functions directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN

# Import constraint computation constants and functions directly
NUM_CLASSES = 3
CONSTRAINT_SCALE_FACTOR = 10
UNLIMITED_CONSTRAINT = 1e10
GRADUATE_CLASS_ID = 2
EXCLUDED_COURSE_ID = 1


def compute_global_constraints(data, target_column, percentage):
    """Compute global constraints (copied from src/training/constraints.py)."""
    constraint = np.zeros(NUM_CLASSES)
    items = data[target_column].value_counts()
    for class_id in items.index:
        constraint[int(class_id)] = np.round(items[class_id] * percentage / CONSTRAINT_SCALE_FACTOR)
    constraint[GRADUATE_CLASS_ID] = UNLIMITED_CONSTRAINT
    return constraint.tolist()


def compute_local_constraints(data, target_column, percentage, groups):
    """Compute local constraints (copied from src/training/constraints.py)."""
    local_constraint = {}
    for group in groups:
        if group == EXCLUDED_COURSE_ID:
            continue
        data_group = data[data['Course'] == group]
        if len(data_group) == 0:
            continue
        constraint = np.zeros(NUM_CLASSES)
        items = data_group[target_column].value_counts()
        for class_id in items.index:
            constraint[int(class_id)] = np.round(items[class_id] * percentage / CONSTRAINT_SCALE_FACTOR)
        constraint[GRADUATE_CLASS_ID] = UNLIMITED_CONSTRAINT
        local_constraint[group] = constraint.tolist()
    return local_constraint


def validate_constraint_sizing():
    """Validate constraint sizes relative to test dataset."""

    print("=" * 80)
    print("CONSTRAINT SIZING VALIDATION")
    print("=" * 80)

    # Load data
    print("\n[1] Loading datasets...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        full_df = pd.concat([train_df, test_df], ignore_index=True)
    except FileNotFoundError as e:
        print(f"  ERROR: Data files not found: {e}")
        print(f"  Expected paths:")
        print(f"    - {TRAIN_PATH}")
        print(f"    - {TEST_PATH}")
        print("\n  Please run this script from an environment with the data files.")
        return

    print(f"  Train size: {len(train_df)}")
    print(f"  Test size:  {len(test_df)}")
    print(f"  Full size:  {len(full_df)}")
    print(f"  Test/Full ratio: {len(test_df)/len(full_df):.2%}")

    # Test constraint percentages
    test_percentages = [0.3, 0.8]

    for percentage in test_percentages:
        print("\n" + "=" * 80)
        print(f"TESTING CONSTRAINT PERCENTAGE: {percentage} ({percentage*100}%)")
        print("=" * 80)

        # Compute global constraints
        global_constraints = compute_global_constraints(full_df, TARGET_COLUMN, percentage)

        print(f"\n[2] Global Constraints (CONSTRAINT_SCALE_FACTOR={CONSTRAINT_SCALE_FACTOR}):")
        print("-" * 80)

        # Analyze each class
        full_class_counts = full_df[TARGET_COLUMN].value_counts().sort_index()
        test_class_counts = test_df[TARGET_COLUMN].value_counts().sort_index()

        class_names = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

        print(f"\n{'Class':<12} {'Full Count':<12} {'Test Count':<12} {'Constraint':<12} {'% of Test':<12} {'% of Full':<12}")
        print("-" * 80)

        for class_id in range(3):
            class_name = class_names.get(class_id, f"Class_{class_id}")
            full_count = full_class_counts.get(class_id, 0)
            test_count = test_class_counts.get(class_id, 0)
            constraint = global_constraints[class_id]

            if constraint > 1e9:  # Unlimited
                constraint_str = "Unlimited"
                pct_test = "N/A"
                pct_full = "N/A"
            else:
                constraint_str = f"{int(constraint)}"
                pct_test = f"{(constraint/test_count)*100:.1f}%" if test_count > 0 else "N/A"
                pct_full = f"{(constraint/full_count)*100:.1f}%" if full_count > 0 else "N/A"

            print(f"{class_name:<12} {full_count:<12} {test_count:<12} {constraint_str:<12} {pct_test:<12} {pct_full:<12}")

        print("\n[Analysis]")
        print(f"  - Constraints computed on FULL dataset ({len(full_df)} samples)")
        print(f"  - Constraints applied to TEST dataset ({len(test_df)} samples)")
        print(f"  - Scale factor: {CONSTRAINT_SCALE_FACTOR}")
        print(f"  - Expected constraint ≈ (full_class_count × {percentage}) / {CONSTRAINT_SCALE_FACTOR}")
        print(f"  - This gives ≈ {percentage*10}% of test set for each class")

        # Compute local constraints
        groups = full_df['Course'].unique()
        local_constraints = compute_local_constraints(full_df, TARGET_COLUMN, percentage, groups)

        print(f"\n[3] Local Constraints by Course:")
        print("-" * 80)
        print(f"Total courses with constraints: {len(local_constraints)}")

        # Analyze a few example groups
        groups_test = test_df['Course']

        print(f"\n{'Course':<10} {'Test Size':<12} {'Class':<12} {'Constraint':<12} {'% of Course':<15}")
        print("-" * 80)

        for group_id in sorted(local_constraints.keys())[:5]:  # Show first 5 courses
            # Get test samples for this group
            test_group = test_df[test_df['Course'] == group_id]
            full_group = full_df[full_df['Course'] == group_id]
            group_test_size = len(test_group)

            constraints = local_constraints[group_id]

            for class_id in range(3):
                class_name = class_names.get(class_id, f"Class_{class_id}")
                constraint = constraints[class_id]

                # Get actual class count in this group (test set)
                test_class_count = len(test_group[test_group[TARGET_COLUMN] == class_id])
                full_class_count = len(full_group[full_group[TARGET_COLUMN] == class_id])

                if constraint > 1e9:  # Unlimited
                    constraint_str = "Unlimited"
                    pct_str = "N/A"
                else:
                    constraint_str = f"{int(constraint)}"
                    # Calculate percentage relative to test group size
                    pct_str = f"{(constraint/group_test_size)*100:.1f}%" if group_test_size > 0 else "N/A"

                if class_id == 0:  # Only print group info on first class
                    print(f"{group_id:<10} {group_test_size:<12} {class_name:<12} {constraint_str:<12} {pct_str:<15}")
                else:
                    print(f"{'':10} {'':12} {class_name:<12} {constraint_str:<12} {pct_str:<15}")

        if len(local_constraints) > 5:
            print(f"... and {len(local_constraints) - 5} more courses")

        print("\n[Summary for Local Constraints]")

        # Calculate statistics across all groups
        all_group_sizes_test = []
        all_constraint_percentages = []

        for group_id, constraints in local_constraints.items():
            test_group = test_df[test_df['Course'] == group_id]
            group_test_size = len(test_group)
            all_group_sizes_test.append(group_test_size)

            for class_id in range(3):
                if class_id == GRADUATE_CLASS_ID:
                    continue  # Skip unlimited class

                constraint = constraints[class_id]
                if constraint < 1e9 and group_test_size > 0:
                    pct = (constraint / group_test_size) * 100
                    all_constraint_percentages.append(pct)

        if all_constraint_percentages:
            print(f"  Average constraint as % of test group size: {np.mean(all_constraint_percentages):.1f}%")
            print(f"  Min: {np.min(all_constraint_percentages):.1f}%, Max: {np.max(all_constraint_percentages):.1f}%")
            print(f"  Expected (percentage × 10): {percentage * 10 * 100:.1f}%")

        print(f"\n  Test group sizes range: {min(all_group_sizes_test)} to {max(all_group_sizes_test)}")
        print(f"  Average test group size: {np.mean(all_group_sizes_test):.1f}")


def demonstrate_constraint_math():
    """Demonstrate constraint calculation logic with example data."""
    print("\n" + "=" * 80)
    print("CONSTRAINT CALCULATION DEMONSTRATION")
    print("=" * 80)

    # Example: Assume typical dataset sizes
    full_dataset_size = 4420  # Example full dataset size
    test_dataset_size = 442   # 10% of full dataset
    train_dataset_size = full_dataset_size - test_dataset_size

    print(f"\nExample Dataset Sizes:")
    print(f"  Full dataset:  {full_dataset_size} samples")
    print(f"  Train dataset: {train_dataset_size} samples ({train_dataset_size/full_dataset_size:.1%})")
    print(f"  Test dataset:  {test_dataset_size} samples ({test_dataset_size/full_dataset_size:.1%})")

    # Example class distribution in full dataset
    full_dropout = 1326
    full_enrolled = 1768
    full_graduate = 1326

    print(f"\nExample Class Distribution in FULL dataset:")
    print(f"  Dropout:  {full_dropout} samples ({full_dropout/full_dataset_size:.1%})")
    print(f"  Enrolled: {full_enrolled} samples ({full_enrolled/full_dataset_size:.1%})")
    print(f"  Graduate: {full_graduate} samples (Unlimited)")

    constraint_percentage = 0.3

    print(f"\n" + "-" * 80)
    print(f"Computing constraints with percentage = {constraint_percentage} (30%)")
    print(f"CONSTRAINT_SCALE_FACTOR = {CONSTRAINT_SCALE_FACTOR}")
    print("-" * 80)

    # Calculate constraints
    dropout_constraint = np.round(full_dropout * constraint_percentage / CONSTRAINT_SCALE_FACTOR)
    enrolled_constraint = np.round(full_enrolled * constraint_percentage / CONSTRAINT_SCALE_FACTOR)

    print(f"\nConstraint Calculations:")
    print(f"  Dropout:  {full_dropout} × {constraint_percentage} / {CONSTRAINT_SCALE_FACTOR} = {dropout_constraint:.0f}")
    print(f"  Enrolled: {full_enrolled} × {constraint_percentage} / {CONSTRAINT_SCALE_FACTOR} = {enrolled_constraint:.0f}")
    print(f"  Graduate: Unlimited")

    # Now show what this means for test set
    test_dropout = full_dropout // 10  # Approximate test set count
    test_enrolled = full_enrolled // 10

    print(f"\n" + "-" * 80)
    print("What this means for TEST dataset (where constraints are applied):")
    print("-" * 80)

    print(f"\nApproximate Test Set Class Counts:")
    print(f"  Dropout:  ~{test_dropout} samples")
    print(f"  Enrolled: ~{test_enrolled} samples")

    print(f"\nConstraint as Percentage of Test Set:")
    print(f"  Dropout:  {dropout_constraint:.0f} / {test_dropout} = {(dropout_constraint/test_dropout)*100:.1f}%")
    print(f"  Enrolled: {enrolled_constraint:.0f} / {test_enrolled} = {(enrolled_constraint/test_enrolled)*100:.1f}%")

    print(f"\n[KEY INSIGHT]")
    print(f"  Constraint percentage: {constraint_percentage} (30%)")
    print(f"  Scale factor: {CONSTRAINT_SCALE_FACTOR}")
    print(f"  Result: Constraints are ≈{constraint_percentage * 10 * 100}% of test set size")
    print(f"  This is because: test_size ≈ full_size / 10, so (full × 0.3) / 10 ≈ test × 0.3")


if __name__ == "__main__":
    # First show the mathematical demonstration
    demonstrate_constraint_math()

    # Then try to validate with actual data
    print("\n\n")
    validate_constraint_sizing()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("\nWhy CONSTRAINT_SCALE_FACTOR = 10:")
    print("  1. Constraints are computed on FULL dataset (train + test)")
    print("  2. Constraints are applied on TEST dataset (≈10% of full)")
    print("  3. Scale factor accounts for this 10:1 ratio")
    print("  4. Result: Constraint of 0.3 → ≈30% of test set capacity")
    print("\nWithout the scale factor:")
    print("  - 0.3 constraint would mean ≈300% of test set (impossible!)")
    print("  - Constraints could never be satisfied")
    print("\nWith scale factor = 10:")
    print("  - 0.3 constraint means ≈30% of test set (reasonable)")
    print("  - Constraints are achievable while maintaining accuracy")
    print("=" * 80)
