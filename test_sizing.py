import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN

NUM_CLASSES = 3
CONSTRAINT_SCALE_FACTOR = 1
UNLIMITED_CONSTRAINT = 1e10
GRADUATE_CLASS_ID = 2
EXCLUDED_COURSE_ID = 1


def compute_global_constraints(data, target_column, percentage):
    constraint = np.zeros(NUM_CLASSES)
    items = data[target_column].value_counts()
    for class_id in items.index:
        constraint[int(class_id)] = np.round(items[class_id] * percentage / CONSTRAINT_SCALE_FACTOR)
    constraint[GRADUATE_CLASS_ID] = UNLIMITED_CONSTRAINT
    return constraint.tolist()


def compute_local_constraints(data, target_column, percentage, groups):
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
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        full_df = pd.concat([train_df, test_df], ignore_index=True)
    except FileNotFoundError:
        print("Data files not found. Using example values.")
        return demonstrate_with_examples()

    print(f"\nDataset: Train={len(train_df)}, Test={len(test_df)}, Full={len(full_df)}")
    print(f"Test/Full ratio: {len(test_df)/len(full_df):.1%}\n")

    for percentage in [0.3, 0.8]:
        print(f"={'='*80}")
        print(f"CONSTRAINT PERCENTAGE: {percentage}")
        print(f"{'='*80}")

        global_constraints = compute_global_constraints(test_df, TARGET_COLUMN, percentage)
        full_counts = full_df[TARGET_COLUMN].value_counts().sort_index()
        test_counts = test_df[TARGET_COLUMN].value_counts().sort_index()

        print(f"\n{'Class':<12} {'Full':<8} {'Test':<8} {'Constraint':<12} {'% of Test':<12}")
        print("-" * 60)

        for class_id in range(3):
            name = ["Dropout", "Enrolled", "Graduate"][class_id]
            full_count = full_counts.get(class_id, 0)
            test_count = test_counts.get(class_id, 0)
            constraint = global_constraints[class_id]

            if constraint > 1e9:
                print(f"{name:<12} {full_count:<8} {test_count:<8} {'Unlimited':<12} {'N/A':<12}")
            else:
                pct = (constraint/test_count)*100 if test_count > 0 else 0
                print(f"{name:<12} {full_count:<8} {test_count:<8} {int(constraint):<12} {pct:.1f}%")

        groups = test_df['Course'].unique()
        local_constraints = compute_local_constraints(test_df, TARGET_COLUMN, percentage, groups)

        print(f"\nLocal Constraints: {len(local_constraints)} courses")

        # Show per-course analysis
        print(f"\n{'Course':<8} {'Class':<10} {'Full':<8} {'Test':<8} {'Split%':<8} {'Constraint':<12} {'% Full':<8} {'% Test':<8}")
        print("-" * 80)

        all_pcts_test = []
        all_pcts_full = []
        for group_id in sorted(local_constraints.keys())[:5]:
            constraints = local_constraints[group_id]
            full_group = full_df[full_df['Course'] == group_id]
            test_group = test_df[test_df['Course'] == group_id]

            for class_id in range(2):
                name = ["Dropout", "Enrolled"][class_id]
                constraint = constraints[class_id]

                if constraint < 1e9:
                    full_count = len(full_group[full_group[TARGET_COLUMN] == class_id])
                    test_count = len(test_group[test_group[TARGET_COLUMN] == class_id])

                    if full_count > 0 and test_count > 0:
                        split_pct = (test_count / full_count) * 100
                        pct_full = (constraint / full_count) * 100
                        pct_test = (constraint / test_count) * 100

                        all_pcts_test.append(pct_test)
                        all_pcts_full.append(pct_full)

                        print(f"{group_id:<8} {name:<10} {full_count:<8} {test_count:<8} {split_pct:<7.1f}% {int(constraint):<12} {pct_full:<7.1f}% {pct_test:<7.1f}%")

        if len(local_constraints) > 5:
            print(f"... and {len(local_constraints) - 5} more courses")

            # Compute for all courses
            for group_id in local_constraints.keys():
                if group_id in sorted(local_constraints.keys())[:5]:
                    continue
                constraints = local_constraints[group_id]
                full_group = full_df[full_df['Course'] == group_id]
                test_group = test_df[test_df['Course'] == group_id]

                for class_id in range(2):
                    constraint = constraints[class_id]
                    if constraint < 1e9:
                        full_count = len(full_group[full_group[TARGET_COLUMN] == class_id])
                        test_count = len(test_group[test_group[TARGET_COLUMN] == class_id])
                        if full_count > 0 and test_count > 0:
                            pct_full = (constraint / full_count) * 100
                            pct_test = (constraint / test_count) * 100
                            all_pcts_test.append(pct_test)
                            all_pcts_full.append(pct_full)

        if all_pcts_full:
            print(f"\nConstraint as % of FULL class count: {np.mean(all_pcts_full):.1f}% (range: {np.min(all_pcts_full):.1f}%-{np.max(all_pcts_full):.1f}%)")
        if all_pcts_test:
            print(f"Constraint as % of TEST class count: {np.mean(all_pcts_test):.1f}% (range: {np.min(all_pcts_test):.1f}%-{np.max(all_pcts_test):.1f}%)")
        print()


def demonstrate_with_examples():
    full_size, test_size = 4420, 442
    full_dropout, full_enrolled = 1326, 1768
    test_dropout, test_enrolled = 132, 176

    print(f"\nDataset: Train=3978, Test={test_size}, Full={full_size}")
    print(f"Test/Full ratio: {test_size/full_size:.1%}\n")

    for percentage in [0.3, 0.8]:
        print(f"={'='*80}")
        print(f"CONSTRAINT PERCENTAGE: {percentage}")
        print(f"{'='*80}")

        dropout_const = int(np.round(full_dropout * percentage / CONSTRAINT_SCALE_FACTOR))
        enrolled_const = int(np.round(full_enrolled * percentage / CONSTRAINT_SCALE_FACTOR))

        print(f"\n{'Class':<12} {'Full':<8} {'Test':<8} {'Constraint':<12} {'% of Test':<12}")
        print("-" * 60)
        print(f"{'Dropout':<12} {full_dropout:<8} {test_dropout:<8} {dropout_const:<12} {(dropout_const/test_dropout)*100:.1f}%")
        print(f"{'Enrolled':<12} {full_enrolled:<8} {test_enrolled:<8} {enrolled_const:<12} {(enrolled_const/test_enrolled)*100:.1f}%")
        print(f"{'Graduate':<12} {1326:<8} {134:<8} {'Unlimited':<12} {'N/A':<12}")
        print()


if __name__ == "__main__":
    validate_constraint_sizing()

