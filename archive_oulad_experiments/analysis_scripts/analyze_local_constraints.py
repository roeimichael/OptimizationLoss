"""
Quick analysis to understand why local constraints fail to satisfy
"""
import pandas as pd
from config.experiment_config import TEST_PATH, TARGET_COLUMN
from src.training.constraints import compute_local_constraints

# Load test data
test_df = pd.read_csv(TEST_PATH)

# Constraint pair [0.5, 0.3]
local_percent = 0.5
groups = test_df['Course'].unique()

# Compute local constraints
local_constraints = compute_local_constraints(test_df, TARGET_COLUMN, local_percent, groups)

print("=" * 80)
print("LOCAL CONSTRAINT ANALYSIS")
print("=" * 80)
print(f"\nTotal courses: {len(local_constraints)}")
print(f"Local percentage: {local_percent * 100}% (50% of each course's actual counts)")

# Analyze each course
course_info = []
for group_id, constraints in local_constraints.items():
    course_data = test_df[test_df['Course'] == group_id]
    total_students = len(course_data)

    # Count actual students per class
    actual_dropout = (course_data[TARGET_COLUMN] == 0).sum()
    actual_enrolled = (course_data[TARGET_COLUMN] == 1).sum()
    actual_graduate = (course_data[TARGET_COLUMN] == 2).sum()

    # Get constraints (50% of actual)
    constraint_dropout = constraints[0]
    constraint_enrolled = constraints[1]

    course_info.append({
        'Course': group_id,
        'Total': total_students,
        'Actual_Dropout': actual_dropout,
        'Constraint_Dropout': constraint_dropout,
        'Actual_Enrolled': actual_enrolled,
        'Constraint_Enrolled': constraint_enrolled,
        'Actual_Graduate': actual_graduate
    })

df = pd.DataFrame(course_info)
df = df.sort_values('Total')

print("\n" + "=" * 80)
print("COURSE BREAKDOWN (sorted by size)")
print("=" * 80)
print(df.to_string(index=False))

print("\n" + "=" * 80)
print("KEY STATISTICS")
print("=" * 80)
print(f"Smallest course: {df['Total'].min()} students")
print(f"Largest course: {df['Total'].max()} students")
print(f"Median course size: {df['Total'].median():.0f} students")
print(f"\nSmallest dropout constraint: {df['Constraint_Dropout'].min()}")
print(f"Smallest enrolled constraint: {df['Constraint_Enrolled'].min()}")

print("\n" + "=" * 80)
print("THE PROBLEM WITH SOFT PREDICTIONS")
print("=" * 80)
print("\nFor a small course with constraint_dropout = 2:")
print("  - Hard predictions: Count of argmax(proba) = dropout")
print("    Example: 2 students predicted as dropout → SATISFIED")
print("  ")
print("  - Soft predictions: Sum of dropout probabilities across ALL students")
print("    Example: 10 students each with 0.25 dropout probability")
print("    Sum = 10 * 0.25 = 2.5 > 2 → VIOLATED!")
print("\nThe model cannot reduce soft predictions below the constraint")
print("without fundamentally changing its probability distribution,")
print("which would hurt predictive accuracy.")

print("\n" + "=" * 80)
print("LOSS FORMULA ANALYSIS")
print("=" * 80)
print("\nWhen soft_count >> constraint:")
print("  loss = E / (E + constraint) where E = soft_count - constraint")
print("  ")
print("Example: constraint=2, soft_count=10")
print("  E = 10 - 2 = 8")
print("  loss = 8 / (8 + 2) = 8/10 = 0.8")
print("  ")
print("Example: constraint=3, soft_count=15")
print("  E = 15 - 3 = 12")
print("  loss = 12 / (12 + 3) = 12/15 = 0.8")
print("\nWhen averaged across many courses with violations, local loss → 1.0")
print("This is why global loss can reach 0 but local loss stays at ~1.0")
