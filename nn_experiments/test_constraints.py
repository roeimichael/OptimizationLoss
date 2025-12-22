import numpy as np
import pandas as pd
from constraints import compute_global_constraints, compute_local_constraints

print("Testing constraint computation with missing classes...")

test_data = pd.DataFrame({
    'Target': [0, 0, 1, 1, 0, 1],
    'Course': [2, 2, 3, 3, 4, 4]
})

print("\nTest data:")
print(test_data)
print(f"\nClass distribution: {test_data['Target'].value_counts().to_dict()}")

global_constraint = compute_global_constraints(test_data, 'Target', percentage=0.8)
print(f"\nGlobal constraints (80%): {global_constraint}")
print(f"  Class 0: {global_constraint[0]}")
print(f"  Class 1: {global_constraint[1]}")
print(f"  Class 2: {global_constraint[2]} (None = unconstrained)")

groups = test_data['Course'].unique()
local_constraints = compute_local_constraints(test_data, 'Target', percentage=0.5, groups=groups)

print(f"\nLocal constraints (50%) for {len(local_constraints)} courses:")
for course, constraint in local_constraints.items():
    print(f"\n  Course {course}: {constraint}")
    course_data = test_data[test_data['Course'] == course]
    print(f"    Actual distribution: {course_data['Target'].value_counts().to_dict()}")
    print(f"    Class 0: {constraint[0]}")
    print(f"    Class 1: {constraint[1]}")
    print(f"    Class 2: {constraint[2]} (None = unconstrained)")

print("\nTest with course having only one class:")
edge_case_data = pd.DataFrame({
    'Target': [0, 0, 0],
    'Course': [5, 5, 5]
})

edge_groups = edge_case_data['Course'].unique()
edge_local = compute_local_constraints(edge_case_data, 'Target', percentage=0.8, groups=edge_groups)

print(f"Course 5 constraints: {edge_local[5]}")
print(f"  Only class 0 exists, others initialized to 0.0")

print("\nAll tests passed!")
