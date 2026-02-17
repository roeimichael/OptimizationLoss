"""Constraint computation utilities.

Constraints are computed from TEST data only.
Binary classification: class 1 (churn) gets a percentage limit.
Class 0 (no churn) is unlimited.

The constraint value = percentage × class_1_count_in_test
"""

import numpy as np
from typing import Dict, List

NUM_CLASSES = 2  # Binary: 0 = no churn, 1 = churn
UNLIMITED = 1e10

# The constrained class (binary: 1 = churn)
CONSTRAINED_CLASS = 1


def compute_global_constraints(data, target_col: str, percentage: float,
                               unlimited_classes: List[int] = None) -> List[float]:
    """
    Compute global constraints based on test data.

    Binary: class 1 (churn) gets a percentage-based limit.
    Class 0 is unlimited.

    Args:
        data: DataFrame with target column (binary labels 0/1)
        target_col: Name of target column
        percentage: Fraction of class 1 count to use as limit
        unlimited_classes: DEPRECATED - kept for backward compatibility, ignored

    Returns:
        List of constraint values per class (index = class ID)
        Example: [1e10, 469] where 469 = percentage * class_1_count
    """
    # Initialize all classes as unlimited
    constraints = [UNLIMITED] * NUM_CLASSES

    # Count constrained class in test data
    constrained_count = (data[target_col] == CONSTRAINED_CLASS).sum()

    # Set the constraint for the constrained class
    constraints[CONSTRAINED_CLASS] = int(np.round(constrained_count * percentage))

    return constraints


def compute_local_constraints(data, target_col: str, percentage: float,
                              group_col: str, unlimited_classes: List[int] = None) -> Dict[int, List[float]]:
    """
    Compute local constraints per group (membership_tier).

    Binary: class 1 (churn) gets a percentage-based limit within each group.
    Class 0 is unlimited.

    Args:
        data: DataFrame with target and group columns (binary labels 0/1)
        target_col: Name of target column
        percentage: Fraction of class 1 count to use as limit
        group_col: Name of group column
        unlimited_classes: DEPRECATED - kept for backward compatibility, ignored

    Returns:
        Dict mapping group_id to list of constraint values per class
    """
    local_constraints = {}

    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        if len(group_data) == 0:
            continue

        # Initialize all classes as unlimited for this group
        constraints = [UNLIMITED] * NUM_CLASSES

        # Count constrained class in this group
        constrained_count = (group_data[target_col] == CONSTRAINED_CLASS).sum()

        # Set the constraint for the constrained class
        constraints[CONSTRAINED_CLASS] = int(np.round(constrained_count * percentage))

        local_constraints[group] = constraints

    return local_constraints
