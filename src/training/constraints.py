"""Constraint computation utilities."""

import numpy as np
from typing import Dict, List

NUM_CLASSES = 5
UNLIMITED = 1e10


def compute_global_constraints(data, target_col: str, percentage: float,
                               unlimited_classes: List[int] = None) -> List[float]:
    """
    Compute global constraints based on class distribution.

    Args:
        data: DataFrame with target column
        target_col: Name of target column
        percentage: Constraint as fraction of class count
        unlimited_classes: List of class IDs (1-indexed) with no constraint

    Returns:
        List of constraint values per class
    """
    unlimited_classes = unlimited_classes or []
    counts = data[target_col].value_counts()

    constraints = np.zeros(NUM_CLASSES)
    for class_id in counts.index:
        constraints[int(class_id) - 1] = np.round(counts[class_id] * percentage)

    for class_id in unlimited_classes:
        constraints[class_id - 1] = UNLIMITED

    return constraints.tolist()


def compute_local_constraints(data, target_col: str, percentage: float,
                              group_col: str, unlimited_classes: List[int] = None) -> Dict[int, List[float]]:
    """
    Compute local constraints per group.

    Args:
        data: DataFrame with target and group columns
        target_col: Name of target column
        percentage: Constraint as fraction of class count
        group_col: Name of group column
        unlimited_classes: List of class IDs (1-indexed) with no constraint

    Returns:
        Dict mapping group_id to list of constraint values per class
    """
    unlimited_classes = unlimited_classes or []
    local_constraints = {}

    for group in data[group_col].unique():
        group_data = data[data[group_col] == group]
        if len(group_data) == 0:
            continue

        counts = group_data[target_col].value_counts()
        constraints = np.zeros(NUM_CLASSES)

        for class_id in counts.index:
            constraints[int(class_id) - 1] = np.round(counts[class_id] * percentage)

        for class_id in unlimited_classes:
            constraints[class_id - 1] = UNLIMITED

        local_constraints[group] = constraints.tolist()

    return local_constraints
