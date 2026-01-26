"""Constraint computation utilities for global and local constraints."""

import numpy as np
from typing import Dict, List

NUM_CLASSES = 3
UNLIMITED_CONSTRAINT = 1e10


def compute_global_constraints(data, target_column: str, percentage: float,
                               unlimited_classes: List[int] = None) -> List[float]:
    """Compute global constraints based on test data class distribution."""
    if unlimited_classes is None:
        unlimited_classes = []

    constraint = np.zeros(NUM_CLASSES)
    items = data[target_column].value_counts()

    for class_id in items.index:
        constraint[int(class_id)] = np.round(items[class_id] * percentage)

    for class_id in unlimited_classes:
        constraint[class_id] = UNLIMITED_CONSTRAINT

    return constraint.tolist()


def compute_local_constraints(data, target_column: str, percentage: float,
                              group_column: str, unlimited_classes: List[int] = None,
                              excluded_groups: List = None) -> Dict[int, List[float]]:
    """Compute local constraints per group based on group-specific class distribution."""
    if unlimited_classes is None:
        unlimited_classes = []
    if excluded_groups is None:
        excluded_groups = []

    local_constraint = {}
    groups = data[group_column].unique()

    for group in groups:
        if group in excluded_groups:
            continue

        data_group = data[data[group_column] == group]
        if len(data_group) == 0:
            continue

        constraint = np.zeros(NUM_CLASSES)
        items = data_group[target_column].value_counts()

        for class_id in items.index:
            constraint[int(class_id)] = np.round(items[class_id] * percentage)

        for class_id in unlimited_classes:
            constraint[class_id] = UNLIMITED_CONSTRAINT

        local_constraint[group] = constraint.tolist()

    return local_constraint
