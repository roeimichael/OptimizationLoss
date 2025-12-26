import numpy as np


def compute_global_constraints(data, target_column, percentage):
    n_classes = 3
    constraint = np.zeros(n_classes)
    items = data[target_column].value_counts()

    # Fill in counts for classes that exist
    for class_id in items.index:
        constraint[int(class_id)] = np.round(items[class_id] * percentage / 10)

    constraint[2] = None  # Graduate class unconstrained
    return constraint.tolist()


def compute_local_constraints(data, target_column, percentage, groups):
    n_classes = 3
    local_constraint = {}

    for group in groups:
        if group == 1:
            continue

        data_group = data[data['Course'] == group]

        # Skip empty groups
        if len(data_group) == 0:
            continue

        # Initialize constraint array with zeros
        constraint = np.zeros(n_classes)
        items = data_group[target_column].value_counts()

        # Fill in counts for classes that exist in this group
        for class_id in items.index:
            constraint[int(class_id)] = np.round(items[class_id] * percentage / 10)

        constraint[2] = None  # Graduate class unconstrained
        local_constraint[group] = constraint.tolist()

    return local_constraint
