import numpy as np


def compute_global_constraints(data, target_column, percentage):
    n_classes = 3
    constraint = np.zeros(n_classes)

    items = data[target_column].value_counts()

    for class_id in items.index:
        constraint[int(class_id)] = np.round(items[class_id] * percentage)

    constraint[2] = None
    return constraint.tolist()


def compute_local_constraints(data, target_column, percentage, groups):
    n_classes = 3
    local_constraint = {}

    for group in groups:
        constraint = np.zeros(n_classes)

        data_group = data[data['Course'] == group]

        if len(data_group) == 0:
            continue

        items = data_group[target_column].value_counts()

        for class_id in items.index:
            constraint[int(class_id)] = np.round(items[class_id] * percentage)

        constraint[2] = None
        local_constraint[group] = constraint.tolist()

    return local_constraint
