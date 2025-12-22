import numpy as np


def compute_global_constraints(data, target_column, percentage):
    items = data[target_column].value_counts()
    const = items * percentage / 10
    global_constraint = np.round(const.sort_index().values)
    global_constraint = global_constraint.tolist()
    global_constraint[2] = None
    return global_constraint


def compute_local_constraints(data, target_column, percentage, groups):
    local_constraint = {}
    for group in groups:
        if group == 1:
            continue
        department = data['Course'] == group
        data_group = data[department]
        items = data_group[target_column].value_counts()
        const = items * percentage / 10
        local_c = np.round(const.sort_index().values)
        local_c[2] = None
        local_constraint[group] = local_c.tolist()
    return local_constraint
