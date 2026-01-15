import numpy as np

# Constants for constraint computation
NUM_CLASSES = 3
CONSTRAINT_SCALE_FACTOR = 1  # Changed from 10 to 1 - use actual percentage
UNLIMITED_CONSTRAINT = 1e10  # Value representing unlimited constraint
GRADUATE_CLASS_ID = 2  # Graduate class is always unlimited
EXCLUDED_COURSE_ID = 1  # Course ID to exclude from local constraints


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
