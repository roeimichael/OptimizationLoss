# Constraint computation from test data distributions.
# Produces global (per-class) and local (per-group per-class) prediction limits.

import numpy as np
from typing import Dict, List, Union

UNLIMITED = 1e10


def _normalize_constrained_classes(constrained_class):
    if isinstance(constrained_class, (list, tuple)):
        return list(constrained_class)
    return [constrained_class]


def compute_global_constraints(data, target_col, percentage, constrained_class=4,
                               num_classes=7, **kwargs):
    classes = _normalize_constrained_classes(constrained_class)
    constraints = [UNLIMITED] * num_classes
    for c in classes:
        count = (data[target_col] == c).sum()
        constraints[c] = int(np.round(count * percentage))
    return constraints


def compute_local_constraints(data, target_col, percentage, group_col,
                              constrained_class=4, num_classes=7, **kwargs):
    classes = _normalize_constrained_classes(constrained_class)
    local = {}
    for group in data[group_col].unique():
        gdata = data[data[group_col] == group]
        if len(gdata) == 0:
            continue
        constraints = [UNLIMITED] * num_classes
        for c in classes:
            count = (gdata[target_col] == c).sum()
            constraints[c] = int(np.round(count * percentage))
        local[group] = constraints
    return local
