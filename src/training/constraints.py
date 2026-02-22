"""Constraint computation from test data for global and per-group limits."""

import numpy as np
from typing import Dict, List

UNLIMITED = 1e10


def compute_global_constraints(data, target_col, percentage, constrained_class=1,
                               num_classes=2, **kwargs):
    """Return per-class constraint limits. Only constrained_class gets a finite limit."""
    constraints = [UNLIMITED] * num_classes
    count = (data[target_col] == constrained_class).sum()
    constraints[constrained_class] = int(np.round(count * percentage))
    return constraints


def compute_local_constraints(data, target_col, percentage, group_col,
                              constrained_class=1, num_classes=2, **kwargs):
    """Return per-group per-class constraint limits."""
    local = {}
    for group in data[group_col].unique():
        gdata = data[data[group_col] == group]
        if len(gdata) == 0:
            continue
        constraints = [UNLIMITED] * num_classes
        count = (gdata[target_col] == constrained_class).sum()
        constraints[constrained_class] = int(np.round(count * percentage))
        local[group] = constraints
    return local
