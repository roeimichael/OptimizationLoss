# Constraint computation from test data distributions.
# Produces global (per-class) and local (per-group per-class) prediction limits.

import logging

import numpy as np

from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _normalize_constrained_classes(constrained_class):
    if isinstance(constrained_class, (list, tuple)):
        return list(constrained_class)
    return [constrained_class]


def _round_to_K(count, percentage, scope_label):
    """Round count*percentage to an integer K and refuse to silently produce K=0
    when there ARE samples to classify. K=0 with count>0 is a config bug
    (asking for 0 budget on a class that exists), and the loss would silently
    skip it -> phantom-satisfied experiment. See AUDIT B12.
    """
    # np.round is banker's rounding: a budget landing exactly on .5 rounds to
    # the EVEN integer, so count=25 at 50% gives 12 and count=35 gives 18. It
    # is applied consistently -- full_panel imports this same function rather
    # than reimplementing it, so trainer and scorer cannot disagree -- but the
    # convention is stated here because .5 budgets do occur at these cap levels.
    K = int(np.round(count * percentage))
    if count > 0 and percentage > 0 and K == 0:
        raise ValueError(
            f"{scope_label}: percentage={percentage} * count={int(count)} "
            f"rounded to K=0. The constraint would vanish silently. "
            f"Pick a larger percentage or move the constrained class.")
    if K == 0:
        # count == 0: the scope holds no true instance of the capped class, so
        # "predict it zero times here" is the correct and tightest budget. It is
        # legitimate but never obvious from a config, and until the loss was
        # fixed it carried no gradient at all -- so say it out loud.
        log.warning("%s: K=0 (this scope has no true instance of the class). "
                    "The budget is real and binding, not a disabled constraint.",
                    scope_label)
    return K


def compute_global_constraints(data, target_col, percentage, constrained_class=4,
                               num_classes=7, **kwargs):
    classes = _normalize_constrained_classes(constrained_class)
    constraints = [UNLIMITED] * num_classes
    for c in classes:
        count = (data[target_col] == c).sum()
        constraints[c] = _round_to_K(count, percentage, f"global K (class {c})")
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
            constraints[c] = _round_to_K(
                count, percentage, f"local K (group {group}, class {c})")
        local[group] = constraints
    return local
