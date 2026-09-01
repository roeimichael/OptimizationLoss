# Constraint computation from test data distributions.
# Produces global (per-class) and local (per-group per-class) prediction limits.

import logging

import numpy as np

from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def normalize_constrained_classes(constrained_class):
    """One capped class or several, always as a list of ints.

    THE single normalizer. There were three, and they disagreed on None:
    this one wrapped it to [None] and every downstream index raised a bare
    "TypeError: list indices must be integers", the loader's copy returned []
    and silently SKIPPED its own class-occurs-in-slice pre-flight check, and
    src/pipeline/data.py had a third inline copy. A config with
    `constrained_class: null` therefore produced a different failure depending
    on which one saw it first, and one of those failures was silence.
    """
    if constrained_class is None:
        raise ValueError(
            "constrained_class is None. Every arm in this project caps at "
            "least one class; a run with no capped class has no constraint to "
            "satisfy and no metric to report. Set it in configs/protocol.yml.")
    if isinstance(constrained_class, (list, tuple)):
        out = list(constrained_class)
    else:
        out = [constrained_class]
    if not out:
        raise ValueError("constrained_class is empty; expected one class or more.")
    return [int(c) for c in out]


def cap_fraction_for(percentage, cls, classes):
    """The cap fraction for ONE capped class. Scalar or one value per class.

    WHY PER-CLASS EXISTS (FRAMEWORK 2(z16), measured 2026-09-01). A cap poses a
    question only where it forces out >= 10 predictions, leaves errors inside K,
    and sits at p@K < 0.99. On iwildcam those windows are **class 2: K/n
    0.70-0.80** and **class 7: K/n 0.90-1.00**, and on MobileNetV3 they DO NOT
    OVERLAP. With one fraction for every capped class the correct two-class
    experiment was literally inexpressible, and every L20/L30/L50 campaign this
    project ran tested a NON-TASK. So this is not a convenience knob.

    A scalar keeps the historical behaviour EXACTLY -- every config written
    before this existed carries one, and must produce byte-identical budgets.

    A sequence is read in the order `constrained_class` lists the classes, and
    its length must match. Silently recycling or truncating would cap the wrong
    class at the wrong level and look completely normal in every log.
    """
    if isinstance(percentage, (int, float)):
        return float(percentage)
    seq = list(percentage)
    if len(seq) != len(classes):
        raise ValueError(
            "cap fraction list has %d entr(ies) for %d constrained class(es) "
            "%s. It is read positionally, so a mismatch would cap the wrong "
            "class at the wrong level."
            % (len(seq), len(classes), classes))
    try:
        return float(seq[list(classes).index(int(cls))])
    except ValueError:
        raise ValueError("class %r is not in constrained_class %s"
                         % (cls, list(classes)))


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
        # K=0 has TWO causes and the message used to assert only one of them.
        # count == 0: the scope holds no true instance of the capped class, so
        # "predict it zero times here" is the correct and tightest budget.
        # percentage == 0: a 0% cap tag, which the raise above cannot reach
        # because it requires percentage > 0. Either is legitimate and neither
        # is obvious from a config, and until the loss was fixed a K=0
        # constraint carried no gradient at all -- so say which one it is.
        cause = ("this scope has no true instance of the class"
                 if count == 0 else
                 "percentage=%s is zero on a count of %d" % (percentage, int(count)))
        log.warning("%s: K=0 (%s). "
                    "The budget is real and binding, not a disabled constraint.",
                    scope_label, cause)
    return K


def compute_global_constraints(data, target_col, percentage, constrained_class,
                               num_classes, **kwargs):
    classes = normalize_constrained_classes(constrained_class)
    constraints = [UNLIMITED] * num_classes
    for c in classes:
        count = (data[target_col] == c).sum()
        pct = cap_fraction_for(percentage, c, classes)
        constraints[c] = _round_to_K(count, pct, f"global K (class {c})")
    return constraints


def compute_local_constraints(data, target_col, percentage, group_col,
                              constrained_class, num_classes, **kwargs):
    classes = normalize_constrained_classes(constrained_class)
    local = {}
    for group in data[group_col].unique():
        gdata = data[data[group_col] == group]
        if len(gdata) == 0:
            continue
        constraints = [UNLIMITED] * num_classes
        for c in classes:
            count = (gdata[target_col] == c).sum()
            pct = cap_fraction_for(percentage, c, classes)
            constraints[c] = _round_to_K(
                count, pct, f"local K (group {group}, class {c})")
        local[group] = constraints
    return local
