import logging
import math
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
            "constrained_class is None. Every arm in this project caps at least one class; a run with no capped class has no constraint to satisfy and no metric to report. Set it in configs/protocol.yml."
        )
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
            "cap fraction list has %d entr(ies) for %d constrained class(es) %s. It is read positionally, so a mismatch would cap the wrong class at the wrong level."
            % (len(seq), len(classes), classes)
        )
    try:
        return float(seq[list(classes).index(int(cls))])
    except ValueError:
        raise ValueError(
            "class %r is not in constrained_class %s" % (cls, list(classes))
        )


def _round_to_K(count, percentage, scope_label):
    """Round count*percentage to an integer K and refuse to silently produce K=0
    when there ARE samples to classify. K=0 with count>0 is a config bug
    (asking for 0 budget on a class that exists), and the loss would silently
    skip it -> phantom-satisfied experiment. See AUDIT B12.
    """
    K = int(np.round(count * percentage))
    if count > 0 and percentage > 0 and (K == 0):
        raise ValueError(
            f"{scope_label}: percentage={percentage} * count={int(count)} rounded to K=0. The constraint would vanish silently. Pick a larger percentage or move the constrained class."
        )
    if K == 0:
        cause = (
            "this scope has no true instance of the class"
            if count == 0
            else "percentage=%s is zero on a count of %d" % (percentage, int(count))
        )
        log.warning(
            "%s: K=0 (%s). The budget is real and binding, not a disabled constraint.",
            scope_label,
            cause,
        )
    return K


SHARE_TOLERANCE = 1e-9


def normalize_group_shares(shares, groups_present):
    """Externally given budget shares, one per group, validated against the data.

    WHY THIS EXISTS. Shifman et al. (2025) Eq. (2) takes Phi[lambda][i] as a
    GIVEN integer bound -- "30 beds for gold members" -- and Eq. (3) takes
    Psi(i) the same way. This project only ever derived Phi from each group's
    OWN true prevalence, which makes the budget proportional to the answer:
    a group holding twice as many positives is handed twice the budget, so the
    cap cannot express a policy that deliberately favours a small group. That
    is the whole point of a local feature, and it was inexpressible.

    A share is a fraction of the capped class's TOTAL local budget, not of the
    group's own count. Shares must cover exactly the groups present and sum to
    one, so the per-group ceilings partition the budget instead of floating
    free of it. Both failures are raised, never defaulted: a missing group
    would silently receive no ceiling (unconstrained, the opposite of the
    intent) and shares summing to less than one would silently shrink the
    budget while every log still reported the configured percentage.
    """
    if not isinstance(shares, dict):
        raise ValueError(
            "group_budget_shares must be a mapping {group -> share}, got %r" % type(shares).__name__
        )
    try:
        out = {int(g): float(v) for g, v in shares.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "group_budget_shares keys must be group ids and values must be numbers: %s" % exc
        )
    present = {int(g) for g in groups_present}
    if set(out) != present:
        missing = sorted(present - set(out))
        extra = sorted(set(out) - present)
        raise ValueError(
            "group_budget_shares must name exactly the groups in the data. "
            "missing=%s extra=%s. A missing group would silently get no local ceiling."
            % (missing, extra)
        )
    negative = sorted(g for g, v in out.items() if v < 0)
    if negative:
        raise ValueError("group_budget_shares has negative share(s) for group(s) %s" % negative)
    total = sum(out.values())
    if abs(total - 1.0) > SHARE_TOLERANCE:
        raise ValueError(
            "group_budget_shares sum to %.12g, not 1. They partition the class budget, "
            "so anything else silently changes the budget the percentage asked for." % total
        )
    return out


def split_budget(total, shares, scope_label):
    """Split an integer budget across groups by share, exactly.

    LARGEST REMAINDER, not round(). Rounding each part independently does not
    reconstruct the total -- three groups at 1/3 of 100 round to 33 each and
    lose an item, and the lost item is invisible in every log. Largest
    remainder is the standard apportionment rule and guarantees
    `sum(parts) == total` for any share vector, which is the invariant that
    makes the local ceilings and the global ceiling agree when the two
    percentages are equal (the protocol Shifman et al. section 3.2 uses).

    Ties break on the lowest group id so the split is deterministic.
    """
    order = sorted(shares)
    raw = [total * shares[g] for g in order]
    parts = [int(math.floor(r)) for r in raw]
    remainder = int(total) - sum(parts)
    if remainder:
        by_frac = sorted(range(len(order)), key=lambda i: (-(raw[i] - parts[i]), order[i]))
        for i in by_frac[:remainder]:
            parts[i] += 1
    assigned = dict(zip(order, parts))
    if sum(parts) != int(total):
        raise AssertionError(
            "%s: split %s does not sum to %d" % (scope_label, assigned, int(total))
        )
    for g, k in assigned.items():
        if k == 0 and total > 0:
            log.warning(
                "%s: group %s receives K=0 of a budget of %d (share %.4g). "
                "The budget is real and binding, not a disabled constraint.",
                scope_label, g, int(total), shares[g],
            )
    return assigned


def compute_global_constraints(
    data, target_col, percentage, constrained_class, num_classes, **kwargs
):
    classes = normalize_constrained_classes(constrained_class)
    constraints = [UNLIMITED] * num_classes
    for c in classes:
        count = (data[target_col] == c).sum()
        pct = cap_fraction_for(percentage, c, classes)
        constraints[c] = _round_to_K(count, pct, f"global K (class {c})")
    return constraints


def compute_local_constraints(
    data,
    target_col,
    percentage,
    group_col,
    constrained_class,
    num_classes,
    group_budget_shares=None,
    **kwargs
):
    """Per-(group, class) ceilings: Phi[lambda][i] of Shifman et al. Eq. (2).

    TWO DERIVATIONS, and the difference is the scientific one.

    `group_budget_shares=None` (the default, and every campaign run before
    2026-09-16): K = round(percentage * count of the class IN THAT GROUP). The
    ceiling is proportional to the group's own prevalence.

    `group_budget_shares={group: share}`: the class's total local budget,
    round(percentage * count of the class over the WHOLE pool), is apportioned
    across groups by externally given shares. This is the paper's reading and
    the one a resource-allocation problem actually has -- 100 beds split 60/30/10
    across membership tiers regardless of how many of each tier applied.

    The default is kept bit-identical on purpose: every completed run in this
    corpus was produced by it, and changing it silently would make the existing
    results incomparable to new ones.
    """
    classes = normalize_constrained_classes(constrained_class)
    if group_budget_shares is not None:
        return _local_constraints_from_shares(
            data, target_col, percentage, group_col, classes, num_classes,
            group_budget_shares,
        )
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
                count, pct, f"local K (group {group}, class {c})"
            )
        local[group] = constraints
    return local


def _local_constraints_from_shares(
    data, target_col, percentage, group_col, classes, num_classes, shares
):
    groups = sorted(int(g) for g in data[group_col].unique())
    shares = normalize_group_shares(shares, groups)
    local = {g: [UNLIMITED] * num_classes for g in groups}
    for c in classes:
        pct = cap_fraction_for(percentage, c, classes)
        total = _round_to_K(
            (data[target_col] == c).sum(), pct, "local budget (class %d, pooled)" % c
        )
        for g, k in split_budget(total, shares, "local K (class %d)" % c).items():
            local[g][c] = k
    return local
