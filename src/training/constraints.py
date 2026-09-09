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


def permute_local_budgets(local_con, constrained_class, seed, log=None):
    """THE CONTROL FOR BUDGET *CONTENT*. Shuffle K across groups, keep the sum.

    🛑 WHAT NO EXISTING CONTROL COVERS. `tralo_coin*` controls the step's
    DIRECTION at matched norm; `tralo_null` controls the COMPUTE (same warm-up,
    same 29 epochs, lambda=0); `tralo_reseed` controls the RNG. Not one of them
    controls whether the constraint reads the budget it is given. This arm
    hands TraLO a budget vector that is a PERMUTATION of the true one across
    groups -- same multiset, same total, same number of K=0 ceilings, same
    machinery byte for byte -- and only the assignment of budgets to groups is
    wrong.

    So the two outcomes are both decisive, which is why it is worth a campaign:

      permuted == true   the machinery is a PERTURBATION, not an enforcement
                         mechanism. It bounds the whole aggregate-penalty
                         family at once and no better dual design can help,
                         because none of them reads anything either.
      permuted <  true   TraLO reads real budget information, the null is only
                         about the DOSE or the delivery, and the conclusion
                         changes.

    ⚠️ PER CLASS, INDEPENDENTLY. Permuting the (group, class) pairs jointly
    would also move budget BETWEEN classes, which changes the per-class totals
    and stops being the same experiment.

    🛑 IT REFUSES TO BE INERT, and that is not decoration. If every group's K
    is identical for a class, a permutation is the IDENTITY and this arm is
    `tralo` under another name -- the exact shape of the six inert flags in
    this project's catalogue. It raises rather than running.

    `seed` is combined with the run seed by the caller, so the four seeds of a
    cell draw FOUR different permutations. The claim is about permuted budgets
    in general, not about one unlucky draw.
    """
    if seed is None:
        return local_con
    import numpy as _np
    classes = normalize_constrained_classes(constrained_class)
    groups = sorted(local_con)
    if len(groups) < 2:
        raise ValueError(
            "permute_local_budgets needs >= 2 groups to permute across; got "
            "%d. With one group the permutation is the identity and the arm "
            "would be `tralo` under another name." % len(groups))
    rng = _np.random.RandomState(int(seed) % (2 ** 31 - 1))
    out = {g: list(local_con[g]) for g in groups}
    moved_any = False
    for c in classes:
        ks = [local_con[g][c] for g in groups]
        if len(set(ks)) == 1:
            raise ValueError(
                "permute_local_budgets is INERT for class %d: all %d groups "
                "have K=%s, so every permutation is the identity and this arm "
                "is `tralo` under another name. Choose a cap or a dataset "
                "whose per-group budgets differ." % (c, len(groups), ks[0]))
        # Re-draw until the permutation actually moves something. With
        # distinct values present this terminates immediately in practice; the
        # bound exists so a pathological case raises instead of spinning.
        for _attempt in range(64):
            perm = rng.permutation(len(groups))
            shuffled = [ks[i] for i in perm]
            if shuffled != ks:
                break
        else:
            raise ValueError(
                "permute_local_budgets drew the identity 64 times for class "
                "%d; refusing to run an arm that is `tralo` under another "
                "name." % c)
        n_moved = sum(1 for a, b in zip(ks, shuffled) if a != b)
        moved_any = moved_any or n_moved > 0
        for g, k in zip(groups, shuffled):
            out[g][c] = k
        if log is not None:
            log.info("permuted local budgets for class %d across %d groups: "
                     "%d group(s) changed, total held at %s (was %s, now %s)",
                     c, len(groups), n_moved, sum(ks), ks, shuffled)
    if not moved_any:
        raise ValueError("permute_local_budgets changed nothing; refusing.")
    return out
