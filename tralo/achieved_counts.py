"""Label-free diagnostic: transfer a model's raw counts as global upper bounds."""

from .global_clipper import allocate


def achieved_caps(probabilities, original_caps, sample_ids):
    """Preserve the constrained class set, replacing its budgets by raw counts.

    These are alternative budgets, not satisfaction of the original policy.
    Unconstrained classes remain unconstrained, including when their count is zero.
    """
    if not isinstance(original_caps, list) or any(
        cap is not None and (type(cap) is not int or cap < 0)
        for cap in original_caps
    ):
        raise ValueError('original caps must be nonnegative integers or None')
    raw = allocate(probabilities, [None] * len(original_caps), sample_ids,
                   'upper_bound_correction')
    return [raw.count(c) if cap is not None else None
            for c, cap in enumerate(original_caps)]
