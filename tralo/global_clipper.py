"""Named global-only diagnostic allocation policies.

These policies consume supplied probabilities and caps. They do not infer
quotas, inspect labels, or claim optimality.
"""

import math


_POLICIES = {"upper_bound_correction", "capped_first"}


def _validate(probabilities, caps, sample_ids, policy):
    if policy not in _POLICIES:
        raise ValueError("unknown policy: %r" % (policy,))
    if not isinstance(probabilities, list) or not probabilities:
        raise ValueError("probabilities must be a nonempty list")
    if not isinstance(probabilities[0], list) or not probabilities[0]:
        raise ValueError("probability rows must be nonempty lists")
    n_classes = len(probabilities[0])
    for row in probabilities:
        if not isinstance(row, list) or len(row) != n_classes:
            raise ValueError("probability rows must have equal length")
        if any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 or p > 1
               for p in row):
            raise ValueError("probabilities must be finite numbers in [0,1]")
        if not math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("each probability row must sum to one")
    if not isinstance(caps, list) or len(caps) != n_classes:
        raise ValueError("caps must have one entry per class")
    if any(cap is not None and (type(cap) is not int or cap < 0) for cap in caps):
        raise ValueError("caps must be nonnegative integers or None")
    if not isinstance(sample_ids, list) or len(sample_ids) != len(probabilities):
        raise ValueError("sample_ids must contain one unique id per sample")
    if any(type(sid) is not str or not sid for sid in sample_ids):
        raise ValueError("sample_ids must be nonempty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")
    if all(cap is not None for cap in caps) and sum(caps) < len(probabilities):
        raise ValueError("finite caps do not provide capacity for every item")
    return n_classes


def _raw(probabilities):
    return [max(range(len(row)), key=lambda c: row[c]) for row in probabilities]


def _pairs(probabilities, sample_ids, items, classes):
    return sorted(
        ((probabilities[i][c], sample_ids[i], c, i) for i in items for c in classes),
        key=lambda pair: (-pair[0], pair[1], pair[2]),
    )


def _assign_pairs(result, counts, caps, pairs):
    for _score, _sid, c, i in pairs:
        if result[i] is not None:
            continue
        if caps[c] is not None and counts[c] >= caps[c]:
            continue
        result[i] = c
        counts[c] += 1


def _upper_bound_correction(probabilities, caps, sample_ids, raw):
    result = [None] * len(raw)
    counts = [0] * len(caps)
    for c, cap in enumerate(caps):
        members = [i for i, prediction in enumerate(raw) if prediction == c]
        keep = members if cap is None or len(members) <= cap else sorted(
            members, key=lambda i: (-probabilities[i][c], sample_ids[i])
        )[:cap]
        for i in keep:
            result[i] = c
            counts[c] += 1
    open_items = [i for i, prediction in enumerate(result) if prediction is None]
    _assign_pairs(result, counts, caps,
                  _pairs(probabilities, sample_ids, open_items, range(len(caps))))
    return result


def _capped_first(probabilities, caps, sample_ids):
    result = [None] * len(probabilities)
    counts = [0] * len(caps)
    capped = [c for c, cap in enumerate(caps) if cap is not None]
    _assign_pairs(result, counts, caps,
                  _pairs(probabilities, sample_ids, range(len(result)), capped))
    open_items = [i for i, prediction in enumerate(result) if prediction is None]
    _assign_pairs(result, counts, caps,
                  _pairs(probabilities, sample_ids, open_items, range(len(caps))))
    return result


def allocate(probabilities, caps, sample_ids, policy):
    """Return one class prediction per item under a named global policy."""
    n_classes = _validate(probabilities, caps, sample_ids, policy)
    raw = _raw(probabilities)
    if policy == "upper_bound_correction":
        result = _upper_bound_correction(probabilities, caps, sample_ids, raw)
    else:
        result = _capped_first(probabilities, caps, sample_ids)
    if any(prediction is None for prediction in result):
        raise ValueError("allocation could not assign every item")
    return result


def allocate_local_capped_first(probabilities, caps, sample_ids, groups, local_caps):
    """Capped-first allocation for one class under pooled and group ceilings.

    Both ceilings are supplied policy inputs. Among feasible capped-class sets,
    select the set with largest sum of capped-class probabilities, breaking ties
    by sample ID. Every remaining item takes its highest-scoring uncapped class.
    No labels or inferred prevalence enter this rule.
    """
    n_classes = _validate(probabilities, caps, sample_ids, "capped_first")
    capped = [c for c, cap in enumerate(caps) if cap is not None]
    if len(capped) != 1 or n_classes < 2:
        raise ValueError("local capped-first requires one capped and one uncapped class")
    c = capped[0]
    if not isinstance(groups, list) or len(groups) != len(probabilities):
        raise ValueError("groups must contain one group per sample")
    if any(type(group) is not str or not group for group in groups):
        raise ValueError("groups must be nonempty strings")
    if not isinstance(local_caps, dict) or set(local_caps) != set(groups):
        raise ValueError("local_caps must cover exactly the supplied groups")
    if any(type(cap) is not int or cap < 0 for cap in local_caps.values()):
        raise ValueError("local caps must be nonnegative integers")

    by_group = {group: [] for group in local_caps}
    for i, group in enumerate(groups):
        by_group[group].append(i)
    order = lambda i: (-probabilities[i][c], sample_ids[i])
    eligible = [i for group, members in by_group.items()
                for i in sorted(members, key=order)[:local_caps[group]]]
    selected = set(sorted(eligible, key=order)[:caps[c]])
    uncapped = [j for j in range(n_classes) if j != c]
    return [c if i in selected else max(uncapped, key=probabilities[i].__getitem__)
            for i in range(len(probabilities))]
