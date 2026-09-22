"""Pure quota counting and validation for explicit predictions."""


def _is_int(value):
    return type(value) is int


def _validate_caps(caps, n_classes, name):
    if not isinstance(caps, list) or len(caps) != n_classes:
        raise ValueError("%s must be a list of length n_classes" % name)
    for cap in caps:
        if cap is not None and (not _is_int(cap) or cap < 0):
            raise ValueError("%s entries must be nonnegative integers or None" % name)


def audit_quotas(predictions, groups, n_classes, global_caps, local_caps):
    """Validate explicit caps, count predictions, and report cap violations.

    This function only audits supplied predictions. It does not allocate, infer
    caps, inspect labels, or mutate any caller-owned input.
    """
    if not _is_int(n_classes) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer")
    if not isinstance(predictions, (list, tuple)) or not isinstance(groups, (list, tuple)):
        raise TypeError("predictions and groups must be lists or tuples")
    if not predictions or len(predictions) != len(groups):
        raise ValueError("predictions and groups must have equal nonzero length")
    _validate_caps(global_caps, n_classes, "global_caps")
    if not isinstance(local_caps, dict):
        raise TypeError("local_caps must be a dict")

    ordered_groups = []
    group_set = set()
    for prediction, group in zip(predictions, groups):
        if not _is_int(prediction) or prediction < 0 or prediction >= n_classes:
            raise ValueError("prediction indices must be integers in [0, n_classes)")
        if not isinstance(group, str) or not group:
            raise ValueError("groups must be nonempty strings")
        if group not in group_set:
            group_set.add(group)
            ordered_groups.append(group)

    if set(local_caps) != group_set:
        raise ValueError("local_caps must contain exactly one entry for each group")
    for group in ordered_groups:
        _validate_caps(local_caps[group], n_classes, "local_caps[%r]" % group)

    global_counts = [0] * n_classes
    local_counts = {group: [0] * n_classes for group in ordered_groups}
    for prediction, group in zip(predictions, groups):
        global_counts[prediction] += 1
        local_counts[group][prediction] += 1

    violations = []
    for class_index, cap in enumerate(global_caps):
        if cap is not None and global_counts[class_index] > cap:
            violations.append({
                "scope": "global", "group": None, "class": class_index,
                "count": global_counts[class_index], "cap": cap,
            })
    for group in ordered_groups:
        for class_index, cap in enumerate(local_caps[group]):
            if cap is not None and local_counts[group][class_index] > cap:
                violations.append({
                    "scope": "local", "group": group, "class": class_index,
                    "count": local_counts[group][class_index], "cap": cap,
                })
    return {
        "global_counts": global_counts,
        "local_counts": local_counts,
        "violations": violations,
        "feasible": not violations,
    }
