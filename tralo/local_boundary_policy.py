"""Label-free, bounded acceptance rule for a proposed constraint direction.

This pure policy does not calculate gradients, alter parameters, infer quotas,
or inspect labels. A caller supplies normalized directional derivatives and a
deterministic probe of the *same* snapshot at a requested displacement radius.
The accepted radius is an observed safety decision, not an F1 optimization.
"""

import math
from collections.abc import Mapping


_VIOLATION_TOLERANCE = 1e-6
_PARTITION_REL_TOLERANCE = 1e-5
_PARTITION_ABS_TOLERANCE = 1e-5


def _nonnegative_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _scope_values(values, groups, name, validator):
    if not isinstance(values, Mapping) or set(values) != set(groups):
        raise ValueError(f"{name} must cover exactly the local scopes")
    return {group: validator(values[group], f"{name}[{group!r}]") for group in groups}


def _normalized_violation(soft, cap):
    return max(0.0, (soft - cap) / max(cap, 1))


def _check_partition(pooled, local, name):
    if not math.isclose(pooled, math.fsum(local.values()),
                        rel_tol=_PARTITION_REL_TOLERANCE,
                        abs_tol=_PARTITION_ABS_TOLERANCE):
        raise ValueError(f"{name} pooled soft count must equal country sum")


def choose_boundary_step(pooled_hard, pooled_soft, pooled_cap,
                         local_soft, local_caps, pooled_derivative,
                         local_derivatives, probe, *, max_radius=0.1,
                         max_halvings=12):
    """Return an observed radius or a recorded zero-step fallback.

    Derivatives are wrt displacement radius of normalized soft residuals
    ``(soft_count - cap) / max(cap, 1)``. Local caps are upper bounds; their
    sum need not equal or be below the pooled cap. The callback returns a map
    with ``pooled_hard``, ``pooled_soft``, and ``local_soft`` at each radius.
    It must replay the same model/data without changing RNG or mutable state.
    The caller remains responsible for enforcing that replay contract.
    """
    pooled_hard = _nonnegative_integer(pooled_hard, "pooled_hard")
    pooled_soft = _nonnegative_number(pooled_soft, "pooled_soft")
    pooled_cap = _nonnegative_integer(pooled_cap, "pooled_cap")
    if not isinstance(local_caps, Mapping) or not local_caps or any(
            type(group) is not str or not group for group in local_caps):
        raise ValueError("local_caps must map nonempty scope names to caps")
    groups = sorted(local_caps)
    caps = _scope_values(local_caps, groups, "local_caps", _nonnegative_integer)
    local = _scope_values(local_soft, groups, "local_soft", _nonnegative_number)
    _check_partition(pooled_soft, local, "initial")
    derivatives = _scope_values(local_derivatives, groups, "local_derivatives",
                                _finite_number)
    pooled_derivative = _finite_number(pooled_derivative, "pooled_derivative")
    max_radius = _finite_number(max_radius, "max_radius")
    if not 0 < max_radius <= 0.1:
        raise ValueError("max_radius must be in (0, 0.1]")
    if type(max_halvings) is not int or not 0 <= max_halvings <= 12:
        raise ValueError("max_halvings must be an integer in [0, 12]")
    if not callable(probe):
        raise ValueError("probe must be callable")

    # Prefix local names so even a real group named "pooled" cannot overwrite
    # the distinct pooled scope in evidence or derivative checks.
    local_keys = {group: f"local:{group}" for group in groups}
    initial_violations = {"pooled": _normalized_violation(pooled_soft, pooled_cap)}
    initial_violations.update({local_keys[group]: _normalized_violation(local[group], caps[group])
                               for group in groups})
    initial_total = sum(initial_violations.values())
    result = dict(applied=False, radius=0.0, initial_radius=0.0,
                  reason="no_positive_violation", probes=[],
                  initial_positive_violations=initial_violations,
                  initial_total_positive_violation=initial_total,
                  pooled_hard_floor=max(0, min(pooled_hard, pooled_cap) - 1),
                  pooled_soft_floor=max(0.0, min(pooled_soft, pooled_cap) - 1.0))
    if initial_total == 0.0:
        return result

    scope_derivatives = {"pooled": pooled_derivative}
    scope_derivatives.update({local_keys[group]: derivatives[group] for group in groups})
    conflicts = [name for name in sorted(initial_violations)
                 if initial_violations[name] > 0 and scope_derivatives[name] >= 0]
    if conflicts:
        result.update(reason="conflicting_direction",
                      conflicting_scopes=conflicts)
        return result

    candidate_limits = [initial_violations[name] / -scope_derivatives[name]
                        for name in initial_violations if initial_violations[name] > 0]
    radius = min(max_radius, min(candidate_limits))
    if not math.isfinite(radius) or radius <= 0:
        result["reason"] = "unrepresentable_initial_radius"
        return result
    result["initial_radius"] = radius

    for halving in range(max_halvings + 1):
        if radius <= 0:
            break
        observed = probe(radius)
        if not isinstance(observed, Mapping) or set(observed) != {
                "pooled_hard", "pooled_soft", "local_soft"}:
            raise ValueError("probe must return pooled_hard, pooled_soft, local_soft")
        hard = _nonnegative_integer(observed["pooled_hard"], "probe.pooled_hard")
        soft = _nonnegative_number(observed["pooled_soft"], "probe.pooled_soft")
        by_group = _scope_values(observed["local_soft"], groups, "probe.local_soft",
                                 _nonnegative_number)
        _check_partition(soft, by_group, "probe")
        candidate_violations = {"pooled": _normalized_violation(soft, pooled_cap)}
        candidate_violations.update({local_keys[group]: _normalized_violation(by_group[group], caps[group])
                                     for group in groups})
        total = sum(candidate_violations.values())
        rejections = []
        if hard < result["pooled_hard_floor"]:
            rejections.append("pooled_hard_floor")
        if soft < result["pooled_soft_floor"]:
            rejections.append("pooled_soft_floor")
        for name in sorted(candidate_violations):
            if candidate_violations[name] > initial_violations[name] + _VIOLATION_TOLERANCE:
                rejections.append(f"worsened_soft_violation:{name}")
        if not total <= initial_total - _VIOLATION_TOLERANCE:
            rejections.append("insufficient_total_violation_reduction")
        result["probes"].append(dict(
            halving=halving, radius=radius, pooled_hard=hard, pooled_soft=soft,
            local_soft=by_group, positive_violations=candidate_violations,
            total_positive_violation=total, accepted=not rejections,
            rejections=rejections))
        if not rejections:
            result.update(applied=True, radius=radius, reason="accepted")
            return result
        radius /= 2.0

    result["reason"] = "no_acceptable_probe"
    return result
