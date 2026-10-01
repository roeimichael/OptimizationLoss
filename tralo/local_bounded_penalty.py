"""TraLO's saturating soft-count penalty over a pooled and metadata scope.

The group key is external tabular metadata. It is used to build constraints,
not injected into the image classifier or inferred from individual labels.
This module defines the actual scalar objective and its logit derivative; a
normalized count direction or hard-cap line search is a different algorithm.
"""

import math

import torch


def _checked(probabilities, groups, capped_class, global_cap, local_caps,
             multipliers, rho):
    if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2 or
            probabilities.shape[0] < 1 or probabilities.shape[1] < 2 or
            not torch.is_floating_point(probabilities) or
            not bool(torch.isfinite(probabilities).all())):
        raise ValueError("finite nonempty probability/logit matrix required")
    if (not isinstance(groups, (list, tuple)) or len(groups) != len(probabilities) or
            any(type(group) is not str or not group or group == "global"
                for group in groups)):
        raise ValueError("one nonempty metadata group per image required")
    if type(capped_class) is not int or not 0 <= capped_class < probabilities.shape[1]:
        raise ValueError("invalid capped class")
    if (type(global_cap) is not int or global_cap < 0 or
            not isinstance(local_caps, dict) or not set(local_caps).issubset(set(groups)) or
            any(type(cap) is not int or cap < 0 for cap in local_caps.values())):
        raise ValueError("integer global and observed local caps required")
    if (not isinstance(multipliers, dict) or
            set(multipliers) != {"global", *local_caps} or
            any(type(value) not in (int, float) or not math.isfinite(value) or
                    value < 0 for value in multipliers.values())):
        raise ValueError("one finite nonnegative multiplier per scope required")
    if type(rho) not in (int, float) or not math.isfinite(rho) or rho < 0:
        raise ValueError("finite nonnegative rho required")


def _scope_indices(groups, local_caps):
    return {"global": list(range(len(groups))), **{
        group: [index for index, item in enumerate(groups) if item == group]
        for group in sorted(local_caps)}}


def bounded_local_count_penalty(logits, groups, capped_class, global_cap,
                                local_caps, multipliers, rho):
    """Differentiable pooled-plus-group extension of TraLO's bounded penalty."""
    _checked(logits, groups, capped_class, global_cap, local_caps,
             multipliers, rho)
    q = logits.softmax(1)[:, capped_class]
    caps = {"global": global_cap, **local_caps}
    total = logits.sum() * 0.0
    for scope, indices in _scope_indices(groups, local_caps).items():
        e = torch.relu(q[indices].sum() - caps[scope]) / max(caps[scope], 1)
        total = total + multipliers[scope] * (
            e / (1.0 + e) + rho * e.square() / (1.0 + e.square()))
    return total


def bounded_local_logit_gradient(probabilities, groups, capped_class,
                                 global_cap, local_caps, multipliers, rho):
    """Exact d(penalty)/d(logits), with zero subgradient at a cap boundary."""
    _checked(probabilities, groups, capped_class, global_cap, local_caps,
             multipliers, rho)
    if (bool((probabilities < 0).any()) or
            not torch.allclose(probabilities.sum(1),
                               torch.ones_like(probabilities[:, 0]), atol=1e-6)):
        raise ValueError("invalid probability rows")
    q = probabilities[:, capped_class]
    caps = {"global": global_cap, **local_caps}
    coefficients = {}
    for scope, indices in _scope_indices(groups, local_caps).items():
        cap = caps[scope]
        soft = q[indices].sum()
        if bool(soft > cap):
            e = (soft - cap) / max(cap, 1)
            coefficients[scope] = (multipliers[scope] / max(cap, 1) *
                (1.0 / (1.0 + e).square() +
                 2.0 * rho * e / (1.0 + e.square()).square()))
        else:
            coefficients[scope] = soft.new_zeros(())
    zero = q.new_zeros(())
    weights = torch.stack([coefficients["global"] + coefficients.get(group, zero)
                           for group in groups])
    result = -weights[:, None] * q[:, None] * probabilities
    result[:, capped_class] += weights * q
    return result
