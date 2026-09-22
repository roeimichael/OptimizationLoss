"""Small, label-free global soft-count constraint primitives."""

import math


def _number(value, name, nonnegative=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("%s must be a finite number" % name)
    if nonnegative and value < 0:
        raise ValueError("%s must be nonnegative" % name)
    return float(value)


def _validate_caps(caps, n_classes):
    if not isinstance(caps, list) or len(caps) != n_classes:
        raise ValueError("caps must be a list with one entry per class")
    for cap in caps:
        if cap is not None and (type(cap) is not int or cap < 0):
            raise ValueError("caps must contain nonnegative integers or None")


def bounded_count_penalty(logits, caps, multipliers, rho):
    """Return a scalar penalty on soft class counts exceeding integer caps."""
    try:
        import torch
    except ImportError as exc:  # lazy dependency by design
        raise RuntimeError("bounded_count_penalty requires torch") from exc
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.shape[0] < 1 or logits.shape[1] < 1):
        raise ValueError("logits must be a nonempty 2-D tensor")
    if not torch.is_floating_point(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating-point values")
    if not isinstance(multipliers, torch.Tensor) or multipliers.ndim != 1:
        raise ValueError("multipliers must be a 1-D tensor")
    n_classes = logits.shape[1]
    _validate_caps(caps, n_classes)
    if multipliers.shape[0] != n_classes or multipliers.device != logits.device:
        raise ValueError("multipliers must match class count and logits device")
    if not bool(torch.isfinite(multipliers).all()) or bool((multipliers < 0).any()):
        raise ValueError("multipliers must be finite and nonnegative")
    rho = _number(rho, "rho")
    soft = torch.softmax(logits, dim=1).sum(dim=0)
    total = logits.sum() * 0.0
    for c, cap in enumerate(caps):
        if cap is None:
            continue
        excess = torch.relu(soft[c] - cap)
        scale = float(max(cap, 1))
        e = excess / scale
        total = total + multipliers[c] * (e / (1.0 + e) + rho * e.square() / (1.0 + e.square()))
    return total


def advance_controller(hard_counts, caps, multipliers, rho, rho_step, lambda_step, frozen):
    """Ratchet duals from hard counts, then freeze once every cap is met."""
    if not isinstance(hard_counts, list) or not isinstance(multipliers, list):
        raise TypeError("hard_counts and multipliers must be lists")
    if len(hard_counts) != len(caps) or len(multipliers) != len(caps):
        raise ValueError("hard_counts, caps, and multipliers must have equal length")
    if not isinstance(frozen, bool):
        raise TypeError("frozen must be bool")
    _validate_caps(caps, len(caps))
    for count in hard_counts:
        if type(count) is not int or count < 0:
            raise ValueError("hard_counts must be nonnegative integers")
    for value in multipliers:
        _number(value, "multiplier")
    rho = _number(rho, "rho")
    rho_step = _number(rho_step, "rho_step")
    lambda_step = _number(lambda_step, "lambda_step")
    old = [float(value) for value in multipliers]
    if frozen:
        return old, rho, True
    new = old[:]
    for c, cap in enumerate(caps):
        if cap is not None and hard_counts[c] > cap:
            new[c] += lambda_step
    new_frozen = all(cap is None or count <= cap for count, cap in zip(hard_counts, caps))
    new_rho = rho if new_frozen else rho + rho_step
    return new, new_rho, new_frozen
