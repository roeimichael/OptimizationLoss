"""Full-pool logit gradients for image training with metadata quotas.

No deployment labels are required. A two-pass evaluation avoids keeping the
entire image pool's autograd graph in memory. The first pass fixes soft-count
coefficients; the second applies their exact vector-Jacobian product to model
parameters. No per-step direction normalization erases TraLO's saturation.
"""

import math

import torch

from .local_bounded_penalty import bounded_local_logit_gradient


def phr_local_logit_gradient(probabilities, groups, capped_class, global_cap,
                             local_caps, dual, rho):
    """Derivative of a pooled/group positive-part PHR penalty at fixed duals."""
    if (not isinstance(dual, dict) or set(dual) != {"global", *local_caps} or
            any(not isinstance(value, (int, float)) or not math.isfinite(value) or
                value < 0 for value in dual.values()) or
            not math.isfinite(rho) or rho <= 0 or
            len(groups) != len(probabilities) or not torch.isfinite(probabilities).all()):
        raise ValueError("one nonnegative dual per scope and positive rho required")
    q = probabilities[:, capped_class]
    caps = {"global": global_cap, **local_caps}
    coefficients = {}
    updated = {}
    for scope, cap in caps.items():
        indices = (list(range(len(groups))) if scope == "global" else
                   [i for i, group in enumerate(groups) if group == scope])
        residual = float((q[indices].sum() - cap) / max(cap, 1))
        updated[scope] = max(0.0, dual[scope] + rho * residual)
        coefficients[scope] = updated[scope] / max(cap, 1)
    weights = torch.stack([q.new_tensor(coefficients["global"] +
                                       coefficients.get(group, 0.0)) for group in groups])
    gradient = -weights[:, None] * q[:, None] * probabilities
    gradient[:, capped_class] += weights * q
    return gradient, updated


def pooled_logit_gradient(probabilities, groups, quota, method, *,
                          multipliers=None, dual=None, rho=0.5,
                          capped_class=1):
    """Calculate one fixed full-pool gradient and the next dual state."""
    if method == "tralo":
        if dual is not None or multipliers is None:
            raise ValueError("TraLO needs fixed multipliers and no dual")
        return bounded_local_logit_gradient(
            probabilities, groups, capped_class, quota["global_cap"],
            quota["local_caps"], multipliers, rho), None
    if method == "phr":
        if multipliers is not None or dual is None:
            raise ValueError("PHR needs duals and no TraLO multipliers")
        return phr_local_logit_gradient(
            probabilities, groups, capped_class, quota["global_cap"],
            quota["local_caps"], dual, rho)
    raise ValueError("unknown local constraint method")


def streaming_parameter_gradient(model, batches, groups, quota, method, *,
                                  multipliers=None, dual=None, rho=0.5,
                                  capped_class=1, parity_atol=1e-5):
    """Accumulate exact constrained parameter gradient over a repeatable pool.

    `batches` is a zero-argument factory yielding `(images, sample_ids)` in
    the same order both passes. Group order must match that sample order.
    The model is held in eval mode; no optimizer or parameter update occurs.
    """
    if model.training:
        raise ValueError("constraint pass requires frozen evaluation behavior")
    first, identities = [], []
    with torch.no_grad():
        for images, sample_ids in batches():
            probabilities = model(images).softmax(1).detach().cpu()
            first.append(probabilities)
            identities.extend(sample_ids)
    if not first or len(identities) != len(groups) or len(set(identities)) != len(identities):
        raise RuntimeError("empty, duplicate or misaligned constraint pool")
    probabilities = torch.cat(first)
    logit_gradient, new_dual = pooled_logit_gradient(
        probabilities, groups, quota, method, multipliers=multipliers,
        dual=dual, rho=rho, capped_class=capped_class)
    if not torch.isfinite(logit_gradient).all():
        raise RuntimeError("nonfinite constraint logit gradient")
    model.zero_grad(set_to_none=True)
    offset = 0
    worst_gap = 0.0
    for images, sample_ids in batches():
        if identities[offset:offset + len(sample_ids)] != list(sample_ids):
            raise RuntimeError("constraint pool order changed between passes")
        logits = model(images)
        actual = logits.detach().softmax(1).cpu()
        gap = float((actual - probabilities[offset:offset + len(sample_ids)]).abs().max())
        worst_gap = max(worst_gap, gap)
        if gap > parity_atol:
            raise RuntimeError("fixed-weight constraint pass changed probabilities")
        torch.autograd.backward(logits, logit_gradient[offset:offset + len(sample_ids)].to(
            device=logits.device, dtype=logits.dtype))
        offset += len(sample_ids)
    if offset != len(groups):
        raise RuntimeError("constraint second pass omitted rows")
    norms = [torch.sum(param.grad.detach().double().square())
             for param in model.parameters() if param.grad is not None]
    norm = float(torch.sqrt(torch.stack(norms).sum())) if norms else 0.0
    if not math.isfinite(norm):
        raise RuntimeError("constraint gradient is nonfinite")
    return {"parameter_gradient_norm": norm,
            "active": norm > 0,
            "fixed_weight_probability_max_abs_gap": worst_gap,
            "sample_count": len(groups), "next_dual": new_dual}


def apply_fixed_correction(model, step_size, max_displacement):
    """Take an unnormalized loss-gradient step, or skip if its dose is unsafe.

    The fixed scalar is chosen before development quality is read. A per-step
    normalization would erase the bounded penalty's magnitude information.
    """
    if (not math.isfinite(step_size) or step_size <= 0 or
            not math.isfinite(max_displacement) or max_displacement <= 0):
        raise ValueError("positive finite correction scale and dose limit required")
    gradients = [(param, param.grad) for param in model.parameters()
                 if param.grad is not None]
    squared = [grad.detach().double().square().sum() for _, grad in gradients]
    norm = float(torch.sqrt(torch.stack(squared).sum())) if squared else 0.0
    proposed = step_size * norm
    if not math.isfinite(proposed):
        raise RuntimeError("nonfinite proposed parameter correction")
    if proposed > max_displacement:
        return {"applied": False, "reason": "dose_limit", "raw_gradient_norm": norm,
                "proposed_displacement_norm": proposed, "actual_displacement_norm": 0.0}
    actual_squares = []
    originals = []
    with torch.no_grad():
        for param, grad in gradients:
            before = param.detach().clone()
            originals.append((param, before))
            param.add_(grad, alpha=-step_size)
            actual_squares.append((param.detach().double() - before.double()).square().sum())
    actual = float(torch.sqrt(torch.stack(actual_squares).sum())) if actual_squares else 0.0
    if not math.isfinite(actual) or actual > max_displacement * (1 + 1e-5):
        with torch.no_grad():
            for param, before in originals:
                param.copy_(before)
        raise RuntimeError("actual correction displacement exceeded dose limit")
    return {"applied": norm > 0, "reason": "applied" if norm > 0 else "slack",
            "raw_gradient_norm": norm, "proposed_displacement_norm": proposed,
            "actual_displacement_norm": actual}
