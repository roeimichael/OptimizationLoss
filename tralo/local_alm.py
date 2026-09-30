"""Label-free PHR residuals and analytic logit gradient for pooled/local caps.

Scope order is pooled, then country codes in lexical order. These functions do
not apply a model update or imply a converged augmented-Lagrangian solve.
"""

import math

import torch

from .alm import augmented_penalty, update_dual
from .knee_end_to_end import infer
from .targeted_step import _place


def _checked(probabilities, groups, capped_class, global_cap, local_caps, dual, rho):
    if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2 or
            len(probabilities) == 0 or
            probabilities.shape[1] < 2 or len(groups) != len(probabilities) or
            type(capped_class) is not int or
            not 0 <= capped_class < probabilities.shape[1]):
        raise ValueError("invalid probability matrix or capped class")
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
        raise ValueError("probabilities must be finite and nonnegative")
    if not bool(torch.allclose(probabilities.sum(1), torch.ones_like(probabilities[:, 0]),
                               atol=1e-6, rtol=0)):
        raise ValueError("probability rows must sum to one")
    if (not isinstance(groups, (list, tuple)) or
            any(type(group) is not str or not group for group in groups) or
            not isinstance(local_caps, dict) or set(local_caps) != set(groups) or
            type(global_cap) is not int or global_cap < 0 or
            any(type(cap) is not int or cap < 0 for cap in local_caps.values())):
        raise ValueError("invalid pooled/local caps or group alignment")
    names = ["global"] + sorted(local_caps)
    if (not isinstance(dual, torch.Tensor) or dual.ndim != 1 or
            len(dual) != len(names) or
            dual.device != probabilities.device or dual.dtype != probabilities.dtype or
            not bool(torch.isfinite(dual).all()) or bool((dual < 0).any()) or
            not math.isfinite(rho) or rho <= 0):
        raise ValueError("invalid PHR multipliers or rho")
    return names


def residuals_and_gradient(probabilities, groups, capped_class, global_cap,
                           local_caps, dual, rho):
    """Return signed residuals, PHR value, and d(PHR)/d(logits).

    `probabilities` are softmax outputs of the current logits. The gradient is
    evaluated at those probabilities and can be replayed through image chunks.
    """
    names = _checked(probabilities, groups, capped_class, global_cap, local_caps,
                     dual, rho)
    q = probabilities[:, capped_class]
    counts = [q.sum()] + [q[[g == name for g in groups]].sum() for name in names[1:]]
    caps = [global_cap] + [local_caps[name] for name in names[1:]]
    scales = probabilities.new_tensor([max(cap, 1) for cap in caps])
    residuals = (torch.stack(counts) - probabilities.new_tensor(caps)) / scales
    penalty = augmented_penalty(residuals, dual, rho)
    coefficients = (dual + rho * residuals).clamp_min(0) / scales
    by_group = {name: coefficients[i + 1] for i, name in enumerate(names[1:])}
    weights = torch.stack([coefficients[0] + by_group[group] for group in groups])
    gradient = -weights[:, None] * q[:, None] * probabilities
    gradient[:, capped_class] += weights * q
    return residuals, penalty, gradient


def projected_dual(residuals, dual, rho):
    """Advance the PHR dual after every snapshot, including a no-step snapshot."""
    if not isinstance(residuals, torch.Tensor) or not bool(torch.isfinite(residuals).all()):
        raise ValueError("PHR residuals must be finite")
    return update_dual(residuals.detach(), dual, rho)


def _scope_counts(probabilities, groups, capped_class, local_caps):
    calls = probabilities.argmax(1).tolist()
    hard = {group: 0 for group in local_caps}
    soft = {group: 0.0 for group in local_caps}
    for i, group in enumerate(groups):
        hard[group] += int(calls[i] == capped_class)
        soft[group] += float(probabilities[i, capped_class])
    return sum(hard.values()), dict(sorted(hard.items())), dict(sorted(soft.items()))


def snapshot_phr_step(model, chunks, groups, capped_class, global_cap,
                      local_caps, dual, rho=0.5, radius=0.1):
    """Take one fixed-dose PHR step on a side model and advance its dual.

    The input model is a disposable copy of a common PTO snapshot. Duals are
    CPU tensors so their scope order and values can be logged independent of GPU.
    """
    if not isinstance(chunks, (list, tuple)) or not chunks:
        raise ValueError("chunks must be replayable and nonempty")
    if type(radius) is not float or radius != 0.1:
        raise ValueError("the named study requires a 0.1 parameter radius")
    if not isinstance(dual, torch.Tensor) or dual.device.type != "cpu":
        raise ValueError("PHR dual must be a CPU tensor")
    before = infer(model, chunks)
    residuals, penalty, upstream = residuals_and_gradient(
        before, groups, capped_class, global_cap, local_caps, dual, rho)
    hard_before, hard_local_before, soft_local_before = _scope_counts(
        before, groups, capped_class, local_caps)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    origin = [p.detach().clone() for p in params]
    was_training = model.training
    record = dict(applied=False, displacement=0.0, penalty_before=float(penalty),
                  residuals_before=residuals.tolist(), dual_before=dual.tolist(),
                  hard_before_global=hard_before, hard_before_local=hard_local_before,
                  soft_before_local=soft_local_before, rho=rho, radius=radius)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        start = 0
        for images in chunks:
            logits = model(images.to(params[0].device))
            end = start + len(images)
            if not torch.allclose(logits.detach().softmax(1).cpu(), before[start:end],
                                  atol=1e-7, rtol=1e-6):
                raise RuntimeError("PHR replay changed logits at fixed weights")
            logits.backward(upstream[start:end].to(logits.device))
            start = end
        if start != len(before):
            raise RuntimeError("PHR cohort length changed")
        gradients = [p.grad.detach().clone() if p.grad is not None else
                     torch.zeros_like(p) for p in params]
        if any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise RuntimeError("nonfinite PHR parameter gradient")
        norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
        record["gradient_norm"] = norm
        if norm > 0:
            direction = [-g / norm for g in gradients]
            # Record every capacity-normalized scope's exact parameter-space
            # directional derivative at the untouched PTO snapshot.
            scope_names = ["global"] + sorted(local_caps)
            scope_dots = {name: 0.0 for name in scope_names}
            start = 0
            for images in chunks:
                logits = model(images.to(params[0].device))
                q = logits.softmax(1)[:, capped_class]
                end = start + len(images)
                terms = [q.sum() / max(global_cap, 1)]
                terms += [q[torch.tensor([g == group for g in groups[start:end]],
                                         device=q.device)].sum() / max(local_caps[group], 1)
                          for group in scope_names[1:]]
                for index, (name, term) in enumerate(zip(scope_names, terms)):
                    scope_grads = torch.autograd.grad(term, params,
                                                      retain_graph=index + 1 < len(terms),
                                                      allow_unused=True)
                    scope_dots[name] += sum(float((g.double() * d.double()).sum())
                                            for g, d in zip(scope_grads, direction)
                                            if g is not None)
                start = end
            record["scope_directional_derivatives"] = scope_dots
            _place(params, origin, direction, radius)
            if any(not bool(torch.isfinite(p).all()) for p in params):
                raise RuntimeError("nonfinite parameter after PHR step")
            moved = math.sqrt(sum(float((p.detach() - old).double().square().sum())
                                  for p, old in zip(params, origin)))
            if abs(moved - radius) > 1e-5:
                raise RuntimeError("PHR parameter dose differs from fixed radius")
            record.update(applied=True, displacement=moved,
                          tensor_displacement_norms=[float((p.detach()-old).double().norm())
                                                     for p, old in zip(params, origin)])
            record["activation_reason"] = "finite_nonzero_phr_gradient"
        else:
            record["scope_directional_derivatives"] = {}
            record["activation_reason"] = "zero_phr_gradient"
        after = infer(model, chunks)
        hard_after, hard_local_after, soft_local_after = _scope_counts(
            after, groups, capped_class, local_caps)
        after_residuals, after_penalty, _ = residuals_and_gradient(
            after, groups, capped_class, global_cap, local_caps, dual, rho)
        next_dual = projected_dual(after_residuals, dual, rho)
        record.update(residuals_after=after_residuals.tolist(),
                      penalty_after=float(after_penalty), dual_after=next_dual.tolist(),
                      soft_before_global=float(before[:, capped_class].sum()),
                      soft_after_global=float(after[:, capped_class].sum()),
                      soft_after_local=soft_local_after,
                      hard_after_global=hard_after, hard_after_local=hard_local_after)
        return record, next_dual
    except BaseException:
        _place(params, origin, [torch.zeros_like(p) for p in params], 0.0)
        raise
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
