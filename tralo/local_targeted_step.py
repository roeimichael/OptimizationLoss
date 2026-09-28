"""Label-free side step for one class with pooled and group hard ceilings.

The direction combines active soft-count gradients. Their sum need not descend
every active scope, so exact parameter-space directional derivatives reject
conflicting directions. A near-origin replay records the observed soft change.
A bounded scan accepts the first observed radius satisfying every hard ceiling.
Hard counts need not be monotone; no minimum-radius claim is made.
"""

import math

import torch

from .knee_end_to_end import infer
from .targeted_step import _place


def _validate(groups, n, capped_class, n_classes, global_cap, local_caps):
    if not isinstance(groups, (list, tuple)) or len(groups) != n:
        raise ValueError("groups must contain one group per image")
    if any(type(group) is not str or not group for group in groups):
        raise ValueError("groups must be nonempty strings")
    if not isinstance(local_caps, dict) or set(local_caps) != set(groups):
        raise ValueError("local_caps must cover exactly the observed groups")
    if type(capped_class) is not int or not 0 <= capped_class < n_classes:
        raise ValueError("invalid capped class")
    if type(global_cap) is not int or global_cap < 0:
        raise ValueError("global_cap must be a nonnegative integer")
    if any(type(cap) is not int or cap < 0 for cap in local_caps.values()):
        raise ValueError("local caps must be nonnegative integers")


def _counts(probabilities, groups, c):
    calls = probabilities.argmax(1).tolist()
    local = {group: 0 for group in set(groups)}
    for call, group in zip(calls, groups):
        local[group] += int(call == c)
    return sum(local.values()), local


def local_count_logit_gradient(probabilities, groups, c, global_cap, local_caps,
                               global_active, local_active):
    """Analytic gradient of the active, normalized soft-count sum wrt logits."""
    if probabilities.ndim != 2 or len(probabilities) != len(groups):
        raise ValueError("probabilities and groups must align")
    _validate(groups, len(probabilities), c, probabilities.shape[1], global_cap, local_caps)
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("nonfinite probabilities")
    if not isinstance(global_active, bool) or not set(local_active) <= set(local_caps):
        raise ValueError("invalid active scopes")
    weights = probabilities.new_tensor([
        (1.0 / max(global_cap, 1) if global_active else 0.0) +
        (1.0 / max(local_caps[group], 1) if group in local_active else 0.0)
        for group in groups
    ])
    q = probabilities[:, c]
    gradient = -weights[:, None] * q[:, None] * probabilities
    gradient[:, c] += weights * q
    return gradient


def _random_direction(unit, generator):
    step = []
    for u in unit:
        if u is None:
            step.append(None)
            continue
        share = float(u.double().norm())
        if share == 0.0:
            step.append(torch.zeros_like(u))
            continue
        noise = torch.randn(u.shape, generator=generator, dtype=torch.float64)
        step.append((noise * (share / float(noise.norm()))).to(dtype=u.dtype, device=u.device))
    return step


def local_targeted_step(model, chunks, groups, capped_class, global_cap, local_caps,
                        sham_generator=None, r0=1e-3, max_doublings=30,
                        scan_points=24, max_radius=0.1, fixed_radius=None,
                        global_only_direction=False):
    """Try one joint direction on a side copy; restore parameters on failure.

    The sham uses the real direction's sampled radius and per-tensor norms. Its
    output is allowed to violate the ceilings; it is a dose control, not a
    feasible allocator. Deployment must use the same joint allocator for arms.
    """
    if not isinstance(chunks, (list, tuple)) or not chunks:
        raise ValueError("chunks must be a nonempty replayable sequence")
    if not math.isfinite(r0) or r0 <= 0 or type(max_doublings) is not int or max_doublings < 0:
        raise ValueError("invalid radius search")
    if not math.isfinite(max_radius) or max_radius < r0:
        raise ValueError("max_radius must be finite and at least r0")
    if type(scan_points) is not int or scan_points < 1:
        raise ValueError("scan_points must be positive")
    if fixed_radius is not None and (not math.isfinite(fixed_radius) or
                                     not 0 < fixed_radius <= max_radius):
        raise ValueError("fixed_radius must be positive and within max_radius")
    if global_only_direction and fixed_radius is None:
        raise ValueError("global-only comparison requires a matched fixed radius")
    probabilities = infer(model, chunks)
    _validate(groups, len(probabilities), capped_class, probabilities.shape[1],
              global_cap, local_caps)
    before_global, before_local = _counts(probabilities, groups, capped_class)
    active_global = global_only_direction or before_global > global_cap
    active_local = set() if global_only_direction else {
        group for group, count in before_local.items() if count > local_caps[group]}
    out = dict(hard_before_global=before_global, hard_before_local=before_local,
               soft_before_global=float(probabilities[:, capped_class].sum()),
               soft_before_local={group: float(probabilities[[i for i, g in enumerate(groups)
                                                               if g == group], capped_class].sum())
                                  for group in sorted(local_caps)},
               active_global=active_global, active_local=sorted(active_local),
               applied=False, displacement=0.0, evaluations=1)
    if not active_global and not active_local:
        return out

    params = [p for p in model.parameters() if p.requires_grad]
    origin = [p.detach().clone() for p in params]
    was_training = model.training
    try:
        model.eval()
        device = params[0].device
        scopes = ([('global', None, max(global_cap, 1))] if active_global else []) + [
            ('local', group, max(local_caps[group], 1)) for group in sorted(active_local)]
        scope_grads = {(kind, group): [torch.zeros_like(p) for p in params]
                       for kind, group, _ in scopes}
        start = 0
        for images in chunks:
            logits = model(images.to(device))
            end = start + len(images)
            if not torch.allclose(logits.detach().softmax(1).cpu(), probabilities[start:end],
                                  atol=1e-7, rtol=1e-6):
                raise RuntimeError("replay changed logits at fixed weights")
            p = logits.softmax(1)[:, capped_class]
            for index, (kind, group, scale) in enumerate(scopes):
                mask = (torch.tensor([g == group for g in groups[start:end]], device=device)
                        if kind == 'local' else None)
                term = (p[mask] if mask is not None else p).sum() / scale
                grads = torch.autograd.grad(term, params,
                                            retain_graph=index + 1 < len(scopes), allow_unused=True)
                accum = scope_grads[(kind, group)]
                for a, g in zip(accum, grads):
                    if g is not None:
                        a.add_(g.detach())
            start = end
        if start != len(probabilities):
            raise RuntimeError("constraint cohort length changed")
        grads = [sum(scope_grads[key][i] for key in scope_grads)
                 for i in range(len(params))]
        if any(not bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError("invalid joint constraint gradient")
        norm = math.sqrt(sum(float(g.double().square().sum()) for g in grads))
        if not norm > 0.0:
            raise RuntimeError("joint constraint gradient is zero")
        unit = [-g / norm for g in grads]
        derivatives = {('global' if kind == 'global' else group):
                       sum(float((g.double() * d.double()).sum())
                           for g, d in zip(scope_grads[(kind, group)], unit))
                       for kind, group, _ in scopes}
        out['scope_directional_derivatives'] = derivatives
        if any(value >= 0 for value in derivatives.values()):
            raise RuntimeError('joint direction fails first-order scope descent: ' + str(derivatives))

        _place(params, origin, unit, r0)
        near = infer(model, chunks)
        near_global = float(near[:, capped_class].sum())
        near_local = {group: float(near[[i for i, g in enumerate(groups) if g == group],
                                         capped_class].sum()) for group in local_caps}
        out["directional_soft_delta_global"] = near_global - out["soft_before_global"]
        out["directional_soft_delta_local"] = {
            group: near_local[group] - out["soft_before_local"][group] for group in local_caps}

        def probe(radius):
            _place(params, origin, unit, radius)
            values = infer(model, chunks)
            global_count, local_counts = _counts(values, groups, capped_class)
            feasible = global_count <= global_cap and all(
                local_counts[group] <= cap for group, cap in local_caps.items())
            return feasible, global_count, local_counts

        evaluations = 2
        if fixed_radius is None:
            hi = r0
            for _ in range(max_doublings):
                feasible, _, _ = probe(hi)
                evaluations += 1
                if feasible:
                    break
                if hi == max_radius:
                    raise RuntimeError("joint search reached its fixed displacement ceiling")
                hi = min(2 * hi, max_radius)
            else:
                raise RuntimeError("no sampled joint displacement meets every hard cap")

            # Scan from zero; a hard count need not stay feasible as radius grows.
            candidate = None
            for index in range(1, scan_points + 1):
                radius = hi * index / scan_points
                feasible, _, _ = probe(radius)
                evaluations += 1
                if feasible:
                    candidate = radius
                    break
            if candidate is None:
                raise RuntimeError("bounded joint scan found no feasible displacement")
        else:
            candidate = fixed_radius
        step = unit if sham_generator is None else _random_direction(unit, sham_generator)
        _place(params, origin, step, candidate)
        after = infer(model, chunks)
        after_global, after_local = _counts(after, groups, capped_class)
        if any(not bool(torch.isfinite(p).all()) for p in params):
            raise RuntimeError("nonfinite parameter after joint step")
        moved = math.sqrt(sum(float((p.detach() - o).double().square().sum())
                              for p, o in zip(params, origin)))
        tensor_norms = [float((p.detach() - o).double().norm())
                        for p, o in zip(params, origin)]
        out.update(applied=True, radius=candidate, displacement=moved,
                   tensor_displacement_norms=tensor_norms,
                   evaluations=evaluations + 1, hard_after_global=after_global,
                   hard_after_local=after_local,
                   soft_after_global=float(after[:, capped_class].sum()),
                   soft_after_local={group: float(after[[i for i, g in enumerate(groups)
                                                         if g == group], capped_class].sum())
                                     for group in sorted(local_caps)})
        if fixed_radius is None and sham_generator is None and (after_global > global_cap or any(
                after_local[group] > cap for group, cap in local_caps.items())):
            raise RuntimeError("joint step lost feasibility after final placement")
        return out
    except BaseException:
        _place(params, origin, [torch.zeros_like(p) for p in params], 0.0)
        raise
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
