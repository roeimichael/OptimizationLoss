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
from .local_boundary_policy import choose_boundary_step
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


def _boundary_step(model, chunks, groups, capped_class, global_cap, local_caps,
                   max_radius):
    """Calibrate the existing hard-active direction using unlabeled replay.

    This path is separate from the historical radius search so an opt-in
    experiment cannot change its fixed-radius or feasibility behavior.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("boundary step requires trainable parameters")
    devices = sorted({p.device.index if p.device.index is not None else torch.cuda.current_device()
                      for p in params if p.device.type == "cuda"})
    was_training = model.training
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    origin = [p.detach().clone() for p in params]
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            probabilities = infer(model, chunks)
            _validate(groups, len(probabilities), capped_class, probabilities.shape[1],
                      global_cap, local_caps)
            before_global, before_local = _counts(probabilities, groups, capped_class)
            group_indices = {group: [i for i, name in enumerate(groups) if name == group]
                             for group in sorted(local_caps)}

            def soft_counts(values):
                return (float(values[:, capped_class].sum()),
                        {group: float(values[indices, capped_class].sum())
                         for group, indices in group_indices.items()})

            before_soft_global, before_soft_local = soft_counts(probabilities)
            active_global = before_global > global_cap
            active_local = sorted(group for group, count in before_local.items()
                                  if count > local_caps[group])
            out = dict(hard_before_global=before_global, hard_before_local=before_local,
                       soft_before_global=before_soft_global,
                       soft_before_local=before_soft_local,
                       active_global=active_global, active_local=active_local,
                       applied=False, radius=0.0, displacement=0.0, evaluations=1)
            if not active_global and not active_local:
                out.update(skip_reason="no_hard_active_scope",
                           boundary_policy=dict(applied=False, radius=0.0,
                                                reason="no_hard_active_scope", probes=[]),
                           hard_after_global=before_global,
                           hard_after_local=before_local,
                           soft_after_global=before_soft_global,
                           soft_after_local=before_soft_local,
                           tensor_displacement_norms=[0.0 for _ in params])
                if any(not torch.equal(buffer, base) for buffer, base in buffers):
                    raise RuntimeError("boundary replay changed model buffers")
                return out

            if (before_soft_global <= global_cap and all(
                    before_soft_local[group] <= cap
                    for group, cap in local_caps.items())):
                decision = choose_boundary_step(
                    before_global, before_soft_global, global_cap,
                    before_soft_local, local_caps, 0.0,
                    {group: 0.0 for group in local_caps},
                    lambda _radius: None, max_radius=max_radius)
                out.update(skip_reason=decision['reason'], boundary_policy=decision,
                           hard_after_global=before_global,
                           hard_after_local=before_local,
                           soft_after_global=before_soft_global,
                           soft_after_local=before_soft_local,
                           tensor_displacement_norms=[0.0 for _ in params])
                if any(not torch.equal(buffer, base) for buffer, base in buffers):
                    raise RuntimeError("boundary replay changed model buffers")
                return out

            # Every normalized scope derivative is needed for the soft rule,
            # including a country whose hard calls do not exceed its cap.
            scopes = [("global", None, max(global_cap, 1))] + [
                ("local", group, max(local_caps[group], 1)) for group in sorted(local_caps)]
            scope_grads = {(kind, group): [torch.zeros_like(p) for p in params]
                           for kind, group, _ in scopes}
            device = params[0].device
            start = 0
            for images in chunks:
                logits = model(images.to(device))
                end = start + len(images)
                if not torch.allclose(logits.detach().softmax(1).cpu(), probabilities[start:end],
                                      atol=1e-7, rtol=1e-6):
                    raise RuntimeError("boundary replay changed logits at fixed weights")
                p = logits.softmax(1)[:, capped_class]
                for index, (kind, group, scale) in enumerate(scopes):
                    if kind == "local":
                        mask = torch.tensor([g == group for g in groups[start:end]],
                                            device=device)
                        term = p[mask].sum() / scale
                    else:
                        term = p.sum() / scale
                    gradients = torch.autograd.grad(term, params,
                                                    retain_graph=index + 1 < len(scopes),
                                                    allow_unused=True)
                    for acc, gradient in zip(scope_grads[(kind, group)], gradients):
                        if gradient is not None:
                            acc.add_(gradient.detach())
                start = end
            if start != len(probabilities):
                raise RuntimeError("boundary constraint cohort length changed")

            active_keys = ([('global', None)] if active_global else []) + [
                ('local', group) for group in active_local]
            joint = [sum(scope_grads[key][i] for key in active_keys)
                     for i in range(len(params))]
            if any(not bool(torch.isfinite(gradient).all()) for gradient in joint):
                raise RuntimeError("invalid boundary joint gradient")
            norm = math.sqrt(sum(float(gradient.double().square().sum())
                                 for gradient in joint))
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError("boundary joint gradient is zero or nonfinite")
            unit = [-gradient / norm for gradient in joint]
            derivatives = {
                ('pooled' if kind == 'global' else f'local:{group}'):
                sum(float((gradient.double() * direction.double()).sum())
                    for gradient, direction in zip(scope_grads[(kind, group)], unit))
                for kind, group, _ in scopes
            }
            out.update(gradient_norm=norm, scope_directional_derivatives=derivatives)

            local_probe_hard = []

            def probe(radius):
                _place(params, origin, unit, radius)
                values = infer(model, chunks)
                hard_global, hard_local = _counts(values, groups, capped_class)
                soft_global, soft_local = soft_counts(values)
                local_probe_hard.append(hard_local)
                return dict(pooled_hard=hard_global, pooled_soft=soft_global,
                            local_soft=soft_local)

            decision = choose_boundary_step(
                before_global, before_soft_global, global_cap,
                before_soft_local, local_caps, derivatives['pooled'],
                {group: derivatives[f'local:{group}'] for group in local_caps},
                probe, max_radius=max_radius)
            for record, hard_local in zip(decision['probes'], local_probe_hard):
                record['local_hard'] = hard_local
            out.update(boundary_policy=decision, evaluations=1 + len(decision['probes']))
            if decision['applied']:
                _place(params, origin, unit, decision['radius'])
                after = infer(model, chunks)
                hard_global, hard_local = _counts(after, groups, capped_class)
                soft_global, soft_local = soft_counts(after)
                accepted = decision['probes'][-1]
                if (hard_global != accepted['pooled_hard'] or
                        hard_local != accepted['local_hard'] or
                        not math.isclose(soft_global, accepted['pooled_soft'],
                                         rel_tol=1e-7, abs_tol=1e-7) or
                        any(not math.isclose(soft_local[group], accepted['local_soft'][group],
                                             rel_tol=1e-7, abs_tol=1e-7)
                            for group in local_caps)):
                    raise RuntimeError("accepted boundary probe did not replay exactly")
                if any(not bool(torch.isfinite(param).all()) for param in params):
                    raise RuntimeError("nonfinite parameter after boundary step")
                tensor_norms = [float((param.detach() - base).double().norm())
                                for param, base in zip(params, origin)]
                moved = math.sqrt(sum(value * value for value in tensor_norms))
                if not math.isfinite(moved):
                    raise RuntimeError("nonfinite boundary displacement")
                if not math.isclose(moved, decision['radius'], rel_tol=1e-4,
                                    abs_tol=1e-5):
                    raise RuntimeError("boundary model dose disagrees with selected radius")
                out.update(applied=True, radius=decision['radius'], displacement=moved,
                           tensor_displacement_norms=tensor_norms,
                           evaluations=out['evaluations'] + 1,
                           hard_after_global=hard_global, hard_after_local=hard_local,
                           soft_after_global=soft_global, soft_after_local=soft_local)
            else:
                _place(params, origin, unit, 0.0)
                out.update(skip_reason=decision['reason'],
                           hard_after_global=before_global,
                           hard_after_local=before_local,
                           soft_after_global=before_soft_global,
                           soft_after_local=before_soft_local,
                           tensor_displacement_norms=[0.0 for _ in params])
            if any(not torch.equal(buffer, base) for buffer, base in buffers):
                raise RuntimeError("boundary replay changed model buffers")
            return out
        except BaseException:
            _place(params, origin, [torch.zeros_like(p) for p in params], 0.0)
            raise
        finally:
            with torch.no_grad():
                for buffer, base in buffers:
                    buffer.copy_(base)
            model.train(was_training)


def local_targeted_step(model, chunks, groups, capped_class, global_cap, local_caps,
                        sham_generator=None, r0=1e-3, max_doublings=30,
                        scan_points=24, max_radius=0.1, fixed_radius=None,
                        global_only_direction=False, require_common_descent=True,
                        boundary_calibrated=False):
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
    if type(require_common_descent) is not bool:
        raise ValueError("require_common_descent must be a bool")
    if type(boundary_calibrated) is not bool:
        raise ValueError("boundary_calibrated must be a bool")
    if boundary_calibrated:
        if sham_generator is not None or fixed_radius is not None or global_only_direction:
            raise ValueError("boundary calibration requires the joint real direction")
        return _boundary_step(model, chunks, groups, capped_class, global_cap,
                              local_caps, max_radius)
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
        out['gradient_norm'] = norm
        unit = [-g / norm for g in grads]
        # A group may itself be named "global" or "pooled". Keep its identity
        # separate from the pooled constraint before checking common descent.
        derivatives = {('pooled' if kind == 'global' else 'local:' + group):
                       sum(float((g.double() * d.double()).sum())
                           for g, d in zip(scope_grads[(kind, group)], unit))
                       for kind, group, _ in scopes}
        out['scope_derivative_schema'] = 'pooled-local-v1'
        out['scope_directional_derivatives'] = derivatives
        if require_common_descent and any(value >= 0 for value in derivatives.values()):
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
