"""Exact two-pass gradient for a cohort-wide, label-free soft-count penalty.

Both passes evaluate the same parameters. BatchNorm stays in evaluation mode;
one optimizer step happens only after every chunk has contributed its gradient.
"""

import torch

from .global_constraint import _validate_caps


def count_logit_gradient(probabilities, caps, multipliers, rho):
    """Analytic d(bound_count_penalty)/d(logits) for every cohort member."""
    if probabilities.ndim != 2 or len(probabilities) == 0:
        raise ValueError("probabilities must be a nonempty matrix")
    n_classes = probabilities.shape[1]
    _validate_caps(caps, n_classes)
    if multipliers.shape != (n_classes,) or multipliers.device != probabilities.device:
        raise ValueError("multipliers must match probabilities")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(multipliers).all():
        raise ValueError("nonfinite count-gradient input")
    if (probabilities < 0).any() or not torch.allclose(
            probabilities.sum(1), torch.ones_like(probabilities[:, 0]), atol=1e-6):
        raise ValueError("invalid probability rows")
    if (multipliers < 0).any() or not (0 <= rho < float('inf')):
        raise ValueError("invalid penalty coefficients")
    soft = probabilities.sum(0)
    coefficients = torch.zeros_like(soft)
    for c, cap in enumerate(caps):
        if cap is None:
            continue
        scale = float(max(cap, 1))
        e = torch.relu(soft[c] - cap) / scale
        if soft[c] > cap:
            coefficients[c] = multipliers[c] / scale * (
                1 / (1 + e).square() + 2 * rho * e / (1 + e.square()).square())
    return probabilities * (coefficients - (probabilities * coefficients).sum(1, keepdim=True))


def streamed_step(model, chunks, caps, multipliers, rho, optimizer):
    """Apply one full-cohort constraint update from replayable image chunks.

    Chunks contain images only, never evaluation labels. The caller supplies a
    finite, ordered sequence so both passes see identical inputs and weights.
    Returns pre-step probabilities and applied-update status for the audit log.
    """
    if not isinstance(chunks, (list, tuple)) or not chunks:
        raise ValueError("chunks must be a nonempty replayable sequence")
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            parts = [model(images.to(device)).softmax(1) for images in chunks]
        probabilities = torch.cat(parts)
        gradient = count_logit_gradient(probabilities, caps, multipliers, rho)
        if not torch.isfinite(gradient).all():
            raise RuntimeError("nonfinite constraint logit gradient")
        if not bool((gradient != 0).any()):
            return dict(probabilities=probabilities.detach(), applied=False,
                        soft_counts=probabilities.sum(0).detach())
        optimizer.zero_grad(set_to_none=True)
        start = 0
        for images, expected in zip(chunks, parts):
            logits = model(images.to(device))
            replay = logits.detach().softmax(1)
            if not torch.allclose(replay, expected, atol=1e-7, rtol=1e-6):
                raise RuntimeError("constraint pass changed logits at fixed weights")
            end = start + len(images)
            logits.backward(gradient[start:end])
            start = end
        if start != len(probabilities):
            raise RuntimeError("constraint cohort length changed")
        if any(p.grad is None or not torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad):
            raise RuntimeError("invalid constraint parameter gradient")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("nonfinite constraint parameter")
        return dict(probabilities=probabilities.detach(), applied=True,
                    soft_counts=probabilities.sum(0).detach())
    finally:
        model.train(was_training)
