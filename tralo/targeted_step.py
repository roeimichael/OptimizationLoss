"""A constraint step sized to land exactly on the hard cap, and its sham.

Pilot seed 1701 of experiments/claude_controller_sham_protocol_20260925.md showed the
published dose overshoots ~10x: ONE separate-Adam step (lr 3e-5, displacement 0.10) moved
the grade-3 soft count 82.6 -> 8.2 against a cap of 76, and CalibratedSGD at the same
norm moved it to 0.2. A random direction of the same norm moved it +4.7. The direction is
informative; the size is wrong, and with one capped class the multiplier and rho only
scale the gradient, so the controller cannot fix the size either.

targeted_step takes TraLO's direction -- steepest descent of the capped class's soft
count, obtained from the verified streamed_step gradient -- and chooses the SMALLEST
displacement r along it that brings the hard count to the cap (bracket by doubling,
then bisection). It triggers on the hard count, not the soft count (audit D2).
The sham finds the same r along the real direction, then moves r in a seeded random
direction with the real step's per-tensor norms. Only development IMAGES are used;
no labels enter.
"""

import math

import torch

from .knee_end_to_end import infer
from .streamed_constraint import streamed_step


class _Capture:
    """Optimizer stand-in: streamed_step fills .grad, step() copies it, weights stay put."""

    def __init__(self, params):
        self.params = list(params)
        self.grads = None

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def step(self):
        self.grads = [None if p.grad is None else p.grad.detach().clone() for p in self.params]


def _hard(model, chunks, c):
    return int((infer(model, chunks).argmax(1) == c).sum())


@torch.no_grad()
def _place(params, origin, direction, r):
    for p, o, d in zip(params, origin, direction):
        if d is not None:
            p.copy_(o + r * d)


def targeted_step(model, chunks, caps, sham_generator=None, r0=1e-3, max_doublings=30, iterations=20):
    constrained = [c for c, cap in enumerate(caps) if cap is not None]
    if len(constrained) != 1:
        raise ValueError('targeted_step is defined for exactly one capped class')
    c = constrained[0]
    cap = caps[c]
    before = _hard(model, chunks, c)
    out = dict(hard_before=before, applied=False, displacement=0.0, evaluations=1)
    if before <= cap:
        return out
    params = [p for p in model.parameters() if p.requires_grad]
    capture = _Capture(params)
    # cap 0 on the capped class: the penalty gradient is then a positive multiple of the
    # gradient of that class's soft count, whatever lambda and rho are.
    direction_caps = [None] * len(caps)
    direction_caps[c] = 0
    device = params[0].device
    streamed_step(model, chunks, direction_caps, torch.full((len(caps),), 1.0, device=device), 0.0, capture)
    grads = capture.grads
    if grads is None:
        raise RuntimeError('streamed_step returned no constraint gradient')
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in grads if g is not None))
    if not norm > 0.0:
        raise RuntimeError('capped class has no soft-count gradient')
    unit = [None if g is None else -g / norm for g in grads]
    origin = [p.detach().clone() for p in params]
    evaluations = 1
    lo, hi = 0.0, r0
    for _ in range(max_doublings):
        _place(params, origin, unit, hi)
        evaluations += 1
        if _hard(model, chunks, c) <= cap:
            break
        lo, hi = hi, 2 * hi
    else:
        _place(params, origin, unit, 0.0)
        raise RuntimeError('no displacement along the constraint direction meets the cap')
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        _place(params, origin, unit, mid)
        evaluations += 1
        if _hard(model, chunks, c) <= cap:
            hi = mid
        else:
            lo = mid
    if sham_generator is None:
        step = unit
    else:
        step = []
        for u in unit:
            if u is None:
                step.append(None)
                continue
            share = float(u.double().norm())
            noise = torch.randn(u.shape, generator=sham_generator, dtype=torch.float64)
            step.append((noise * (share / float(noise.norm()))).to(dtype=u.dtype, device=u.device)
                        if share > 0.0 else torch.zeros_like(u))
    _place(params, origin, step, hi)
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise RuntimeError('nonfinite parameter after the targeted step')
    moved = math.sqrt(sum(float((p.detach() - o).double().square().sum()) for p, o in zip(params, origin)))
    out.update(applied=True, displacement=moved, radius=hi, radius_violating=lo, evaluations=evaluations + 1,
               hard_after=_hard(model, chunks, c))
    return out
