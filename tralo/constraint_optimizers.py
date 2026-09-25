"""Optimizers for the constraint step that fix audit finding D1, plus the sham control.

D1 (analysis/CODE_AUDIT_20260925.md): a dedicated Adam is invariant to the scale of its
gradient, so with one constraint step per epoch the multiplier, rho and violation depth
all cancel and every step is ~lr * sign(g). The TraLO controller never acted.

CalibratedSGD makes the step proportional to the gradient, so the multiplier and the
violation depth set the dose, as the method intends. Its learning rate is fixed ONCE,
label-free, at the first applied step, so that this first displacement has the norm an
Adam first step of `adam_lr` would have, epsilon included (see calibrate).
That matches the initial dose to the published Adam arm and lets every later step scale
with the controller.

ShamOptimizer is the missing control (analysis/FINDINGS_20260925.md, A3): end-to-end
training amplifies ANY perturbation chaotically, so TraLO minus Null cannot separate
"the constraint's information" from "a perturbation of that size". The sham takes the
real constraint gradient, keeps each parameter tensor's norm, replaces its direction with
a seeded random one, then steps the same inner optimizer. Same dose at every step,
controller included; no information in the direction.

Both wrappers expose zero_grad()/step() so tralo.streamed_constraint.streamed_step, whose
two-pass gradient is verified exact, is used unchanged.
"""

import math

import torch


def _grads(params):
    return [p for p in params if p.requires_grad and p.grad is not None]


class CalibratedSGD:
    def __init__(self, params, adam_lr, adam_eps=1e-8):
        if not (isinstance(adam_lr, (int, float)) and math.isfinite(adam_lr) and adam_lr > 0):
            raise ValueError('adam_lr must be a positive finite number')
        self.params = [p for p in params]
        self.adam_lr = float(adam_lr)
        self.adam_eps = float(adam_eps)
        self.lr = None
        self.steps = 0
        self.last_displacement = None

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def calibrate(self):
        """Fix lr from the CURRENT gradients if not yet fixed; returns the lr (or None)."""
        if self.lr is None:
            live = _grads(self.params)
            norm_sq = sum(float(p.grad.double().square().sum()) for p in live)
            if norm_sq > 0.0:
                # A fresh Adam's first update is exactly lr * g / (|g| + eps) per coordinate
                # (bias-corrected m = g, v = g^2). Match that norm, epsilon included: on a
                # ResNet many coordinates have |g| << eps, so sqrt(nonzero) would overstate it.
                eps = self.adam_eps
                adam_sq = sum(float((p.grad.double() / (p.grad.double().abs() + eps)).square().sum())
                              for p in live)
                self.lr = self.adam_lr * math.sqrt(adam_sq) / math.sqrt(norm_sq)
        return self.lr

    @torch.no_grad()
    def step(self):
        live = _grads(self.params)
        norm_sq = sum(float(p.grad.double().square().sum()) for p in live)
        if norm_sq == 0.0:
            self.last_displacement = 0.0
            return
        self.calibrate()
        for p in live:
            p.add_(p.grad, alpha=-self.lr)
        self.steps += 1
        self.last_displacement = self.lr * math.sqrt(norm_sq)


class ShamOptimizer:
    """Norm-matched, direction-randomised constraint step through an inner optimizer."""

    def __init__(self, inner, seed):
        if type(seed) is not int:
            raise TypeError('seed must be an int')
        self.inner = inner
        self.params = inner.params
        self.generator = torch.Generator().manual_seed(seed)

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        # Calibrate on the REAL gradient: random noise has no exact zeros, so calibrating
        # after the swap would count more nonzero coordinates and change the dose.
        self.inner.calibrate()
        for p in _grads(self.params):
            norm = p.grad.double().norm()
            if norm == 0:
                continue
            noise = torch.randn(p.grad.shape, generator=self.generator, dtype=torch.float64)
            noise = noise * (norm / noise.norm())
            p.grad.copy_(noise.to(dtype=p.grad.dtype, device=p.grad.device))
        self.inner.step()

    @property
    def lr(self):
        return self.inner.lr

    @property
    def last_displacement(self):
        return self.inner.last_displacement
