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

from dataclasses import dataclass
import hashlib
import math
import weakref

import torch

from .knee_end_to_end import infer
from .streamed_constraint import streamed_step


@dataclass(frozen=True, eq=False)
class _RadiusCalibration:
    """Ephemeral native-search result; never serialized or accepted by a CLI."""
    binding: str
    gradient_sha256: str | None
    radius: float
    radius_violating: float
    evaluations: int


_issued_calibrations = weakref.WeakKeyDictionary()


def _calibration_signature(token):
    return (token.binding, token.gradient_sha256, token.radius, token.radius_violating, token.evaluations)


def _publish_calibration(destination, token):
    _issued_calibrations[token] = _calibration_signature(token)
    destination.append(token)


def _tensor_digest(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(repr((name, tuple(value.shape), str(value.dtype), str(value.device))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _calibration_binding(model, chunks, caps, settings):
    identity = (type(model).__module__, type(model).__qualname__, model.training,
                tuple((name, p.requires_grad) for name, p in model.named_parameters()),
                tuple(caps), settings, _tensor_digest(model.state_dict().items()),
                _tensor_digest(model.named_buffers()),
                _tensor_digest((str(i), x) for i, x in enumerate(chunks)))
    return hashlib.sha256(repr(identity).encode()).hexdigest()


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


def _counts(model, chunks, c):
    probabilities = infer(model, chunks)
    return (int((probabilities.argmax(1) == c).sum()),
            float(probabilities[:, c].sum()))


@torch.no_grad()
def _place(params, origin, direction, r):
    for p, o, d in zip(params, origin, direction):
        if d is not None:
            p.copy_(o + r * d)


def targeted_step(model, chunks, caps, sham_generator=None, r0=1e-3, max_doublings=30, iterations=20,
                  *, calibration_out=None, calibration=None):
    """Optionally reuse a native search only for its identical-copy seeded sham.

    The sham still recomputes and authenticates the exact real gradient. Only
    its redundant hard-count search is omitted; placement and RNG are unchanged.
    """
    if calibration_out is not None and (type(calibration_out) is not list or calibration_out
                                         or sham_generator is not None or calibration is not None):
        raise ValueError('calibration export requires an empty native-only destination')
    if calibration is not None and (type(calibration) is not _RadiusCalibration
                                     or _issued_calibrations.get(calibration) != _calibration_signature(calibration)
                                     or sham_generator is None):
        raise ValueError('calibration reuse requires a native token and seeded sham')
    binding = None
    if calibration_out is not None or calibration is not None:
        binding = _calibration_binding(model, chunks, caps, (r0, max_doublings, iterations))
    if calibration is not None and calibration.binding != binding:
        raise ValueError('calibration copy/input/cap/search identity differs')
    constrained = [c for c, cap in enumerate(caps) if cap is not None]
    if len(constrained) != 1:
        raise ValueError('targeted_step is defined for exactly one capped class')
    c = constrained[0]
    cap = caps[c]
    before, soft_before = _counts(model, chunks, c)
    out = dict(hard_before=before, hard_after=before,
               soft_before=soft_before, soft_after=soft_before,
               gradient_norm=0.0, applied=False, displacement=0.0, evaluations=1)
    if before <= cap:
        if calibration is not None and calibration.gradient_sha256 is not None:
            raise ValueError('calibration activation differs')
        if calibration_out is not None:
            _publish_calibration(calibration_out, _RadiusCalibration(binding, None, 0.0, 0.0, 1))
        if calibration is not None:
            out.update(radius_calibration_reused=True, radius_calibration_evaluations=calibration.evaluations)
        return out
    params = [p for p in model.parameters() if p.requires_grad]
    capture = _Capture(params)
    # With one active class, the penalty gradient is a positive scalar times the
    # gradient of its soft count. Normalizing below discards lambda, rho, and the
    # penalty curve's magnitude; this step tests the count direction and the
    # hard-cap radius search, not the original penalty shape.
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
    out['gradient_norm'] = norm
    unit = [None if g is None else -g / norm for g in grads]
    gradient_sha256 = (_tensor_digest((str(i), g) for i, g in enumerate(grads) if g is not None)
                       if binding is not None else None)
    if calibration is not None and calibration.gradient_sha256 != gradient_sha256:
        raise ValueError('calibration real gradient differs')
    origin = [p.detach().clone() for p in params]
    evaluations = 1
    if calibration is not None:
        lo, hi = calibration.radius_violating, calibration.radius
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi):
            raise ValueError('calibration radius bracket is invalid')
        out.update(radius_calibration_reused=True, radius_calibration_evaluations=calibration.evaluations)
    else:
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
    hard_after, soft_after = _counts(model, chunks, c)
    out.update(applied=True, displacement=moved, radius=hi, radius_violating=lo, evaluations=evaluations + 1,
               hard_after=hard_after, soft_after=soft_after)
    if calibration_out is not None:
        _publish_calibration(calibration_out, _RadiusCalibration(binding, gradient_sha256, hi, lo, evaluations + 1))
    return out
