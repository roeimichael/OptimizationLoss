import logging
import math
from contextlib import contextmanager
import torch

log = logging.getLogger(__name__)


@contextmanager
def constraint_autocast(amp_dtype, use_amp, fp32):
    if fp32 or not use_amp:
        with torch.amp.autocast("cuda", enabled=False):
            yield
    else:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
            yield


def constraint_backward(loss, scaler, fp32):
    if scaler is not None and (not fp32):
        scaler.scale(loss).backward()
    else:
        loss.backward()


def finish_constraint_step(
    model, optimizer, scaler, clip, mode="clip", fp32=False, diagnostics=None
):
    if scaler is not None and (not fp32):
        scale_before = scaler.get_scale()
        scaler.unscale_(optimizer)
    amp_overflow = False
    raw = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
    raw_norm = float(raw)
    applied = bool(torch.isfinite(raw) and raw > 0)
    if applied and mode == "normalize" and (raw_norm < clip):
        scale = clip / (raw_norm + 1e-12)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
    observed = []
    if diagnostics is not None:
        observed = [
            (p, p.detach().clone()) for p in model.parameters() if p.grad is not None
        ]
        transformed_norm = math.sqrt(
            sum((float(p.grad.detach().double().square().sum()) for (p, _) in observed))
        )
    if applied:
        if scaler is not None and (not fp32):
            scaler.step(optimizer)
        else:
            optimizer.step()
    if scaler is not None and (not fp32):
        scaler.update()
        amp_overflow = scaler.get_scale() < scale_before
        if applied:
            applied = not amp_overflow
    if diagnostics is not None:
        (delta_sq, descent_dot) = (0.0, 0.0)
        for p, before in observed:
            delta = p.detach().double() - before.double()
            delta_sq += float(delta.square().sum())
            descent_dot -= float((delta * p.grad.detach().double()).sum())
        delta_norm = math.sqrt(delta_sq)
        denom = transformed_norm * delta_norm
        alignment = descent_dot / denom if denom > 0 and math.isfinite(denom) else None
        diagnostics.update(
            pre_clip_grad_norm=raw_norm if math.isfinite(raw_norm) else None,
            transformed_grad_norm=transformed_norm
            if math.isfinite(transformed_norm)
            else None,
            parameter_delta_norm=delta_norm if math.isfinite(delta_norm) else None,
            descent_alignment=alignment,
            nonfinite_gradient=not math.isfinite(raw_norm) or amp_overflow,
            amp_overflow_detected=amp_overflow,
            optimizer_step_applied=bool(applied),
        )
    return (raw_norm, applied)
