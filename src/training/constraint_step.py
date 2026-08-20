"""The constraint optimizer step -- ONE implementation, shared by all four arms.

WHY IT IS SHARED. Each trained arm hand-rolled this block, and the copies
drifted. Measured on `results/vit_diag` (ViTB16 x dermmnist x L30_G30, seed 1,
same warm-up model, same `constraint_grad_clip: 1.0` in all three configs):

    arm       raw constraint grad norm      clip binds     steps applied
    tralo     0.638 .. 1826.5               6 of 7         >=7 of 29
    fioretto  17,667 .. 80,827              18 of 18       18 of 29
    hounie    0.005 .. 0.1105               0 of 29        28 of 29, none clipped

At the last epoch of that run fioretto's constraint loss is 4390.838 and
hounie's is 0.004204 -- a factor of 1.04e6, and their gradient norms differ by
1.02e6. The cause is faithful to the two papers (`hounie_rcl` divides the
violation by N to match its dual's scale, `fioretto_ldf` sums), but the
CONSEQUENCE is not a method difference: one absolute clip applied to two
natural scales six orders apart means tralo and fioretto each deliver a
unit-norm step while hounie delivers its raw ~0.05-norm step. Roughly a 20x
dose difference between arms, and no config gate can see it -- every config
says 1.0.

So `normalize` exists. It rescales the constraint gradient to EXACTLY
`constraint_grad_clip` instead of merely capping it, which makes the constraint
step size a protocol constant identical across arms and leaves each method's
DIRECTION as its actual contribution -- which is the thing the comparison is
supposed to be about. For tralo and fioretto it changes almost nothing (their
clip already bound on essentially every epoch); for hounie it is the whole
difference between taking a step and not.

`clip` remains the default so every existing result stays reproducible.

AND THE fp32 PASS. fioretto lost 10 of its 29 constraint epochs to non-finite
gradients -- 6 NaN and 4 inf, raw count before any dropna (an analysis that
calls dropna() first sees only the 4 inf and reports "4 of 29"). On the FP16
path a NaN norm fails the `> 0` gate and an inf norm is skipped inside
`scaler.step`, so either way no update lands and the run still reports
`status: completed`. It recurs rather than settling because the 126 CE steps
per epoch grow the loss scale back up between constraint steps, and fioretto's
constraint loss is ~1e4 times CE's.

Running the constraint pass in fp32 without the scaler decouples it from that
loop entirely. It is one backward per epoch over the test set in chunks, so the
cost is small against 126 CE steps, and it is method-neutral -- it changes no
formula, only the precision the formula is evaluated in.
"""

import logging
from contextlib import contextmanager

import torch

log = logging.getLogger(__name__)


@contextmanager
def constraint_autocast(amp_dtype, use_amp, fp32):
    """Autocast for the constraint forward. fp32=True disables it."""
    if fp32 or not use_amp:
        with torch.amp.autocast("cuda", enabled=False):
            yield
    else:
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
            yield


def constraint_backward(loss, scaler, fp32):
    """Backward for the constraint term.

    In fp32 the scaler is bypassed: its only job is to keep fp16 intermediates
    in range, and it is what couples this step to the CE pass's loss scale.
    """
    if scaler is not None and not fp32:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def finish_constraint_step(model, optimizer, scaler, clip, mode="clip",
                           fp32=False):
    """Bound the constraint gradient and take the step.

    Returns (raw_norm, applied). `raw_norm` is the true pre-clip norm, so a log
    written from it still records what the method actually produced -- which is
    the only way to see that a clip bound, or never did.
    """
    if scaler is not None and not fp32:
        scaler.unscale_(optimizer)

    # returns the total norm BEFORE clipping, and caps in place
    raw = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
    raw_norm = float(raw)

    applied = bool(torch.isfinite(raw) and raw > 0)
    if applied and mode == "normalize" and raw_norm < clip:
        # clip_grad_norm_ only shrinks. Scale UP so the delivered step is
        # exactly `clip` for every arm, not just the ones that overshoot it.
        scale = clip / (raw_norm + 1e-12)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    if applied:
        if scaler is not None and not fp32:
            scaler.step(optimizer)
        else:
            optimizer.step()
    if scaler is not None and not fp32:
        scaler.update()
    return raw_norm, applied
