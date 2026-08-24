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

# Counter for the random-direction control's generator, so its seed varies per
# call without drawing from the global RNG. See _randomize_direction.
_RANDDIR_CALLS = 0


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


def _randomize_direction(model, clip, seed_tensor):
    """Replace the constraint gradient with a random vector of the same norm.

    THE CONTROL FOR "DID THE DIRECTION MATTER?". Measured 2026-08-20 on
    dermmnist multi-class: the constraint costs exactly 4 correct capped-class
    predictions out of 89, at every one of three seeds (d capF1 -0.0149,
    -0.0150, -0.0149, sd 0.0001) -- while the underlying count trajectories are
    wildly different between those seeds (class 4 ends at 57, 201 and 439).

    A loss that is constant while the path that produced it is not looks like
    damage from perturbing a fitted ranking at all, rather than from perturbing
    it in the penalty's particular direction. If a random step of the same norm
    costs the same 4 items, the constraint contributed nothing a coin could not
    have, and no amount of shape or dose tuning will change that.

    Deterministic AND side-effect free. The seed comes from `initial_seed()`
    plus a call counter, neither of which DRAWS from the global RNG. Seeding it
    with `torch.randint` instead -- the obvious way, and how this was first
    written -- consumes a global draw, so the control run's dropout masks and
    batch order diverge from the real arm's as well as its step direction. That
    makes the control vary two things when its entire purpose is to vary one.
    """
    global _RANDDIR_CALLS
    _RANDDIR_CALLS += 1
    gen = torch.Generator(device=seed_tensor.device)
    gen.manual_seed((torch.initial_seed() + 7919 * _RANDDIR_CALLS) % (2 ** 31 - 1))
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            p.grad.normal_(generator=gen)
            total += float(p.grad.pow(2).sum())
    total = total ** 0.5
    if total > 0:
        scale = clip / total
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)


def head_parameter_ids(model, n_classes):
    """ids of the classifier head's parameters -- the Linear that emits logits.

    IDENTIFIED, NOT HARDCODED. The four backbones name their head differently
    (`classifier` on MobileNetV2/V3, `fc` on RegNetY400MF, `heads` on ViTB16),
    so a name list would be a landmine the day a fifth arrives. All four end in
    `nn.Linear(feat, n_classes)`, so that is the rule.

    It REFUSES on ambiguity rather than guessing. `out_features == n_classes`
    could in principle match an intermediate layer; if it matches more than one
    Linear, or none, the head is not determined and a silently-wrong choice
    would confine the constraint to the wrong parameters while every config and
    log still read `head_only: true` -- an inert flag with a plausible name,
    which is this project's most frequent defect.
    """
    hits = [m for m in model.modules()
            if isinstance(m, torch.nn.Linear) and m.out_features == n_classes]
    if len(hits) != 1:
        raise ValueError(
            "head_only needs exactly one Linear with out_features == %d to "
            "identify the classifier head; found %d. Name the head explicitly "
            "for this backbone rather than letting the constraint land on an "
            "arbitrary layer." % (n_classes, len(hits)))
    return {id(prm) for prm in hits[0].parameters()}


def snapshot_grads(model):
    """The CE gradient still sitting on the parameters, cloned. (list or None)

    Called right after the CE loop and BEFORE the `zero_grad` that opens the
    constraint pass, so it costs one clone and no extra forward/backward.

    ⚠️ IT IS ONE MINIBATCH, not the epoch's CE direction. That is the cheap
    estimate, and it is the honest description of what `ortho_project` removes:
    the component along the LAST CE step actually taken, which is also the
    step whose progress the constraint is most likely to undo. A full-epoch
    reference would cost a second pass over the training set every epoch.

    Returns None if any grad is non-finite -- on the FP16 path `scaler.step`
    skips a non-finite update, and projecting against a reference that was
    never applied would remove a direction the model never moved in.
    """
    ref = []
    for prm in model.parameters():
        if prm.grad is None:
            ref.append(None)
            continue
        if not torch.isfinite(prm.grad).all():
            return None
        ref.append(prm.grad.detach().clone())
    return ref if any(r is not None for r in ref) else None


def project_out(model, ref):
    """Remove the component of the constraint gradient along `ref`, in place.

    `g <- g - (<g,r>/<r,r>) r`, the projection onto the orthogonal complement,
    so enforcing the cap cannot undo the CE progress just made. Returns the
    coefficient removed, which is what `Ortho Fired` logs: a run whose
    coefficient is 0.0 every epoch did nothing, and `ortho_project` would then
    be an inert flag -- this project's most frequent failure mode, four
    occurrences and counting.

    THE ORDER MATTERS AND IT IS DELIBERATE. This runs BEFORE the norm bound in
    `finish_constraint_step`, so the projected gradient is renormalised to
    exactly `clip` afterwards under `mode="normalize"`. The projected and
    unprojected arms therefore deliver the SAME step size and differ only in
    DIRECTION -- the same argument that makes `random_direction` a legal
    control. Projecting after the bound would shorten the treatment's step and
    confound direction with dose, which is the trap that made the hounie
    baseline meaningless.
    """
    params = [prm for prm in model.parameters()]
    dot = 0.0
    nrm = 0.0
    for prm, r in zip(params, ref):
        if r is None or prm.grad is None:
            continue
        dot = dot + float((prm.grad * r).sum())
        nrm = nrm + float((r * r).sum())
    if nrm <= 0.0:
        return 0.0
    a = dot / nrm
    with torch.no_grad():
        for prm, r in zip(params, ref):
            if r is not None and prm.grad is not None:
                prm.grad.sub_(r, alpha=a)
    return a


def finish_constraint_step(model, optimizer, scaler, clip, mode="clip",
                           fp32=False, step_rule="shared", lr=None,
                           random_direction=False, ortho_ref=None,
                           head_ids=None):
    """Bound the constraint gradient and take the step.

    Returns (raw_norm, applied). `raw_norm` is the true pre-clip norm, so a log
    written from it still records what the method actually produced -- which is
    the only way to see that a clip bound, or never did.
    """
    if scaler is not None and not fp32:
        scaler.unscale_(optimizer)

    # BEFORE the bound, so the projected step is renormalised to the same size
    # as the unprojected one and the arms differ in direction alone.
    if head_ids is not None:
        # BEFORE the bound, like the projection, so `normalize` renormalises
        # what is left to exactly `clip`. THE CONSEQUENCE IS DELIBERATE AND
        # MUST BE READ WITH THE RESULT: the whole step norm is then delivered
        # to the head, so this arm moves the head MORE than `tralo` does. It
        # answers "does confining the constraint to the head remove the
        # damage", not "does freezing the backbone help, all else equal".
        # Holding the per-parameter step instead would leave the arm taking a
        # far smaller total step than its control, which confounds support
        # with dose -- the trap that made the hounie baseline meaningless.
        with torch.no_grad():
            for prm in model.parameters():
                if prm.grad is not None and id(prm) not in head_ids:
                    prm.grad.zero_()

    if ortho_ref is not None:
        # LOGGED, NOT RETURNED. The return arity is gated
        # (`test_..._unpacks_finish_constraint_steps_two_returns`) because
        # `applied` was once dropped by a caller and 10 of 29 epochs vanished
        # silently. The liveness gate for this flag is the project's standard
        # one -- `scripts/flag_live tralo tralo_ortho`, md5 over the raw
        # predictions (CLAUDE.md rule 3) -- and a coefficient in the log is a
        # per-epoch diagnostic beside it, not a substitute for it.
        log.info("ortho_project: removed CE component, coef=%.6g",
                 project_out(model, ortho_ref))

    # returns the total norm BEFORE clipping, and caps in place
    raw = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
    raw_norm = float(raw)

    applied = bool(torch.isfinite(raw) and raw > 0)
    if applied and random_direction:
        # Same norm, no information. Everything downstream is unchanged.
        #
        # !! THE NORM TO MATCH IS THE DELIVERED ONE, NOT `clip`. Under the
        # default mode="clip" the treatment delivers min(raw, clip), so
        # rescaling the coin to `clip` over-doses it on every epoch where the
        # clip did not bind -- 20x for hounie, whose raw norms are 0.005-0.11
        # against clip 1.0. The control would then differ from the treatment in
        # DOSE as well as information, which is the one thing it exists to hold
        # fixed, and the bias runs in the direction that flatters the treatment.
        _randomize_direction(model, clip if mode == "normalize"
                             else min(raw_norm, clip), raw)
    elif applied and mode == "normalize" and raw_norm < clip:
        # clip_grad_norm_ only shrinks. Scale UP so the delivered step is
        # exactly `clip` for every arm, not just the ones that overshoot it.
        scale = clip / (raw_norm + 1e-12)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

    if applied:
        if step_rule == "sgd":
            # Plain SGD, deliberately NOT the Adam the CE pass just took 126
            # steps with. Measured in this project: sharing that Adam leaves
            # cos(parameter update, constraint gradient) at 0.009-0.017, i.e.
            # the "constraint step" is ~98% a 127th CE step.
            #
            # This is NOT the rejected `separate_constraint_optimizer` arm. That
            # one used a dedicated ADAM, whose 1/sqrt(v) gives a step of norm
            # ~lr*sqrt(N) -- about 8,900x larger at ViT-B/16 scale -- so it
            # confounded direction with an enormous dose increase and cost
            # AP -0.0938. Here the step is lr*||g|| exactly, and with
            # mode="normalize" that is lr*clip = the smallest step in the
            # sweep, with the direction fully recovered.
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-lr)
        elif scaler is not None and not fp32:
            scaler.step(optimizer)
        else:
            optimizer.step()
    if scaler is not None and not fp32:
        # The scaler still owns the CE pass, so its bookkeeping runs either way.
        scaler.update()
    return raw_norm, applied
