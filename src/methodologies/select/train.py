"""select: a jointly-trained SELECTION head. Path 1c of docs/FRAMEWORK.md section 4.

WHY THIS IS NOT A FOURTH SCORE-PUSHING ARM. `rank`, `rankpair` and `budget_margin` all
add a term that moves the SCORE ORDERING while leaving the classification loss untouched.
All three are null, which is good evidence that the ranking cannot be fixed by pushing on
scores. This does something else: it reweights the CLASSIFICATION loss so the model is
optimised to be accurate **on the items it selects**. The representation changes, not the
offsets.

CREDIT. The mechanism is SelectiveNet (Geifman & El-Yaniv, ICML 2019): a selection head
`g` trained jointly with the classifier `f`, optimising the risk over the covered domain
plus a quadratic penalty pulling coverage to a target. Their reported baseline is "a
threshold over the prediction confidence of a pre-trained network" -- which is exactly our
`clip` arm -- and they beat it. That is why this direction is worth one campaign.

WHAT IS OURS, and it is where any novelty has to live. SelectiveNet covers ONE global
fraction of ALL items with a single scalar coverage target. Here:
  - coverage is PER CAPPED CLASS, from that class's transductive budget `K_c / n_test`;
  - the capped classes are COUPLED -- they compete for the same items through the softmax,
    which is the see-saw that makes every count penalty move items between classes rather
    than out of them;
  - and there are group-local caps on top of the global one.
So `g` is per (item, capped class), not per item, and there is one coverage target per
class rather than one for the run.

HOW THE BUDGET ENTERS. `tau_c = K_c / n_test` is the fraction of the TEST set the budget
allows for class c. It is a single scalar per class -- the same information the count
penalty gets -- but it is applied per item, at training time, where labels exist. That is
the whole bet: the budget is not new information (train and test prevalence agree by
construction, see section 4), so the only way to win is to spend it better, and the only
place labels exist is the training set.

!! THE BAR, pre-registered: it must move **ccP**. An arm that moves AUROC and not ccP has
reproduced `budget_margin` and the shipped penalty for a third time -- both improve
AUROC/ECE/Brier/NLL while ccP FALLS, i.e. they improve the ordering everywhere the cap
does not read. AP is watched as an overfitting guard, not as an endpoint: the covered set
is a small self-selected subset, which is how `joint_objective` held the cap on 98.8% of
epochs and lost 0.067 AP.

No post-hoc behaviour changes: `g` is a training-time device. Allocation at test time is
the same allocator every other trained arm uses, so an arm-vs-arm delta is attributable to
the representation and not to a different filling rule.
"""

import logging

import torch
import torch.nn as nn

from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.pipeline.setup import setup_runtime
from src.utils.constants import CONSTRAINT_CHUNK_SIZE
from src.pipeline.warmup import make_ce_criterion, make_dataloader, make_optimizer
from src.training.logging import log_progress_to_csv, write_csv_header
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)

EPSILON = 1e-8


def head_and_feature_dim(model):
    """The final head module and the width of what feeds it.

    A forward-pre-hook on this module yields the penultimate features without
    modifying any of the four model classes -- which matters because the warm-up
    cache is keyed on the model definition, and changing it would silently
    invalidate every cached warm-up.

    The four backbones name it differently (`heads` on ViT, `fc` on RegNet,
    `classifier` on both MobileNets), so it is looked up rather than hardcoded.
    """
    bb = getattr(model, "backbone", model)
    for name in ("heads", "fc", "classifier"):
        head = getattr(bb, name, None)
        if isinstance(head, nn.Module):
            linear = [m for m in head.modules() if isinstance(m, nn.Linear)]
            if linear:
                return head, linear[0].in_features
    raise ValueError(
        "could not find a head module on %s. The selection head needs the "
        "penultimate features; add this backbone's head attribute to the lookup "
        "rather than guessing a feature width." % type(model).__name__)


def coverage_targets(global_con, local_con, capped, n_test, num_classes):
    """tau_c = K_c / n_test, the fraction of the WHOLE TEST SET the budget allows.

    !! THE TWO SCOPES ARE NOT INTERCHANGEABLE AND MUST NOT BE min()'d.
    `global_con[c]` is one ceiling over all n_test items. `local_con[g][c]` is a
    ceiling over group g ALONE. The selection head's coverage is `g.mean()` over
    a batch drawn from every group, so the numerator has to be a whole-test
    quantity too. Taking `min` across groups picks the SMALLEST GROUP's budget
    and divides it by the whole test set: on derm L50_G30 that is 9/2004 instead
    of 67/2004, a 7.4x over-tightening, and it makes tau move with the LOCAL tag
    while the global cap is unchanged -- so a `G < L` cap sweep would be sweeping
    the smallest group rather than the constraint under test.

    The correct whole-test ceiling is `min(K_global, sum_g K_local_g)`, which is
    the same quantity the framework's "the sum of the locals already bounds the
    count" note is about. A class capped only LOCALLY still gets a finite target
    from the sum; a class capped only globally gets it from `global_con`.
    """
    out = {}
    for c in capped:
        k = float(global_con[c]) if c < len(global_con) else UNLIMITED
        # The local sum is a whole-test ceiling ONLY when EVERY group caps this
        # class. If even one group leaves it unlimited, that group can absorb
        # any number of items and the sum of the others bounds nothing -- using
        # it anyway would reinstate the smallest-group bug in a new branch (one
        # group capped at 9 out of a global 67 would give tau = 9/n again).
        groups = list((local_con or {}).values())
        finite = [float(b[c]) for b in groups
                  if c < len(b) and b[c] is not None and float(b[c]) < UNLIMITED]
        if groups and len(finite) == len(groups):
            k = min(k, sum(finite))
        if k >= UNLIMITED:
            raise ValueError(
                "class %d has no finite budget in either scope, so it has no "
                "coverage target and this arm would train it unconstrained "
                "while reporting a selection phase." % c)
        out[c] = min(1.0, max(k, 0.0) / max(1, n_test))
    return out


def selective_loss(g, probs, y, cls, tau, cov_weight, cov_ema=None,
                   risk_ema=None):
    """Selective risk over the covered set, plus a pull toward the budget.

    `g` in [0,1] is how much this item is selected for class `cls`. The risk is
    the per-item binary loss "is this really class cls", weighted by g -- that
    is what makes it a risk over the COVERED domain rather than a reweighted sum.

    The coverage term is two-sided on purpose. `relu(cov - tau)` would be flat
    below the target, which is exactly the shape that gave the count penalty no
    fixed point at K and produced bang-bang oscillation (measured 2026-08-21:
    the gradient vanishes the epoch the cap is met and CE snaps the count
    straight back). A quadratic in (cov - tau) has a minimum AT the target.

    !! BOTH ESTIMATORS ARE STABILISED, and the reason is a measured dosing
    trap this project has hit before. SelectiveNet's coverage targets are
    0.70-1.00. Ours is `K_c / n_test` = 62/2014 = 0.031 -- a 23x extrapolation.
    At batch_size 64 that is ~2 COVERED ITEMS PER BATCH, so:

      - dividing the risk by `g.sum()` (expectation ~2) is a ratio estimator
        whose denominator is a small random variable: enormous variance, and
        when g.sum() falls near zero EPSILON dominates and the gradient blows
        up. Normalising by the EXPECTED covered mass `n * tau` instead keeps
        the same expectation at cov = tau with a CONSTANT denominator.
      - `(g.mean() - tau)**2` has expectation Var(cov) + bias**2, and at ~2
        items Var dominates, so the penalty sits on a noise floor and its
        gradient is mostly noise. The value therefore comes from a running
        estimate across batches while the GRADIENT comes from this batch --
        the same detach construction the chunked soft count already uses.

    Without these the arm takes a near-zero, high-variance step, reports a null
    and writes `completed` -- indistinguishable from the method not working.
    That is exactly how `cut_temp: 0.02` produced a silent null with 1.4-1.9
    items inside its window.
    """
    is_c = (y == cls).float()
    # WRITTEN OUT, not F.binary_cross_entropy. CUDA autocast BANS that op --
    # "unsafe to autocast" -- and this is called from inside the autocast block,
    # so every `select` run died in 11 s with a header-only training_log and was
    # reset to `pending` by the dispatcher. NOT binary_cross_entropy_with_logits
    # either: `probs` is a SOFTMAX probability, and the sigmoid of logits[:, cls]
    # is a different quantity, so that "fix" would silently change the loss.
    # `probs` is fp32 (logits.float() upstream) and p is already clamped, so this
    # is bit-identical to the banned call -- verified, max abs diff 0.0.
    p_c = probs[:, cls].clamp(EPSILON, 1 - EPSILON)
    per_item = -(is_c * p_c.log() + (1.0 - is_c) * (1.0 - p_c).log())
    cov = g.mean()
    # CENTRED, and that is not cosmetic. Swapping g.sum() for the expected
    # covered mass fixes the variance but also removes the CENTRING the ratio
    # estimator had for free: d risk / d g_i becomes per_i / (n*tau), which is
    # positive for EVERY item, so the risk term degenerates into a pure "cover
    # nothing" force and only the coverage penalty holds it up. Measured on a
    # 4000-item synthetic at tau = 0.031: equilibrium coverage falls from
    # 0.74*tau (ratio) to 0.60*tau (fixed denominator), and undershooting the
    # budget is the one regime where the two-allocator confound bites.
    # Subtracting the DETACHED mean loss over the covered set restores it --
    # easy items get a negative gradient and are pulled in, hard ones pushed
    # out -- which is what a selective risk is supposed to do. Detached, so it
    # shifts the gradient without carrying the small random denominator into
    # it; smoothed for the same reason the coverage value is.
    base = ((g * per_item).sum() / (g.sum() + EPSILON)).detach()
    if risk_ema is not None:
        base = torch.as_tensor(risk_ema, dtype=base.dtype, device=base.device)
    risk = (g * (per_item - base)).sum() / (g.numel() * tau + EPSILON)
    cov_eff = cov if cov_ema is None else (cov_ema + cov - cov.detach())
    return risk + cov_weight * (cov_eff - tau) ** 2, float(cov), float(base)


def train(inputs: TrainInputs) -> TrainOutputs:
    hp = inputs.hyperparams
    device = inputs.device
    model = inputs.model
    num_classes = inputs.num_classes
    capped = sorted(inputs.constrained_classes)

    eta = float(hp.get("select_eta", 1.0))
    cov_weight = float(hp.get("select_cov_weight", 32.0))
    warmup_epochs = int(hp["warmup_epochs"])
    n_epochs = int(hp["constraint_epochs"])
    lr = float(hp["lr_constraint"])

    use_amp, amp_dtype, scaler = setup_runtime(device)
    head, feat_dim = head_and_feature_dim(model)
    sel = nn.Linear(feat_dim, len(capped)).to(device)

    feats = {}
    handle = head.register_forward_pre_hook(
        lambda _m, args: feats.__setitem__("x", args[0]))

    criterion_ce = make_ce_criterion(inputs.config, inputs.y_train, num_classes, device)
    # The selection head is trained WITH the backbone: the point is that the
    # representation moves. Optimising `sel` alone would make this a post-hoc
    # reweighting of a frozen model, which is a different (and already-null) arm.
    optimizer = make_optimizer(list(model.parameters()) + list(sel.parameters()),
                               lr, device)
    loader = make_dataloader(inputs.X_train, inputs.y_train, hp["batch_size"])

    tau = coverage_targets(inputs.global_con, inputs.local_con, capped,
                           len(inputs.y_test), num_classes)
    log.info("select: coverage targets %s (eta=%g cov_weight=%g)",
             {c: round(t, 4) for c, t in tau.items()}, eta, cov_weight)

    # THE DOSE CHECK, before a single step. tau here is K_c/n_test ~ 0.03 against
    # SelectiveNet's published 0.70-1.00, so the covered mass per batch is tiny
    # and both estimators degrade. Say the number out loud rather than letting
    # the arm report a silent null: this project has twice shipped a treatment
    # whose effective sample was ~2 items and read the result as "no effect".
    bs = int(hp["batch_size"])
    per_batch = {c: bs * tau[c] for c in capped}
    log.info("select: covered items per batch of %d = %s",
             bs, {c: round(v, 2) for c, v in per_batch.items()})
    if min(per_batch.values()) < 5.0:
        log.warning(
            "select: only %.2f covered items per batch (tau=%.4f, batch=%d). "
            "The risk and coverage estimators are stabilised for exactly this "
            "regime, but the DOSE is still small -- read a null from this arm as "
            "'underpowered' before reading it as 'the method does not work'.",
            min(per_batch.values()), min(tau.values()), bs)
    cov_ema = {c: None for c in capped}
    risk_ema = {c: None for c in capped}
    ema_beta = float(hp.get("select_cov_ema", 0.9))
    write_csv_header(str(inputs.csv_log_path), num_classes,
                     local_constraints=inputs.local_con)

    try:
        for epoch in range(warmup_epochs, warmup_epochs + n_epochs):
            model.train()
            sel.train()
            tot_ce = tot_sel = correct = seen = 0.0
            cov_seen = {c: 0.0 for c in capped}
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(xb)
                    ce = criterion_ce(logits, yb)
                    # OUTSIDE autocast on purpose. autocast casts an
                    # nn.Linear's input down regardless of a preceding
                    # .float(), so `g_all` would be bf16/fp16 -- and at
                    # tau ~ 0.03 the bf16 ULP is 1.22e-4, i.e. 0.4% of the
                    # target, on the very quantity the coverage penalty
                    # squares. The head is one Linear; the cost is nil.
                    with torch.amp.autocast("cuda", enabled=False):
                        g_all = torch.sigmoid(sel(feats["x"].float()))
                    probs = torch.softmax(logits.float(), dim=1)
                    sel_loss = logits.new_zeros(())
                    batch_base = {}
                    for j, c in enumerate(capped):
                        li, cov, base = selective_loss(
                            g_all[:, j], probs, yb, c, tau[c], cov_weight,
                            cov_ema[c], risk_ema[c])
                        sel_loss = sel_loss + li
                        cov_seen[c] += cov * len(yb)
                        batch_base[c] = base
                    loss = ce + eta * sel_loss
                # Running coverage estimate, updated after the loss is built so
                # it never enters the graph: `cov_ema[c]` read above is the
                # estimate from PRIOR batches (value), this batch supplies the
                # gradient. Seeded from the first batch rather than 0.0, which
                # would otherwise spend the early epochs pulling coverage up
                # from a level no batch ever had.
                for j, c in enumerate(capped):
                    cur = float(g_all[:, j].mean())
                    cov_ema[c] = (cur if cov_ema[c] is None
                                  else ema_beta * cov_ema[c] + (1 - ema_beta) * cur)
                    b = float(batch_base[c])
                    risk_ema[c] = (b if risk_ema[c] is None
                                   else ema_beta * risk_ema[c] + (1 - ema_beta) * b)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                tot_ce += float(ce) * len(yb)
                tot_sel += float(sel_loss) * len(yb)
                correct += float((logits.argmax(1) == yb).sum())
                seen += len(yb)

            counts, soft = _test_counts(model, inputs, device, hp, num_classes)
            log_progress_to_csv(
                str(inputs.csv_log_path), epoch, tot_ce / max(1.0, seen),
                correct / max(1.0, seen),
                global_loss=tot_sel / max(1.0, seen),
                global_counts=counts, global_soft=soft,
                constraints=inputs.global_con, num_classes=num_classes,
                global_satisfied=all(counts[c] <= inputs.global_con[c] for c in capped),
                local_constraints=inputs.local_con)
            log.info("select epoch %d: ce=%.4f sel=%.4f acc=%.4f cov=%s",
                     epoch, tot_ce / max(1.0, seen), tot_sel / max(1.0, seen),
                     correct / max(1.0, seen),
                     {c: round(cov_seen[c] / max(1.0, seen), 4) for c in capped})
    finally:
        # A live hook on a model that outlives this function would fire during
        # evaluation and keep a reference to every batch's features.
        handle.remove()

    return TrainOutputs(model=model,
                        summary={"select_eta": eta, "select_cov_weight": cov_weight,
                                 "coverage_targets": {str(c): tau[c] for c in capped}})


def _test_counts(model, inputs, device, hp, num_classes):
    """Hard counts on the test set, for the same log every other arm writes."""
    chunk = int(hp.get("constraint_chunk_size", CONSTRAINT_CHUNK_SIZE))
    model.eval()
    preds = []
    with torch.no_grad():
        X = inputs.X_test.to(device)
        for i in range(0, len(X), chunk):
            preds.append(model(X[i:i + chunk]).argmax(1).cpu())
    pred = torch.cat(preds)
    counts = {c: int((pred == c).sum()) for c in range(num_classes)}
    return counts, {c: float(counts[c]) for c in range(num_classes)}
