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
import torch.nn.functional as F

from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.pipeline.setup import setup_runtime
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
    """tau_c = K_c / n_test, the fraction of the test set the budget allows.

    Uses the tightest binding scope. A class capped only LOCALLY has an
    UNLIMITED global budget, and reading `global_con` alone would hand it
    tau = 1.0 -- i.e. "cover everything", which is no constraint at all and
    would make the arm silently untreated on exactly the `G < L` sweeps the
    framework prescribes.
    """
    out = {}
    for c in capped:
        k = float(global_con[c]) if c < len(global_con) else UNLIMITED
        for bounds in (local_con or {}).values():
            if c < len(bounds) and bounds[c] is not None:
                lim = float(bounds[c])
                if lim < UNLIMITED:
                    k = min(k, lim) if k < UNLIMITED else lim
        if k >= UNLIMITED:
            raise ValueError(
                "class %d has no finite budget in either scope, so it has no "
                "coverage target and this arm would train it unconstrained "
                "while reporting a selection phase." % c)
        out[c] = min(1.0, max(k, 0.0) / max(1, n_test))
    return out


def selective_loss(g, probs, y, cls, tau, cov_weight):
    """Selective risk over the covered set, plus a pull toward the budget.

    `g` in [0,1] is how much this item is selected for class `cls`. The risk is
    the per-item binary loss "is this really class cls", weighted by g and
    normalised BY THE COVERAGE -- that normalisation is what makes it a risk over
    the covered domain rather than a reweighted sum, and it is why the model
    cannot reduce the loss just by selecting nothing.

    The coverage term is two-sided on purpose. `relu(cov - tau)` would be flat
    below the target, which is exactly the shape that gave the count penalty no
    fixed point at K and produced bang-bang oscillation (measured 2026-08-21:
    the gradient vanishes the epoch the cap is met and CE snaps the count
    straight back). A quadratic in (cov - tau) has a minimum AT the target.
    """
    is_c = (y == cls).float()
    per_item = F.binary_cross_entropy(probs[:, cls].clamp(EPSILON, 1 - EPSILON),
                                      is_c, reduction="none")
    cov = g.mean()
    risk = (g * per_item).sum() / (g.sum() + EPSILON)
    return risk + cov_weight * (cov - tau) ** 2, float(cov)


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
                    g_all = torch.sigmoid(sel(feats["x"].float()))
                    probs = torch.softmax(logits.float(), dim=1)
                    sel_loss = logits.new_zeros(())
                    for j, c in enumerate(capped):
                        li, cov = selective_loss(g_all[:, j], probs, yb, c,
                                                 tau[c], cov_weight)
                        sel_loss = sel_loss + li
                        cov_seen[c] += cov * len(yb)
                    loss = ce + eta * sel_loss
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
    chunk = int(hp.get("constraint_chunk_size", 128))
    model.eval()
    preds = []
    with torch.no_grad():
        X = inputs.X_test.to(device)
        for i in range(0, len(X), chunk):
            preds.append(model(X[i:i + chunk]).argmax(1).cpu())
    pred = torch.cat(preds)
    counts = {c: int((pred == c).sum()) for c in range(num_classes)}
    return counts, {c: float(counts[c]) for c in range(num_classes)}
