"""TraLO: cross-entropy plus a bounded penalty on the predicted count.

    L = CE + sum over capped (class, scope) of  lambda * penalty(soft_count, K)

    penalty(s, K) = E/(E+S) + rho * (E/S)^2 / (1 + (E/S)^2),
    with E = relu(s - K) and S = max(K, 1).

lambda ratchets per capped (class, scope) while the constraint is violated and
freezes on satisfaction. The transductive passes run in eval mode; pass 1 is
FP32 and computes the counts, pass 2 is AMP and carries the gradient, with a
detach construction that yields the exact full-N gradient from one chunk at a
time. A unit-norm gradient clip follows, and it is load-bearing: without it the
predicted count collapses to zero.

Everything else this file used to describe -- an undershoot hinge, a
`bounded_only` / `undershoot_hinge` mode switch, a KL anchor to the warm-up
distribution, a CE-saturation skip -- was DELETED from the pipeline (FRAMEWORK
section 2f). Each was measured and each made results worse. No config can
re-enable them.
"""

import logging
import time

import torch
import torch.nn.functional as F

from src.losses import MulticlassTransductiveLoss
from src.methodologies.dual_common import read_step_config
from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.pipeline.setup import setup_runtime
from src.pipeline.warmup import make_ce_criterion, make_dataloader, make_optimizer
from src.training.constraint_step import (
    constraint_autocast, constraint_backward, finish_constraint_step)
from src.losses.transductive_loss import (margin_window, margins,
                                          uniform_grad_count, window_temp)
from src.training.logging import log_progress_to_csv, write_csv_header
from src.training.metrics import compute_prediction_statistics
from src.training.reordering import capped_scores, reordering_report
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)

def train(inputs: TrainInputs) -> TrainOutputs:
    config = inputs.config
    hp = inputs.hyperparams
    SOFT_COUNT_MODE = str(hp.get("soft_count_mode", "sum"))
    if SOFT_COUNT_MODE not in ("sum", "margin", "uniform"):
        # Checked BEFORE read_step_config so a typo is reported as a typo
        # rather than as whichever required key happens to be read first.
        raise ValueError(
            "soft_count_mode must be one of sum / margin / uniform, got %r. "
            "An unrecognised mode silently ran `sum` under a different arm "
            "name, which is this project's most frequent defect."
            % SOFT_COUNT_MODE)
    step_cfg = read_step_config(hp)
    CUT_WINDOW_ITEMS = int(hp.get("cut_window_items", 5))
    STRAIGHT_THROUGH = bool(hp.get("straight_through", False))
    if SOFT_COUNT_MODE == "margin" and not STRAIGHT_THROUGH:
        # The fourth corner of the 2x2, and it is not an arm. Pass 1 would
        # accumulate the PLAIN sum into total_global_soft while pass 2 windows
        # each chunk, so the detach construction cancels a windowed chunk out
        # of a plain total and silently yields a THIRD semantics: value
        # sum_i p_ic, gradient from the window. Measured, the windowed count
        # over-counts (56.6 against a hard 45), so the penalty pushes past
        # feasibility -- the joint arm's failure mode. Refuse it rather than
        # run it: the generator cannot emit this, but a hand-written config
        # can, and a silent third semantics is how this project loses weeks.
        raise ValueError(
            "soft_count_mode: margin requires straight_through: true. "
            "Windowing the gradient while the penalty reads sum_i p_ic is "
            "neither arm and is not a configuration this project runs.")
    # THE RESEED CONTROL (`tralo_reseed`). ONE draw from the global generator
    # and nothing else -- no extra parameter, no extra step, no change to the
    # loss. Everything downstream that consumes the global RNG (the
    # DataLoader's shuffle seed, drawn afresh per epoch, and CPU dropout) then
    # runs on a different stream, so this arm is `tralo_null` re-randomised.
    #
    # WHY HERE AND NOT EARLIER. run_experiment re-seeds AFTER the warm-up, so
    # this is the first RNG consumer of the constraint phase and the draw
    # cannot reach the warm-up. That matters: `tralo_reseed` shares
    # `base_model_id` with `tralo` and `tralo_null` on purpose, and exactly one
    # of the three trains the cached model. A draw made before the warm-up
    # would change WHAT GETS CACHED depending on which arm ran first, so the
    # matched control would stop being matched -- silently, and differently on
    # every machine.
    #
    # WHY ONE DRAW IS THE RIGHT DOSE. It reproduces the perturbation that
    # produced the measured floor: `select_null` (`select_eta: 0`) is a pure-CE
    # run on the same seed and warm-up cache whose only side effect is
    # constructing an nn.Linear selection head, i.e. a handful of draws from
    # the global generator before its first batch. Reseeding that way moves the
    # capped-class hard count by RMS 83-95 items while turning the constraint
    # ON moves it 75-95. The stream is either the same or it is not; a second
    # draw would not make it "more reseeded".
    if bool(hp.get("rng_reseed", False)):
        torch.rand(1)
        log.info("rng_reseed: one draw taken from the global generator. This "
                 "is the RESEED CONTROL -- zero dose, perturbed stream.")
    # Hoisted: the per-epoch snapshot clone is gated on this, and a
    # state_dict() copied to CPU each epoch for a checkpoint nothing
    # reads is ~344 MB per epoch on ViTB16.
    allow_restore = _required(hp, "enable_checkpoint_restore", bool)
    device = inputs.device
    num_classes = inputs.num_classes
    model = inputs.model
    csv_log_path = str(inputs.csv_log_path)

    use_amp, amp_dtype, scaler = setup_runtime(device)

    warmup_epochs = hp["warmup_epochs"]
    constraint_epochs = _required(hp, "constraint_epochs", int)
    total_epochs = warmup_epochs + constraint_epochs
    lambda_step = hp["lambda_step"]
    stable_count_threshold = _required(hp, "stable_count_threshold", int)


    criterion_ce = make_ce_criterion(config, inputs.y_train, num_classes, device)
    lr_constraint = _required(hp, "lr_constraint", float)
    optimizer = make_optimizer(model.parameters(), lr_constraint, device)
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, hp["batch_size"])
    X_test = inputs.X_test.to(device)
    group_ids = torch.LongTensor(inputs.group_ids).to(device)
    global_con = inputs.global_con
    local_con = inputs.local_con

    # Bounded TraLO penalty machine.
    criterion_constraint = MulticlassTransductiveLoss(
        global_constraints=global_con, local_constraints=local_con,
        num_classes=num_classes,
        initial_rho=hp.get("initial_rho", 0.5),
        penalty_shape=str(hp.get("penalty_shape", "rational_bounded")),
    ).to(device)

    # Union of both scopes. Deriving this from global_con alone silently drops a
    # class that is capped locally but not globally: it gets no lambda, so both
    # L_Global and L_Local stay at exactly 0.0 for the whole run and the arm
    # trains as plain CE while reporting a constraint phase. The duals derive
    # theirs from inputs.constrained_classes and would honour it, so this was
    # also an arm-vs-arm asymmetry. It is latent while the generator always sets
    # both scopes together -- and it is exactly what breaks when we sweep G < L
    # to make the global scope the thing under test.
    constrained_classes = sorted(
        {c for c in range(num_classes) if global_con[c] < UNLIMITED}
        | {c for bounds in local_con.values()
           for c in range(num_classes) if bounds[c] < UNLIMITED})

    # REQUIRED, not defaulted. `tralo_null` is defined by zeroing exactly these
    # two plus lambda_step -- and lambda_step already raises when missing. A
    # hand-written null that omitted either of these would have silently trained
    # with a LIVE lambda = 0.01 while calling itself a control, which is the one
    # failure a control cannot survive.
    init_g = _required(hp, "lambda_global")
    init_l = _required(hp, "lambda_local")
    for c in constrained_classes:
        criterion_constraint.set_lambda_per_class(c, init_g, scope="global")
    for gid, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                criterion_constraint.set_lambda_per_class(c, init_l, scope="local", group_id=gid)


    rho_target = hp.get("rho_target", 100.0)
    initial_rho = hp.get("initial_rho", 0.5)
    rho_step = (rho_target - initial_rho) / max(constraint_epochs, 1)
    rho_frozen = False


    satisfaction_epoch = None
    stable_count = 0
    constraint_steps_applied = constraint_steps_attempted = 0
    best_sat_state = None
    best_sat_epoch = None
    min_excess_state = None
    min_excess_epoch = None
    min_total_excess = float("inf")
    training_start = time.time()

    write_csv_header(csv_log_path, num_classes, local_con)

    chunk_size = _required(hp, "constraint_chunk_size", int)
    # Baseline for the reordering diagnostic: the capped class's test ranking as
    # the WARM-UP left it, before a single constraint step.
    warmup_scores = capped_scores(model, X_test, constrained_classes,
                                  chunk_size)

    cached_train_acc = 0.0

    for epoch in range(warmup_epochs, total_epochs):
        # ---- CE pass ----
        model.train()
        for pg in optimizer.param_groups:
            pg["lr"] = lr_constraint
        epoch_ce = 0.0
        num_batches = max(len(train_loader), 1)
        train_correct, train_total = 0, 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits_ce = model(batch_X)
                loss_ce = criterion_ce(logits_ce, batch_y)
            if scaler:
                scaler.scale(loss_ce).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_ce.backward()
                optimizer.step()
            epoch_ce += loss_ce.item()
            with torch.no_grad():
                train_correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
                train_total += batch_y.size(0)
        cached_train_acc = ((train_correct / train_total)
                            if train_total > 0 else cached_train_acc)

        # ---- Transductive pass 1: aggregate soft + hard counts (no_grad, eval) ----
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        n_test = len(X_test)
        n_chunks = (n_test + chunk_size - 1) // chunk_size

        with torch.no_grad():
            total_global_soft = torch.zeros(num_classes, device=device)
            total_global_hard = torch.zeros(num_classes, device=device)
            total_local_soft = {gid: torch.zeros(num_classes, device=device)
                                for gid in criterion_constraint.local_groups}
            total_local_hard = {gid: torch.zeros(num_classes, device=device)
                                for gid in criterion_constraint.local_groups}
            kept_margins = [] if SOFT_COUNT_MODE == "margin" else None
            for ci in range(n_chunks):
                start = ci * chunk_size
                end = min(start + chunk_size, n_test)
                chunk_logits = model(X_test[start:end])
                chunk_proba = F.softmax(chunk_logits, dim=1)
                chunk_preds = chunk_logits.argmax(dim=1)
                if kept_margins is not None:
                    kept_margins.append(margins(chunk_proba))
                total_global_soft += chunk_proba.sum(dim=0)
                total_global_hard += torch.bincount(
                    chunk_preds, minlength=num_classes).float()
                chunk_gids = group_ids[start:end]
                for gid in total_local_soft:
                    mask = (chunk_gids == gid)
                    if mask.any():
                        total_local_soft[gid] += chunk_proba[mask].sum(dim=0)
                        total_local_hard[gid] += torch.bincount(
                            chunk_preds[mask], minlength=num_classes).float()

        # ---- soft_count_mode / straight_through ----
        # T is DERIVED, per class, per epoch, so the window always holds
        # `cut_window_items` items. A fixed T is not a fixed dose: measured on
        # the stored dermmnist evidence, the T holding ~20 items spans
        # 0.182 .. 0.502 ACROSS SEEDS OF ONE CELL, and margins keep growing
        # through the constraint phase as CE converges.
        #
        # And the count the penalty READS is the hard one. The existing chunked
        # construction below is `total.detach() - chunk.detach() + chunk`, so
        # seeding `total` with the HARD count makes the value exact while the
        # gradient still comes from the window -- a straight-through estimator
        # that falls out of the construction already here, no extra machinery.
        # Without it the window over-counts (56.6 against a hard 45 at 40
        # items) and the penalty keeps pushing after the cap is satisfied,
        # which is the overshoot measured in scripts/flag_live. With it,
        # `cut_window_items` moves ONLY where the gradient lands.
        cut_temp = None
        if SOFT_COUNT_MODE == "margin":
            cut_temp = window_temp(torch.cat(kept_margins, dim=0),
                                   CUT_WINDOW_ITEMS)
            del kept_margins
        if STRAIGHT_THROUGH:
            # Seed the detach construction below with the HARD count. Its value
            # becomes exact while the gradient still comes from whichever soft
            # count is selected -- a straight-through estimator that falls out
            # of the construction already here.
            #
            # INDEPENDENT OF soft_count_mode ON PURPOSE. These are two separate
            # fixes and bundling them makes the result unattributable: the
            # count's VALUE (sum_i p_ic tracks probability mass, not the hard
            # count) and the gradient's PLACEMENT (p(1-p) vs the margin window)
            # are different defects. Measured on the stored evidence, placement
            # is worth at most 1.30x on the items that must actually flip, so
            # an effect much larger than that is the VALUE fix, not this arm.
            total_global_soft = total_global_hard.clone()
            for gid in total_local_soft:
                total_local_soft[gid] = total_local_hard[gid].clone()

        # Snapshot pre-step state (matches counts above).
        snapshot_global_satisfied = True
        for c in constrained_classes:
            if total_global_hard[c].item() > criterion_constraint.global_constraints[c].item():
                snapshot_global_satisfied = False
                break
        snapshot_local_satisfied = True
        for gid_s, buffer_name_s in criterion_constraint.local_groups.items():
            lc_s = getattr(criterion_constraint, buffer_name_s)
            for c in constrained_classes:
                if c < len(lc_s) and lc_s[c] < UNLIMITED:
                    if total_local_hard[gid_s][c].item() > lc_s[c].item():
                        snapshot_local_satisfied = False
                        break
            if not snapshot_local_satisfied:
                break
        snapshot_total_excess = 0.0
        for c in constrained_classes:
            snapshot_total_excess += max(
                0.0, total_global_hard[c].item()
                - criterion_constraint.global_constraints[c].item())
        for gid_s, buffer_name_s in criterion_constraint.local_groups.items():
            lc_s = getattr(criterion_constraint, buffer_name_s)
            for c in constrained_classes:
                if c < len(lc_s) and lc_s[c] < UNLIMITED:
                    snapshot_total_excess += max(
                        0.0, total_local_hard[gid_s][c].item() - lc_s[c].item())
        snapshot_state = None
        snapshot_is_sat = snapshot_global_satisfied and snapshot_local_satisfied
        # Only clone when a restore could actually use it. Every config
        # the generator emits sets enable_checkpoint_restore=false, and
        # a full state_dict() copied to CPU each epoch is ~344 MB on
        # ViTB16 for a checkpoint nothing reads.
        if allow_restore and (snapshot_is_sat
                              or snapshot_total_excess < min_total_excess):
            snapshot_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # ---- Transductive pass 2: chunked backward ----
        # Compute info values for logging (no grad).
        loss_global_val = criterion_constraint.compute_global_from_counts(total_global_soft).item()
        loss_local_val = criterion_constraint.compute_local_from_counts(total_local_soft).item()
        bounded_total = loss_global_val + loss_local_val
        total_constraint = bounded_total

        has_constraint = total_constraint > 0
        if has_constraint:
            for ci in range(n_chunks):
                start = ci * chunk_size
                end = min(start + chunk_size, n_test)
                with constraint_autocast(amp_dtype, use_amp, step_cfg["fp32"]):
                    chunk_logits = model(X_test[start:end])
                chunk_logits_f = chunk_logits.float()
                chunk_proba = F.softmax(chunk_logits_f, dim=1)
                chunk_loss = torch.tensor(0.0, device=device)
                # Build chunked soft estimates (same g_soft trick as TraLO: each
                # chunk routes gradient only through its own samples, but the
                # value plugged into the penalty is the TOTAL soft count using
                # this chunk's grad-attached partial.)
                # Same window as pass 1. It is a per-item function of that
                # item's own row, so the chunked detach construction still
                # yields the exact full-N gradient.
                if SOFT_COUNT_MODE == "margin":
                    chunk_eff = margin_window(chunk_proba, cut_temp)
                elif SOFT_COUNT_MODE == "uniform":
                    # Value is exactly `p`, so pass 1's plain sum above is
                    # already the right total and the detach construction
                    # below stays exact. Only the per-item gradient changes,
                    # from p(1-p) to a constant -- see uniform_grad_count for
                    # the measurement that forced it.
                    chunk_eff = uniform_grad_count(chunk_proba)
                else:
                    chunk_eff = chunk_proba
                chunk_global = chunk_eff.sum(dim=0)
                chunk_gids = group_ids[start:end]
                chunk_local_soft = {}
                for gid in criterion_constraint.local_groups:
                    mask = (chunk_gids == gid)
                    if mask.any():
                        chunk_local_soft[gid] = chunk_eff[mask].sum(dim=0)
                    else:
                        chunk_local_soft[gid] = torch.zeros(num_classes, device=device)
                g_soft = (total_global_soft.detach()
                          - chunk_eff.sum(dim=0).detach() + chunk_global)
                l_soft = {}
                for gid in total_local_soft:
                    l_soft[gid] = (total_local_soft[gid].detach()
                                   - chunk_local_soft[gid].detach()
                                   + chunk_local_soft[gid])
                # ---- Bounded TraLO term ----
                lg = criterion_constraint.compute_global_from_counts(g_soft)
                ll = criterion_constraint.compute_local_from_counts(l_soft)
                # No /n_chunks. The detach construction above already yields
                # the EXACT full-N gradient, so dividing by the chunk count is
                # pure attenuation -- and n_chunks = ceil(N_test/chunk_size),
                # which made TraLO's effective constraint weight a function of
                # the dataset (derm 8, oct 4, tissue 10 => 2.5x apart) and of a
                # memory knob. That is a confound across the three headline
                # datasets, not a hyperparameter.
                chunk_loss = chunk_loss + lg + ll
                constraint_backward(chunk_loss, scaler, step_cfg["fp32"])

        last_grad_norm = 0.0
        did_backward = has_constraint
        if did_backward:
            last_grad_norm, applied = finish_constraint_step(
                model, optimizer, scaler, **step_cfg)
            # `applied` is False when the constraint gradient came back
            # non-finite, and then no step landed this epoch. Counting it is
            # the only way a reader can tell 29 steps from 19: the run still
            # writes `status: completed` either way.
            constraint_steps_attempted += 1
            constraint_steps_applied += 1 if applied else 0

        avg_ce = epoch_ce / num_batches

        # ---- Satisfaction / checkpoint ----
        global_satisfied = snapshot_global_satisfied
        local_satisfied = snapshot_local_satisfied
        is_satisfied = global_satisfied and local_satisfied
        if is_satisfied:
            stable_count += 1
            if snapshot_state is not None:
                best_sat_state = snapshot_state
                best_sat_epoch = epoch + 1
        else:
            stable_count = 0
        if snapshot_total_excess < min_total_excess and snapshot_state is not None:
            min_total_excess = snapshot_total_excess
            min_excess_state = snapshot_state
            min_excess_epoch = epoch + 1

        # ---- TraLO ratchet (lambda_T) ----
        ratchet_gate = satisfaction_epoch is None
        for c in constrained_classes:
            hard_c = total_global_hard[c].item()
            limit_c = criterion_constraint.global_constraints[c].item()
            if hard_c > limit_c and ratchet_gate:
                old = criterion_constraint.get_lambda_per_class(c, scope="global")
                criterion_constraint.set_lambda_per_class(
                    c, old + lambda_step, scope="global")
        for gid, buffer_name in criterion_constraint.local_groups.items():
            lc = getattr(criterion_constraint, buffer_name)
            for c in constrained_classes:
                if c < len(lc) and lc[c] < UNLIMITED:
                    hard_c = total_local_hard[gid][c].item()
                    if hard_c > lc[c].item() and ratchet_gate:
                        old = criterion_constraint.get_lambda_per_class(
                            c, scope="local", group_id=gid)
                        criterion_constraint.set_lambda_per_class(
                            c, old + lambda_step, scope="local", group_id=gid)

        if is_satisfied and satisfaction_epoch is None:
            satisfaction_epoch = epoch + 1
            if not rho_frozen:
                rho_frozen = True
                log.info("First satisfied at epoch %d, freezing rho=%.3f",
                         epoch + 1, criterion_constraint.get_rho())
        if not rho_frozen:
            criterion_constraint.increment_rho(rho_step)

        if stable_count >= stable_count_threshold:
            log.info("Converged: constraints stable for %d epochs", stable_count)
            break

        # EVERY epoch. This used to log every 5th, which wrote 7 rows for a
        # 29-epoch phase and made the count trajectory unwatchable: a 2-epoch
        # diagnostic run logged exactly one row, from BEFORE the treatment had
        # differentiated, so four configs with four different outputs all wrote
        # byte-identical logs. The duals already log every epoch, so this also
        # removes a cross-arm asymmetry. The block only reads tensors that are
        # already computed and `cached_train_acc` -- no forward pass, no RNG --
        # so density cannot change a result. Verified: identical predictions
        # md5 before and after.
        train_acc = cached_train_acc
        g_counts = {c: int(total_global_hard[c].item()) for c in range(num_classes)}
        l_counts = {gid: {c: int(total_local_hard[gid][c].item())
                          for c in range(num_classes)}
                    for gid in total_local_hard}
        g_soft_d = {c: total_global_soft[c].item() for c in range(num_classes)}
        l_soft_d = {gid: {c: total_local_soft[gid][c].item()
                          for c in range(num_classes)}
                    for gid in total_local_soft}
        mode_tag = "Satisfied" if is_satisfied else "Constraint"
        lam_local = criterion_constraint.lambda_local_per_key
        lam_L_mean = (sum(lam_local.values()) / len(lam_local)
                      if lam_local else 0.0)
        lam_T_mean = (sum(criterion_constraint.lambda_global_per_class.values())
                      / max(1, len(criterion_constraint.lambda_global_per_class)))
        log.info("Epoch %d [%s] ce=%.4f bounded=%.4f "
                 "lam_T=%.3f rho=%.3f acc=%.4f stable=%d g_%s l_%s",
                 epoch + 1, mode_tag, avg_ce, bounded_total,
                 lam_T_mean,
                 criterion_constraint.get_rho(), train_acc, stable_count,
                 "OK" if global_satisfied else "VIOL",
                 "OK" if local_satisfied else "VIOL")
        log_progress_to_csv(
            csv_log_path, epoch, avg_ce, train_acc,
            loss_global_val, loss_local_val,
            g_counts, l_counts, g_soft_d, l_soft_d,
            lam_T_mean, lam_L_mean,
            global_con, global_satisfied, local_satisfied,
            grad_norm=last_grad_norm, local_constraints=local_con)
        model.train()

    elapsed = time.time() - training_start
    log.info("Training complete: %.1fs, satisfaction epoch: %s",
             elapsed, satisfaction_epoch or "N/A")

    # ---- Final eval + checkpoint restore (mirror TraLO) ----
    model.eval()
    g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
        model, X_test, group_ids, num_classes=num_classes)
    final_total_excess = 0.0
    for c in range(num_classes):
        if global_con[c] < UNLIMITED:
            final_total_excess += max(0, g_counts.get(c, 0) - int(global_con[c]))
    if local_con:
        for gid, bounds in local_con.items():
            for c in range(num_classes):
                if bounds[c] < UNLIMITED:
                    final_total_excess += max(0, l_counts.get(gid, {}).get(c, 0) - int(bounds[c]))
    # `violates` is exactly `excess > 0`: the loop that used to compute it
    # separately tested the same condition, term by term, on the same two
    # scopes -- verified identical over 5000 randomized count/cap draws.
    final_violates = final_total_excess > 0

    restored_from_epoch = None
    restore_kind = None
    # The end-of-run restore swaps the trained model for an earlier checkpoint
    # chosen on CONSTRAINT SATISFACTION. Measured cost: -0.0351 AP across four
    # cells (restoreprobe, n=16, within-run), which is ~83% of TraLO's ranking
    # deficit against the post-hoc clipper. The clipper never restores.
    # Default True so every existing config keeps its behaviour bit for bit.
    if not allow_restore:
        log.info("enable_checkpoint_restore=False: keeping the trained model, "
                 "no lowest-excess / best-satisfied swap")
    if allow_restore and best_sat_state is not None and final_violates:
        log.info("Restoring best-satisfied checkpoint from epoch %d", best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            model, X_test, group_ids, num_classes=num_classes)
    elif (allow_restore and min_excess_state is not None
          and final_total_excess > min_total_excess):
        log.info("Restoring lowest-excess checkpoint from epoch %d (excess=%d)",
                 min_excess_epoch, int(min_total_excess))
        model.load_state_dict({k: v.to(device) for k, v in min_excess_state.items()})
        restored_from_epoch = min_excess_epoch
        restore_kind = "min_excess"
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            model, X_test, group_ids, num_classes=num_classes)

    # AFTER the restore: the restored model is the one whose predictions the
    # scorer reads, so it is the one whose ranking must be compared.
    reorder = reordering_report(model, X_test, warmup_scores, constrained_classes,
                                chunk_size)

    final_soft_hard_gap = {c: abs(g_soft.get(c, 0) - g_counts.get(c, 0))
                           for c in constrained_classes}

    return TrainOutputs(
        model=model,
        summary={
            "satisfaction_epoch": satisfaction_epoch,
            "best_sat_epoch": best_sat_epoch,
            "min_excess_epoch": min_excess_epoch,
            "min_total_excess": (None if min_total_excess == float("inf")
                                 else int(min_total_excess)),
            "restored_from_epoch": restored_from_epoch,
            "restore_kind": restore_kind,
            "soft_hard_gap": final_soft_hard_gap,
            # THE DOSE THAT ACTUALLY LANDED -- see dual_common.record_step.
            "constraint_steps_applied": int(constraint_steps_applied),
            "constraint_steps_attempted": int(constraint_steps_attempted),
            # tau near 1.0 with a large bias_shift = the count moved and the
            # RANKING did not, which 9 of the scorer's 13 metrics cannot see.
            "reordering": reorder,
        },
    )
