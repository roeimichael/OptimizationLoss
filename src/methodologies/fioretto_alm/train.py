"""fioretto_alm methodology: augmented-Lagrangian dual update (Track B / B3).

Identical to fioretto_ldf EXCEPT the dual (multiplier) update. Fioretto-LDF
accumulates the positive-part subgradient (lambda += step * excess^+), which
can only grow -- the "linear-penalty windup" the paper discusses. The
augmented-Lagrangian method (Hestenes 1969 / Powell 1969 / Rockafellar 1974)
is the standard literature fix, and R2 asked for it as a baseline. It

  (a) updates the multiplier on the RAW residual with a nonnegativity
      projection, so the multiplier can SHRINK when the constraint goes slack
      (dual descent), and
  (b) adds an augmentation penalty whose coefficient mu grows linearly, giving
      feasibility pressure without requiring the multiplier itself to wind up.

Update rule (advisor handoff B3):

    lambda_c <- max(0, lambda_c + eta (S_c - K_c)) + mu_t (S_c - K_c)^+
    mu_t = alm_mu0 + alm_mu_step * epoch     (linear growth)

where S_c is the soft (probability-sum) count and K_c the cap. The rule is
self-limiting: as the model reaches feasibility the residual (S_c - K_c)^+ -> 0,
the augmentation term vanishes, and the projected ascent term bleeds the
multiplier back down. Applied to both global and per-group (local) caps.

Everything else -- the two-pass transductive constraint gradient, CE-saturation
skip, grad-clip recovery, best-checkpoint restore on the excess axis, and the
5-consecutive-satisfied early stop -- is copied verbatim from fioretto_ldf so
the ALM/Fioretto comparison isolates ONLY the dual rule.
"""

import csv
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.pipeline.setup import setup_runtime
from src.pipeline.warmup import (make_ce_criterion, make_dataloader,
                                 make_optimizer)
from src.training.ce_schedule import CESaturationSkip
from src.training.constraint_step import (
    constraint_autocast, constraint_backward, finish_constraint_step)
from src.training.reordering import capped_scores, reordering_report
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _train_constraints(model, config, inputs, device):
    """Augmented-Lagrangian dual optimization (ALM variant of Fioretto Alg. 1/2)."""
    hp = inputs.hyperparams
    CLIP = _required(hp, "constraint_grad_clip")   # the treatment dose
    # Defaults reproduce the pre-2026-08-20 behaviour EXACTLY, so a config
    # generated before these keys existed still runs and still gives the same
    # numbers. That is why _required is not used here: the danger it guards
    # against is a default that silently disagrees with the protocol, and
    # "clip"/False is the protocol's own historical value.
    CONSTRAINT_GRAD_MODE = str(hp.get("constraint_grad_mode", "clip"))
    CONSTRAINT_FP32 = bool(hp.get("constraint_fp32", False))
    CONSTRAINT_STEP_RULE = str(hp.get("constraint_step_rule", "shared"))
    CONSTRAINT_RANDOM_DIR = bool(hp.get("constraint_random_direction", False))
    LR_CONSTRAINT = _required(hp, "lr_constraint")
    # Hoisted: the per-epoch snapshot clone is gated on this, and a
    # state_dict() copied to CPU each epoch for a checkpoint nothing
    # reads is ~344 MB per epoch on ViTB16.
    allow_restore = _required(hp, "enable_checkpoint_restore", bool)
    constraint_epochs = _required(hp, "constraint_epochs", int)
    # Apples-to-apples: same early-stop policy as TraLO/Fioretto (5 consecutive
    # satisfied epochs). Default matches TraLO.
    stable_count_threshold = _required(hp, "stable_count_threshold", int)
    lr_c = _required(hp, "lr_constraint", float)
    # ALM update hyperparameters. eta falls back to the Fioretto step size so a
    # config cloned from a Fioretto/TraLO cell runs without extra keys.
    eta = _required(hp, "alm_eta", float)
    mu0 = float(hp.get("alm_mu0", 0.01))
    mu_step = float(hp.get("alm_mu_step", 0.01))
    batch_size = hp.get("batch_size", 64)
    # protocol.yml carries this in BOTH the constraint_phase and chunked
    # blocks, so the 256 inline default could only ever fire on a
    # hand-written config -- exactly what _required exists to refuse.
    chunk_size = _required(hp, "constraint_chunk_size", int)

    use_amp, amp_dtype, scaler = setup_runtime(device)

    constrained_classes = inputs.constrained_classes
    num_classes = inputs.num_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids

    # ALM starts the multipliers at 0 (zero-start dual ascent); the augmentation
    # term supplies the initial feasibility pressure once a violation appears.
    lam0 = float(hp.get("fioretto_lambda_init", 0.0))
    lambda_g = {c: lam0 for c in constrained_classes if global_con[c] < UNLIMITED}
    lambda_l = {}
    aug_g, aug_l = {}, {}
    for group_id, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                lambda_l[(group_id, c)] = lam0

    log.info("Fioretto ALM: %d epochs, lr=%.2e, eta=%.4f, mu0=%.4f, mu_step=%.4f, "
             "%d global + %d local multipliers",
             constraint_epochs, lr_c, eta, mu0, mu_step, len(lambda_g), len(lambda_l))

    optimizer = make_optimizer(model.parameters(), lr_c, device)
    # Built from the config, exactly as tralo does. Constructing
    # nn.CrossEntropyLoss() directly here silently ignored
    # class_weighted_ce, so three of the four trained arms would have
    # run a different CE from the fourth the moment that key was
    # turned on -- and no gate could see it, because the key IS read
    # (by tralo) and IS emitted.
    criterion_ce = make_ce_criterion(inputs.config, inputs.y_train,
                                     num_classes, device)
    train_loader = make_dataloader(inputs.X_train, inputs.y_train, batch_size)

    X_test_dev = inputs.X_test.to(device)
    unique_groups = np.unique(groups_np)

    satisfaction_epoch = None
    # Best-checkpoint restore (apples-to-apples with TraLO/Fioretto): snapshot
    # model state BEFORE the constraint step at every epoch that satisfies or
    # improves on the lowest total excess seen so far.
    best_sat_state = None
    best_sat_epoch = None
    min_excess_state = None
    min_excess_epoch = None
    min_total_excess = float("inf")

    log_path = inputs.experiment_path / "training_log.csv"
    last_grad_norm = 0.0
    log_fields = ["epoch", "ce_loss", "constraint_loss", "total_excess",
                  "all_satisfied", "max_lambda_g", "mu_t",
                  # The raw norm BEFORE the unit clip. It is the whole dose
                  # question: FRAMEWORK measures the clip delivering exactly
                  # 1.000 against a raw norm of thousands, which makes the
                  # lambda ratchet a no-op. tralo logged it and these three
                  # discarded it, so the comparison was one arm wide.
                  "grad_norm"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, log_fields).writeheader()

    stable_count = 0  # consecutive satisfied epochs for early-stop parity
    # ONE schedule object, built from the SHARED constraint_phase block, so a
    # campaign cannot run this gate for one arm and not another -- the exact
    # defect that got the original CE-skip deleted.
    ce_skip = CESaturationSkip(hp)
    cached_train_acc = 0.0

    for epoch in range(constraint_epochs):
        epoch_start = time.time()
        mu_t = mu0 + mu_step * epoch  # linearly growing augmentation coefficient

        # ---- Step 1: CE on TRAIN data (batched) ----
        model.train()
        ce_losses = []
        train_correct, train_total = 0, 0
        for batch_X, batch_y in ([] if ce_skip.should_skip()
                                 else train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits_ce = model(batch_X)
                ce_loss = criterion_ce(logits_ce, batch_y)
            ce_losses.append(ce_loss.item())
            if scaler:
                scaler.scale(ce_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                ce_loss.backward()
                optimizer.step()
            with torch.no_grad():
                train_correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
                train_total += batch_y.size(0)

        cached_train_acc = ((train_correct / train_total)
                            if train_total > 0 else cached_train_acc)
        ce_skip.update(cached_train_acc, epoch)

        # ---- Step 2: constraint gradient on TEST data (transductive) ----
        model.eval()
        total_soft = torch.zeros(num_classes, device=device)
        group_soft = {g: torch.zeros(num_classes, device=device) for g in unique_groups}
        all_hard = []
        with torch.no_grad():
            for i in range(0, len(X_test_dev), chunk_size):
                chunk_logits = model(X_test_dev[i:i + chunk_size])
                chunk_proba = F.softmax(chunk_logits, dim=1)
                total_soft += chunk_proba.sum(dim=0)
                all_hard.append(chunk_logits.argmax(dim=1))
                chunk_groups = groups_np[i:i + chunk_size]
                for g in unique_groups:
                    mask = (chunk_groups == g)
                    if mask.any():
                        group_soft[g] += chunk_proba[mask].sum(dim=0)
            hard_preds = torch.cat(all_hard).cpu().numpy()

        # Raw residuals r_c = S_c - K_c (kept signed for the ALM ascent term);
        # violations_* hold the positive part for the loss gate (parity with
        # Fioretto: the constraint LOSS pushes only classes above the cap).
        residual_g = {}
        violations_g = {}
        violated_global = set()
        for c in constrained_classes:
            K = global_con[c]
            if K >= UNLIMITED:
                continue
            excess = total_soft[c].item() - K
            residual_g[c] = excess
            violations_g[c] = max(0.0, excess)
            if excess > 0:
                violated_global.add(c)

        residual_l = {}
        violations_l = {}
        violated_local = set()
        for g in unique_groups:
            bounds = local_con.get(g, [UNLIMITED] * num_classes)
            for c in constrained_classes:
                key = (g, c)
                if key not in lambda_l:
                    continue
                K_local = bounds[c]
                if K_local >= UNLIMITED:
                    continue
                excess = group_soft[g][c].item() - K_local
                residual_l[key] = excess
                violations_l[key] = max(0.0, excess)
                if excess > 0:
                    violated_local.add(key)

        # Hard-count satisfaction from pass-1 predictions BEFORE the step.
        hard_counts = {c: int((hard_preds == c).sum()) for c in constrained_classes}
        total_excess = sum(
            max(0, hard_counts[c] - int(global_con[c]))
            for c in constrained_classes if global_con[c] < UNLIMITED
        )
        if local_con:
            for g_id, bounds in local_con.items():
                for c in constrained_classes:
                    if bounds[c] < UNLIMITED:
                        gc = int(((hard_preds == c) & (groups_np == g_id)).sum())
                        total_excess += max(0, gc - int(bounds[c]))
        all_satisfied = (total_excess == 0)

        snapshot_state = None
        # Only clone when a restore could use it: every generated config
        # sets enable_checkpoint_restore=false, and a full state_dict()
        # copied to CPU each epoch is ~344 MB on ViTB16 for a checkpoint
        # nothing ever reads.
        if allow_restore and (all_satisfied or total_excess < min_total_excess):
            snapshot_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

        # MUST consult the SAME weights the chunk loss below uses. Reading
        # `lambda_*` alone made ALM's whole augmentation unreachable whenever
        # the multipliers start at 0: `w = lambda + aug` is what enters the
        # loss, so with lambda pinned at 0 and aug climbing, has_work stayed
        # False on every epoch while `training_log.csv` faithfully wrote a
        # rising mu_t. That is a treatment that logs itself and never happens
        # -- the exact shape of this project's four inert flags.
        has_work = (
            any(lambda_g.get(c, 0) + aug_g.get(c, 0.0) > 0
                for c in violated_global) or
            any(lambda_l.get(k, 0) + aug_l.get(k, 0.0) > 0
                for k in violated_local)
        )
        constraint_loss_val = 0.0
        did_backward = False
        if has_work:
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, len(X_test_dev), chunk_size):
                with constraint_autocast(amp_dtype, use_amp, CONSTRAINT_FP32):
                    chunk_logits = model(X_test_dev[i:i + chunk_size])
                    chunk_proba = F.softmax(chunk_logits, dim=1)
                    chunk_loss = torch.zeros(1, device=device)
                    for c in violated_global:
                        w_g = lambda_g[c] + aug_g.get(c, 0.0)
                        if w_g > 0:
                            chunk_loss = chunk_loss + w_g * chunk_proba[:, c].sum()
                    chunk_groups = groups_np[i:i + chunk_size]
                    for key in violated_local:
                        g, c = key
                        w_l = lambda_l[key] + aug_l.get(key, 0.0)
                        if w_l > 0:
                            mask = (chunk_groups == g)
                            if mask.any():
                                chunk_loss = chunk_loss + w_l * chunk_proba[mask, c].sum()
                if chunk_loss.item() > 0:
                    constraint_backward(chunk_loss, scaler, CONSTRAINT_FP32)
                    constraint_loss_val += chunk_loss.item()
                    did_backward = True
            if did_backward:
                last_grad_norm, _applied = finish_constraint_step(
                    model, optimizer, scaler, CLIP,
                    mode=CONSTRAINT_GRAD_MODE, fp32=CONSTRAINT_FP32,
                    step_rule=CONSTRAINT_STEP_RULE, lr=LR_CONSTRAINT,
                random_direction=CONSTRAINT_RANDOM_DIR)

        # ---- Step 3: augmented-Lagrangian dual update ----
        # lambda_c <- max(0, lambda_c + eta * r_c) + mu_t * (r_c)^+   (r_c signed)
        # Hestenes/Powell: the MULTIPLIER is lam <- max(0, lam + eta*r). The
        # augmentation mu_t*r+ is a property of the current iterate and is added
        # to the PRIMAL weight at use time (see aug_g / aug_l below), never
        # stored back into lam -- storing it compounds it every epoch.
        for c, r in residual_g.items():
            lambda_g[c] = max(0.0, lambda_g[c] + eta * r)
            aug_g[c] = mu_t * max(0.0, r)
        for key, r in residual_l.items():
            lambda_l[key] = max(0.0, lambda_l[key] + eta * r)
            aug_l[key] = mu_t * max(0.0, r)

        if all_satisfied and satisfaction_epoch is None:
            satisfaction_epoch = epoch + 1
            log.info("Fioretto ALM: first satisfaction at epoch %d", epoch + 1)
        if all_satisfied:
            stable_count += 1
            if snapshot_state is not None:
                best_sat_state = snapshot_state
                best_sat_epoch = epoch + 1
        else:
            stable_count = 0
        if total_excess < min_total_excess and snapshot_state is not None:
            min_total_excess = total_excess
            min_excess_state = snapshot_state
            min_excess_epoch = epoch + 1

        row = {
            "epoch": epoch,
            "ce_loss": round(np.mean(ce_losses), 6),
            "constraint_loss": round(constraint_loss_val, 6),
            "total_excess": total_excess,
            "all_satisfied": int(all_satisfied),
            "max_lambda_g": round(max(lambda_g.values()) if lambda_g else 0, 6),
            "mu_t": round(mu_t, 6),
            "grad_norm": round(float(last_grad_norm), 6),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, log_fields).writerow(row)

        if epoch < 5 or (epoch + 1) % 25 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join(f"c{c}={lambda_g[c]:.3f}" for c in sorted(lambda_g))
            log.info("Fioretto ALM %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s stable=%d "
                     "mu=%.3f lam=[%s] [%.1fs]",
                     epoch + 1, constraint_epochs, np.mean(ce_losses),
                     constraint_loss_val, total_excess, all_satisfied,
                     stable_count, mu_t, lam_str, time.time() - epoch_start)

        if stable_count >= stable_count_threshold:
            log.info("Fioretto ALM: converged (constraints stable for %d epochs at ep %d)",
                     stable_count, epoch + 1)
            break

    return (satisfaction_epoch, best_sat_state, best_sat_epoch,
            min_excess_state, min_excess_epoch, min_total_excess,
            ce_skip.summary())


def train(inputs: TrainInputs) -> TrainOutputs:
    hp = inputs.hyperparams
    allow_restore = _required(hp, "enable_checkpoint_restore", bool)
    model = inputs.model
    device = inputs.device

    # Baseline for the reordering diagnostic, captured BEFORE a single
    # constraint step. This used to exist only in tralo/train.py, so for these
    # three arms "did the constraint phase reorder anything, or only shift a
    # bias the scorer cannot see" was unanswerable -- the same asymmetry that
    # made the CE-skip flag a 0.22 cc-F1 artifact.
    _reorder_chunk = _required(inputs.hyperparams, "constraint_chunk_size", int)
    _warmup_scores = capped_scores(model, inputs.X_test, inputs.constrained_classes,
                                   _reorder_chunk)

    (satisfaction_epoch, best_sat_state, best_sat_epoch,
     min_excess_state, min_excess_epoch, min_total_excess, ce_skip_summary
     ) = _train_constraints(model, inputs.config, inputs, device)

    # Apples-to-apples checkpoint restore (mirrors TraLO/Fioretto): selection on
    # the constraint-excess axis, NOT on F1.
    constrained_classes = inputs.constrained_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids
    X_test_dev = inputs.X_test.to(device)

    model.eval()
    with torch.no_grad():
        chunk_size = _required(inputs.hyperparams, "constraint_chunk_size", int)
        all_hard = []
        for i in range(0, len(X_test_dev), chunk_size):
            chunk_logits = model(X_test_dev[i:i + chunk_size])
            all_hard.append(chunk_logits.argmax(dim=1))
        hard_preds = torch.cat(all_hard).cpu().numpy()
    final_total_excess = sum(
        max(0, int((hard_preds == c).sum()) - int(global_con[c]))
        for c in constrained_classes if global_con[c] < UNLIMITED
    )
    if local_con:
        for g_id, bounds in local_con.items():
            for c in constrained_classes:
                if bounds[c] < UNLIMITED:
                    gc = int(((hard_preds == c) & (groups_np == g_id)).sum())
                    final_total_excess += max(0, gc - int(bounds[c]))
    final_violates = final_total_excess > 0

    restored_from_epoch = None
    restore_kind = None
    # Same gate as fioretto_ldf / hounie_rcl / tralo. Without it ALM restored a
    # best-satisfied checkpoint unconditionally while the other duals could be
    # denied that, so a comparison against them was not apples to apples.
    if not allow_restore:
        log.info("ALM: enable_checkpoint_restore=False, keeping the trained model")
    if allow_restore and best_sat_state is not None and final_violates:
        log.info("Fioretto ALM: final violates; restoring best-satisfied checkpoint from epoch %d",
                 best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
    elif (allow_restore and min_excess_state is not None
          and final_total_excess > min_total_excess):
        log.info("Fioretto ALM: final excess=%d > min seen excess=%d (epoch %d); "
                 "restoring lowest-excess checkpoint",
                 int(final_total_excess), int(min_total_excess), min_excess_epoch)
        model.load_state_dict({k: v.to(device) for k, v in min_excess_state.items()})
        restored_from_epoch = min_excess_epoch
        restore_kind = "min_excess"

    # AFTER the restore: the restored model is the one whose
    # predictions the scorer reads.
    _reorder = reordering_report(model, inputs.X_test, _warmup_scores,
                                 inputs.constrained_classes,
                                 _reorder_chunk)

    return TrainOutputs(
        model=model,
        summary={
            # WHETHER and WHEN the CE gate fired. "never fired" and
            # "fired and did nothing" are different results that look
            # identical in the metrics.
            "ce_skip": ce_skip_summary,
            "satisfaction_epoch": satisfaction_epoch,
            "best_sat_epoch": best_sat_epoch,
            "min_excess_epoch": min_excess_epoch,
            "min_total_excess": (None if min_total_excess == float("inf")
                                 else int(min_total_excess)),
            "restored_from_epoch": restored_from_epoch,
            "restore_kind": restore_kind,
            # tau near 1.0 with a large bias_shift = the count moved and
            # the RANKING did not, which 9 of 13 scored metrics cannot see.
            "reordering": _reorder,
        },
    )
