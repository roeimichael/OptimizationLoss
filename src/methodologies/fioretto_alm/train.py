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

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.pipeline.contracts import TrainInputs, TrainOutputs, _required
from src.pipeline.setup import setup_runtime
from src.training.constraint_step import (
    constraint_autocast, constraint_backward, finish_constraint_step)
from src.methodologies.dual_common import (Checkpoints, ce_epoch,
                                          count_excess, dual_setup,
                                          open_epoch_log, read_step_config,
                                          run_dual_arm, transductive_counts)
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _train_constraints(model, inputs, device):
    """Augmented-Lagrangian dual optimization (ALM variant of Fioretto Alg. 1/2)."""
    hp = inputs.hyperparams
    step_cfg = read_step_config(hp)
    allow_restore = _required(hp, "enable_checkpoint_restore", bool)
    constraint_epochs = _required(hp, "constraint_epochs", int)
    # Apples-to-apples: same early-stop policy as TraLO/Fioretto (5 consecutive
    # satisfied epochs). Default matches TraLO.
    stable_count_threshold = _required(hp, "stable_count_threshold", int)
    lr_c = _required(hp, "lr_constraint", float)
    # ALM update hyperparameters. eta falls back to the Fioretto step size so a
    # config cloned from a Fioretto/TraLO cell runs without extra keys.
    eta = _required(hp, "alm_eta", float)
    mu0 = _required(hp, "alm_mu0")
    mu_step = _required(hp, "alm_mu_step")
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

    optimizer, criterion_ce, train_loader = dual_setup(
        model, inputs, device, lr_c, batch_size)

    X_test_dev = inputs.X_test.to(device)
    unique_groups = np.unique(groups_np)

    ck = Checkpoints(allow_restore, "Fioretto ALM")

    log_fields = ["epoch", "train_acc", "ce_loss", "constraint_loss",
                  "total_excess",
                  "all_satisfied", "max_lambda_g", "mu_t",
                  "grad_norm"]
    write_row = open_epoch_log(inputs.experiment_path, log_fields)

    stable_count = 0  # consecutive satisfied epochs for early-stop parity

    for epoch in range(constraint_epochs):
        # Reset EVERY epoch. Hoisted above the loop it was carried
        # forward: the arms only reassign it inside `if did_backward`,
        # so an epoch where the constraint went slack logged the
        # PREVIOUS epoch's norm as if it were this one's. tralo resets
        # per-epoch and logs 0.0, so a tralo-vs-dual dose comparison off
        # this column was asymmetric by construction -- and grad_norm is
        # the column this project uses to decide whether two arms got a
        # comparable dose. Verified inert on results/dualbar2 (no slack
        # epoch occurs there), so no stored number moves.
        last_grad_norm = 0.0
        epoch_start = time.time()
        mu_t = mu0 + mu_step * epoch  # linearly growing augmentation coefficient

        # ---- Step 1: CE on TRAIN data (batched) ----
        ce_losses, train_acc = ce_epoch(model, train_loader, optimizer, criterion_ce,
                             device, amp_dtype, use_amp, scaler)


        # ---- Step 2: constraint gradient on TEST data (transductive) ----
        total_soft, group_soft, hard_preds = transductive_counts(
            model, X_test_dev, groups_np, unique_groups, num_classes,
            chunk_size, device)

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
        total_excess = count_excess(hard_preds, groups_np, constrained_classes,
                                    global_con, local_con)
        all_satisfied = (total_excess == 0)

        snapshot_state = ck.snapshot(model, all_satisfied, total_excess)

        # MUST consult the SAME weights the chunk loss below uses. Reading
        # `lambda_*` alone made ALM's whole augmentation unreachable whenever
        # the multipliers start at 0: `w = lambda + aug` is what enters the
        # loss, so with lambda pinned at 0 and aug climbing, has_work stayed
        # False on every epoch while `training_log.csv` faithfully wrote a
        # rising mu_t. That is a treatment that logs itself and never happens
        # -- the exact shape of this project's four inert flags.
        # THE AUGMENTATION IS A PROPERTY OF THE CURRENT ITERATE, so it must be
        # built from THIS epoch's residuals -- which are complete above -- and
        # not left to Step 3, which runs AFTER the pass that uses it. Doing it
        # there weighted epoch e by `lambda_e + mu_{e-1} * r_{e-1}^+`: a current
        # multiplier plus a one-epoch-stale augmentation, on the term that
        # DOMINATES lambda early. Under `constraint_grad_mode: clip` the
        # magnitude error is cancelled, but the SCOPE MIX is not -- global and
        # local residuals follow different trajectories, so the lag reweights
        # one scope against the other and that survives the clip.
        for c, r in residual_g.items():
            aug_g[c] = mu_t * max(0.0, r)
        for key, r in residual_l.items():
            aug_l[key] = mu_t * max(0.0, r)

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
                with constraint_autocast(amp_dtype, use_amp, step_cfg["fp32"]):
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
                    constraint_backward(chunk_loss, scaler, step_cfg["fp32"])
                    constraint_loss_val += chunk_loss.item()
                    did_backward = True
            if did_backward:
                last_grad_norm, _applied = finish_constraint_step(
                    model, optimizer, scaler, **step_cfg)

        # ---- Step 3: augmented-Lagrangian dual update ----
        # Hestenes/Powell: the MULTIPLIER ascends, lam <- max(0, lam + eta*r).
        # ONLY the multiplier is updated here. The augmentation mu_t*(r)^+ is
        # built from the current iterate ABOVE and added to the primal weight
        # at use time -- never stored back into lam, which would compound it
        # every epoch, and never deferred to here, which would lag it by one.
        for c, r in residual_g.items():
            lambda_g[c] = max(0.0, lambda_g[c] + eta * r)
        for key, r in residual_l.items():
            lambda_l[key] = max(0.0, lambda_l[key] + eta * r)

        ck.record(snapshot_state, all_satisfied, total_excess, epoch)
        # Apples-to-apples early stop: N consecutive satisfied epochs.
        stable_count = stable_count + 1 if all_satisfied else 0

        row = {
            "epoch": epoch,
            "train_acc": round(train_acc, 4),
            "ce_loss": round(np.mean(ce_losses), 6),
            "constraint_loss": round(constraint_loss_val, 6),
            "total_excess": total_excess,
            "all_satisfied": int(all_satisfied),
            "max_lambda_g": round(max(lambda_g.values()) if lambda_g else 0, 6),
            "mu_t": round(mu_t, 6),
            "grad_norm": round(float(last_grad_norm), 6),
        }
        write_row(row)

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

    return ck


def train(inputs: TrainInputs) -> TrainOutputs:
    return run_dual_arm(inputs, _train_constraints, "Fioretto ALM")
