"""hounie_rcl methodology: Resilient Constrained Learning.

Faithful reimplementation of the algorithm from:

    Hounie, Ribeiro, Chamon. "Resilient Constrained Learning."
    NeurIPS 2023. arXiv:2306.02426.

Algorithm 1 (generic) + Algorithm 2 (federated specialisation) collapsed onto
TraLO's prediction-count constraint task. The mapping from their notation to
this code is documented in `archive/benchmarks/hounie/` (reference implementation).

Per epoch, three updates:

    theta:  primal SGD on    L = L_ce + sum_i lam_i * (l_i - u_i)
    u:      grad ascent on   max_u  lam_i*u_i - h(u)  (with h = alpha*||u||^2)
            (this line read `-h(u) - lam_i*u_i`, whose gradient is
             -2*alpha*u - lam and does NOT give the arrow below. The
             arrow and the code are correct; the objective statement was
             not. Fixed point u* = lam/(2*alpha).)
            -> u_i <- max(0, u_i + eta_u * (lam_i - 2*alpha*u_i))
    lam:    dual ascent on   E[l_i] - u_i
            -> lam_i <- max(0, lam_i + eta_lam * (E[l_i] - u_i))

Constraint losses for prediction-count case:

    l_c(f_theta(x))      = softmax_c(f_theta(x)) - K_c / N
    l_{g,c}(f_theta(x))  = softmax_c(f_theta(x)) - K_{g,c} / N_g

So mean over the test set ((1/N) sum l_c) equals (count_soft_c - K_c) / N.
Constraint satisfied iff count_soft_c <= K_c.

No posthoc, no best_excess pick - paper takes the final-epoch model. Posthoc is
applied at the runner level for fair comparison with TraLO/Fioretto.
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
                                          count_excess, count_fields,
                                          count_row, dual_setup,
                                          open_epoch_log, read_step_config,
                                          run_dual_arm, transductive_counts)
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _train_constraints(model, inputs: TrainInputs, device):
    hp = inputs.hyperparams
    step_cfg = read_step_config(hp)
    allow_restore = _required(hp, "enable_checkpoint_restore", bool)
    constraint_epochs = _required(hp, "constraint_epochs", int)
    lr_c = _required(hp, "lr_constraint", float)
    # Default dual-step bumped 10x for apples-to-apples convergence speed.
    # At 0.01 (original) lambda grows ~0.01/epoch when (count_soft-K)/N ~= 0.04
    # -> constraint contribution to L_total is ~1e-3, ~25x weaker than CE.
    # The model effectively trains CE-only for 100+ epochs before lambda
    # builds up. With 0.1 lambda hits meaningful magnitude by ep 10.
    eta_lambda = _required(hp, "hounie_eta_lambda", float)
    eta_u = _required(hp, "hounie_eta_u", float)
    alpha = float(hp.get("hounie_alpha", 10.0))
    if abs(1.0 - 2.0 * eta_u * alpha) >= 1.0:
        raise ValueError(
            f"hounie_rcl: eta_u={eta_u} with alpha={alpha} gives stability "
            f"factor {1.0 - 2.0 * eta_u * alpha:+.3f}; |factor| >= 1 means the "
            f"perturbation u oscillates or diverges instead of converging to "
            f"lambda/(2*alpha). The paper's value is eta_u=0.01.")
    batch_size = hp.get("batch_size", 64)
    # protocol.yml carries this in BOTH the constraint_phase and chunked
    # blocks, so the 256 inline default could only ever fire on a
    # hand-written config -- exactly what _required exists to refuse.
    chunk_size = _required(hp, "constraint_chunk_size", int)
    # Apples-to-apples early stop: 5 consecutive satisfied epochs (matches TraLO).
    stable_count_threshold = _required(hp, "stable_count_threshold", int)

    use_amp, amp_dtype, scaler = setup_runtime(device)

    constrained_classes = inputs.constrained_classes
    num_classes = inputs.num_classes
    global_con = inputs.global_con
    local_con = inputs.local_con
    groups_np = inputs.group_ids

    n_test = len(inputs.X_test)
    unique_groups = np.unique(groups_np)
    group_sizes = {int(g): int((groups_np == g).sum()) for g in unique_groups}

    # K thresholds per active constraint (in absolute counts).
    K_global = {c: float(global_con[c])
                for c in constrained_classes if global_con[c] < UNLIMITED}
    K_local = {}
    for g in unique_groups:
        bounds = local_con.get(int(g), [UNLIMITED] * num_classes)
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                K_local[(int(g), c)] = float(bounds[c])

    # Multipliers and slack variables, one per active constraint.
    lam_g = {c: 0.0 for c in K_global}
    lam_l = {key: 0.0 for key in K_local}
    u_g = {c: 0.0 for c in K_global}
    u_l = {key: 0.0 for key in K_local}

    log.info(
        "Hounie RCL: %d constraint epochs, lr=%.2e eta_lam=%.4f eta_u=%.4f alpha=%.2f, "
        "%d global + %d local constraints",
        constraint_epochs, lr_c, eta_lambda, eta_u, alpha,
        len(lam_g), len(lam_l),
    )

    optimizer, criterion_ce, train_loader = dual_setup(
        model, inputs, device, lr_c, batch_size)

    X_test_dev = inputs.X_test.to(device)

    log_fields = ["epoch", "train_acc", "ce_loss", "constraint_loss",
                  "total_excess",
                  "all_satisfied", "max_lam_g", "max_u_g", "h_u",
                  "grad_norm"]
    # The per-class counts every reader needs, named as tralo names them.
    log_fields = log_fields + count_fields(constrained_classes)
    write_row = open_epoch_log(inputs.experiment_path, log_fields)

    ck = Checkpoints(allow_restore, "Hounie")
    stable_count = 0


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

        # ---- Step 1: CE on TRAIN (theta SGD on L_ce) ----
        ce_losses, train_acc = ce_epoch(model, train_loader, optimizer, criterion_ce,
                             device, amp_dtype, use_amp, scaler)


        # ---- Step 2: soft-count gradient on TEST (theta SGD on Σ_i lam_i * E[l_i]) ----
        total_soft, group_soft, hard_preds = transductive_counts(
            model, X_test_dev, groups_np, unique_groups, num_classes,
            chunk_size, device)

        # Compute hard-count satisfaction from pass-1 predictions BEFORE the
        # constraint step. Required so the snapshot below reflects the exact
        # model that produced these counts.
        total_excess_pre = count_excess(hard_preds, groups_np, constrained_classes,
                                        global_con, local_con)
        all_satisfied_pre = (total_excess_pre == 0)
        snapshot_state = ck.snapshot(model, all_satisfied_pre, total_excess_pre)

        # Second pass: weighted gradient if any lam > 0.
        constraint_loss_val = 0.0
        has_active = (any(v > 0 for v in lam_g.values())
                      or any(v > 0 for v in lam_l.values()))
        did_backward = False
        if has_active:
            optimizer.zero_grad(set_to_none=True)
            for i in range(0, n_test, chunk_size):
                with constraint_autocast(amp_dtype, use_amp, step_cfg["fp32"]):
                    chunk_logits = model(X_test_dev[i:i + chunk_size])
                    chunk_proba = F.softmax(chunk_logits.float(), dim=1)
                    chunk_loss = torch.zeros(1, device=device)
                    # AUDIT BUGFIX: divide by n_test / N_g to match the dual ascent
                    # scale (mean_l = sum/N), so primal d/dtheta and dual lambda
                    # update are on the same scale. Without this, primal gradient
                    # is N-times stronger than intended -> over-suppresses the
                    # constrained class and inflates ECE.
                    for c, lam in lam_g.items():
                        if lam > 0:
                            chunk_loss = chunk_loss + lam * chunk_proba[:, c].sum() / n_test
                    chunk_groups = groups_np[i:i + chunk_size]
                    for (g, c), lam in lam_l.items():
                        if lam > 0:
                            mask = (chunk_groups == g)
                            if mask.any():
                                N_g = max(1, group_sizes[g])
                                chunk_loss = chunk_loss + lam * chunk_proba[mask, c].sum() / N_g
                if chunk_loss.item() > 0:
                    constraint_backward(chunk_loss, scaler, step_cfg["fp32"])
                    constraint_loss_val += chunk_loss.item()
                    did_backward = True
            if did_backward:
                # Grad clip + grad_norm>0 gate + scaler.update() always called
                # (mirrors TraLO recovery pattern).
                last_grad_norm, applied = finish_constraint_step(
                    model, optimizer, scaler, **step_cfg)
                # `applied` is False when the constraint gradient
                # was non-finite, and then NO step landed this
                # epoch. Dropping it on the floor is how this arm
                # ran a 62%-length constraint phase while writing
                # `status: completed`.
                ck.record_step(applied)

        # ---- Step 3: dual ascent on lambda (paper Eq. 5 / Alg. 2) ----
        # E[l_i] = (count_soft_i - K_i) / N_i  (per-constraint normalisation).
        for c, K in K_global.items():
            mean_l = (total_soft[c].item() - K) / n_test
            lam_g[c] = max(0.0, lam_g[c] + eta_lambda * (mean_l - u_g[c]))
        for (g, c), K in K_local.items():
            N_g = max(1, group_sizes[g])
            mean_l = (group_soft[g][c].item() - K) / N_g
            lam_l[(g, c)] = max(0.0, lam_l[(g, c)] + eta_lambda * (mean_l - u_l[(g, c)]))

        # ---- Step 4: perturbation update on u (h(u) = alpha * ||u||^2) ----
        # u_i <- max(0, u_i + eta_u * (lam_i - 2 * alpha * u_i)).
        #
        # NOTE, deliberate: this reads the lambda just written in Step 3
        # (Gauss-Seidel), while the archived derivation writes both updates
        # from the previous iterate (Jacobi). Measured at the protocol values
        # (eta_lambda=eta_u=0.01, alpha=10, 29 epochs) over four violation
        # profiles, the worst relative lambda difference after the full run is
        # 4.7e-04, and the two schemes share the SAME fixed point by
        # construction (at a fixed point lam_t == lam_{t-1}, so they coincide).
        # That is far below the FP16/BF16 cross-server spread this project
        # already accepts. Left as-is on purpose: switching to Jacobi would
        # change every hounie number by ~0.05% for no benefit and break
        # bit-identity with the runs already in the corpus.
        for c in K_global:
            u_g[c] = max(0.0, u_g[c] + eta_u * (lam_g[c] - 2.0 * alpha * u_g[c]))
        for key in K_local:
            u_l[key] = max(0.0, u_l[key] + eta_u * (lam_l[key] - 2.0 * alpha * u_l[key]))

        # ---- Bookkeeping ---- (uses the pre-step satisfaction state computed
        # earlier, which is what the snapshot reflects).
        total_excess = total_excess_pre
        all_satisfied = all_satisfied_pre
        ck.record(snapshot_state, all_satisfied, total_excess, epoch)
        # Apples-to-apples early stop: N consecutive satisfied epochs.
        stable_count = stable_count + 1 if all_satisfied else 0

        h_u = alpha * (sum(v ** 2 for v in u_g.values())
                       + sum(v ** 2 for v in u_l.values()))

        row = {
            "epoch": epoch,
            "train_acc": round(train_acc, 4),
            "ce_loss": round(np.mean(ce_losses), 6),
            "constraint_loss": round(constraint_loss_val, 6),
            "total_excess": total_excess,
            "all_satisfied": int(all_satisfied),
            "max_lam_g": round(max(lam_g.values()) if lam_g else 0.0, 6),
            "max_u_g": round(max(u_g.values()) if u_g else 0.0, 6),
            "h_u": round(h_u, 6),
            "grad_norm": round(float(last_grad_norm), 6),
        }
        row.update(count_row(hard_preds, total_soft,
                             constrained_classes, global_con))
        write_row(row)

        if epoch < 5 or (epoch + 1) % 10 == 0 or epoch == constraint_epochs - 1:
            lam_str = " ".join(f"c{c}={lam_g[c]:.3f}" for c in sorted(lam_g))
            u_str = " ".join(f"c{c}={u_g[c]:.3f}" for c in sorted(u_g))
            log.info(
                "Hounie %d/%d: CE=%.4f cstr=%.4f excess=%d sat=%s stable=%d "
                "lam=[%s] u=[%s] h_u=%.4f [%.1fs]",
                epoch + 1, constraint_epochs,
                np.mean(ce_losses), constraint_loss_val, total_excess,
                all_satisfied, stable_count, lam_str, u_str, h_u,
                time.time() - epoch_start,
            )

        if stable_count >= stable_count_threshold:
            log.info("Hounie: converged (constraints stable for %d epochs at ep %d)",
                     stable_count, epoch + 1)
            break

    return ck


def train(inputs: TrainInputs) -> TrainOutputs:
    return run_dual_arm(inputs, _train_constraints, "Hounie")
