"""TraLO-Fioretto hybrid methodology.

Mixes TraLO's bounded saturated penalty (per-class lambda ratchet) with
an undershoot hinge to control post-satisfaction parking behaviour.

  bounded_only:     L = CE + Sum_c lambda_T_c * bounded(E_c)
                    TraLO bounded penalty alone (above K only).

  undershoot_hinge: L = CE + Sum_c lambda_T_c * [ bounded(E_c)
                                                + beta * relu(K_c - soft_count_c) / K_c ]
                    Bounded penalty above K + linear hinge pushing back UP
                    when below K. Asymmetric pair that parks near K from
                    both sides without overshooting either way.

bounded(E_c) = E/(E+K) + rho * (E/K)^2 / (1 + (E/K)^2),   E = relu(soft_count - K).

Apples-to-apples machinery mirrors TraLO: eval mode in transductive passes,
CE saturation skip, grad clip + norm gate, best_sat/min_excess restore,
5-consecutive early stop.
"""

import logging
import os
import time

import torch
import torch.nn.functional as F

from src.losses import MulticlassTransductiveLoss
from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.pipeline.setup import setup_runtime
from src.pipeline.warmup import make_ce_criterion, make_dataloader, make_optimizer
from src.training.logging import log_progress_to_csv, write_csv_header
from src.training.metrics import compute_prediction_statistics
from src.utils.constants import UNLIMITED
from src.utils.error_handler import logger

log = logging.getLogger(__name__)

CONSTRAINT_CHUNK_SIZE = 256
VALID_MODES = ("bounded_only", "undershoot_hinge")


def class_margin(logits, c):
    """logit(p_c) = z_c - logsumexp_{j != c} z_j.

    Strictly monotone in p_c, so ranking the pool by this quantity is exactly
    the ranking a budget-K allocator uses.  This is the margin the cut
    objective acts on.
    """
    keep = [j for j in range(logits.shape[1]) if j != c]
    return logits[:, c] - torch.logsumexp(logits[:, keep], dim=1)


def argmax_margin(logits, c):
    """z_c - max_{j != c} z_j.  Positive iff argmax == c.

    This is the predicate satisfaction is verified on, so it is the one the
    count surrogate should smooth (soft_count_mode="sigmoid").
    """
    keep = [j for j in range(logits.shape[1]) if j != c]
    return logits[:, c] - logits[:, keep].max(dim=1).values


def build_cut_plan(m, idx, K, gamma):
    """Detached cut geometry for one (scope, class) cap.

    m    margins for the participating samples
    idx  their positions in the pool
    K    the cap
    Returns None when the cap cannot define a cut (K outside the pool).
    """
    n = m.numel()
    if K < 1 or K >= n:
        return None
    ms, _ = torch.sort(m, descending=True)
    theta = 0.5 * (ms[K - 1] + ms[K])
    med = m.median()
    scale = (m - med).abs().median().clamp_min(1e-6)
    sign = torch.where(m > theta, 1.0, -1.0)
    active = (gamma - sign * (m - theta) / scale) > 0
    return dict(idx=idx, sign=sign, theta=theta, scale=scale,
                n_act=int(active.sum().item()),
                n_keep_act=int((active & (sign > 0)).sum().item()),
                K=int(K), n=int(n))


@logger()
def train(inputs: TrainInputs) -> TrainOutputs:
    config = inputs.config
    hp = inputs.hyperparams
    device = inputs.device
    num_classes = inputs.num_classes
    model = inputs.model
    csv_log_path = str(inputs.csv_log_path)

    use_amp, amp_dtype, scaler = setup_runtime(device)

    warmup_epochs = hp["warmup_epochs"]
    constraint_epochs = hp.get("constraint_epochs", 150)
    total_epochs = warmup_epochs + constraint_epochs
    lambda_step = hp["lambda_step"]
    stable_count_threshold = int(hp.get("stable_count_threshold", 5))

    hybrid_mode = hp.get("hybrid_mode", "undershoot_hinge")
    if hybrid_mode not in VALID_MODES:
        raise ValueError(f"hybrid_mode must be one of {VALID_MODES}, got {hybrid_mode!r}")
    # Adam state hangover diagnosis (hybrid_v2 reveal): after the descent
    # phase, Adam has accumulated momentum in the "decrease soft_4" direction.
    # Once bounded penalty disengages and only the hinge gradient is acting,
    # that small positive-soft gradient can't overcome accumulated negative-
    # soft momentum, so soft keeps drifting down.
    #   reset_optimizer_at_sat: rebuild the optimizer with fresh state at
    #     first satisfaction. Clears m/v buffers entirely.
    #   post_sat_optimizer: "adam" (default) | "sgd" — switch optimizer
    #     family at first satisfaction. SGD has no momentum so post-sat
    #     gradient effects are purely current-step.
    reset_optimizer_at_sat = bool(hp.get("reset_optimizer_at_sat", False))
    post_sat_optimizer = str(hp.get("post_sat_optimizer", "adam")).lower()
    if post_sat_optimizer not in ("adam", "sgd"):
        raise ValueError(f"post_sat_optimizer must be adam|sgd, got {post_sat_optimizer!r}")
    # Ablation flag: when True the per-class lambda ratchet keeps incrementing
    # and rho keeps schedule-ramping even after first satisfaction. Defaults
    # to False = freeze on satisfy (the published TraLO behaviour).
    disable_freeze_on_satisfy = bool(hp.get("disable_freeze_on_satisfy", False))
    fior_beta = float(hp.get("fior_beta", 0.0))
    alpha_kl = float(hp.get("alpha_kl", 0.0))

    # ---------------- GEOM arm: the cut objective ------------------------
    # The count constraint splits into two parts that are worth very
    # different amounts.  WHERE the cut sits is one scalar (the dual
    # potential of the entropic projection onto {at most K in class c};
    # verified to 5e-11 that that projection is exactly
    # sigmoid(logit(p_ic) - f)).  Moving it is free: a per-class bias shift
    # already accounts for ~30% of the incumbent penalty's displacement
    # energy and costs +0.0003 AP.  WHETHER the cut is resolved -- whether
    # the samples straddling rank K are separated -- is a representation
    # property, and it is the only part a loss can earn.
    #
    #   m_i     = logit(p_ic) = z_ic - logsumexp_{j!=c} z_ij
    #             (strictly monotone in p_ic, so ranking by m == the
    #              budget allocator's own ranking)
    #   theta   = (m_(K) + m_(K+1)) / 2          detached, per epoch
    #   s       = MAD_i(m_i)                     detached, per epoch
    #   y_i     = +1 if rank(i) <= K else -1
    #   L_cut   = (1/n_act) sum_i relu(gamma - y_i (m_i - theta)/s)
    #
    # Shift-invariant by construction: adding a constant to every m_i moves
    # theta by the same constant, so inflating a competitor class -- the
    # incumbent's free escape route -- is exactly a null direction.
    #
    #   cut_loss   off   | hinge (the above) | otce (Asano-style control:
    #                      CE onto the budget pseudo-label, which in this
    #                      single-cap setting IS the entropic-OT target)
    #   cut_gamma  margin demanded at the cut, in MAD units
    #   cut_scope  global | both (adds the per-group caps)
    cut_loss = str(hp.get("cut_loss", "off")).lower()
    if cut_loss not in ("off", "hinge", "otce"):
        raise ValueError(f"cut_loss must be off|hinge|otce, got {cut_loss!r}")
    cut_gamma = float(hp.get("cut_gamma", 1.0))
    cut_weight = float(hp.get("cut_weight", 1.0))
    cut_scope = str(hp.get("cut_scope", "global")).lower()
    if cut_scope not in ("global", "both"):
        raise ValueError(f"cut_scope must be global|both, got {cut_scope!r}")
    # Count what verification actually counts.  The incumbent penalty
    # constrains sum_i p_ic, whose gradient weight p(1-p) is a function of
    # confidence, not of the decision; satisfaction is checked on argmax.
    # "sigmoid" swaps the counted quantity for sigmoid(mtilde_i / tau) with
    # mtilde_i = z_ic - max_{j!=c} z_ij, the smoothed indicator of the
    # predicate that is actually verified.
    soft_count_mode = str(hp.get("soft_count_mode", "prob")).lower()
    if soft_count_mode not in ("prob", "sigmoid"):
        raise ValueError(f"soft_count_mode must be prob|sigmoid, got {soft_count_mode!r}")
    count_tau = float(hp.get("count_tau", 0.25))

    criterion_ce = make_ce_criterion(config, inputs.y_train, num_classes, device)
    lr_constraint = hp.get("lr_constraint", 1e-5)
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
        alpha_kl=alpha_kl,
        penalty_mode=hp.get("penalty_mode", "both"),
    ).to(device)

    constrained_classes = [c for c in range(num_classes) if global_con[c] < UNLIMITED]
    init_g = hp.get("lambda_global", 0.01)
    init_l = hp.get("lambda_local", 0.01)
    for c in constrained_classes:
        criterion_constraint.set_lambda_per_class(c, init_g, scope="global")
    for gid, bounds in local_con.items():
        for c in constrained_classes:
            if bounds[c] < UNLIMITED:
                criterion_constraint.set_lambda_per_class(c, init_l, scope="local", group_id=gid)

    log.info("Hybrid mode=%s | beta=%.3f alpha_kl=%.3f",
             hybrid_mode, fior_beta, alpha_kl)

    # KL anchor: cache warmup logits if alpha_kl > 0.
    warmup_logits_cache = None
    if alpha_kl > 0:
        from src.utils.inference import chunked_forward
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            warmup_logits_cache = chunked_forward(model, X_test).float().detach()
        log.info("Cached warmup logits for KL anchor: shape=%s", warmup_logits_cache.shape)

    rho_target = hp.get("rho_target", 100.0)
    initial_rho = hp.get("initial_rho", 0.5)
    rho_step = (rho_target - initial_rho) / max(constraint_epochs, 1)
    rho_frozen = False

    enable_ce_skip = bool(hp.get("enable_ce_skip", True))
    ce_skip_counter = 0
    skip_ce = False

    satisfaction_epoch = None
    stable_count = 0
    best_sat_state = None
    best_sat_epoch = None
    min_excess_state = None
    min_excess_epoch = None
    min_total_excess = float("inf")
    training_start = time.time()

    write_csv_header(csv_log_path, num_classes, local_con)

    for epoch in range(warmup_epochs, total_epochs):
        # ---- CE pass (skipped after saturation) ----
        model.train()
        for pg in optimizer.param_groups:
            pg["lr"] = lr_constraint
        epoch_ce = 0.0
        num_batches = max(len(train_loader), 1)
        train_correct, train_total = 0, 0
        if skip_ce:
            num_batches = 1
        for batch_X, batch_y in (train_loader if not skip_ce else []):
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
        cached_train_acc = train_correct / train_total if train_total > 0 else 1.0
        if enable_ce_skip and not skip_ce:
            if cached_train_acc >= 0.995:
                ce_skip_counter += 1
                if ce_skip_counter >= 2:
                    skip_ce = True
                    log.info("Epoch %d: CE saturated (acc=%.4f), disabling CE",
                             epoch + 1, cached_train_acc)
            else:
                ce_skip_counter = 0

        # ---- Transductive pass 1: aggregate soft + hard counts (no_grad, eval) ----
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        chunk_size = hp.get("constraint_chunk_size", CONSTRAINT_CHUNK_SIZE)
        n_test = len(X_test)
        n_chunks = (n_test + chunk_size - 1) // chunk_size

        with torch.no_grad():
            total_global_soft = torch.zeros(num_classes, device=device)
            total_global_hard = torch.zeros(num_classes, device=device)
            total_local_soft = {gid: torch.zeros(num_classes, device=device)
                                for gid in criterion_constraint.local_groups}
            total_local_hard = {gid: torch.zeros(num_classes, device=device)
                                for gid in criterion_constraint.local_groups}
            # Per-sample margins for the cut objective (detached; the geometry
            # theta/scale/sign is fixed once per epoch from these).
            pool_margin = {c: torch.zeros(n_test, device=device)
                           for c in constrained_classes} if cut_loss != "off" else {}
            pool_alt = {c: torch.zeros(n_test, dtype=torch.long, device=device)
                        for c in constrained_classes} if cut_loss == "otce" else {}
            for ci in range(n_chunks):
                start = ci * chunk_size
                end = min(start + chunk_size, n_test)
                chunk_logits = model(X_test[start:end])
                chunk_logits_f1 = chunk_logits.float()
                chunk_proba = F.softmax(chunk_logits, dim=1)
                chunk_preds = chunk_logits.argmax(dim=1)
                chunk_count = chunk_proba
                if soft_count_mode == "sigmoid":
                    chunk_count = chunk_proba.clone()
                    for c in constrained_classes:
                        chunk_count[:, c] = torch.sigmoid(
                            argmax_margin(chunk_logits_f1, c) / count_tau)
                for c in pool_margin:
                    pool_margin[c][start:end] = class_margin(chunk_logits_f1, c)
                for c in pool_alt:
                    other = chunk_logits_f1.clone()
                    other[:, c] = float("-inf")
                    pool_alt[c][start:end] = other.argmax(dim=1)
                total_global_soft += chunk_count.sum(dim=0)
                total_global_hard += torch.bincount(
                    chunk_preds, minlength=num_classes).float()
                chunk_gids = group_ids[start:end]
                for gid in total_local_soft:
                    mask = (chunk_gids == gid)
                    if mask.any():
                        total_local_soft[gid] += chunk_count[mask].sum(dim=0)
                        total_local_hard[gid] += torch.bincount(
                            chunk_preds[mask], minlength=num_classes).float()

            # ---- cut geometry, one plan per (scope, class) cap ----
            cut_plans = []
            if cut_loss != "off":
                all_idx = torch.arange(n_test, device=device)
                for c in constrained_classes:
                    K_c = criterion_constraint.global_constraints[c].item()
                    if K_c < UNLIMITED:
                        p = build_cut_plan(pool_margin[c], all_idx,
                                           int(round(K_c)), cut_gamma)
                        if p is not None:
                            p["cls"] = c; p["scope"] = "global"
                            cut_plans.append(p)
                    if cut_scope != "both":
                        continue
                    for gid, bname in criterion_constraint.local_groups.items():
                        lc = getattr(criterion_constraint, bname)
                        if c >= len(lc) or lc[c] >= UNLIMITED:
                            continue
                        gmask = (group_ids == gid)
                        gidx = all_idx[gmask]
                        p = build_cut_plan(pool_margin[c][gmask], gidx,
                                           int(round(lc[c].item())), cut_gamma)
                        if p is not None:
                            p["cls"] = c; p["scope"] = f"local{gid}"
                            cut_plans.append(p)
                # scatter sign / target into pool-length vectors for chunked use
                for p in cut_plans:
                    sgn = torch.zeros(n_test, device=device)
                    sgn[p["idx"]] = p["sign"]
                    p["sign_full"] = sgn
                    if cut_loss == "otce":
                        tgt = torch.full((n_test,), -1, dtype=torch.long, device=device)
                        c = p["cls"]
                        tgt[p["idx"]] = torch.where(
                            p["sign"] > 0,
                            torch.full_like(p["idx"], c),
                            pool_alt[c][p["idx"]])
                        p["target_full"] = tgt
                        p["n_part"] = int(p["idx"].numel())

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
        if snapshot_is_sat or snapshot_total_excess < min_total_excess:
            snapshot_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # ---- Transductive pass 2: chunked backward ----
        # Compute info values for logging (no grad).
        loss_global_val = criterion_constraint.compute_global_from_counts(total_global_soft).item()
        loss_local_val = criterion_constraint.compute_local_from_counts(total_local_soft).item()
        bounded_total = loss_global_val + loss_local_val
        # Undershoot hinge contribution (info only; gradient computed per-chunk below).
        pen_total = 0.0
        if hybrid_mode == "undershoot_hinge":
            for c in constrained_classes:
                K = criterion_constraint.global_constraints[c].item()
                if K <= 0:
                    continue
                lam = criterion_constraint.get_lambda_per_class(c, scope="global")
                pen_total += lam * fior_beta * max(0.0, (K - total_global_soft[c].item()) / K)
            for gid_s, bname_s in criterion_constraint.local_groups.items():
                lc_s = getattr(criterion_constraint, bname_s)
                for c in constrained_classes:
                    if c < len(lc_s) and lc_s[c] < UNLIMITED:
                        K = lc_s[c].item()
                        if K <= 0:
                            continue
                        lam = criterion_constraint.get_lambda_per_class(
                            c, scope="local", group_id=gid_s)
                        pen_total += lam * fior_beta * max(
                            0.0, (K - total_local_soft[gid_s][c].item()) / K)
        total_constraint = bounded_total + pen_total

        has_constraint = total_constraint > 0
        has_kl = alpha_kl > 0 and warmup_logits_cache is not None
        has_cut = cut_loss != "off" and cut_weight > 0 and len(cut_plans) > 0
        loss_cut_val = 0.0
        loss_kl_val = 0.0
        if has_constraint or has_kl or has_cut:
            for ci in range(n_chunks):
                start = ci * chunk_size
                end = min(start + chunk_size, n_test)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    chunk_logits = model(X_test[start:end])
                chunk_logits_f = chunk_logits.float()
                chunk_proba = F.softmax(chunk_logits_f, dim=1)
                chunk_count = chunk_proba
                if soft_count_mode == "sigmoid":
                    chunk_count = chunk_proba.clone()
                    for c in constrained_classes:
                        chunk_count[:, c] = torch.sigmoid(
                            argmax_margin(chunk_logits_f, c) / count_tau)
                chunk_loss = torch.tensor(0.0, device=device)
                # Build chunked soft estimates (same g_soft trick as TraLO: each
                # chunk routes gradient only through its own samples, but the
                # value plugged into the penalty is the TOTAL soft count using
                # this chunk's grad-attached partial.)
                chunk_global = chunk_count.sum(dim=0)
                chunk_gids = group_ids[start:end]
                chunk_local_soft = {}
                for gid in criterion_constraint.local_groups:
                    mask = (chunk_gids == gid)
                    if mask.any():
                        chunk_local_soft[gid] = chunk_count[mask].sum(dim=0)
                    else:
                        chunk_local_soft[gid] = torch.zeros(num_classes, device=device)
                g_soft = (total_global_soft.detach()
                          - chunk_count.sum(dim=0).detach() + chunk_global)
                l_soft = {}
                for gid in total_local_soft:
                    l_soft[gid] = (total_local_soft[gid].detach()
                                   - chunk_local_soft[gid].detach()
                                   + chunk_local_soft[gid])
                # ---- Bounded TraLO term ----
                lg = criterion_constraint.compute_global_from_counts(g_soft)
                ll = criterion_constraint.compute_local_from_counts(l_soft)
                chunk_loss = chunk_loss + (lg + ll) / n_chunks
                # ---- Undershoot hinge: lambda_T_c * beta * relu(K - soft)/K ----
                if hybrid_mode == "undershoot_hinge":
                    for c in constrained_classes:
                        K_c = criterion_constraint.global_constraints[c].item()
                        if K_c <= 0:
                            continue
                        lam = criterion_constraint.get_lambda_per_class(c, scope="global")
                        if lam <= 0 or fior_beta <= 0:
                            continue
                        chunk_loss = chunk_loss + (
                            lam * fior_beta * F.relu(K_c - g_soft[c]) / K_c / n_chunks)
                    for gid_k, bname_k in criterion_constraint.local_groups.items():
                        lc_k = getattr(criterion_constraint, bname_k)
                        for c in constrained_classes:
                            if c < len(lc_k) and lc_k[c] < UNLIMITED:
                                K_c = lc_k[c].item()
                                if K_c <= 0:
                                    continue
                                lam = criterion_constraint.get_lambda_per_class(
                                    c, scope="local", group_id=gid_k)
                                if lam <= 0 or fior_beta <= 0:
                                    continue
                                chunk_loss = chunk_loss + (
                                    lam * fior_beta * F.relu(K_c - l_soft[gid_k][c])
                                    / K_c / n_chunks)
                # ---- Cut objective -------------------------------------
                # A genuine per-sample sum: each chunk contributes only its own
                # samples, so it is NOT divided by n_chunks (unlike the count
                # penalty, which every chunk recomputes in full).
                if has_cut:
                    for p in cut_plans:
                        sgn = p["sign_full"][start:end]
                        part = sgn != 0
                        if not bool(part.any()):
                            continue
                        if cut_loss == "hinge":
                            m_chunk = class_margin(chunk_logits_f, p["cls"])
                            u = (m_chunk - p["theta"]) / p["scale"]
                            h = F.relu(cut_gamma - sgn * u) * part
                            term = cut_weight * h.sum() / max(p["n_act"], 1)
                        else:   # otce: CE onto the budget pseudo-label
                            tgt = p["target_full"][start:end]
                            logp = F.log_softmax(chunk_logits_f, dim=1)
                            ce = -logp.gather(1, tgt.clamp_min(0).unsqueeze(1)).squeeze(1)
                            term = cut_weight * (ce * part).sum() / max(p["n_part"], 1)
                        chunk_loss = chunk_loss + term
                        loss_cut_val += float(term.item())
                # ---- KL anchor against warmup distribution ----
                if has_kl:
                    log_p_cur = F.log_softmax(chunk_logits_f, dim=1)
                    p_cur = chunk_proba
                    log_p_warm = F.log_softmax(warmup_logits_cache[start:end], dim=1)
                    kl_chunk = (p_cur * (log_p_cur - log_p_warm)).sum(dim=1).mean()
                    chunk_loss = chunk_loss + alpha_kl * kl_chunk / n_chunks
                    loss_kl_val += kl_chunk.item() / n_chunks
                if scaler:
                    scaler.scale(chunk_loss).backward()
                else:
                    chunk_loss.backward()

        did_backward = has_constraint or has_kl or has_cut
        if scaler and did_backward:
            try:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if grad_norm > 0:
                    scaler.step(optimizer)
                scaler.update()
            except (AssertionError, RuntimeError):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if grad_norm > 0:
                    optimizer.step()
        elif not scaler and did_backward:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if grad_norm > 0:
                optimizer.step()

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
        ratchet_gate = (satisfaction_epoch is None) or disable_freeze_on_satisfy
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
            if not rho_frozen and not disable_freeze_on_satisfy:
                rho_frozen = True
                log.info("First satisfied at epoch %d, freezing rho=%.3f",
                         epoch + 1, criterion_constraint.get_rho())
            elif disable_freeze_on_satisfy:
                log.info("First satisfied at epoch %d, NOT freezing (ablation)",
                         epoch + 1)
            # Clear Adam state from the descent phase (hybrid_v2 diagnosis).
            # Adam accumulated "decrease soft_4" momentum during E51-Esat
            # while bounded penalty was pushing soft down. Post-sat, the
            # hinge gradient is too small to overcome that residual momentum
            # on its own, so soft keeps drifting down. Fresh m/v buffers let
            # the post-sat penalty actually steer the model.
            if reset_optimizer_at_sat or post_sat_optimizer != "adam":
                if post_sat_optimizer == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr_constraint)
                    log.info("Switched to SGD (no momentum) at sat E%d", epoch + 1)
                else:
                    optimizer = make_optimizer(model.parameters(), lr_constraint, device)
                    log.info("Reset Adam state at sat E%d", epoch + 1)
        if not rho_frozen:
            criterion_constraint.increment_rho(rho_step)

        # ---- cut diagnostics sidecar (pre-registered kill checks) ----
        # n_act_frac  must stay in [0.05, 0.25]; if it collapses toward 0 while
        #             margin_std inflates, the hinge was discharged by blowing
        #             up the score scale instead of resolving the cut.
        # margin_std  the leak detector for that failure mode.
        if cut_loss != "off" and cut_plans:
            import csv as _csv
            diag = str(csv_log_path).replace(".csv", "") + "_cut_diagnostics.csv"
            new = not os.path.exists(diag)
            with open(diag, "a", newline="") as fh:
                w = _csv.writer(fh)
                if new:
                    w.writerow(["epoch", "scope", "cls", "K", "n", "theta", "scale",
                                "n_act", "n_act_frac", "n_keep_act", "margin_std",
                                "loss_cut", "hard_count"])
                for p in cut_plans:
                    mp = pool_margin[p["cls"]]
                    w.writerow([epoch + 1, p["scope"], p["cls"], p["K"], p["n"],
                                float(p["theta"].item()), float(p["scale"].item()),
                                p["n_act"], p["n_act"] / max(p["n"], 1),
                                p["n_keep_act"], float(mp.std().item()),
                                loss_cut_val,
                                int(total_global_hard[p["cls"]].item())])

        if stable_count >= stable_count_threshold:
            log.info("Converged: constraints stable for %d epochs", stable_count)
            break

        if (epoch + 1) % 5 == 0 or is_satisfied or epoch == warmup_epochs:
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
            lam_T_mean = (sum(criterion_constraint.lambda_global_per_class.values())
                          / max(1, len(criterion_constraint.lambda_global_per_class)))
            log.info("Epoch %d [%s] ce=%.4f bounded=%.4f pen=%.4f "
                     "lam_T=%.3f rho=%.3f acc=%.4f stable=%d g_%s l_%s",
                     epoch + 1, mode_tag, avg_ce, bounded_total, pen_total,
                     lam_T_mean,
                     criterion_constraint.get_rho(), train_acc, stable_count,
                     "OK" if global_satisfied else "VIOL",
                     "OK" if local_satisfied else "VIOL")
            log_progress_to_csv(
                csv_log_path, epoch, avg_ce, train_acc,
                loss_global_val, loss_local_val,
                g_counts, l_counts, g_soft_d, l_soft_d,
                lam_T_mean, 0.0,
                global_con, global_satisfied, local_satisfied,
                kl_loss=0.0, local_constraints=local_con)
        model.train()

    elapsed = time.time() - training_start
    log.info("Training complete: %.1fs, satisfaction epoch: %s",
             elapsed, satisfaction_epoch or "N/A")

    # ---- Final eval + checkpoint restore (mirror TraLO) ----
    model.eval()
    g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
        model, X_test, group_ids, num_classes=num_classes)
    final_violates = False
    for c in range(num_classes):
        if global_con[c] < UNLIMITED and g_counts.get(c, 0) > int(global_con[c]):
            final_violates = True
            break
    if not final_violates and local_con:
        for gid, bounds in local_con.items():
            for c in range(num_classes):
                if bounds[c] < UNLIMITED and l_counts.get(gid, {}).get(c, 0) > int(bounds[c]):
                    final_violates = True
                    break
            if final_violates:
                break
    final_total_excess = 0.0
    for c in range(num_classes):
        if global_con[c] < UNLIMITED:
            final_total_excess += max(0, g_counts.get(c, 0) - int(global_con[c]))
    if local_con:
        for gid, bounds in local_con.items():
            for c in range(num_classes):
                if bounds[c] < UNLIMITED:
                    final_total_excess += max(0, l_counts.get(gid, {}).get(c, 0) - int(bounds[c]))

    restored_from_epoch = None
    restore_kind = None
    if best_sat_state is not None and final_violates:
        log.info("Restoring best-satisfied checkpoint from epoch %d", best_sat_epoch)
        model.load_state_dict({k: v.to(device) for k, v in best_sat_state.items()})
        restored_from_epoch = best_sat_epoch
        restore_kind = "fully_satisfied"
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            model, X_test, group_ids, num_classes=num_classes)
    elif min_excess_state is not None and final_total_excess > min_total_excess:
        log.info("Restoring lowest-excess checkpoint from epoch %d (excess=%d)",
                 min_excess_epoch, int(min_total_excess))
        model.load_state_dict({k: v.to(device) for k, v in min_excess_state.items()})
        restored_from_epoch = min_excess_epoch
        restore_kind = "min_excess"
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            model, X_test, group_ids, num_classes=num_classes)

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
            "hybrid_mode": hybrid_mode,
        },
    )
