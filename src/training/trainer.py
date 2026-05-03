# Constraint-aware trainer: CE warmup phase then constraint optimization with lambda toggle.
# Two-pass constraint computation for memory efficiency on high-resolution images.
# Includes AMP, fused Adam, monotone lambda ratchet + first-satisfaction freeze.

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.losses import MulticlassTransductiveLoss
from src.pipeline.warmup import make_ce_criterion, make_dataloader, make_optimizer
from src.training.logging import write_csv_header, log_progress_to_csv
from src.training.metrics import compute_prediction_statistics
from src.utils.error_handler import logger
from src.utils.inference import chunked_forward
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)

CONSTRAINT_CHUNK_SIZE = 256


class ConstraintTrainer:

    def __init__(self, config: Dict[str, Any], experiment_path: str, device: torch.device,
                 num_classes: int = 7):
        self.config = config
        self.hyperparams = config['hyperparams']
        self.experiment_path = Path(experiment_path)
        self.device = device
        self.num_classes = num_classes
        self.csv_log_path = self.experiment_path / 'training_log.csv'
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion_ce = nn.CrossEntropyLoss()
        self.from_cache = False
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False  # Disabled: Blackwell sm_120 VBIOS temp threshold bug causes crashes with autotuning
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            gpu_arch = torch.cuda.get_device_capability(0)[0]
            use_bf16 = gpu_arch >= 8 and torch.cuda.is_bf16_supported()
            if use_bf16:
                self.amp_dtype = torch.bfloat16
                self.scaler = None
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler('cuda')
            log.info("AMP enabled: dtype=%s (gpu_arch=%d)", self.amp_dtype, gpu_arch)
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

    def _cache_warmup_logits(self, X_test: torch.Tensor) -> torch.Tensor:
        """Cache RAW warmup logits so that KL anchor can be applied
        symmetrically to BOTH current and warmup distributions inside the KL
        term. Previously we cached softmax(logits/T) which softened only the
        warmup side, turning KL(p_current || p_warmup) into a pull toward
        uniform when T>1. AUDIT C2.
        """
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            warmup_logits = chunked_forward(self.model, X_test).float()
        log.info("Cached warmup logits: shape=%s", warmup_logits.shape)
        return warmup_logits.detach()

    @logger()
    def train_constraints(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          X_test: torch.Tensor, groups_test: np.ndarray,
                          global_con: list, local_con: Dict[int, list],
                          actual_warmup_epochs: int = 50) -> nn.Module:
        hp = self.hyperparams
        warmup_epochs = actual_warmup_epochs
        total_epochs = warmup_epochs + hp.get('constraint_epochs', 300)
        lambda_step = hp['lambda_step']
        self.criterion_ce = make_ce_criterion(self.config, y_train, self.num_classes, self.device)
        self.optimizer = make_optimizer(self.model.parameters(), hp.get('lr_constraint', 1e-5), self.device)
        log.info("Reset optimizer for constraint phase (lr=%.2e)", hp.get('lr_constraint', 1e-5))
        lr_constraint = hp.get('lr_constraint', 1e-5)
        train_loader = make_dataloader(X_train, y_train, hp['batch_size'])
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test).to(self.device)
        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con, local_constraints=local_con,
            num_classes=self.num_classes,
            initial_rho=hp.get('initial_rho', 0.5), alpha_kl=hp.get('alpha_kl', 0.0),
        ).to(self.device)
        log.info("Using FULL test set (%d samples) for constraint gradient", len(X_test))
        alpha_kl = hp.get('alpha_kl', 0.0)
        warmup_logits_cache = None
        if alpha_kl > 0:
            warmup_logits_cache = self._cache_warmup_logits(X_test)
        satisfaction_epoch = None
        stable_count = 0
        training_start = time.time()
        constrained_classes = [c for c in range(self.num_classes) if global_con[c] < UNLIMITED]
        rho_frozen = False
        # Per-class lambda init.
        init_g = hp.get('lambda_global', 0.01)
        init_l = hp.get('lambda_local', 0.01)
        for c in constrained_classes:
            criterion_constraint.set_lambda_per_class(c, init_g, scope='global')
        for gid, bounds in local_con.items():
            for c in constrained_classes:
                if bounds[c] < UNLIMITED:
                    criterion_constraint.set_lambda_per_class(c, init_l, scope='local', group_id=gid)
        log.info("Per-class lambdas: %d global + %d local",
                 len(criterion_constraint.lambda_global_per_class),
                 len(criterion_constraint.lambda_local_per_key))
        # Linear rho step
        constraint_epochs = hp.get('constraint_epochs', 300)
        rho_target = hp.get('rho_target', 100.0)
        initial_rho = hp.get('initial_rho', 0.5)
        rho_step = (rho_target - initial_rho) / max(constraint_epochs, 1)
        log.info("Constraint training: epochs %d to %d", warmup_epochs + 1, total_epochs)
        write_csv_header(str(self.csv_log_path), self.num_classes, local_con)

        for epoch in range(warmup_epochs, total_epochs):
            self.model.train()
            current_lr = lr_constraint
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_constraint
            epoch_ce = 0.0
            num_batches = len(train_loader)
            train_correct, train_total = 0, 0
            n_ce_batches = len(train_loader)
            for bi, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    logits_ce = self.model(batch_X)
                    loss_ce = self.criterion_ce(logits_ce, batch_y)
                if self.scaler:
                    self.scaler.scale(loss_ce).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_ce.backward()
                    self.optimizer.step()
                epoch_ce += loss_ce.item()
                with torch.no_grad():
                    train_correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
                    train_total += batch_y.size(0)
            cached_train_acc = train_correct / train_total if train_total > 0 else 1.0

            # Pass A (bookkeeping) and Pass B (constraint gradient) both run on
            # X_test in transductive mode -- switch to eval() so dropout is OFF
            # and BatchNorm running stats are NOT polluted by test inputs.
            # Train mode is restored at the bottom of the epoch (line ~709).
            # AUDIT C1.
            self.model.eval()
            self.optimizer.zero_grad(set_to_none=True)
            chunk_size = hp.get('constraint_chunk_size', CONSTRAINT_CHUNK_SIZE)
            n_test = len(X_test)
            n_chunks = (n_test + chunk_size - 1) // chunk_size

            with torch.no_grad():
                total_global_soft = torch.zeros(self.num_classes, device=self.device)
                total_global_hard = torch.zeros(self.num_classes, device=self.device)
                total_local_soft = {gid: torch.zeros(self.num_classes, device=self.device)
                                    for gid in criterion_constraint.local_groups}
                total_local_hard = {gid: torch.zeros(self.num_classes, device=self.device)
                                    for gid in criterion_constraint.local_groups}
                for ci in range(n_chunks):
                    start = ci * chunk_size
                    end = min(start + chunk_size, n_test)
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                        chunk_logits = self.model(X_test[start:end])
                    chunk_proba = F.softmax(chunk_logits.float(), dim=1)
                    chunk_preds = chunk_logits.argmax(dim=1)
                    total_global_soft += chunk_proba.sum(dim=0)
                    total_global_hard += torch.bincount(
                        chunk_preds, minlength=self.num_classes).float()
                    chunk_gids = group_ids[start:end]
                    for gid in total_local_soft:
                        mask = (chunk_gids == gid)
                        if mask.any():
                            total_local_soft[gid] += chunk_proba[mask].sum(dim=0)
                            total_local_hard[gid] += torch.bincount(
                                chunk_preds[mask], minlength=self.num_classes).float()

            loss_global_val = criterion_constraint.compute_global_from_counts(
                total_global_soft).item()
            loss_local_val = criterion_constraint.compute_local_from_counts(
                total_local_soft).item()
            total_constraint = loss_global_val + loss_local_val

            loss_kl_val = 0.0
            has_constraint = total_constraint > 0
            has_kl = alpha_kl > 0 and warmup_logits_cache is not None
            if has_constraint or has_kl:
                for ci in range(n_chunks):
                    start = ci * chunk_size
                    end = min(start + chunk_size, n_test)
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                        chunk_logits = self.model(X_test[start:end])
                    chunk_logits_f = chunk_logits.float()
                    chunk_proba = F.softmax(chunk_logits_f, dim=1)
                    chunk_loss = torch.tensor(0.0, device=self.device)
                    if has_constraint:
                        chunk_global = chunk_proba.sum(dim=0)
                        chunk_local = {}
                        chunk_gids = group_ids[start:end]
                        for gid in criterion_constraint.local_groups:
                            mask = (chunk_gids == gid)
                            if mask.any():
                                chunk_local[gid] = chunk_proba[mask].sum(dim=0)
                            else:
                                chunk_local[gid] = torch.zeros(
                                    self.num_classes, device=self.device)
                        g_soft = total_global_soft.detach() - chunk_proba.sum(dim=0).detach() + chunk_global
                        l_soft = {}
                        for gid in total_local_soft:
                            l_soft[gid] = total_local_soft[gid].detach() - chunk_local[gid].detach() + chunk_local[gid]
                        lg = criterion_constraint.compute_global_from_counts(g_soft)
                        ll = criterion_constraint.compute_local_from_counts(l_soft)
                        chunk_loss = chunk_loss + (lg + ll) / n_chunks
                    if has_kl:
                        # KL(p_cur || p_warm), anchor current predictions to warmup distribution.
                        log_p_cur = F.log_softmax(chunk_logits_f, dim=1)
                        p_cur = F.softmax(chunk_logits_f, dim=1)
                        log_p_warm = F.log_softmax(warmup_logits_cache[start:end], dim=1)
                        kl_chunk = (p_cur * (log_p_cur - log_p_warm)).sum(dim=1).mean()
                        chunk_loss = chunk_loss + alpha_kl * kl_chunk / n_chunks
                        loss_kl_val += kl_chunk.item() / n_chunks
                    if self.scaler:
                        self.scaler.scale(chunk_loss).backward()
                    else:
                        chunk_loss.backward()

            # Unscale + step for constraint gradients. The scaler is needed
            # here because the model forward runs in FP16 (autocast) and
            # small constraint gradients near the satisfaction boundary can
            # underflow without scaling. When constraint loss is exactly
            # zero (satisfied epoch), the scaler may crash with "No inf
            # checks" — we catch that and fall back to a plain step.
            did_backward = has_constraint or has_kl
            if self.scaler and did_backward:
                try:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)
                    if grad_norm > 0:
                        self.scaler.step(self.optimizer)
                    self.scaler.update()
                except (AssertionError, RuntimeError):
                    # Scaler state issue — plain step fallback.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)
                    if grad_norm > 0:
                        self.optimizer.step()
            elif not self.scaler and did_backward:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                if grad_norm > 0:
                    self.optimizer.step()

            loss_global = torch.tensor(loss_global_val)
            loss_local = torch.tensor(loss_local_val)
            loss_kl = torch.tensor(loss_kl_val)
            avg_ce = epoch_ce / num_batches
            avg_global = loss_global.detach().item()
            avg_local = loss_local.detach().item()
            avg_kl = loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl

            global_satisfied = True
            for c in range(self.num_classes):
                if c < len(criterion_constraint.global_constraints) and \
                        criterion_constraint.global_constraints[c] < UNLIMITED:
                    if total_global_hard[c].item() > criterion_constraint.global_constraints[c].item():
                        global_satisfied = False
                        break
            local_satisfied = True
            for gid, buffer_name in criterion_constraint.local_groups.items():
                lc = getattr(criterion_constraint, buffer_name)
                for c in range(self.num_classes):
                    if c < len(lc) and lc[c] < UNLIMITED:
                        if total_local_hard[gid][c].item() > lc[c].item():
                            local_satisfied = False
                            break
                if not local_satisfied:
                    break

            is_satisfied = global_satisfied and local_satisfied
            if is_satisfied:
                stable_count += 1
            else:
                stable_count = 0

            # Per-class lambda ratchet: each constrained class/group gets its own
            # lambda that increments only when THAT specific constraint is violated.
            # Lambdas + rho freeze on first satisfaction.
            for c in constrained_classes:
                if c < len(criterion_constraint.global_constraints) and \
                        criterion_constraint.global_constraints[c] < UNLIMITED:
                    hard_c = total_global_hard[c].item()
                    limit_c = criterion_constraint.global_constraints[c].item()
                    if hard_c > limit_c and satisfaction_epoch is None:
                        old = criterion_constraint.get_lambda_per_class(c, scope='global')
                        criterion_constraint.set_lambda_per_class(c, old + lambda_step, scope='global')
            for gid, buffer_name in criterion_constraint.local_groups.items():
                lc = getattr(criterion_constraint, buffer_name)
                for c in constrained_classes:
                    if c < len(lc) and lc[c] < UNLIMITED:
                        hard_c = total_local_hard[gid][c].item()
                        limit_c = lc[c].item()
                        if hard_c > limit_c and satisfaction_epoch is None:
                            old = criterion_constraint.get_lambda_per_class(c, scope='local', group_id=gid)
                            criterion_constraint.set_lambda_per_class(c, old + lambda_step, scope='local', group_id=gid)
            if is_satisfied and satisfaction_epoch is None:
                satisfaction_epoch = epoch + 1
                if not rho_frozen:
                    rho_frozen = True
                    log.info("First satisfied at epoch %d, freezing rho=%.3f and per-class lambdas",
                             epoch + 1, criterion_constraint.get_rho())

            # Linear rho increment
            if not rho_frozen:
                criterion_constraint.increment_rho(rho_step)

            # Convergence check
            if stable_count >= 5:
                log.info("Converged: constraints stable for %d epochs (lambdas frozen)", stable_count)
                break

            if (epoch + 1) % 5 == 0 or is_satisfied or epoch == warmup_epochs:
                train_acc = cached_train_acc
                g_counts = {c: int(total_global_hard[c].item()) for c in range(self.num_classes)}
                l_counts = {}
                for gid in total_local_hard:
                    l_counts[gid] = {c: int(total_local_hard[gid][c].item()) for c in range(self.num_classes)}
                g_soft = {c: total_global_soft[c].item() for c in range(self.num_classes)}
                l_soft = {}
                for gid in total_local_soft:
                    l_soft[gid] = {c: total_local_soft[gid][c].item() for c in range(self.num_classes)}
                mode = "Satisfied" if is_satisfied else "Constraint"
                kl_str = f" kl={avg_kl:.4f}" if alpha_kl > 0 else ""
                lam_g_mean = (sum(criterion_constraint.lambda_global_per_class.values())
                              / max(1, len(criterion_constraint.lambda_global_per_class)))
                lam_l_mean = (sum(criterion_constraint.lambda_local_per_key.values())
                              / max(1, len(criterion_constraint.lambda_local_per_key)))
                log.info("Epoch %d [%s] lr=%.2e ce=%.4f g=%.4f l=%.4f%s "
                         "lambda_mean(g=%.3f l=%.3f rho=%.3f) acc=%.4f g_%s l_%s",
                         epoch + 1, mode, current_lr, avg_ce, avg_global, avg_local, kl_str,
                         lam_g_mean, lam_l_mean,
                         criterion_constraint.get_rho(), train_acc,
                         "OK" if global_satisfied else "VIOL",
                         "OK" if local_satisfied else "VIOL")
                for c in range(self.num_classes):
                    if global_con[c] < UNLIMITED:
                        log.info("  Global class %d: pred=%d limit=%d", c, g_counts.get(c, 0), int(global_con[c]))
                for gid in sorted(l_counts.keys()):
                    group_name = f"group_{gid}"
                    for c in range(self.num_classes):
                        if local_con and gid in local_con and local_con[gid][c] < UNLIMITED:
                            log.info("  Local %s class %d: pred=%d limit=%d",
                                     group_name, c,
                                     l_counts.get(gid, {}).get(c, 0),
                                     int(local_con[gid][c]))
                log_progress_to_csv(
                    str(self.csv_log_path), epoch, avg_ce, train_acc, avg_global, avg_local,
                    g_counts, l_counts, g_soft, l_soft,
                    lam_g_mean, lam_l_mean,
                    global_con, global_satisfied, local_satisfied,
                    kl_loss=avg_kl, local_constraints=local_con)
            self.model.train()

        elapsed = time.time() - training_start
        log.info("Training complete: %.1fs, satisfaction epoch: %s",
                 elapsed, satisfaction_epoch or "N/A")
        self.model.eval()
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            self.model, X_test, group_ids, num_classes=self.num_classes)
        log.info("=== Final prediction summary ===")
        for c in range(self.num_classes):
            limit = int(global_con[c]) if global_con[c] < UNLIMITED else 'INF'
            log.info("  Global class %d: hard=%d soft=%.2f limit=%s",
                     c, g_counts.get(c, 0), g_soft.get(c, 0), limit)
        for gid in sorted(l_counts.keys()):
            group_name = f"group_{gid}"
            for c in range(self.num_classes):
                if local_con and gid in local_con and local_con[gid][c] < UNLIMITED:
                    log.info("  Local %s class %d: hard=%d soft=%.2f limit=%d",
                             group_name, c,
                             l_counts.get(gid, {}).get(c, 0),
                             l_soft.get(gid, {}).get(c, 0.0),
                             int(local_con[gid][c]))
        # Expose constraint metrics for Track 1 evaluation
        self.satisfaction_epoch = satisfaction_epoch
        self.final_soft_hard_gap = {}
        constrained = [c for c in range(self.num_classes) if global_con[c] < UNLIMITED]
        for c in constrained:
            self.final_soft_hard_gap[c] = abs(g_soft.get(c, 0) - g_counts.get(c, 0))

        return self.model
