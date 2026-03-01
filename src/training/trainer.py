"""Constraint-aware trainer: warmup-until-saturation then constraint optimization."""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.models import get_model
from src.losses import MulticlassTransductiveLoss
from src.training.metrics import compute_train_accuracy, compute_prediction_statistics
from src.training.logging import log_progress_to_csv, write_csv_header
from src.training.schedulers import LearningRateScheduler
from src.training.model_cache import save_to_cache, load_from_cache
from src.utils.error_handler import logger

log = logging.getLogger(__name__)

# Max samples per forward pass to avoid GPU OOM on 224x224 images
# Override via config hyperparams: constraint_chunk_size, inference_chunk_size
INFERENCE_CHUNK_SIZE = 512  # no_grad inference (safe for 22GB GPUs)
CONSTRAINT_CHUNK_SIZE = 256  # with-gradients constraint step
UNLIMITED = 1e9  # constraint value meaning "no limit"


def _chunked_forward(model, X, chunk_size=INFERENCE_CHUNK_SIZE):
    """Forward pass in chunks to avoid GPU OOM on large batches (no_grad only)."""
    if len(X) <= chunk_size:
        return model(X)
    chunks = [model(X[i:i + chunk_size]) for i in range(0, len(X), chunk_size)]
    return torch.cat(chunks, dim=0)


class ConstraintTrainer:
    """Trains a model with CE warmup then constraint optimization with lambda ratchet."""

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

    @logger()
    def setup_model(self, input_dim: int, base_model_id: str) -> None:
        """Initialize or load model from cache."""
        self.model = load_from_cache(
            base_model_id, self.config, input_dim, self.num_classes, self.device)

        if self.model is None:
            log.info("Creating new model: %s (%d classes)", self.config['model_name'], self.num_classes)
            self.model = get_model(
                self.config['model_name'], input_dim=input_dim,
                hidden_dims=self.hyperparams.get('hidden_dims'),
                n_classes=self.num_classes, dropout=self.hyperparams['dropout'],
                pretrained=self.hyperparams.get('pretrained', False)
            ).to(self.device)
            self.from_cache = False
        else:
            log.info("Loaded cached model: %s", base_model_id)
            self.from_cache = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(X, y)
        # Multi-worker loading on Linux (server), single-worker on Windows (local dev)
        use_workers = os.name != 'nt'
        n_workers = 2 if use_workers else 0
        return DataLoader(dataset, batch_size=self.hyperparams['batch_size'],
                          shuffle=True, num_workers=n_workers,
                          pin_memory=True,
                          persistent_workers=use_workers and n_workers > 0)

    @logger()
    def train_warmup(self, X_train: torch.Tensor, y_train: torch.Tensor, base_model_id: str) -> int:
        """CE-only warmup for exactly warmup_epochs. Returns epoch count."""
        if self.from_cache:
            return self.hyperparams['warmup_epochs']

        # Class-weighted CE: inverse-frequency weighting for imbalanced data
        if self.hyperparams.get('class_weighted_ce', False):
            class_counts = torch.bincount(y_train, minlength=self.num_classes).float()
            class_weights = (1.0 / class_counts.clamp(min=1)).to(self.device)
            class_weights = class_weights / class_weights.sum() * self.num_classes  # normalize
            self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
            log.info("Using class-weighted CE: weights=%s", class_weights.cpu().numpy().round(3))

        train_loader = self._create_dataloader(X_train, y_train)
        warmup_epochs = self.hyperparams['warmup_epochs']
        log_interval = max(1, warmup_epochs // 5)

        log.info("Warmup: training for %d epochs (CE only)", warmup_epochs)

        for epoch in range(warmup_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion_ce(self.model(batch_X), batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % log_interval == 0 or epoch == warmup_epochs - 1:
                avg_loss = epoch_loss / len(train_loader)
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                log_progress_to_csv(str(self.csv_log_path), epoch, avg_loss, train_acc,
                                    num_classes=self.num_classes)
                log.info("Warmup %d/%d: loss=%.4f acc=%.4f",
                         epoch + 1, warmup_epochs, avg_loss, train_acc)

        save_to_cache(self.model, base_model_id, self.config)
        return warmup_epochs

    def _cache_warmup_probabilities(self, X_test: torch.Tensor,
                                     kl_temperature: float = 1.0) -> torch.Tensor:
        """Cache warmup softmax probabilities for KL-divergence regularization.

        kl_temperature > 1.0 softens the reference distribution, giving the
        constraint optimizer more freedom to deviate while still being anchored.
        """
        self.model.eval()
        with torch.no_grad():
            warmup_logits = _chunked_forward(self.model, X_test)
            warmup_proba = F.softmax(warmup_logits / kl_temperature, dim=1)
        log.info("Cached warmup probabilities: shape=%s kl_temp=%.1f",
                 warmup_proba.shape, kl_temperature)
        return warmup_proba.detach()

    @logger()
    def train_constraints(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          X_test: torch.Tensor, groups_test: np.ndarray,
                          global_con: list, local_con: Dict[int, list],
                          actual_warmup_epochs: int = 50) -> nn.Module:
        """Run constraint optimization phase after warmup.

        Uses FULL test set for constraint gradient (feasible at 64x64).
        Constraint loss is computed once per epoch (not per batch) to avoid
        redundant forward passes and keep training efficient.
        """
        hp = self.hyperparams
        warmup_epochs = actual_warmup_epochs
        total_epochs = warmup_epochs + hp.get('constraint_epochs', 350)
        lambda_step = hp['lambda_step']

        # Fix 4: Fresh optimizer for constraint phase (clean Adam momentum)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=hp.get('lr_constraint', 1e-5))
        log.info("Reset optimizer for constraint phase (lr=%.2e)", hp.get('lr_constraint', 1e-5))

        lr_scheduler = LearningRateScheduler(
            optimizer=self.optimizer, warmup_lr=hp.get('lr', 1e-3),
            drop_lr=hp.get('lr_constraint', 1e-5), warmup_epochs=warmup_epochs)

        train_loader = self._create_dataloader(X_train, y_train)
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test).to(self.device)

        # Single constraint loss on full test set (no proxy needed at 64x64)
        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con, local_constraints=local_con,
            lambda_global=hp['lambda_global'], lambda_local=hp['lambda_local'],
            num_classes=self.num_classes, use_sum=hp.get('use_sum_loss', True),
            initial_rho=hp.get('initial_rho', 0.5), alpha_kl=hp.get('alpha_kl', 0.0)
        ).to(self.device)

        log.info("Using FULL test set (%d samples) for constraint gradient", len(X_test))

        # KL-divergence regularization
        alpha_kl = hp.get('alpha_kl', 0.0)
        kl_temperature = hp.get('kl_temperature', 1.0)
        warmup_proba = None
        if alpha_kl > 0:
            warmup_proba = self._cache_warmup_probabilities(X_test, kl_temperature)

        # Training state
        constraints_satisfied = False
        satisfaction_epoch = None
        stable_count = 0
        training_start = time.time()

        # Fix 1: CE saturation detection — skip Phase 1 when train acc is stuck at 100%
        ce_skip_counter = 0
        skip_ce = False

        # Fix 3: Stagnation detection — early stop if hard count plateaus
        # Track distance-to-limit rather than absolute hard count, since
        # the count may start high and needs to come DOWN to the limit.
        constrained_class = None
        constraint_limit = None
        for c in range(self.num_classes):
            if global_con[c] < 1e9:
                constrained_class = c
                constraint_limit = int(global_con[c])
                break
        best_hard_count = float('inf')
        stagnation_counter = 0
        STAGNATION_PATIENCE = 100  # epochs without improvement before stopping

        log.info("Constraint training: epochs %d to %d", warmup_epochs + 1, total_epochs)

        # Write full CSV header now that we know the local constraint schema
        write_csv_header(str(self.csv_log_path), self.num_classes, local_con)

        for epoch in range(warmup_epochs, total_epochs):
            self.model.train()
            current_lr = lr_scheduler.step(epoch)

            # ── Phase 1: CE on train batches (skipped when accuracy saturated) ──
            epoch_ce = 0.0
            num_batches = len(train_loader)
            train_correct, train_total = 0, 0
            if not skip_ce:
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    self.optimizer.zero_grad()
                    logits_ce = self.model(batch_X)
                    loss_ce = self.criterion_ce(logits_ce, batch_y)
                    loss_ce.backward()
                    self.optimizer.step()
                    epoch_ce += loss_ce.item()
                    # Track accuracy during CE pass (avoids separate eval pass)
                    with torch.no_grad():
                        train_correct += (logits_ce.argmax(dim=1) == batch_y).sum().item()
                        train_total += batch_y.size(0)
            cached_train_acc = train_correct / train_total if train_total > 0 else 1.0

            # ── Phase 2: Constraint on full test set (gradient accumulation) ──
            # Two-pass approach for memory efficiency at 224x224:
            #   Pass 1 (no_grad): compute total soft counts
            #   Pass 2 (with grad): per-chunk backward with scaled loss
            self.model.train()
            self.optimizer.zero_grad()

            chunk_size = hp.get('constraint_chunk_size', CONSTRAINT_CHUNK_SIZE)
            n_test = len(X_test)
            n_chunks = (n_test + chunk_size - 1) // chunk_size

            # Pass 1: compute total soft+hard counts (no_grad) for loss value + satisfaction
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
                    chunk_logits = self.model(X_test[start:end])
                    chunk_proba = F.softmax(chunk_logits, dim=1)
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

            # Compute constraint losses from total counts (for logging + loss check)
            loss_global_val = criterion_constraint.compute_global_from_counts(
                total_global_soft).item()
            loss_local_val = criterion_constraint.compute_local_from_counts(
                total_local_soft).item()

            total_constraint = (criterion_constraint.lambda_global * loss_global_val +
                                criterion_constraint.lambda_local * loss_local_val)

            # Pass 2: per-chunk backward for constraint + KL (single forward pass)
            # Merges constraint and KL gradients into one loop to halve GPU work.
            loss_kl_val = 0.0
            has_constraint = total_constraint > 0
            has_kl = alpha_kl > 0 and warmup_proba is not None

            if has_constraint or has_kl:
                for ci in range(n_chunks):
                    start = ci * chunk_size
                    end = min(start + chunk_size, n_test)
                    chunk_logits = self.model(X_test[start:end])
                    chunk_proba = F.softmax(chunk_logits, dim=1)
                    chunk_loss = torch.tensor(0.0, device=self.device)

                    # Constraint gradient
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
                        chunk_loss = chunk_loss + (criterion_constraint.lambda_global * lg +
                                      criterion_constraint.lambda_local * ll) / n_chunks

                    # KL gradient (reuses same chunk_logits — no extra forward pass)
                    if has_kl:
                        log_p = F.log_softmax(chunk_logits, dim=1)
                        p_current = chunk_proba  # already computed above
                        p_warmup = warmup_proba[start:end].clamp(min=1e-8)
                        kl_chunk = (p_current * (log_p - torch.log(p_warmup))).sum(dim=1).mean()
                        chunk_loss = chunk_loss + alpha_kl * kl_chunk / n_chunks
                        loss_kl_val += kl_chunk.item() / n_chunks

                    chunk_loss.backward()

            # Clip and step
            has_grad = any(p.grad is not None and p.grad.norm().item() > 0
                          for p in self.model.parameters())
            if has_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Store for logging
            loss_global = torch.tensor(loss_global_val)
            loss_local = torch.tensor(loss_local_val)
            loss_kl = torch.tensor(loss_kl_val)

            avg_ce = epoch_ce / num_batches
            avg_global = loss_global.detach().item()
            avg_local = loss_local.detach().item()
            avg_kl = loss_kl.item() if isinstance(loss_kl, torch.Tensor) else loss_kl

            # ── End-of-epoch: check satisfaction using Pass 1 hard counts ──
            # No extra forward pass needed — hard counts were collected in Pass 1
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

            # First satisfaction: freeze lambdas
            if is_satisfied and not constraints_satisfied:
                constraints_satisfied = True
                satisfaction_epoch = epoch + 1
                log.info("Constraints satisfied at epoch %d, freezing lambdas (g=%.3f, l=%.3f)",
                         satisfaction_epoch, criterion_constraint.lambda_global,
                         criterion_constraint.lambda_local)

            # Early stop when stable
            if constraints_satisfied and stable_count >= 5:
                log.info("Converged: constraints stable for %d epochs", stable_count)
                break

            # Lambda ratchet (only before satisfaction)
            if not constraints_satisfied:
                if not global_satisfied:
                    new_g = criterion_constraint.lambda_global + lambda_step
                    criterion_constraint.set_lambda(lambda_global=new_g)
                if not local_satisfied:
                    new_l = criterion_constraint.lambda_local + lambda_step
                    criterion_constraint.set_lambda(lambda_local=new_l)
                # Adaptive ALM: increase rho every 25 epochs
                if (epoch - warmup_epochs) % 25 == 0 and epoch > warmup_epochs:
                    criterion_constraint.update_rho(factor=1.5)

            # Logging every 5 epochs or on state change
            if (epoch + 1) % 5 == 0 or is_satisfied or epoch == warmup_epochs:
                # Reuse cached data from Phase 1 and Pass 1 (no extra forward passes)
                train_acc = cached_train_acc
                g_counts = {c: int(total_global_hard[c].item()) for c in range(self.num_classes)}
                l_counts = {}
                for gid in total_local_hard:
                    l_counts[gid] = {c: int(total_local_hard[gid][c].item()) for c in range(self.num_classes)}
                g_soft = {c: total_global_soft[c].item() for c in range(self.num_classes)}
                l_soft = {}
                for gid in total_local_soft:
                    l_soft[gid] = {c: total_local_soft[gid][c].item() for c in range(self.num_classes)}

                # Fix 1: CE saturation detection
                if train_acc >= 0.995:
                    ce_skip_counter += 1
                    if ce_skip_counter >= 2 and not skip_ce:
                        skip_ce = True
                        log.info("CE saturated (acc>=0.995 for %d checks), skipping Phase 1",
                                 ce_skip_counter)
                else:
                    ce_skip_counter = 0
                    skip_ce = False

                # Fix 3: Stagnation detection for constrained class
                # Track whether hard count is getting closer to the constraint limit
                if constrained_class is not None and not constraints_satisfied:
                    current_hard = g_counts.get(constrained_class, 0)
                    if current_hard < best_hard_count:
                        best_hard_count = current_hard
                        stagnation_counter = 0
                        log.info("  New best hard count for class %d: %d (limit=%s)",
                                 constrained_class, best_hard_count, constraint_limit)
                    else:
                        stagnation_counter += 5  # increments by log interval (5 epochs)

                    if stagnation_counter >= STAGNATION_PATIENCE:
                        log.info("Stagnation: hard count hasn't improved for %d epochs "
                                 "(best=%d, current=%d, limit=%s). Stopping.",
                                 stagnation_counter, best_hard_count, current_hard,
                                 constraint_limit)
                        break

                mode = "Refinement" if constraints_satisfied else "Constraint"
                kl_str = f" kl={avg_kl:.4f}" if alpha_kl > 0 else ""
                log.info("Epoch %d [%s] lr=%.2e ce=%.4f g=%.4f l=%.4f%s "
                         "lambda(g=%.3f l=%.3f rho=%.3f) acc=%.4f g_%s l_%s",
                         epoch + 1, mode, current_lr, avg_ce, avg_global, avg_local, kl_str,
                         criterion_constraint.lambda_global, criterion_constraint.lambda_local,
                         criterion_constraint.get_rho(), train_acc,
                         "OK" if global_satisfied else "VIOL",
                         "OK" if local_satisfied else "VIOL")

                for c in range(self.num_classes):
                    if global_con[c] < 1e9:
                        log.info("  Global class %d: pred=%d limit=%d", c, g_counts.get(c, 0), int(global_con[c]))

                # Log per-group (sex) constraint status
                for gid in sorted(l_counts.keys()):
                    group_name = f"group_{gid}"
                    for c in range(self.num_classes):
                        if local_con and gid in local_con and local_con[gid][c] < 1e9:
                            log.info("  Local %s class %d: pred=%d limit=%d",
                                     group_name, c,
                                     l_counts.get(gid, {}).get(c, 0),
                                     int(local_con[gid][c]))

                log_progress_to_csv(
                    str(self.csv_log_path), epoch, avg_ce, train_acc, avg_global, avg_local,
                    g_counts, l_counts, g_soft, l_soft,
                    criterion_constraint.lambda_global, criterion_constraint.lambda_local,
                    global_con, global_satisfied, local_satisfied,
                    kl_loss=avg_kl, local_constraints=local_con)

            self.model.train()

        elapsed = time.time() - training_start
        log.info("Training complete: %.1fs, satisfaction epoch: %s",
                 elapsed, satisfaction_epoch or "N/A")

        # Final validation
        self.model.eval()
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            self.model, X_test, group_ids, num_classes=self.num_classes)
        log.info("=== Final prediction summary ===")
        for c in range(self.num_classes):
            limit = int(global_con[c]) if global_con[c] < 1e9 else 'INF'
            log.info("  Global class %d: hard=%d soft=%.2f limit=%s",
                     c, g_counts.get(c, 0), g_soft.get(c, 0), limit)
        for gid in sorted(l_counts.keys()):
            group_name = f"group_{gid}"
            for c in range(self.num_classes):
                if local_con and gid in local_con and local_con[gid][c] < 1e9:
                    log.info("  Local %s class %d: hard=%d soft=%.2f limit=%d",
                             group_name, c,
                             l_counts.get(gid, {}).get(c, 0),
                             l_soft.get(gid, {}).get(c, 0.0),
                             int(local_con[gid][c]))

        return self.model
