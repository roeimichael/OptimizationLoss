# Constraint-aware trainer: CE warmup phase then constraint optimization with lambda ratchet.
# Two-pass constraint computation for memory efficiency on high-resolution images.
# Includes AMP, fused Adam, CE saturation skip, and stagnation early stopping.

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
from src.utils.inference import chunked_forward

log = logging.getLogger(__name__)

CONSTRAINT_CHUNK_SIZE = 256
UNLIMITED = 1e10


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
            torch.backends.cudnn.benchmark = True
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

    @logger()
    def setup_model(self, input_dim: int, base_model_id: str) -> None:
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
        self.optimizer = self._make_optimizer(self.model.parameters(), self.hyperparams['lr'])

    def _make_optimizer(self, params, lr):
        use_fused = torch.cuda.is_available() and hasattr(torch.optim.Adam, 'fused')
        try:
            return torch.optim.Adam(params, lr=lr, fused=use_fused)
        except Exception:
            return torch.optim.Adam(params, lr=lr)

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(X, y)
        use_workers = os.name != 'nt'
        n_workers = 2 if use_workers else 0
        return DataLoader(dataset, batch_size=self.hyperparams['batch_size'],
                          shuffle=True, num_workers=n_workers,
                          pin_memory=True,
                          persistent_workers=use_workers and n_workers > 0)

    @logger()
    def train_warmup(self, X_train: torch.Tensor, y_train: torch.Tensor, base_model_id: str) -> int:
        if self.from_cache:
            return self.hyperparams['warmup_epochs']
        if self.hyperparams.get('class_weighted_ce', False):
            class_counts = torch.bincount(y_train, minlength=self.num_classes).float()
            class_weights = (1.0 / class_counts.clamp(min=1)).to(self.device)
            class_weights = class_weights / class_weights.sum() * self.num_classes
            self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
            log.info("Using class-weighted CE: weights=%s", class_weights.cpu().numpy().round(3))
        train_loader = self._create_dataloader(X_train, y_train)
        warmup_epochs = self.hyperparams['warmup_epochs']
        log_interval = max(1, warmup_epochs // 5)
        n_batches = len(train_loader)
        log.info("Warmup: %d epochs, %d batches/epoch (batch_size=%d, samples=%d)",
                 warmup_epochs, n_batches, self.hyperparams['batch_size'], len(X_train))
        log.info("AMP: enabled=%s dtype=%s scaler=%s", self.use_amp, self.amp_dtype,
                 self.scaler is not None)
        epoch_times = []
        for epoch in range(warmup_epochs):
            epoch_start = time.time()
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = self.criterion_ce(self.model(batch_X), batch_y)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                epoch_loss += loss.item()
            epoch_elapsed = time.time() - epoch_start
            epoch_times.append(epoch_elapsed)
            if epoch < 3 or (epoch + 1) % log_interval == 0 or epoch == warmup_epochs - 1:
                avg_loss = epoch_loss / n_batches
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                log_progress_to_csv(str(self.csv_log_path), epoch, avg_loss, train_acc,
                                    num_classes=self.num_classes)
                log.info("Warmup %d/%d: loss=%.4f acc=%.4f [%.2fs/epoch]",
                         epoch + 1, warmup_epochs, avg_loss, train_acc, epoch_elapsed)
        avg_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        log.info("Warmup done: avg=%.2fs/epoch total=%.1fs", avg_epoch, sum(epoch_times))
        save_to_cache(self.model, base_model_id, self.config)
        return warmup_epochs

    def _cache_warmup_probabilities(self, X_test: torch.Tensor,
                                     kl_temperature: float = 1.0) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            warmup_logits = chunked_forward(self.model, X_test)
            warmup_proba = F.softmax(warmup_logits.float() / kl_temperature, dim=1)
        log.info("Cached warmup probabilities: shape=%s kl_temp=%.1f",
                 warmup_proba.shape, kl_temperature)
        return warmup_proba.detach()

    @logger()
    def train_constraints(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          X_test: torch.Tensor, groups_test: np.ndarray,
                          global_con: list, local_con: Dict[int, list],
                          actual_warmup_epochs: int = 50) -> nn.Module:
        hp = self.hyperparams
        warmup_epochs = actual_warmup_epochs
        total_epochs = warmup_epochs + hp.get('constraint_epochs', 350)
        lambda_step = hp['lambda_step']
        self.optimizer = self._make_optimizer(
            self.model.parameters(), hp.get('lr_constraint', 1e-5))
        log.info("Reset optimizer for constraint phase (lr=%.2e)", hp.get('lr_constraint', 1e-5))
        lr_scheduler = LearningRateScheduler(
            optimizer=self.optimizer, warmup_lr=hp.get('lr', 1e-3),
            drop_lr=hp.get('lr_constraint', 1e-5), warmup_epochs=warmup_epochs)
        train_loader = self._create_dataloader(X_train, y_train)
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test).to(self.device)
        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con, local_constraints=local_con,
            lambda_global=hp['lambda_global'], lambda_local=hp['lambda_local'],
            num_classes=self.num_classes, use_sum=hp.get('use_sum_loss', True),
            initial_rho=hp.get('initial_rho', 0.5), alpha_kl=hp.get('alpha_kl', 0.0)
        ).to(self.device)
        log.info("Using FULL test set (%d samples) for constraint gradient", len(X_test))
        alpha_kl = hp.get('alpha_kl', 0.0)
        kl_temperature = hp.get('kl_temperature', 1.0)
        warmup_proba = None
        if alpha_kl > 0:
            warmup_proba = self._cache_warmup_probabilities(X_test, kl_temperature)
        constraints_satisfied = False
        satisfaction_epoch = None
        stable_count = 0
        training_start = time.time()
        ce_skip_counter = 0
        skip_ce = False
        constrained_classes = [c for c in range(self.num_classes) if global_con[c] < 1e9]
        constrained_class = constrained_classes[0] if constrained_classes else None
        constraint_limit = int(global_con[constrained_class]) if constrained_class is not None else None
        best_hard_count = float('inf')
        stagnation_counter = 0
        STAGNATION_PATIENCE = 100
        log.info("Constraint training: epochs %d to %d", warmup_epochs + 1, total_epochs)
        write_csv_header(str(self.csv_log_path), self.num_classes, local_con)

        for epoch in range(warmup_epochs, total_epochs):
            self.model.train()
            current_lr = lr_scheduler.step(epoch)
            epoch_ce = 0.0
            num_batches = len(train_loader)
            train_correct, train_total = 0, 0
            if not skip_ce:
                for batch_X, batch_y in train_loader:
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

            self.model.train()
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
            total_constraint = (criterion_constraint.lambda_global * loss_global_val +
                                criterion_constraint.lambda_local * loss_local_val)

            loss_kl_val = 0.0
            has_constraint = total_constraint > 0
            has_kl = alpha_kl > 0 and warmup_proba is not None
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
                        chunk_loss = chunk_loss + (criterion_constraint.lambda_global * lg +
                                      criterion_constraint.lambda_local * ll) / n_chunks
                    if has_kl:
                        log_p = F.log_softmax(chunk_logits_f, dim=1)
                        p_current = chunk_proba
                        p_warmup = warmup_proba[start:end].clamp(min=1e-8)
                        kl_chunk = (p_current * (log_p - torch.log(p_warmup))).sum(dim=1).mean()
                        chunk_loss = chunk_loss + alpha_kl * kl_chunk / n_chunks
                        loss_kl_val += kl_chunk.item() / n_chunks
                    if self.scaler:
                        self.scaler.scale(chunk_loss).backward()
                    else:
                        chunk_loss.backward()

            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if grad_norm > 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
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
            if is_satisfied and not constraints_satisfied:
                constraints_satisfied = True
                satisfaction_epoch = epoch + 1
                log.info("Constraints satisfied at epoch %d, freezing lambdas (g=%.3f, l=%.3f)",
                         satisfaction_epoch, criterion_constraint.lambda_global,
                         criterion_constraint.lambda_local)
            if constraints_satisfied and stable_count >= 5:
                log.info("Converged: constraints stable for %d epochs", stable_count)
                break
            if not constraints_satisfied:
                if not global_satisfied:
                    new_g = criterion_constraint.lambda_global + lambda_step
                    criterion_constraint.set_lambda(lambda_global=new_g)
                if not local_satisfied:
                    new_l = criterion_constraint.lambda_local + lambda_step
                    criterion_constraint.set_lambda(lambda_local=new_l)
                if (epoch - warmup_epochs) % 25 == 0 and epoch > warmup_epochs:
                    criterion_constraint.update_rho(factor=1.5)

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
                if train_acc >= 0.995:
                    ce_skip_counter += 1
                    if ce_skip_counter >= 2 and not skip_ce:
                        skip_ce = True
                        log.info("CE saturated (acc>=0.995 for %d checks), skipping Phase 1",
                                 ce_skip_counter)
                else:
                    ce_skip_counter = 0
                    skip_ce = False
                if constrained_classes and not constraints_satisfied:
                    total_excess = sum(
                        max(0, g_counts.get(c, 0) - int(global_con[c]))
                        for c in constrained_classes
                    )
                    if total_excess < best_hard_count:
                        best_hard_count = total_excess
                        stagnation_counter = 0
                        log.info("  New best total excess: %d", best_hard_count)
                    else:
                        stagnation_counter += 5
                    if stagnation_counter >= STAGNATION_PATIENCE:
                        log.info("Stagnation: total excess hasn't improved for %d epochs "
                                 "(best=%d, current=%d). Stopping.",
                                 stagnation_counter, best_hard_count, total_excess)
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
