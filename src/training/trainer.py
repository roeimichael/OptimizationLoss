"""Constraint-aware trainer: warmup-until-saturation then constraint optimization."""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.models import get_model
from src.losses import MulticlassTransductiveLoss
from src.training.metrics import compute_train_accuracy, compute_prediction_statistics
from src.training.logging import log_progress_to_csv, write_csv_header
from src.training.schedulers import LearningRateScheduler
from src.training.model_cache import save_to_cache, load_from_cache
from src.utils.error_handler import logger

log = logging.getLogger(__name__)


class ConstraintTrainer:
    """Trains a model with CE warmup then constraint optimization with lambda ratchet."""

    def __init__(self, config: Dict[str, Any], experiment_path: str, device: torch.device,
                 num_classes: int = 2):
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
                hidden_dims=self.hyperparams['hidden_dims'],
                n_classes=self.num_classes, dropout=self.hyperparams['dropout']
            ).to(self.device)
            self.from_cache = False
        else:
            log.info("Loaded cached model: %s", base_model_id)
            self.from_cache = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)

    @logger()
    def train_warmup(self, X_train: torch.Tensor, y_train: torch.Tensor, base_model_id: str) -> int:
        """CE-only warmup until accuracy saturates. Returns actual epoch count."""
        if self.from_cache:
            return self.hyperparams['warmup_epochs']

        train_loader = self._create_dataloader(X_train, y_train)
        hp = self.hyperparams
        max_epochs = hp.get('max_warmup_epochs', 500)
        threshold = hp.get('warmup_saturation_threshold', 0.001)
        patience = hp.get('warmup_saturation_patience', 5)
        check_interval = 5

        best_acc = 0.0
        no_improve = 0
        actual_epochs = 0

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion_ce(self.model(batch_X), batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            actual_epochs = epoch + 1

            if actual_epochs % check_interval == 0:
                avg_loss = epoch_loss / len(train_loader)
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                log_progress_to_csv(str(self.csv_log_path), epoch, avg_loss, train_acc,
                                    num_classes=self.num_classes)

                if actual_epochs % 25 == 0:
                    log.info("Warmup %d: loss=%.4f acc=%.4f (best=%.4f, patience=%d/%d)",
                             actual_epochs, avg_loss, train_acc, best_acc, no_improve, patience)

                if train_acc > best_acc + threshold:
                    best_acc = train_acc
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    log.info("Warmup saturated at epoch %d (best acc=%.4f)", actual_epochs, best_acc)
                    break

        if no_improve < patience:
            log.info("Warmup reached max %d epochs (best acc=%.4f)", max_epochs, best_acc)

        save_to_cache(self.model, base_model_id, self.config)
        return actual_epochs

    def _cache_warmup_probabilities(self, X_test: torch.Tensor) -> torch.Tensor:
        """Cache warmup softmax probabilities for KL-divergence regularization."""
        self.model.eval()
        with torch.no_grad():
            warmup_proba = F.softmax(self.model(X_test), dim=1)
        log.info("Cached warmup probabilities: shape=%s", warmup_proba.shape)
        return warmup_proba.detach()

    @logger()
    def train_constraints(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          X_test: torch.Tensor, groups_test: pd.Series,
                          global_con: list, local_con: Dict[int, list],
                          actual_warmup_epochs: int = 50) -> nn.Module:
        """Run constraint optimization phase after warmup."""
        hp = self.hyperparams
        warmup_epochs = actual_warmup_epochs
        total_epochs = warmup_epochs + hp.get('constraint_epochs', 350)
        lambda_step = hp['lambda_step']

        lr_scheduler = LearningRateScheduler(
            optimizer=self.optimizer, warmup_lr=hp.get('lr', 1e-3),
            drop_lr=hp.get('lr_constraint', 1e-5), warmup_epochs=warmup_epochs)

        train_loader = self._create_dataloader(X_train, y_train)
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test.values).to(self.device)

        # Full test set constraint loss
        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con, local_constraints=local_con,
            lambda_global=hp['lambda_global'], lambda_local=hp['lambda_local'],
            num_classes=self.num_classes, use_sum=hp.get('use_sum_loss', True),
            initial_rho=hp.get('initial_rho', 0.5), alpha_kl=hp.get('alpha_kl', 0.0)
        ).to(self.device)

        # Mini test proxy for efficient per-batch constraint updates
        proxy_size = min(len(X_test), hp['batch_size'] * 10)
        proxy_indices = torch.randperm(len(X_test))[:proxy_size]
        X_test_proxy = X_test[proxy_indices]
        group_ids_proxy = group_ids[proxy_indices]
        proxy_scale = proxy_size / len(X_test)

        global_con_proxy = [c * proxy_scale if c < 1e9 else c for c in global_con]
        local_con_proxy = {
            gid: [c * proxy_scale if c < 1e9 else c for c in cons]
            for gid, cons in local_con.items()
        } if local_con else None

        criterion_proxy = MulticlassTransductiveLoss(
            global_constraints=global_con_proxy, local_constraints=local_con_proxy,
            lambda_global=hp['lambda_global'], lambda_local=hp['lambda_local'],
            num_classes=self.num_classes, use_sum=hp.get('use_sum_loss', True),
            initial_rho=hp.get('initial_rho', 0.5), alpha_kl=hp.get('alpha_kl', 0.0)
        ).to(self.device)

        log.info("Proxy: %d/%d samples (scale=%.3f)", proxy_size, len(X_test), proxy_scale)

        # KL-divergence regularization
        alpha_kl = hp.get('alpha_kl', 0.0)
        warmup_proba_full = None
        warmup_proba_proxy = None
        if alpha_kl > 0:
            warmup_proba_full = self._cache_warmup_probabilities(X_test)
            warmup_proba_proxy = warmup_proba_full[proxy_indices]

        # Training state
        constraints_satisfied = False
        satisfaction_epoch = None
        stable_count = 0
        training_start = time.time()

        log.info("Constraint training: epochs %d to %d", warmup_epochs + 1, total_epochs)

        # Write full CSV header now that we know the local constraint schema
        write_csv_header(str(self.csv_log_path), self.num_classes, local_con)

        for epoch in range(warmup_epochs, total_epochs):
            self.model.train()
            current_lr = lr_scheduler.step(epoch)

            epoch_ce, epoch_global, epoch_local, epoch_kl = 0.0, 0.0, 0.0, 0.0
            num_batches = len(train_loader)

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                loss_ce = self.criterion_ce(self.model(batch_X), batch_y)

                proxy_logits = self.model(X_test_proxy)
                _, _, loss_global, loss_local, loss_kl = criterion_proxy(
                    proxy_logits, y_true=None, group_ids=group_ids_proxy,
                    warmup_proba=warmup_proba_proxy)

                constraint_loss = (
                    criterion_proxy.lambda_global * loss_global +
                    criterion_proxy.lambda_local * loss_local +
                    criterion_proxy.alpha_kl * loss_kl)

                loss = loss_ce + constraint_loss
                epoch_ce += loss_ce.item()
                epoch_global += loss_global.item()
                epoch_local += loss_local.item()
                epoch_kl += loss_kl.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            avg_ce = epoch_ce / num_batches
            avg_global = epoch_global / num_batches
            avg_local = epoch_local / num_batches
            avg_kl = epoch_kl / num_batches

            # End-of-epoch validation on full test set
            self.model.eval()
            with torch.no_grad():
                test_logits = self.model(X_test)
                criterion_constraint(test_logits, y_true=None, group_ids=group_ids,
                                     warmup_proba=warmup_proba_full)

            global_satisfied = criterion_constraint.global_constraints_satisfied
            local_satisfied = criterion_constraint.local_constraints_satisfied
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
                    criterion_proxy.set_lambda(lambda_global=new_g)
                if not local_satisfied:
                    new_l = criterion_constraint.lambda_local + lambda_step
                    criterion_constraint.set_lambda(lambda_local=new_l)
                    criterion_proxy.set_lambda(lambda_local=new_l)
                # Adaptive ALM: increase rho every 25 epochs
                if (epoch - warmup_epochs) % 25 == 0 and epoch > warmup_epochs:
                    criterion_constraint.update_rho(factor=1.5)
                    criterion_proxy.update_rho(factor=1.5)

            # Logging every 10 epochs or on state change
            if (epoch + 1) % 10 == 0 or is_satisfied or epoch == warmup_epochs:
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
                    self.model, X_test, group_ids, num_classes=self.num_classes)

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
                        log.info("  Class %d: pred=%d limit=%d", c, g_counts.get(c, 0), int(global_con[c]))

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
        for c in range(self.num_classes):
            limit = int(global_con[c]) if global_con[c] < 1e9 else 'INF'
            log.info("  Final class %d: hard=%d soft=%.2f limit=%s",
                     c, g_counts.get(c, 0), g_soft.get(c, 0), limit)

        return self.model
