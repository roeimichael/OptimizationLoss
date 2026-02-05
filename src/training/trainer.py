"""
Constraint-aware neural network trainer.

This module provides the ConstraintTrainer class for training neural networks
with both global and local constraint satisfaction objectives.
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.models import get_model
from src.losses import MulticlassTransductiveLoss
from src.training.metrics import compute_train_accuracy, compute_prediction_statistics
from src.training.logging import log_progress_to_csv, print_progress, save_run_status
from src.utils.error_handler import logger, safe_execute
from src.utils.filesystem_manager import save_stop_reason


class ConstraintTrainer:
    """
    Trainer for constraint-aware neural network optimization.

    Handles warmup training (standard CE loss) followed by constraint
    optimization training with adaptive lambda weighting.
    """

    def __init__(self, config: Dict[str, Any], experiment_path: str, device: torch.device):
        self.config = config
        self.hyperparams = config['hyperparams']
        self.experiment_path = Path(experiment_path)
        self.device = device
        self.csv_log_path = self.experiment_path / 'training_log.csv'
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion_ce = nn.CrossEntropyLoss()
        self.from_cache = False

    @logger()
    def setup_model(self, input_dim: int, base_model_id: str) -> None:
        """Initialize or load model from cache."""
        self.model = self._load_from_cache(base_model_id, input_dim)

        if self.model is None:
            print(f"[INIT] Creating new model: {self.config['model_name']}")
            self.model = get_model(
                self.config['model_name'],
                input_dim=input_dim,
                hidden_dims=self.hyperparams['hidden_dims'],
                n_classes=5,
                dropout=self.hyperparams['dropout']
            ).to(self.device)
            self.from_cache = False
        else:
            print(f"[CACHE] Loaded model: {base_model_id}")
            self.from_cache = True

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        """Create a DataLoader from tensors."""
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)

    @logger()
    def train_warmup(self, X_train: torch.Tensor, y_train: torch.Tensor, base_model_id: str) -> None:
        """Run warmup training phase with standard cross-entropy loss."""
        if self.from_cache:
            return

        print("WARMUP TRAINING")
        train_loader = self._create_dataloader(X_train, y_train)
        warmup_epochs = self.hyperparams['warmup_epochs']

        for epoch in range(warmup_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion_ce(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(train_loader)
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                log_progress_to_csv(str(self.csv_log_path), epoch, avg_loss, train_acc)

                if (epoch + 1) % 50 == 0:
                    print(f"Warmup Epoch {epoch + 1}/{warmup_epochs}: Loss={avg_loss:.4f}, Acc={train_acc:.4f}")

        self._save_to_cache(base_model_id)

    @logger()
    def train_constraints(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        groups_test: pd.Series,
        global_con: list,
        local_con: Dict[int, list]
    ) -> nn.Module:
        """
        Run constraint optimization training phase.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (used for transductive constraint computation)
            groups_test: Group assignments for local constraints
            global_con: Global constraint values per class
            local_con: Local constraint values per group per class

        Returns:
            Trained model
        """
        print("CONSTRAINT OPTIMIZATION TRAINING")
        train_loader = self._create_dataloader(X_train, y_train)
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test.values).to(self.device)

        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con,
            local_constraints=local_con,
            lambda_global=self.hyperparams['lambda_global'],
            lambda_local=self.hyperparams['lambda_local']
        ).to(self.device)

        warmup_epochs = self.hyperparams['warmup_epochs']
        total_epochs = self.hyperparams['epochs']
        lambda_step = self.hyperparams['lambda_step']

        avg_global = 0.0
        avg_local = 0.0

        for epoch in range(warmup_epochs, total_epochs):
            self.model.train()
            epoch_ce_loss = 0.0
            epoch_global_loss = 0.0
            epoch_local_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                train_logits = self.model(batch_X)
                loss_ce = self.criterion_ce(train_logits, batch_y)

                test_logits = self.model(X_test)
                _, _, loss_global, loss_local = criterion_constraint(
                    test_logits, y_true=None, group_ids=group_ids
                )

                ce_weight = 1.0 + criterion_constraint.lambda_global + criterion_constraint.lambda_local
                loss = (
                    ce_weight * loss_ce +
                    criterion_constraint.lambda_global * loss_global +
                    criterion_constraint.lambda_local * loss_local
                )
                loss.backward()
                self.optimizer.step()

                epoch_ce_loss += loss_ce.item()
                epoch_global_loss += loss_global.item()
                epoch_local_loss += loss_local.item()

            avg_ce = epoch_ce_loss / len(train_loader)
            avg_global = epoch_global_loss / len(train_loader)
            avg_local = epoch_local_loss / len(train_loader)

            print(
                f"Epoch {epoch + 1}: CE={avg_ce:.4f}, "
                f"Global={avg_global:.4f}(l={criterion_constraint.lambda_global:.2f}), "
                f"Local={avg_local:.4f}(l={criterion_constraint.lambda_local:.2f})"
            )

            new_lambda_global = criterion_constraint.lambda_global
            new_lambda_local = criterion_constraint.lambda_local
            if not criterion_constraint.global_constraints_satisfied:
                new_lambda_global += lambda_step
            if not criterion_constraint.local_constraints_satisfied:
                new_lambda_local += lambda_step
            criterion_constraint.set_lambda(lambda_global=new_lambda_global, lambda_local=new_lambda_local)

            if (epoch + 1) % 3 == 0 or (epoch + 1) == warmup_epochs + 1:
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
                    self.model, X_test, group_ids
                )

                log_progress_to_csv(
                    str(self.csv_log_path), epoch, avg_ce, train_acc, avg_global, avg_local,
                    g_counts, l_counts, g_soft, l_soft,
                    criterion_constraint.lambda_global, criterion_constraint.lambda_local,
                    global_con, criterion_constraint.global_constraints_satisfied,
                    criterion_constraint.local_constraints_satisfied
                )
                print_progress(
                    epoch, avg_ce, avg_global, avg_local, criterion_constraint.lambda_global,
                    criterion_constraint.lambda_local, train_acc, g_counts, g_soft, global_con,
                    criterion_constraint.global_constraints_satisfied,
                    criterion_constraint.local_constraints_satisfied
                )

            if criterion_constraint.global_constraints_satisfied and criterion_constraint.local_constraints_satisfied:
                self._save_convergence_status(epoch, avg_global, avg_local, criterion_constraint)
                break
        else:
            self._save_failure_status(total_epochs, avg_global, avg_local, criterion_constraint)

        return self.model

    def _save_convergence_status(
        self,
        epoch: int,
        avg_global: float,
        avg_local: float,
        criterion: MulticlassTransductiveLoss
    ) -> None:
        """Save status when training converges successfully."""
        print(f"\n[CONVERGED] Both constraints satisfied at epoch {epoch + 1}")
        print(f"  Final loss: Global={avg_global:.6f}, Local={avg_local:.6f}")
        print(f"  Lambda values: Global={criterion.lambda_global:.2f}, Local={criterion.lambda_local:.2f}")

        save_run_status(
            str(self.experiment_path),
            status='converged',
            epoch=epoch + 1,
            global_satisfied=True,
            local_satisfied=True,
            details=f"Converged at epoch {epoch + 1}. Global loss: {avg_global:.6f}, Local loss: {avg_local:.6f}"
        )

        save_stop_reason(
            str(self.experiment_path),
            status='converged',
            reason=f"Both constraints satisfied at epoch {epoch + 1}",
            exception_type=None,
            final_epoch=epoch + 1,
            global_satisfied=True,
            local_satisfied=True
        )

    def _save_failure_status(
        self,
        total_epochs: int,
        avg_global: float,
        avg_local: float,
        criterion: MulticlassTransductiveLoss
    ) -> None:
        """Save status when training fails to converge."""
        global_sat = criterion.global_constraints_satisfied
        local_sat = criterion.local_constraints_satisfied

        print(f"\n[FAILED] Reached maximum epochs ({total_epochs}) without convergence")
        print(f"  Final loss: Global={avg_global:.6f}, Local={avg_local:.6f}")
        print(f"  Constraint status: Global={'Satisfied' if global_sat else 'Not Satisfied'}, "
              f"Local={'Satisfied' if local_sat else 'Not Satisfied'}")

        save_run_status(
            str(self.experiment_path),
            status='failed',
            epoch=total_epochs,
            global_satisfied=global_sat,
            local_satisfied=local_sat,
            details=f"Reached max epochs. Global loss: {avg_global:.6f}, Local loss: {avg_local:.6f}"
        )

        if global_sat and not local_sat:
            reason = f"Reached {total_epochs} epochs with only Global constraint satisfied"
        elif not global_sat and local_sat:
            reason = f"Reached {total_epochs} epochs with only Local constraint satisfied"
        elif not global_sat and not local_sat:
            reason = f"Reached {total_epochs} epochs without satisfying either constraint"
        else:
            reason = f"Reached {total_epochs} epochs"

        save_stop_reason(
            str(self.experiment_path),
            status='failed',
            reason=reason,
            exception_type=None,
            final_epoch=total_epochs,
            global_satisfied=global_sat,
            local_satisfied=local_sat
        )

    def _get_cache_path(self, base_model_id: str) -> Path:
        """Get the file path for a cached model."""
        cache_dir = Path('model_cache')
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{base_model_id}.pt"

    def _save_to_cache(self, base_model_id: str) -> None:
        """Save current model to cache."""
        path = self._get_cache_path(base_model_id)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'base_model_id': base_model_id,
            'config': self.config,
            'saved_at': time.strftime('%Y-%m-%d')
        }, path)
        print(f"[CACHE] Model saved: {base_model_id}")

    def _load_from_cache(self, base_model_id: str, input_dim: int) -> Optional[nn.Module]:
        """Attempt to load model from cache."""
        path = self._get_cache_path(base_model_id)

        if not path.exists():
            return None

        ckpt = safe_execute(
            torch.load, path,
            map_location=self.device,
            default=None,
            context=f"Loading cached model {base_model_id}"
        )

        if ckpt is None:
            return None

        if ckpt.get('base_model_id') != base_model_id:
            return None

        model = get_model(
            self.config['model_name'],
            input_dim=input_dim,
            n_classes=5,
            hidden_dims=self.hyperparams['hidden_dims'],
            dropout=self.hyperparams['dropout']
        ).to(self.device)

        load_result = safe_execute(
            model.load_state_dict, ckpt['model_state_dict'],
            default=None,
            context=f"Loading state dict for {base_model_id}"
        )

        if load_result is None:
            return None

        return model
