import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.models import get_model
from src.losses import MulticlassTransductiveLoss
from src.losses.lambda_adjusting import create_lambda_adjuster
from src.training.metrics import compute_train_accuracy, compute_prediction_statistics
from src.training.logging import log_progress_to_csv, print_progress, save_run_status


class ConstraintTrainer:
    def __init__(self, config: Dict[str, Any], experiment_path: str, device: torch.device):
        self.config = config
        self.hyperparams = config['hyperparams']
        self.experiment_path = Path(experiment_path)
        self.device = device
        self.csv_log_path = self.experiment_path / 'training_log.csv'

        # Initialize placeholders
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion_ce = nn.CrossEntropyLoss()

    def setup_model(self, input_dim: int, base_model_id: str) -> None:
        self.model = self._load_from_cache(base_model_id, input_dim)
        if self.model is None:
            print(f"[INIT] Creating new model: {self.config['model_name']}")
            self.model = get_model(
                self.config['model_name'],
                input_dim=input_dim,
                hidden_dims=self.hyperparams['hidden_dims'],
                n_classes=3,
                dropout=self.hyperparams['dropout']
            ).to(self.device)
            self.from_cache = False
        else:
            print(f"[CACHE] Loaded model: {base_model_id}")
            self.from_cache = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)

    def train_warmup(self, X_train: torch.Tensor, y_train: torch.Tensor, base_model_id: str) -> None:
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

    def train_constraints(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor,
                          groups_test: pd.Series, global_con: list, local_con: Dict[int, list]) -> nn.Module:
        print("CONSTRAINT OPTIMIZATION TRAINING")
        train_loader = self._create_dataloader(X_train, y_train)
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test.values).to(self.device)

        # Initialize lambda adjuster based on strategy
        lambda_strategy = self.hyperparams.get('lambda_strategy', 'linear')
        lambda_adjuster = create_lambda_adjuster(
            strategy=lambda_strategy,
            lambda_step=self.hyperparams['lambda_step'],
            lambda_max=50.0
        )
        print(f"Using lambda adjustment strategy: {lambda_strategy}")

        # Initialize constraint loss with base lambdas
        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con,
            local_constraints=local_con,
            lambda_global=self.hyperparams['lambda_global'],
            lambda_local=self.hyperparams['lambda_local']
        ).to(self.device)

        # Initialize sustained convergence checker
        from src.training.sustained_convergence import SustainedConvergenceChecker
        convergence_window = self.hyperparams.get('convergence_window', 1)
        convergence_required = self.hyperparams.get('convergence_required', 1)
        convergence_checker = SustainedConvergenceChecker(
            window_size=convergence_window,
            required_satisfied=convergence_required
        )
        print(f"Convergence criterion: {convergence_required}/{convergence_window} recent epochs must be satisfied")

        warmup_epochs = self.hyperparams['warmup_epochs']
        total_epochs = self.hyperparams['epochs']
        threshold = self.hyperparams['constraint_threshold']
        lambda_initialized = False

        for epoch in range(warmup_epochs, total_epochs):
            self.model.train()
            epoch_ce_loss = 0.0
            epoch_global_loss = 0.0
            epoch_local_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                # Training loss
                train_logits = self.model(batch_X)
                loss_ce = self.criterion_ce(train_logits, batch_y)

                # Constraint loss on test set (computed fresh each batch)
                test_logits = self.model(X_test)
                _, _, loss_global, loss_local = criterion_constraint(test_logits, y_true=None, group_ids=group_ids)

                # Scale CE loss to maintain balance with constraint losses
                # CE weight equals 1 + sum of lambdas to ensure CE remains competitive
                ce_weight = 1.0 + criterion_constraint.lambda_global + criterion_constraint.lambda_local

                loss = (ce_weight * loss_ce +
                        criterion_constraint.lambda_global * loss_global +
                        criterion_constraint.lambda_local * loss_local)
                loss.backward()
                self.optimizer.step()

                epoch_ce_loss += loss_ce.item()
                epoch_global_loss += loss_global.item()
                epoch_local_loss += loss_local.item()

            avg_ce = epoch_ce_loss / len(train_loader)
            avg_global = epoch_global_loss / len(train_loader)
            avg_local = epoch_local_loss / len(train_loader)

            print(
                f"Epoch {epoch + 1}: CE={avg_ce:.4f}, Global={avg_global:.4f}(λ={criterion_constraint.lambda_global:.2f}), "
                f"Local={avg_local:.4f}(λ={criterion_constraint.lambda_local:.2f})")


            # Initialize lambdas based on strategy (only once, after first constraint epoch)
            if not lambda_initialized:
                lambda_initialized = True
                new_lambda_global, new_lambda_local = lambda_adjuster.initialize_lambdas(
                    avg_global, avg_local,
                    criterion_constraint.lambda_global, criterion_constraint.lambda_local
                )
                if lambda_strategy == 'balanced':
                    print(f"  [BALANCED INIT] Adjusted lambdas: Global={new_lambda_global:.4f}, Local={new_lambda_local:.4f}")
                    print(f"  [BALANCED INIT] Loss ratio: Global={avg_global:.4f} / Local={avg_local:.4f}")
                criterion_constraint.set_lambda(lambda_global=new_lambda_global, lambda_local=new_lambda_local)

            # Adjust lambdas using the selected strategy
            new_lambda_global, new_lambda_local = lambda_adjuster.adjust_lambdas(
                criterion_constraint.lambda_global,
                criterion_constraint.lambda_local,
                criterion_constraint.global_constraints_satisfied,
                criterion_constraint.local_constraints_satisfied,
                avg_global,
                avg_local,
                threshold
            )
            criterion_constraint.set_lambda(lambda_global=new_lambda_global, lambda_local=new_lambda_local)

            if (epoch + 1) % 3 == 0 or (epoch + 1) == warmup_epochs + 1:
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(self.model, X_test, group_ids)

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
                    criterion_constraint.global_constraints_satisfied, criterion_constraint.local_constraints_satisfied
                )

            # Check for sustained convergence
            should_stop, reason = convergence_checker.update(
                criterion_constraint.global_constraints_satisfied,
                criterion_constraint.local_constraints_satisfied
            )

            if should_stop:
                print(f"\n[CONVERGED] {reason}")
                print(f"  Epoch {epoch + 1}")
                print(f"  Final loss: Global={avg_global:.6f}, Local={avg_local:.6f}")
                print(f"  Lambda values: Global={criterion_constraint.lambda_global:.2f}, Local={criterion_constraint.lambda_local:.2f}")

                # Save converged status to run_status.json
                save_run_status(
                    str(self.experiment_path),
                    status='converged',
                    epoch=epoch + 1,
                    global_satisfied=True,
                    local_satisfied=True,
                    details=f"Converged at epoch {epoch + 1}. {reason}. Global loss: {avg_global:.6f}, Local loss: {avg_local:.6f}"
                )

                # Import here to avoid circular dependency
                from src.utils.filesystem_manager import save_stop_reason
                # Save stop reason to config.json
                save_stop_reason(
                    str(self.experiment_path),
                    status='converged',
                    reason=f"Sustained convergence: {reason} at epoch {epoch + 1}",
                    exception_type=None,
                    final_epoch=epoch + 1,
                    global_satisfied=True,
                    local_satisfied=True
                )
                break
            else:
                # Optionally print convergence progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    rate = convergence_checker.get_satisfaction_rate()
                    print(f"  [CONV PROGRESS] Satisfaction rate: {rate*100:.1f}% ({sum(convergence_checker.history)}/{len(convergence_checker.history)})")
        else:
            # Loop completed without break - reached max epochs without convergence
            print(f"\n[FAILED] Reached maximum epochs ({total_epochs}) without full convergence")
            print(f"  Final loss: Global={avg_global:.6f}, Local={avg_local:.6f}")
            print(f"  Constraint status: Global={'Satisfied' if criterion_constraint.global_constraints_satisfied else 'Not Satisfied'}, "
                  f"Local={'Satisfied' if criterion_constraint.local_constraints_satisfied else 'Not Satisfied'}")

            # Save failed status to run_status.json
            save_run_status(
                str(self.experiment_path),
                status='failed',
                epoch=total_epochs,
                global_satisfied=criterion_constraint.global_constraints_satisfied,
                local_satisfied=criterion_constraint.local_constraints_satisfied,
                details=f"Reached max epochs without both constraints satisfied. Global loss: {avg_global:.6f}, Local loss: {avg_local:.6f}"
            )

            # Import here to avoid circular dependency
            from src.utils.filesystem_manager import save_stop_reason
            # Determine specific failure reason
            if criterion_constraint.global_constraints_satisfied and not criterion_constraint.local_constraints_satisfied:
                reason = f"Reached {total_epochs} epochs with only Global constraint satisfied (Local constraint not satisfied)"
            elif not criterion_constraint.global_constraints_satisfied and criterion_constraint.local_constraints_satisfied:
                reason = f"Reached {total_epochs} epochs with only Local constraint satisfied (Global constraint not satisfied)"
            elif not criterion_constraint.global_constraints_satisfied and not criterion_constraint.local_constraints_satisfied:
                reason = f"Reached {total_epochs} epochs without satisfying either Global or Local constraints"
            else:
                reason = f"Reached {total_epochs} epochs (unexpected state)"

            # Save stop reason to config.json
            save_stop_reason(
                str(self.experiment_path),
                status='failed',
                reason=reason,
                exception_type=None,
                final_epoch=total_epochs,
                global_satisfied=criterion_constraint.global_constraints_satisfied,
                local_satisfied=criterion_constraint.local_constraints_satisfied
            )

        return self.model

    def _get_cache_path(self, base_model_id: str) -> Path:
        cache_dir = Path('model_cache')
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{base_model_id}.pt"

    def _save_to_cache(self, base_model_id: str) -> None:
        path = self._get_cache_path(base_model_id)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'base_model_id': base_model_id,
            'config': self.config,
            'saved_at': time.strftime('%Y-%m-%d')
        }, path)
        print(f"[CACHE] Model saved: {base_model_id}")

    def _load_from_cache(self, base_model_id: str, input_dim: int) -> Optional[nn.Module]:
        path = self._get_cache_path(base_model_id)
        if not path.exists():
            return None
        try:
            ckpt = torch.load(path, map_location=self.device)
            if ckpt['base_model_id'] != base_model_id:
                return None

            model = get_model(
                self.config['model_name'],
                input_dim=input_dim,
                n_classes=3,
                hidden_dims=self.hyperparams['hidden_dims'],
                dropout=self.hyperparams['dropout']
            ).to(self.device)
            model.load_state_dict(ckpt['model_state_dict'])
            return model
        except Exception as e:
            print(f"[WARNING] Failed to load cached model {base_model_id}: {e}")
            return None
