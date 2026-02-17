"""Constraint-aware neural network trainer with temperature scaling and adaptive scheduling.

Features:
- Temperature scaling: Makes predictions sharper during constraint phase
- 3-phase learning rate: Warmup -> Drop -> Recovery
- Conditional loss switching: Pure CE after constraints satisfied
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
from src.training.logging import log_progress_to_csv, print_progress
from src.training.schedulers import (
    TemperatureScheduler,
    LearningRateScheduler,
    TemperatureScaledModel
)
from src.utils.error_handler import logger, safe_execute


class ConstraintTrainer:
    """Trainer for constraint-aware neural network optimization.

    Binary classification: labels 0 (no churn) and 1 (churn).
    Standard CrossEntropyLoss is used.

    Training Phases (400 epochs total):
        - Warmup (0-49): temp=1.0, base LR, standard CE loss
        - Drop Phase (50-249): temp 1.5->0.5, low LR, constraint loss
        - Convergence (250-399): temp=0.5, recovery LR if needed

    Post-satisfaction: Switches to pure CE loss for accuracy refinement.
    """

    def __init__(self, config: Dict[str, Any], experiment_path: str, device: torch.device):
        self.config = config
        self.hyperparams = config['hyperparams']
        self.experiment_path = Path(experiment_path)
        self.device = device
        self.csv_log_path = self.experiment_path / 'training_log.csv'
        self.model: Optional[nn.Module] = None
        self.temp_model: Optional[TemperatureScaledModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion_ce = nn.CrossEntropyLoss()  # Binary labels 0/1
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
                n_classes=2,  # Binary classification
                dropout=self.hyperparams['dropout']
            ).to(self.device)
            self.from_cache = False
        else:
            print(f"[CACHE] Loaded model: {base_model_id}")
            self.from_cache = True

        # Wrap model with temperature scaling
        use_temp_scaling = self.hyperparams.get('use_temperature_scaling', True)
        if use_temp_scaling:
            self.temp_model = TemperatureScaledModel(
                self.model,
                initial_temp=1.0,
                learnable=self.hyperparams.get('learnable_temperature', False)
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.temp_model.parameters(),
                lr=self.hyperparams['lr']
            )
        else:
            self.temp_model = None
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hyperparams['lr']
            )

    def _get_forward_model(self) -> nn.Module:
        """Get the model to use for forward passes (temp-scaled or base)."""
        return self.temp_model if self.temp_model is not None else self.model

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
        print(f"[LABELS] y_train range: {y_train.min().item()}-{y_train.max().item()}")
        train_loader = self._create_dataloader(X_train, y_train)
        warmup_epochs = self.hyperparams['warmup_epochs']

        forward_model = self._get_forward_model()

        # Set temperature to 1.0 during warmup
        if self.temp_model is not None:
            self.temp_model.set_temperature(1.0)
            print("[TEMP] Temperature=1.0 (warmup phase)")

        for epoch in range(warmup_epochs):
            forward_model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = forward_model(batch_X)
                loss = self.criterion_ce(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(train_loader)
                train_acc = compute_train_accuracy(forward_model, train_loader, self.device)
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
        """Run constraint optimization training phase with advanced scheduling."""
        print("CONSTRAINT OPTIMIZATION TRAINING (Advanced)")

        # Get hyperparams with defaults
        hp = self.hyperparams
        warmup_epochs = hp['warmup_epochs']
        total_epochs = hp.get('epochs', 200)
        drop_epochs = hp.get('drop_epochs', 100)
        conv_epochs = hp.get('conv_epochs', 50)
        lambda_step = hp['lambda_step']

        # Initialize schedulers
        temp_scheduler = TemperatureScheduler(
            warmup_epochs=warmup_epochs,
            drop_epochs=drop_epochs,
            conv_epochs=conv_epochs,
            warmup_temp=hp.get('warmup_temp', 1.0),
            drop_start_temp=hp.get('drop_start_temp', 1.5),
            drop_end_temp=hp.get('drop_end_temp', 0.5),
            conv_temp=hp.get('conv_temp', 0.5)
        )

        lr_scheduler = LearningRateScheduler(
            optimizer=self.optimizer,
            warmup_lr=hp.get('lr', 1e-3),
            drop_lr=hp.get('lr_constraint', 1e-5),
            warmup_epochs=warmup_epochs,
            drop_epochs=drop_epochs,
            recovery_multiplier=hp.get('recovery_lr_multiplier', 2.0),
            recovery_interval=hp.get('recovery_interval', 25)
        )


        # Setup data
        train_loader = self._create_dataloader(X_train, y_train)
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test.values).to(self.device)

        # Initialize constraint loss with Augmented Lagrangian and Gumbel-Softmax
        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con,
            local_constraints=local_con,
            lambda_global=hp['lambda_global'],
            lambda_local=hp['lambda_local'],
            margin=hp.get('constraint_margin', 0.0),
            use_sum=hp.get('use_sum_loss', True),
            initial_rho=hp.get('initial_rho', 1.0),
            gumbel_temp=hp.get('gumbel_temp', 0.1)
        ).to(self.device)

        # Create mini test proxy for efficient every-batch constraint updates
        # This gives constraint gradient equal footing with CE gradient
        proxy_size = min(len(X_test), hp['batch_size'] * 10)
        proxy_indices = torch.randperm(len(X_test))[:proxy_size]
        X_test_proxy = X_test[proxy_indices]
        group_ids_proxy = group_ids[proxy_indices]
        proxy_scale = proxy_size / len(X_test)

        # Scale global constraints for proxy (proportional to proxy size)
        global_con_proxy = [c * proxy_scale if c < 1e9 else c for c in global_con]
        local_con_proxy = {
            gid: [c * proxy_scale if c < 1e9 else c for c in cons]
            for gid, cons in local_con.items()
        } if local_con else None

        # Create separate constraint loss for proxy
        criterion_constraint_proxy = MulticlassTransductiveLoss(
            global_constraints=global_con_proxy,
            local_constraints=local_con_proxy,
            lambda_global=hp['lambda_global'],
            lambda_local=hp['lambda_local'],
            margin=hp.get('constraint_margin', 0.0),
            use_sum=hp.get('use_sum_loss', True),
            initial_rho=hp.get('initial_rho', 1.0),
            gumbel_temp=hp.get('gumbel_temp', 0.1)
        ).to(self.device)

        print(f"[PROXY] Using {proxy_size}/{len(X_test)} samples, scale={proxy_scale:.3f}")
        print(f"[PROXY] Constraint limit: {global_con[1]} -> {global_con_proxy[1]:.1f}")

        forward_model = self._get_forward_model()

        # Training state
        constraints_satisfied = False
        satisfaction_epoch = None
        stable_count = 0  # Count consecutive epochs with satisfied constraints
        refinement_epochs = hp.get('refinement_epochs', 20)  # Epochs to refine after satisfaction

        print(f"Total epochs: {total_epochs} (warmup={warmup_epochs}, drop={drop_epochs}, conv={conv_epochs})")
        print(f"Will refine for {refinement_epochs} epochs after constraint satisfaction")

        # Timing for bottleneck analysis
        training_start_time = time.time()
        epoch_times = []

        for epoch in range(warmup_epochs, total_epochs):
            epoch_start_time = time.time()
            forward_model.train()

            # Update temperature
            current_temp = temp_scheduler.get_temperature(epoch)
            if self.temp_model is not None:
                self.temp_model.set_temperature(current_temp)

            # Update learning rate
            current_lr = lr_scheduler.step(epoch, constraints_satisfied)

            epoch_ce_loss = 0.0
            epoch_global_loss = 0.0
            epoch_local_loss = 0.0
            num_batches = len(train_loader)

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                # CE loss on training batch
                train_logits = forward_model(batch_X)
                loss_ce = self.criterion_ce(train_logits, batch_y)

                # Constraint loss on proxy (EVERY batch, not 1/10)
                proxy_logits = forward_model(X_test_proxy)
                _, _, loss_global, loss_local = criterion_constraint_proxy(
                    proxy_logits, y_true=None, group_ids=group_ids_proxy
                )

                # Combined loss: CE + constraint (both contribute every batch)
                constraint_loss = (
                    criterion_constraint_proxy.lambda_global * loss_global +
                    criterion_constraint_proxy.lambda_local * loss_local
                )

                # Total loss: CE + constraint compete fairly
                loss = loss_ce + constraint_loss

                epoch_ce_loss += loss_ce.item()
                epoch_global_loss += loss_global.item()
                epoch_local_loss += loss_local.item()

                loss.backward()

                # DEBUG: Verify gradients are reaching model (first constraint batch of first constraint epoch)
                if batch_idx == 0 and epoch == hp['warmup_epochs']:
                    grad_max = 0.0
                    grad_name = None
                    for name, param in forward_model.named_parameters():
                        if param.grad is not None:
                            g = param.grad.abs().max().item()
                            if g > grad_max:
                                grad_max = g
                                grad_name = name
                    if grad_max > 1e-8:
                        print(f"[GRAD OK] {grad_name}: max={grad_max:.6f}")
                    else:
                        print("[GRAD ZERO] No significant gradients found!")

                torch.nn.utils.clip_grad_norm_(forward_model.parameters(), max_norm=1.0)
                self.optimizer.step()

            avg_ce = epoch_ce_loss / num_batches
            avg_global = epoch_global_loss / num_batches
            avg_local = epoch_local_loss / num_batches

            # End-of-epoch validation
            forward_model.eval()
            with torch.no_grad():
                test_logits = forward_model(X_test)
                criterion_constraint(test_logits, y_true=None, group_ids=group_ids)

            global_satisfied = criterion_constraint.global_constraints_satisfied
            local_satisfied = criterion_constraint.local_constraints_satisfied
            is_satisfied = global_satisfied and local_satisfied

            # Track constraint satisfaction stability
            if is_satisfied:
                stable_count += 1
            else:
                stable_count = 0

            # Check for first satisfaction - reduce lambdas significantly
            # The goal is to maintain constraints but allow accuracy to improve
            if is_satisfied and not constraints_satisfied:
                constraints_satisfied = True
                satisfaction_epoch = epoch + 1
                temp_scheduler.freeze(current_temp)

                # Reduce lambdas to a maintenance level (10% of current)
                # This keeps light pressure to prevent violation but allows accuracy improvement
                maintenance_factor = hp.get('maintenance_lambda_factor', 0.1)
                new_lambda_g = criterion_constraint.lambda_global * maintenance_factor
                new_lambda_l = criterion_constraint.lambda_local * maintenance_factor
                criterion_constraint.set_lambda(lambda_global=new_lambda_g, lambda_local=new_lambda_l)
                criterion_constraint_proxy.set_lambda(lambda_global=new_lambda_g, lambda_local=new_lambda_l)

                print(f"\n[SATISFIED] Constraints satisfied at epoch {satisfaction_epoch}")
                print(f"[REDUCE] Reducing lambdas to maintenance level ({maintenance_factor*100:.0f}%)")
                print(f"[TEMP] Temperature frozen at {current_temp:.3f}")
                print(f"[LAMBDA] Reduced to g={criterion_constraint.lambda_global:.3f}, l={criterion_constraint.lambda_local:.3f}")

            # Early stop when constraints satisfied and stable for 5 epochs
            if constraints_satisfied and stable_count >= 5:
                print(f"\n[CONVERGED] Constraints satisfied for {stable_count} epochs. Stopping training.")
                break

            # Update lambdas and rho based on satisfaction status
            if not constraints_satisfied:
                # Before first satisfaction: increase lambdas if not satisfied
                if not global_satisfied:
                    new_lambda_g = criterion_constraint.lambda_global + lambda_step
                    criterion_constraint.set_lambda(lambda_global=new_lambda_g)
                    criterion_constraint_proxy.set_lambda(lambda_global=new_lambda_g)
                if not local_satisfied:
                    new_lambda_l = criterion_constraint.lambda_local + lambda_step
                    criterion_constraint.set_lambda(lambda_local=new_lambda_l)
                    criterion_constraint_proxy.set_lambda(lambda_local=new_lambda_l)
                # Adaptive Augmented Lagrangian: increase quadratic penalty every 25 epochs
                if (epoch - warmup_epochs) % 25 == 0 and epoch > warmup_epochs:
                    criterion_constraint.update_rho(factor=1.5)
                    criterion_constraint_proxy.update_rho(factor=1.5)
            else:
                # After satisfaction: if we drift back into violation, boost lambdas
                if not is_satisfied:
                    boost_factor = hp.get('violation_boost_factor', 2.0)
                    new_lambda_g = criterion_constraint.lambda_global * boost_factor
                    new_lambda_l = criterion_constraint.lambda_local * boost_factor
                    criterion_constraint.set_lambda(lambda_global=new_lambda_g, lambda_local=new_lambda_l)
                    criterion_constraint_proxy.set_lambda(lambda_global=new_lambda_g, lambda_local=new_lambda_l)
                    print(f"[BOOST] Constraints violated in refinement, boosting lambdas by {boost_factor}x")


            # Logging (every 10 epochs or on state change)
            phase = temp_scheduler.get_phase(epoch)
            should_log = (epoch + 1) % 10 == 0 or is_satisfied or (epoch == warmup_epochs)

            if should_log:
                train_acc = compute_train_accuracy(forward_model, train_loader, self.device)
                g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
                    forward_model, X_test, group_ids
                )

                mode_str = "Refinement" if constraints_satisfied else "Constraint"
                recovery_str = " [RECOVERY]" if lr_scheduler.is_in_recovery() else ""

                print(f"\n{'='*70}")
                print(f"Epoch {epoch + 1} | Phase: {phase} | Mode: {mode_str}{recovery_str}")
                print(f"Temp: {current_temp:.3f} | LR: {current_lr:.2e}")
                print(f"CE: {avg_ce:.4f} | Global: {avg_global:.4f} | Local: {avg_local:.4f}")
                print(f"Lambda: g={criterion_constraint.lambda_global:.3f}, l={criterion_constraint.lambda_local:.3f}, rho={criterion_constraint.get_rho():.3f}")
                print(f"Accuracy: {train_acc:.4f}")
                print(f"Global: {'OK' if global_satisfied else 'VIOLATED'} | Local: {'OK' if local_satisfied else 'VIOLATED'}")

                # Constrained class details (Class 1 = churn in binary)
                class1_hard = g_counts.get(1, 0)
                class1_limit = int(global_con[1]) if global_con[1] < 1e9 else 'INF'
                print(f"Class 1: pred={class1_hard}, limit={class1_limit}")
                print('='*70)

                log_progress_to_csv(
                    str(self.csv_log_path), epoch, avg_ce, train_acc, avg_global, avg_local,
                    g_counts, l_counts, g_soft, l_soft,
                    criterion_constraint.lambda_global, criterion_constraint.lambda_local,
                    global_con, global_satisfied, local_satisfied
                )

            forward_model.train()

            # Track epoch timing
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

        # Final validation
        total_training_time = time.time() - training_start_time
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        print(f"\n[TIMING] Total training: {total_training_time:.1f}s | Avg epoch: {avg_epoch_time:.2f}s | Epochs: {len(epoch_times)}")
        print("\n[FINAL VALIDATION]")
        forward_model.eval()
        g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(
            forward_model, X_test, group_ids
        )

        print(f"Final Temperature: {self.temp_model.get_temperature() if self.temp_model else 1.0:.3f}")
        print(f"Satisfaction Epoch: {satisfaction_epoch if satisfaction_epoch else 'Not satisfied'}")
        print(f"Class 1 predictions: hard={g_counts.get(1, 0)}, soft={g_soft.get(1, 0):.2f}")
        print(f"Class 1 constraint: {int(global_con[1]) if global_con[1] < 1e9 else 'INF'}")

        for c in range(2):  # Binary: 0 and 1
            limit = int(global_con[c]) if global_con[c] < 1e9 else 'INF'
            status = 'OK' if (isinstance(limit, str) or g_counts.get(c, 0) <= limit) else 'VIOLATED'
            print(f"  Class {c}: hard={g_counts.get(c, 0):>5}, limit={str(limit):>5}, {status}")

        # Return base model (without temperature wrapper for inference)
        return self.model

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

        if ckpt is None or ckpt.get('base_model_id') != base_model_id:
            return None

        model = get_model(
            self.config['model_name'],
            input_dim=input_dim,
            n_classes=2,  # Binary classification
            hidden_dims=self.hyperparams['hidden_dims'],
            dropout=self.hyperparams['dropout']
        ).to(self.device)

        load_result = safe_execute(
            model.load_state_dict, ckpt['model_state_dict'],
            default=None,
            context=f"Loading state dict for {base_model_id}"
        )

        return model if load_result is not None else None
