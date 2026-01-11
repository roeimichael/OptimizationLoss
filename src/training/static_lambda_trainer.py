"""Static Lambda Trainer for constraint-based optimization.

This trainer implements a fixed-lambda training methodology where lambda values
remain constant throughout training, as opposed to the adaptive approach where
lambdas increase over time.

Key differences from ConstraintTrainer:
- No warmup phase (trains with constraints from epoch 0)
- Lambda values are static (do not increase)
- Fixed training duration (300 epochs)
- Raises error if constraints not satisfied after training
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


class ConstraintsNotMetError(Exception):
    """Raised when constraints are not satisfied after training completion.

    This exception is raised when the model completes training but fails to
    satisfy the global and/or local constraints within the specified threshold.
    """

    def __init__(self, message: str, global_satisfied: bool, local_satisfied: bool,
                 final_global_loss: float, final_local_loss: float):
        super().__init__(message)
        self.global_satisfied = global_satisfied
        self.local_satisfied = local_satisfied
        self.final_global_loss = final_global_loss
        self.final_local_loss = final_local_loss


class StaticLambdaTrainer:
    """Trainer for static lambda constraint optimization experiments.

    This trainer implements a training methodology where:
    1. No warmup phase - constraints applied from epoch 0
    2. Lambda values remain constant throughout training
    3. Training runs for exactly 300 epochs (configurable)
    4. After training, verifies constraints are satisfied
    5. Raises ConstraintsNotMetError if constraints not met
    """

    def __init__(self, config: Dict[str, Any], experiment_path: str, device: torch.device):
        """Initialize StaticLambdaTrainer.

        Args:
            config: Experiment configuration dictionary
            experiment_path: Path to experiment directory
            device: PyTorch device (CPU or CUDA)
        """
        self.config = config
        self.hyperparams = config['hyperparams']
        self.experiment_path = Path(experiment_path)
        self.device = device
        self.csv_log_path = self.experiment_path / 'training_log.csv'

        # Initialize placeholders
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion_ce = nn.CrossEntropyLoss()
        self.from_cache = False

    def setup_model(self, input_dim: int, base_model_id: str) -> None:
        """Setup model architecture and optimizer.

        Args:
            input_dim: Number of input features
            base_model_id: Unique identifier for model caching
        """
        # Note: Static lambda experiments don't use cached warmup models
        # since there is no warmup phase
        print(f"[INIT] Creating new model: {self.config['model_name']}")
        self.model = get_model(
            self.config['model_name'],
            input_dim=input_dim,
            hidden_dims=self.hyperparams['hidden_dims'],
            n_classes=3,
            dropout=self.hyperparams['dropout']
        ).to(self.device)
        self.from_cache = False

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams['lr']
        )

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        """Create DataLoader for training data.

        Args:
            X: Input features
            y: Target labels

        Returns:
            DataLoader instance
        """
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=True
        )

    def train_static_lambda(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        groups_test: pd.Series,
        global_con: list,
        local_con: Dict[int, list]
    ) -> nn.Module:
        """Train model with static lambda values.

        This method trains the model for a fixed number of epochs with constant
        lambda values. Unlike the adaptive approach, lambda values do not change
        during training. After training completes, it verifies that constraints
        are satisfied and raises an error if they are not.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (for constraint evaluation)
            groups_test: Group IDs for local constraints
            global_con: Global constraint values [c0, c1, c2]
            local_con: Local constraint values {group_id: [c0, c1, c2]}

        Returns:
            Trained model

        Raises:
            ConstraintsNotMetError: If constraints not satisfied after training
        """
        print("\n" + "=" * 80)
        print("STATIC LAMBDA TRAINING")
        print("=" * 80)
        print(f"Lambda Global: {self.hyperparams['lambda_global']}")
        print(f"Lambda Local: {self.hyperparams['lambda_local']}")
        print(f"Training Epochs: {self.hyperparams['epochs']}")
        print(f"No warmup phase - constraints applied from epoch 0")
        print("=" * 80)

        train_loader = self._create_dataloader(X_train, y_train)
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test.values).to(self.device)

        # Create constraint loss with STATIC lambda values
        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con,
            local_constraints=local_con,
            lambda_global=self.hyperparams['lambda_global'],
            lambda_local=self.hyperparams['lambda_local'],
            use_ce=False
        ).to(self.device)

        total_epochs = self.hyperparams['epochs']
        threshold = self.hyperparams['constraint_threshold']

        # Train for exactly 'epochs' iterations
        for epoch in range(total_epochs):
            self.model.train()
            epoch_ce_loss = 0.0

            # Training step
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()

                # Compute training loss (cross-entropy)
                train_logits = self.model(batch_X)
                loss_ce = self.criterion_ce(train_logits, batch_y)

                # Compute constraint losses on test set
                self.model.eval()
                test_logits = self.model(X_test)
                self.model.train()

                _, _, loss_global, loss_local = criterion_constraint(
                    test_logits,
                    y_true=None,
                    group_ids=group_ids
                )

                # Total loss: CE + (constant lambda_global × global) + (constant lambda_local × local)
                # Note: Lambda values do NOT change during training
                loss = loss_ce + \
                       (criterion_constraint.lambda_global * loss_global) + \
                       (criterion_constraint.lambda_local * loss_local)

                loss.backward()
                self.optimizer.step()
                epoch_ce_loss += loss_ce.item()

            # Logging
            avg_ce = epoch_ce_loss / len(train_loader)
            avg_global = loss_global.item()
            avg_local = loss_local.item()

            # Log progress every 3 epochs or at epoch 1
            if (epoch + 1) % 3 == 0 or (epoch + 1) == 1:
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
                    epoch, avg_ce, avg_global, avg_local,
                    criterion_constraint.lambda_global, criterion_constraint.lambda_local,
                    train_acc, g_counts, g_soft, global_con,
                    criterion_constraint.global_constraints_satisfied,
                    criterion_constraint.local_constraints_satisfied
                )

            # Optional: Early stopping if constraints satisfied
            # (Can be disabled to always train for full 300 epochs)
            if avg_global <= threshold and avg_local <= threshold:
                print(f"\n✓ CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
                print(f"  Global loss: {avg_global:.6f} <= {threshold}")
                print(f"  Local loss: {avg_local:.6f} <= {threshold}")
                # Still continue training to full epochs for fair comparison

        # After training completes, verify constraints are satisfied
        print("\n" + "=" * 80)
        print("VERIFYING CONSTRAINT SATISFACTION")
        print("=" * 80)

        # Final check
        self.model.eval()
        with torch.no_grad():
            final_test_logits = self.model(X_test)
            _, _, final_global_loss, final_local_loss = criterion_constraint(
                final_test_logits,
                y_true=None,
                group_ids=group_ids
            )

        global_satisfied = criterion_constraint.global_constraints_satisfied
        local_satisfied = criterion_constraint.local_constraints_satisfied

        print(f"Final Global Loss: {final_global_loss.item():.6f} "
              f"(threshold: {threshold})")
        print(f"Global Satisfied: {global_satisfied}")
        print(f"Final Local Loss: {final_local_loss.item():.6f} "
              f"(threshold: {threshold})")
        print(f"Local Satisfied: {local_satisfied}")

        # Raise error if constraints not satisfied
        if not (global_satisfied and local_satisfied):
            error_msg = (
                f"Constraints not satisfied after {total_epochs} epochs. "
                f"Global: {global_satisfied} (loss={final_global_loss.item():.6f}), "
                f"Local: {local_satisfied} (loss={final_local_loss.item():.6f})"
            )
            print(f"\n✗ {error_msg}")
            raise ConstraintsNotMetError(
                error_msg,
                global_satisfied=global_satisfied,
                local_satisfied=local_satisfied,
                final_global_loss=final_global_loss.item(),
                final_local_loss=final_local_loss.item()
            )

        print("\n✓ ALL CONSTRAINTS SATISFIED!")
        return self.model
