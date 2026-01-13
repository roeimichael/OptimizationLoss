"""
Scheduled Growth with Loss Gates Trainer.

This trainer implements adaptive lambda adjustment with scheduled checkpoints,
starting from a low initial value. Lambda only increases if the model fails
to improve constraint satisfaction, giving the model time to adjust.

Key features:
- Starts with low initial lambda and increases throughout training
- Scheduled growth: Check constraint improvement every N epochs
- Loss gates: Only increase λ if loss didn't improve (or got worse)
- Early stopping: Stops when constraints satisfied for consecutive epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple
import numpy as np

from ..losses import compute_constraints_loss


class ScheduledGrowthTrainer:
    """
    Trainer with scheduled growth and loss gate adaptive lambda.

    Algorithm:
        1. Start with initial_lambda (low value)
        2. Every check_frequency epochs:
           - If constraint_loss >= previous_constraint_loss:
             λ = λ × growth_factor
           - Else: λ stays the same (model is improving)
        3. Early stopping: Stop if constraints satisfied for N consecutive epochs

    Args:
        config: Experiment configuration dictionary
        experiment_path: Path to save experiment results
        device: Device to run training on
    """

    def __init__(self, config: Dict[str, Any], experiment_path: str, device: torch.device):
        self.config = config
        self.experiment_path = experiment_path
        self.device = device

        # Extract hyperparameters
        self.total_epochs = config.get('epochs', 1000)
        self.initial_lambda = config.get('initial_lambda', 0.001)
        self.growth_factor = config.get('growth_factor', 1.1)  # 10% increase
        self.check_frequency = config.get('check_frequency', 10)  # Check every 10 epochs
        self.max_lambda = config.get('max_lambda', 1.0)
        self.constraint_threshold = config.get('constraint_threshold', 1e-6)
        self.early_stop_patience = config.get('early_stop_patience', 10)


    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        groups_test: np.ndarray,
        global_con: Tuple[float, float],
        local_con: Dict[int, Tuple[float, float]]
    ) -> nn.Module:
        """
        Train model with scheduled growth and loss gates.

        Args:
            model: Neural network model
            X_train: Training features
            y_train: Training labels
            X_test: Test features (for constraint evaluation)
            groups_test: Test group memberships
            global_con: Global constraint (min_grad_ratio, max_dropout_ratio)
            local_con: Local per-group constraints

        Returns:
            Trained model
        """
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        groups_test_t = torch.LongTensor(groups_test).to(self.device)

        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()

        # Initialize lambda values
        lambda_global = self.initial_lambda
        lambda_local = self.initial_lambda

        # Tracking for scheduled growth
        previous_total_constraint_loss = float('inf')
        last_check_epoch = 0

        # Early stopping tracking
        consecutive_satisfied = 0

        # Training log
        from ..training.logging import TrainingLogger
        logger = TrainingLogger(self.experiment_path)

        for epoch in range(1, self.total_epochs + 1):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            logits = model(X_train_t)
            loss_pred = criterion(logits, y_train_t)

            # Compute training accuracy
            _, predicted = torch.max(logits, 1)
            train_acc = (predicted == y_train_t).float().mean().item()

            # Apply constraints with current lambda from epoch 1
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test_t)
                test_probs = torch.softmax(test_logits, dim=1)

            # Compute constraint losses
            loss_global, loss_local, global_satisfied, local_satisfied, constraint_info = \
                compute_constraints_loss(
                    test_probs,
                    groups_test_t,
                    global_con,
                    local_con,
                    self.constraint_threshold
                )

            model.train()

            # Total loss with current lambda
            total_loss = loss_pred + lambda_global * loss_global + lambda_local * loss_local

            # Track total constraint loss
            current_total_constraint_loss = loss_global.item() + loss_local.item()

            # Scheduled lambda update (check every N epochs)
            epochs_since_check = epoch - last_check_epoch
            if epochs_since_check >= self.check_frequency:
                # Compare with previous constraint loss
                if current_total_constraint_loss >= previous_total_constraint_loss:
                    # Loss didn't improve (or got worse) - increase lambda
                    lambda_global_new = lambda_global * self.growth_factor
                    lambda_local_new = lambda_local * self.growth_factor

                    # Clip to max bounds
                    lambda_global = min(lambda_global_new, self.max_lambda)
                    lambda_local = min(lambda_local_new, self.max_lambda)

                    print(f"\n[Epoch {epoch}] Constraint loss didn't improve "
                          f"({previous_total_constraint_loss:.6f} → {current_total_constraint_loss:.6f})")
                    print(f"  → Increasing lambda: λ_g={lambda_global:.4f}, λ_l={lambda_local:.4f}\n")
                else:
                    # Loss improved - keep lambda the same
                    print(f"\n[Epoch {epoch}] Constraint loss improved "
                          f"({previous_total_constraint_loss:.6f} → {current_total_constraint_loss:.6f})")
                    print(f"  → Keeping lambda: λ_g={lambda_global:.4f}, λ_l={lambda_local:.4f}\n")

                # Update tracking variables
                previous_total_constraint_loss = current_total_constraint_loss
                last_check_epoch = epoch

            # Check early stopping
            if global_satisfied and local_satisfied:
                consecutive_satisfied += 1
            else:
                consecutive_satisfied = 0

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Log progress
            logger.log_epoch(
                epoch=epoch,
                train_acc=train_acc,
                loss_pred=loss_pred.item(),
                loss_global=loss_global.item(),
                loss_local=loss_local.item(),
                lambda_global=lambda_global,
                lambda_local=lambda_local,
                constraint_info=constraint_info
            )

            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.total_epochs} | "
                      f"Acc: {train_acc:.4f} | L_pred: {loss_pred.item():.4f} | "
                      f"L_global: {loss_global.item():.6f} | L_local: {loss_local.item():.6f} | "
                      f"λ_g: {lambda_global:.4f} | λ_l: {lambda_local:.4f} | "
                      f"G_sat: {int(global_satisfied)} | L_sat: {int(local_satisfied)}")

            # Early stopping
            if consecutive_satisfied >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (constraints satisfied)")
                break

        logger.close()
        return model
