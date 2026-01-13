"""
Loss-Proportional Adaptive Lambda Trainer.

This trainer implements adaptive lambda adjustment where lambda grows proportionally
to the current constraint loss. This allows the model to learn patterns first (during
warmup) and then gradually increase constraint penalties based on violation severity.

Key features:
- Warmup phase: Pure prediction training (λ=0) for initial epochs
- Adaptive phase: λ increases proportional to constraint loss magnitude
- Early stopping: Stops when constraints satisfied for consecutive epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple
import numpy as np

from ..losses import compute_constraints_loss


class LossProportionalTrainer:
    """
    Trainer with loss-proportional adaptive lambda adjustment.

    Algorithm:
        1. Warmup (epochs 1 to warmup_epochs): λ = 0, train on L_pred only
        2. Adaptive (epochs warmup_epochs+1 onwards):
           - λ_global_new = λ_global_old + α × L_global
           - λ_local_new = λ_local_old + α × L_local
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
        self.warmup_epochs = config.get('warmup_epochs', 250)
        self.total_epochs = config.get('epochs', 1000)
        self.lambda_lr = config.get('lambda_learning_rate', 0.01)  # α parameter
        self.initial_lambda = config.get('initial_lambda', 0.001)
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
        Train model with loss-proportional adaptive lambda.

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

            # Phase determination
            in_warmup = (epoch <= self.warmup_epochs)

            if in_warmup:
                # Warmup: No constraints
                total_loss = loss_pred
                loss_global = torch.tensor(0.0)
                loss_local = torch.tensor(0.0)
                global_satisfied = False
                local_satisfied = False
            else:
                # Adaptive phase: Apply constraints with current lambda
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

                # Update lambda proportionally to constraint losses
                # λ_new = λ_old + α × L_constraint
                lambda_global_new = lambda_global + self.lambda_lr * loss_global.item()
                lambda_local_new = lambda_local + self.lambda_lr * loss_local.item()

                # Clip to max bounds
                lambda_global = min(lambda_global_new, self.max_lambda)
                lambda_local = min(lambda_local_new, self.max_lambda)

                # Check early stopping
                if global_satisfied and local_satisfied:
                    consecutive_satisfied += 1
                else:
                    consecutive_satisfied = 0

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Log progress
            if not in_warmup:
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
            else:
                # Log warmup with zeros for constraint values
                logger.log_epoch(
                    epoch=epoch,
                    train_acc=train_acc,
                    loss_pred=loss_pred.item(),
                    loss_global=0.0,
                    loss_local=0.0,
                    lambda_global=0.0,
                    lambda_local=0.0,
                    constraint_info=None
                )

            # Print progress
            if epoch % 10 == 0 or epoch == self.warmup_epochs + 1:
                if in_warmup:
                    print(f"Epoch {epoch}/{self.total_epochs} [WARMUP] | "
                          f"Acc: {train_acc:.4f} | L_pred: {loss_pred.item():.4f}")
                else:
                    print(f"Epoch {epoch}/{self.total_epochs} [ADAPTIVE] | "
                          f"Acc: {train_acc:.4f} | L_pred: {loss_pred.item():.4f} | "
                          f"L_global: {loss_global.item():.6f} | L_local: {loss_local.item():.6f} | "
                          f"λ_g: {lambda_global:.4f} | λ_l: {lambda_local:.4f} | "
                          f"G_sat: {int(global_satisfied)} | L_sat: {int(local_satisfied)}")

            # Early stopping
            if not in_warmup and consecutive_satisfied >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (constraints satisfied)")
                break

        logger.close()
        return model
