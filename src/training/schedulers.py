"""Temperature and Learning Rate Schedulers for Constraint Training.

Temperature Schedule (200 epochs total):
    - Warmup (0-49): temp=1.0 (fixed)
    - Drop Phase (50-149): Linear decay from 1.5 to 0.5
    - Convergence Phase (150-199): Stable at 0.5

Learning Rate Schedule (3 levels):
    - Warmup (0-49): base_lr (e.g., 1e-3)
    - Drop Phase (50-149): low_lr (e.g., 1e-5)
    - Recovery (150+): If not satisfied, multiply LR every 25 epochs
"""

import torch
import torch.nn as nn
from typing import Optional


class TemperatureScheduler:
    """Dynamic temperature scheduler for logit scaling."""

    def __init__(
        self,
        warmup_epochs: int = 50,
        drop_epochs: int = 100,
        conv_epochs: int = 50,
        warmup_temp: float = 1.0,
        drop_start_temp: float = 1.5,
        drop_end_temp: float = 0.5,
        conv_temp: float = 0.5,
        min_temp: float = 0.1
    ):
        """
        Args:
            warmup_epochs: Number of warmup epochs (temp=warmup_temp)
            drop_epochs: Number of drop phase epochs (linear decay)
            conv_epochs: Number of convergence epochs (temp=conv_temp)
            warmup_temp: Temperature during warmup
            drop_start_temp: Temperature at start of drop phase
            drop_end_temp: Temperature at end of drop phase
            conv_temp: Temperature during convergence
            min_temp: Minimum temperature (clamp value)
        """
        self.warmup_epochs = warmup_epochs
        self.drop_epochs = drop_epochs
        self.conv_epochs = conv_epochs
        self.warmup_temp = warmup_temp
        self.drop_start_temp = drop_start_temp
        self.drop_end_temp = drop_end_temp
        self.conv_temp = conv_temp
        self.min_temp = min_temp

        self.total_epochs = warmup_epochs + drop_epochs + conv_epochs
        self.drop_start_epoch = warmup_epochs
        self.conv_start_epoch = warmup_epochs + drop_epochs

        self._frozen = False
        self._frozen_temp: Optional[float] = None

    def get_temperature(self, epoch: int) -> float:
        """Get temperature for given epoch."""
        if self._frozen and self._frozen_temp is not None:
            return max(self._frozen_temp, self.min_temp)

        if epoch < self.drop_start_epoch:
            # Warmup phase
            return self.warmup_temp
        elif epoch < self.conv_start_epoch:
            # Drop phase: linear decay
            progress = (epoch - self.drop_start_epoch) / self.drop_epochs
            temp = self.drop_start_temp + progress * (self.drop_end_temp - self.drop_start_temp)
            return max(temp, self.min_temp)
        else:
            # Convergence phase
            return max(self.conv_temp, self.min_temp)

    def freeze(self, current_temp: float):
        """Freeze temperature at current value (called when constraints satisfied)."""
        self._frozen = True
        self._frozen_temp = current_temp

    def unfreeze(self):
        """Unfreeze temperature scheduling."""
        self._frozen = False
        self._frozen_temp = None

    def get_phase(self, epoch: int) -> str:
        """Get current training phase name."""
        if epoch < self.drop_start_epoch:
            return "warmup"
        elif epoch < self.conv_start_epoch:
            return "drop"
        else:
            return "convergence"


class LearningRateScheduler:
    """3-level learning rate scheduler with recovery mode."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_lr: float = 1e-3,
        drop_lr: float = 1e-5,
        warmup_epochs: int = 50,
        drop_epochs: int = 100,
        recovery_multiplier: float = 2.0,
        recovery_interval: int = 25,
        max_lr: float = 1e-2
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_lr: Learning rate during warmup
            drop_lr: Learning rate during drop phase
            warmup_epochs: Number of warmup epochs
            drop_epochs: Number of drop phase epochs
            recovery_multiplier: LR multiplier for recovery mode
            recovery_interval: Epochs between recovery LR increases
            max_lr: Maximum learning rate (clamp)
        """
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.drop_lr = drop_lr
        self.warmup_epochs = warmup_epochs
        self.drop_epochs = drop_epochs
        self.recovery_multiplier = recovery_multiplier
        self.recovery_interval = recovery_interval
        self.max_lr = max_lr

        self.conv_start_epoch = warmup_epochs + drop_epochs
        self._in_recovery = False
        self._recovery_lr = drop_lr
        self._current_lr = warmup_lr

    def step(self, epoch: int, constraints_satisfied: bool = False):
        """Update learning rate based on epoch and satisfaction status."""
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.warmup_lr
        elif epoch < self.conv_start_epoch:
            # Drop phase
            lr = self.drop_lr
        else:
            # Convergence/Recovery phase
            if constraints_satisfied:
                # Keep current LR for refinement
                lr = self._current_lr
            else:
                # Recovery mode: increase LR periodically
                epochs_in_recovery = epoch - self.conv_start_epoch
                if epochs_in_recovery > 0 and epochs_in_recovery % self.recovery_interval == 0:
                    self._recovery_lr = min(
                        self._recovery_lr * self.recovery_multiplier,
                        self.max_lr
                    )
                    self._in_recovery = True
                lr = self._recovery_lr

        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self._current_lr = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._current_lr

    def is_in_recovery(self) -> bool:
        """Check if in recovery mode."""
        return self._in_recovery


class TemperatureScaledModel(nn.Module):
    """Wrapper that adds temperature scaling to any model."""

    def __init__(self, base_model: nn.Module, initial_temp: float = 1.0, learnable: bool = False):
        """
        Args:
            base_model: The underlying model
            initial_temp: Initial temperature value
            learnable: If True, temperature is a learnable parameter
        """
        super().__init__()
        self.base_model = base_model
        self.learnable = learnable

        if learnable:
            self.temperature = nn.Parameter(torch.tensor([initial_temp]))
        else:
            self.register_buffer('temperature', torch.tensor([initial_temp]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling."""
        logits = self.base_model(x)
        # Scale logits by temperature (lower temp = sharper predictions)
        temp = self.temperature.clamp(min=0.1)
        return logits / temp

    def set_temperature(self, temp: float):
        """Set temperature value."""
        if self.learnable:
            self.temperature.data.fill_(temp)
        else:
            self.temperature.fill_(temp)

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


