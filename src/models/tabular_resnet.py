"""Residual Network for tabular data.

ResNet architecture adapted for tabular data with residual connections.
Based on "Deep Residual Learning for Image Recognition" (He et al., 2016)
adapted for tabular feature spaces.
"""

from typing import List
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with skip connection for tabular data."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input [batch_size, dim]

        Returns:
            Output [batch_size, dim]
        """
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out


class TabularResNet(nn.Module):
    """Residual Network for tabular data.

    Architecture:
    - Initial projection to hidden dimension
    - Multiple residual blocks with skip connections
    - Optional dimension reduction between blocks
    - Final classification layer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 128, 128],
        n_classes: int = 3,
        dropout: float = 0.3,
        n_blocks_per_stage: int = 2
    ):
        """Initialize TabularResNet.

        Args:
            input_dim: Number of input features
            hidden_dims: Dimensions for each stage (transitions between stages reduce dim)
            n_classes: Number of output classes
            dropout: Dropout probability
            n_blocks_per_stage: Number of residual blocks per stage
        """
        super().__init__()

        # Initial projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Build residual stages
        self.stages = nn.ModuleList()
        prev_dim = hidden_dims[0]

        for i, dim in enumerate(hidden_dims):
            # Dimension transition if needed
            if dim != prev_dim:
                transition = nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                self.stages.append(transition)

            # Residual blocks at this dimension
            for _ in range(n_blocks_per_stage):
                self.stages.append(ResidualBlock(dim, dropout))

            prev_dim = dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits [batch_size, n_classes]
        """
        x = self.input_layer(x)

        for stage in self.stages:
            x = stage(x)

        x = self.output_layer(x)
        return x
