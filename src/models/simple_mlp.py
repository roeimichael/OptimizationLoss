"""Simple Multi-Layer Perceptron for tabular data.

A clean baseline feedforward neural network with configurable depth and width.
"""

from typing import List
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron baseline.

    A straightforward feedforward network with:
    - Configurable hidden layers
    - Batch normalization for stable training
    - ReLU activations
    - Dropout for regularization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        n_classes: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation - logits)
        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Logits [batch_size, n_classes]
        """
        return self.network(x)
