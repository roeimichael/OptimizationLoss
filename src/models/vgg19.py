from typing import List
import torch
import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256, 128, 128, 64, 64], n_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
