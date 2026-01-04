"""
VGG19-inspired architecture for Student Dropout Prediction
Adapted for tabular data with deep sequential layers
"""
import torch
import torch.nn as nn


class VGG19(nn.Module):
    """
    VGG19-inspired network adapted for tabular data
    Deep network with small layers stacked sequentially
    """
    def __init__(self, input_dim, hidden_dims=[256, 256, 128, 128, 64, 64], n_classes=3, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build deep sequential network
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
