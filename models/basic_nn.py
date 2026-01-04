"""
Basic Neural Network for Student Dropout Prediction
Simple feedforward network with configurable layers
"""
import torch
import torch.nn as nn


class BasicNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], n_classes=3, dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
