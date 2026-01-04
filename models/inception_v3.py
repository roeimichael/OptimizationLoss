"""
InceptionV3-inspired architecture for Student Dropout Prediction
Adapted for tabular data with multi-scale feature extraction
"""
import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_dim, dropout=0.3):
        super().__init__()

        # Multiple parallel pathways with different scales
        self.branch1 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.branch2 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.branch3 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.branch4 = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.BatchNorm1d(in_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class InceptionV3(nn.Module):
    """
    InceptionV3-inspired network adapted for tabular data
    Uses parallel pathways to capture features at multiple scales
    """
    def __init__(self, input_dim, hidden_dims=[256, 256], n_classes=3, dropout=0.3):
        super().__init__()

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Inception modules
        self.inception_modules = nn.ModuleList()
        for dim in hidden_dims:
            self.inception_modules.append(InceptionModule(dim, dropout))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x):
        x = self.input_layer(x)

        for module in self.inception_modules:
            x = module(x)

        return self.output_layer(x)
