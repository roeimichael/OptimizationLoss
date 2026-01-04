"""
ResNet56-inspired architecture for Student Dropout Prediction
Adapted for tabular data with residual connections
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        out += residual  # Skip connection
        out = self.relu(out)

        return out


class ResNet56(nn.Module):
    """
    ResNet56-inspired network adapted for tabular data
    Uses residual blocks with skip connections
    """
    def __init__(self, input_dim, hidden_dims=[256, 256, 128, 128], n_classes=3, dropout=0.3):
        super().__init__()

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i + 1]:
                # Same dimension - use residual block
                self.residual_blocks.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                # Different dimensions - use projection
                self.residual_blocks.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ))

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], n_classes)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)
