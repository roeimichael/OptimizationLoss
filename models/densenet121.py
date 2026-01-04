"""
DenseNet121-inspired architecture for Student Dropout Prediction
Adapted for tabular data with dense connections
"""
import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    def __init__(self, in_dim, growth_rate, dropout=0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, growth_rate),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_dim, num_layers, growth_rate, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(DenseLayer(in_dim + i * growth_rate, growth_rate, dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DenseNet121(nn.Module):
    """
    DenseNet121-inspired network adapted for tabular data
    Uses dense connections where each layer receives inputs from all previous layers
    """
    def __init__(self, input_dim, hidden_dims=[128], n_classes=3, dropout=0.3, growth_rate=32):
        super().__init__()

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Dense blocks
        self.dense_blocks = nn.ModuleList()
        current_dim = hidden_dims[0]

        # Create 3 dense blocks with transitions
        num_blocks = 3
        layers_per_block = 4

        for i in range(num_blocks):
            # Dense block
            self.dense_blocks.append(DenseBlock(current_dim, layers_per_block, growth_rate, dropout))
            current_dim += layers_per_block * growth_rate

            # Transition layer (compression)
            if i < num_blocks - 1:
                self.dense_blocks.append(nn.Sequential(
                    nn.BatchNorm1d(current_dim),
                    nn.ReLU(),
                    nn.Linear(current_dim, current_dim // 2),
                    nn.Dropout(dropout)
                ))
                current_dim = current_dim // 2

        # Output layer
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(current_dim),
            nn.ReLU(),
            nn.Linear(current_dim, n_classes)
        )

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.dense_blocks:
            x = block(x)

        return self.output_layer(x)
