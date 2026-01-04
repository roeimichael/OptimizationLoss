from typing import List
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_dim: int, growth_rate: int, dropout: float = 0.3):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, growth_rate),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_dim: int, num_layers: int, growth_rate: int, dropout: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_dim + i * growth_rate, growth_rate, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128], n_classes: int = 3, dropout: float = 0.3, growth_rate: int = 32):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dense_blocks = nn.ModuleList()
        current_dim = hidden_dims[0]
        num_blocks = 3
        layers_per_block = 4
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(current_dim, layers_per_block, growth_rate, dropout))
            current_dim += layers_per_block * growth_rate
            if i < num_blocks - 1:
                self.dense_blocks.append(nn.Sequential(
                    nn.BatchNorm1d(current_dim),
                    nn.ReLU(),
                    nn.Linear(current_dim, current_dim // 2),
                    nn.Dropout(dropout)
                ))
                current_dim = current_dim // 2
        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(current_dim),
            nn.ReLU(),
            nn.Linear(current_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.dense_blocks:
            x = block(x)
        return self.output_layer(x)
